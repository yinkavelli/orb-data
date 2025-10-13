from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .figure import (
    _iter_session_chunks,
    _session_color,
    _timeframe_to_timedelta,
)

REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close")

# ---------------------------------------------------------------------------
# Compatibility: many plotting libs still reference numpy legacy aliases that
# were removed in numpy 2. We create shims so imports keep working.
# ---------------------------------------------------------------------------
if not hasattr(np, "bool"):  # pragma: no cover - compatibility shim
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "int"):  # pragma: no cover - compatibility shim
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):  # pragma: no cover - compatibility shim
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "complex"):  # pragma: no cover - compatibility shim
    np.complex = complex  # type: ignore[attr-defined]
if not hasattr(np, "bool8"):  # pragma: no cover - compatibility shim
    np.bool8 = np.bool_  # type: ignore[attr-defined]


class ChartBackendError(RuntimeError):
    """Wrapper exception when a chart backend cannot be produced."""


@dataclass(frozen=True)
class CandlestickFrame:
    """Prepared container shared between Streamlit and the Bokeh renderer."""

    price: pd.DataFrame
    full: pd.DataFrame
    bar_width_ms: float
    utc_index: pd.DatetimeIndex


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(33, 150, 243, {alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def prepare_candlestick_frame(frame: pd.DataFrame) -> CandlestickFrame:
    """Return a frame suitable for the richer Bokeh rendering."""
    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in frame.columns]
    if missing:
        raise ChartBackendError(f"Missing required price columns: {', '.join(missing)}")

    working = frame.copy()
    if not isinstance(working.index, pd.DatetimeIndex):
        working.index = pd.to_datetime(working.index)
    working.sort_index(inplace=True)

    idx = pd.DatetimeIndex(working.index)
    if getattr(idx, "tz", None) is None:
        utc_index = idx.tz_localize("UTC")
        time_values = idx
    else:
        utc_index = idx.tz_convert("UTC")
        time_values = idx.tz_localize(None)

    working = working.copy()
    working["time"] = pd.DatetimeIndex(time_values)

    price_columns = [col for col in REQUIRED_PRICE_COLUMNS if col in working.columns]
    price = working[price_columns + ["time"]].copy()
    if "volume" in working.columns:
        price["volume"] = working["volume"].astype(float)

    if len(price) >= 2:
        diffs = price["time"].diff().dropna()
        median_delta = diffs.median()
        if pd.isna(median_delta):
            bar_width = 60_000.0
        else:
            bar_width = max(median_delta.total_seconds() * 1000.0 * 0.4, 1.0)
    else:
        bar_width = 60_000.0

    return CandlestickFrame(
        price=price.reset_index(drop=True),
        full=working.reset_index(drop=True),
        bar_width_ms=bar_width,
        utc_index=utc_index,
    )


def _as_naive(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").tz_localize(None)
        return ts
    return pd.Timestamp(ts)


def make_bokeh_candlestick(
    candles: CandlestickFrame,
    *,
    title: str,
    timeframe: str | None = None,
    sessions: Sequence[str] | None = None,
    session_visibility: Dict[str, bool] | None = None,
    show_sessions: bool = True,
    show_prev_levels: bool = True,
    show_buy_volume: bool = False,
    show_sell_volume: bool = False,
    show_day_boundaries: bool = True,
    x_range: Tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
) -> "Figure":
    try:
        from bokeh.models import (
            BoxAnnotation,
            ColumnDataSource,
            HoverTool,
            LinearAxis,
            NumeralTickFormatter,
            PanTool,
            Range1d,
            Span,
            WheelZoomTool,
        )
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ChartBackendError("Bokeh is not installed. Add 'bokeh' to your environment.") from exc

    price = candles.price.copy()
    full = candles.full.copy()
    if price.empty or full.empty:
        raise ChartBackendError("No price data available to render.")

    visibility = session_visibility or {}
    selected_sessions = set(sessions) if sessions is not None else None

    open_values = price["open"].astype(float).to_numpy()
    high_values = price["high"].astype(float).to_numpy()
    low_values = price["low"].astype(float).to_numpy()
    close_values = price["close"].astype(float).to_numpy()
    upper = np.maximum(open_values, close_values)
    lower = np.minimum(open_values, close_values)

    outline_colors = np.where(close_values >= open_values, "#2e7d32", "#c62828").tolist()
    color_series = full.get("volume_spread_color")
    if color_series is None:
        color_series = pd.Series("#B0BEC5", index=full.index)
    body_colors = color_series.fillna("#B0BEC5").astype(str).tolist()

    pattern_priority = [
        ("is_doji", "Doji"),
        ("is_long_legged_doji", "Long-Legged Doji"),
        ("is_spinning_top", "Spinning Top"),
        ("is_hammer", "Hammer"),
        ("is_inverted_hammer", "Inverted Hammer"),
        ("is_shooting_star", "Shooting Star"),
        ("is_marubozu", "Marubozu"),
    ]

    patterns: list[str] = []
    for _, row in full.iterrows():
        label = None
        for column, name in pattern_priority:
            if column in full.columns and bool(row.get(column, False)):
                label = name
                break
        if label is None:
            direction = row.get("candle_direction")
            if pd.isna(direction):
                direction = 1 if row.get("close", 0.0) >= row.get("open", 0.0) else -1
            if direction > 0:
                label = "Bullish"
            elif direction < 0:
                label = "Bearish"
            else:
                label = "Neutral"
        patterns.append(label)

    vs_profile = full.get("volume_spread_profile")
    if vs_profile is None:
        vs_profile = pd.Series(pd.NA, index=full.index, dtype="string")
    vs_profile = vs_profile.astype("string").fillna("No Volume/Spread Profile")

    volume_series = price.get("volume")
    if volume_series is None:
        volume_series = pd.Series(np.nan, index=price.index, dtype=float)
    volume_delta = full.get("volume_delta", pd.Series(0.0, index=full.index)).astype(float).fillna(0.0)
    buy_share = full.get("volume_buy_share", pd.Series(np.nan, index=full.index))
    buy_share_pct = buy_share.astype(float).mul(100.0).clip(0.0, 100.0).fillna(0.0)

    source = ColumnDataSource(
        data=dict(
            time=price["time"],
            open=open_values,
            high=high_values,
            low=low_values,
            close=close_values,
            upper=upper,
            lower=lower,
            color=body_colors,
            outline=outline_colors,
            volume=volume_series.fillna(0.0).to_numpy(),
            profile=vs_profile.to_list(),
            volume_delta=volume_delta.to_list(),
            buy_share=buy_share_pct.to_list(),
            pattern=patterns,
        )
    )

    tools = "wheel_zoom,pan,reset"
    fig = figure(
        x_axis_type="datetime",
        title=title,
        width=1100,
        height=600,
        tools=tools,
        toolbar_location="above",
        background_fill_color="white",
        border_fill_color="white",
    )
    wheel_tool = fig.select_one(WheelZoomTool)
    if wheel_tool is not None:
        wheel_tool.dimensions = "both"
        fig.toolbar.active_scroll = wheel_tool

    body_renderer = fig.vbar(
        x="time",
        width=candles.bar_width_ms,
        top="upper",
        bottom="lower",
        fill_color="color",
        line_color="outline",
        line_width=1.1,
        source=source,
    )
    fig.segment(
        x0="time",
        y0="low",
        x1="time",
        y1="high",
        line_color="#616161",
        line_width=1.0,
        source=source,
    )

    hover = HoverTool(
        renderers=[body_renderer],
        tooltips=[
            ("Time", "@time{%Y-%m-%d %H:%M:%S}"),
            ("Open", "@open{0,0.00000}"),
            ("High", "@high{0,0.00000}"),
            ("Low", "@low{0,0.00000}"),
            ("Close", "@close{0,0.00000}"),
            ("Volume", "@volume{0,0}"),
            ("Volume-Spread", "@profile"),
            ("\u0394Vol", "@volume_delta{0,0}"),
            ("Buy%", "@buy_share{0.0}%"),
            ("Pattern", "@pattern"),
        ],
        formatters={"@time": "datetime"},
        mode="vline",
    )
    fig.add_tools(hover)

    fig.xaxis.axis_label = "Time"
    fig.yaxis.axis_label = "Price"
    fig.xaxis.axis_label_text_color = "#212121"
    fig.yaxis.axis_label_text_color = "#212121"
    fig.xaxis.major_label_text_color = "#424242"
    fig.yaxis.major_label_text_color = "#424242"
    fig.axis.axis_line_color = "#b0b0b0"
    fig.yaxis.formatter = NumeralTickFormatter(format="0,0.00000")
    fig.toolbar.autohide = True
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.legend.visible = False

    # Exponential moving averages
    if "ema_5" in full.columns:
        ema5 = full[["time", "ema_5"]].dropna()
        if not ema5.empty:
            fig.line(ema5["time"], ema5["ema_5"].astype(float), line_color="#00897B", line_width=1.4)
    if "ema_13" in full.columns:
        ema13 = full[["time", "ema_13"]].dropna()
        if not ema13.empty:
            fig.line(ema13["time"], ema13["ema_13"].astype(float), line_color="#3949AB", line_width=1.4)

    shading_windows: list[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    orb_tf_value = None
    if "orb_base_timeframe" in full.columns:
        non_null = full["orb_base_timeframe"].dropna().unique()
        if len(non_null):
            orb_tf_value = str(non_null[0])
    orb_delta = _timeframe_to_timedelta(orb_tf_value)

    if show_sessions:
        session_cols = [col for col in full.columns if col.startswith("session_id_")]
        for sid_col in session_cols:
            session = sid_col.replace("session_id_", "")
            if selected_sessions is not None and session not in selected_sessions:
                continue
            if not visibility.get(session, True):
                continue

            high_col = f"orb_high_{session}"
            low_col = f"orb_low_{session}"
            mid_col = f"orb_mid_{session}"
            if high_col not in full.columns or low_col not in full.columns:
                continue

            session_color = _session_color(session)
            session_alpha = 0.65

            session_frame = full.set_index("time")
            for chunk in _iter_session_chunks(session_frame, session):
                index = chunk.index
                fig.line(index, chunk[high_col], line_color=session_color, line_width=1.6, alpha=0.9)
                fig.line(index, chunk[low_col], line_color=session_color, line_width=1.2, alpha=session_alpha)
                if mid_col in chunk.columns:
                    fig.line(index, chunk[mid_col], line_color=session_color, line_width=1.2, line_dash="dashed", alpha=session_alpha)

                bull_specs = ["L1_bull", "L2_bull", "L3_bull"]
                bear_specs = ["L1_bear", "L2_bear", "L3_bear"]
                for suffix in bull_specs:
                    col = f"{suffix}_{session}"
                    if col in chunk.columns and chunk[col].notna().any():
                        fig.line(index, chunk[col], line_color=session_color, line_width=1.0, line_dash="dotted", alpha=session_alpha)
                for suffix in bear_specs:
                    col = f"{suffix}_{session}"
                    if col in chunk.columns and chunk[col].notna().any():
                        fig.line(index, chunk[col], line_color=session_color, line_width=1.0, line_dash="dotted", alpha=0.45)

                if (
                    orb_delta is not None
                    and timeframe is not None
                    and orb_tf_value is not None
                    and orb_tf_value != timeframe
                ):
                    is_orb_col = f"is_orb_{session}"
                    if is_orb_col in chunk.columns:
                        orb_rows = chunk.index[chunk[is_orb_col].fillna(False)]
                        if len(orb_rows):
                            start = orb_rows[0]
                            end = start + orb_delta
                            shading_windows.append((start, end, session_color))

    for left, right, color in shading_windows:
        fig.add_layout(
            BoxAnnotation(
                left=left,
                right=right,
                fill_color=_hex_to_rgba(color, 0.18),
                line_alpha=0.0,
                level="underlay",
            )
        )

    if show_prev_levels:
        def _line_from_series(series: Optional[pd.Series], color: str, label: str, width: float = 1.3) -> None:
            if series is None:
                return
            s = pd.Series(series).dropna()
            if s.empty:
                return
            aligned_time = full.loc[s.index, "time"]
            fig.line(aligned_time, s.astype(float), line_color=color, line_width=width)

        _line_from_series(full.get("prev_day_high"), "#FDD835", "Prev Day High")
        _line_from_series(full.get("prev_day_low"), "#FDD835", "Prev Day Low")
        _line_from_series(full.get("prev_week_high"), "#212121", "Prev Week High", width=1.6)
        _line_from_series(full.get("prev_week_low"), "#424242", "Prev Week Low", width=1.6)

    volume_max = 0.0
    buy_volume = pd.Series(dtype=float)
    sell_volume = pd.Series(dtype=float)
    if show_buy_volume and "volume_buy" in full.columns:
        buy_volume = full["volume_buy"].astype(float).fillna(0.0)
        volume_max = max(volume_max, float(buy_volume.abs().max()))
    if show_sell_volume and "volume_sell" in full.columns:
        sell_volume = full["volume_sell"].astype(float).fillna(0.0)
        volume_max = max(volume_max, float(sell_volume.abs().max()))

    if volume_max > 0.0 and (show_buy_volume or show_sell_volume):
        lower = -volume_max * 1.2 if show_sell_volume else 0.0
        upper = volume_max * 1.2
        fig.extra_y_ranges = {"volume": Range1d(start=lower, end=upper)}
        fig.add_layout(LinearAxis(y_range_name="volume", axis_label="Volume", axis_label_text_color="#424242"), "right")
        if show_buy_volume and not buy_volume.empty:
            fig.vbar(
                x=full["time"],
                top=buy_volume,
                width=candles.bar_width_ms,
                fill_color="rgba(46, 125, 50, 0.55)",
                line_color=None,
                alpha=0.6,
                y_range_name="volume",
            )
        if show_sell_volume and not sell_volume.empty:
            fig.vbar(
                x=full["time"],
                top=-sell_volume,
                width=candles.bar_width_ms,
                fill_color="rgba(198, 40, 40, 0.55)",
                line_color=None,
                alpha=0.6,
                y_range_name="volume",
            )

    if show_day_boundaries:
        base_days = pd.DatetimeIndex(sorted(pd.unique(candles.utc_index.normalize())))
        if len(base_days):
            local_starts = base_days + pd.Timedelta(hours=4)
            for ts in local_starts:
                span_time = _as_naive(ts)
                if span_time is not None:
                    fig.add_layout(
                        Span(location=span_time, dimension="height", line_dash="dotted", line_color="#BDBDBD", line_width=1)
                    )

    if x_range is not None:
        start = _as_naive(x_range[0])
        end = _as_naive(x_range[1])
        if start is not None and end is not None and start < end:
            fig.x_range.start = start
            fig.x_range.end = end

    return fig
