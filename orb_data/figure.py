from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np

import pandas as pd
import plotly.graph_objects as go

SESSION_COLOR_MAP = {
    "asia": "#1f77b4",
    "europe": "#ff7f0e",
    "us": "#2ca02c",
    "overnight": "#9467bd",
}

_TIMEFRAME_PATTERN = re.compile(r"^(\d+)([mhd])$")


def _timeframe_to_timedelta(value: str | None) -> pd.Timedelta | None:
    if not value:
        return None
    match = _TIMEFRAME_PATTERN.match(value.strip())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    return None


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(33, 150, 243, {alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _coerce_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.MultiIndex):
        if "time" in frame.index.names:
            frame = frame.droplevel([name for name in frame.index.names if name != "time"])
        else:
            frame = frame.copy()
            frame.index = frame.index.get_level_values(-1)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def _select_symbol_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        try:
            return frame.xs(symbol, level="symbol")
        except KeyError:
            return frame.loc[(symbol, slice(None))]
    if "symbol" in frame.columns:
        return frame[frame["symbol"] == symbol]
    return frame


def _session_color(name: str) -> str:
    return SESSION_COLOR_MAP.get(name, "#1565c0")


def _iter_session_chunks(data: pd.DataFrame, session_name: str) -> Iterable[pd.DataFrame]:
    sid_col = f"session_id_{session_name}"
    if sid_col not in data.columns:
        return []
    return [chunk for key, chunk in data.groupby(sid_col, sort=False) if not pd.isna(key) and not chunk.empty]


def make_orb_figure(
    frame: pd.DataFrame,
    *,
    symbol: str,
    title_prefix: str = "ORB",
    timeframe: str | None = None,
    show_sessions: bool = True,
    sessions: list[str] | None = None,
    x_range: tuple[pd.Timestamp | str | None, pd.Timestamp | str | None] | None = None,
    session_visibility: Dict[str, bool] | None = None,
    show_buy_volume: bool = False,
    show_sell_volume: bool = False,
) -> go.Figure:
    data = _select_symbol_frame(frame, symbol).copy()
    if data.empty:
        return go.Figure()

    data = _ensure_datetime_index(data).sort_index()
    visibility = session_visibility or {}
    selected_sessions = set(sessions) if sessions is not None else None

    fig = go.Figure()

    open_values = data["open"].astype(float).to_numpy()
    high_values = data["high"].astype(float).to_numpy()
    low_values = data["low"].astype(float).to_numpy()
    close_values = data["close"].astype(float).to_numpy()
    volume_values = data.get("volume", pd.Series(np.nan, index=data.index)).astype(float).to_numpy()

    body_base = np.minimum(open_values, close_values)
    body_height = np.abs(close_values - open_values)
    body_height = np.where(body_height == 0, 1e-9, body_height)

    if len(data.index) > 1:
        diffs = np.diff(data.index.view("int64"))
        diffs = diffs[diffs > 0]
        median_ns = float(np.median(diffs)) if diffs.size else float(pd.Timedelta(minutes=1).value)
    else:
        median_ns = float(pd.Timedelta(minutes=1).value)
    bar_width_ms = max(median_ns / 1_000_000.0 * 0.4, 1.0)
    body_widths = np.full(len(data.index), bar_width_ms)
    body_widths_list = body_widths.tolist()

    color_series = data.get("volume_spread_color")
    if color_series is None:
        color_series = pd.Series("#B0BEC5", index=data.index)
    body_colors = color_series.fillna("#B0BEC5").astype(str).tolist()

    outline_colors = np.where(close_values >= open_values, "#2e7d32", "#c62828")
    outline_colors_list = outline_colors.tolist()

    pattern_priority: List[Tuple[str, str]] = [
        ("is_doji", "Doji"),
        ("is_long_legged_doji", "Long-Legged Doji"),
        ("is_spinning_top", "Spinning Top"),
        ("is_hammer", "Hammer"),
        ("is_inverted_hammer", "Inverted Hammer"),
        ("is_shooting_star", "Shooting Star"),
        ("is_marubozu", "Marubozu"),
    ]

    patterns: list[str] = []
    for idx, row in data.iterrows():
        label = None
        for column, name in pattern_priority:
            if column in data.columns and bool(row.get(column, False)):
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

    vs_profile = data.get("volume_spread_profile")
    if vs_profile is None:
        vs_profile = pd.Series(pd.NA, index=data.index, dtype="string")
    vs_profile = vs_profile.astype("string").fillna("No Volume/Spread Profile")

    volume_delta = data.get("volume_delta")
    if volume_delta is None:
        volume_delta = pd.Series(0.0, index=data.index, dtype=float)
    volume_delta = volume_delta.astype(float).fillna(0.0)

    buy_share = data.get("volume_buy_share")
    if buy_share is None:
        buy_share = pd.Series(np.nan, index=data.index, dtype=float)
    buy_share_pct = buy_share.astype(float).mul(100.0).clip(lower=0.0, upper=100.0).fillna(0.0)

    customdata = np.array(
        list(
            zip(
                open_values,
                high_values,
                low_values,
                close_values,
                volume_values,
                vs_profile.to_numpy(dtype=object),
                volume_delta.to_numpy(),
                buy_share_pct.to_numpy(),
                patterns,
            )
        )
    )

    body_hover = (
        "<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>"
        "Open: %{customdata[0]:,.2f}<br>"
        "High: %{customdata[1]:,.2f}<br>"
        "Low: %{customdata[2]:,.2f}<br>"
        "Close: %{customdata[3]:,.2f}<br>"
        "Volume: %{customdata[4]:,.0f}<br>"
        "Volume\u2013Spread: %{customdata[5]}<br>"
        "\u0394Vol: %{customdata[6]:,.0f}<br>"
        "Buy%: %{customdata[7]:.1f}%<br>"
        "Pattern: %{customdata[8]}<extra></extra>"
    )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=body_height,
            base=body_base,
            width=body_widths_list,
            marker=dict(
                color=body_colors,
                line=dict(color=outline_colors_list, width=1.5),
            ),
            offsetgroup="candles",
            showlegend=False,
            hovertemplate=body_hover,
            customdata=customdata,
            name="Price",
        )
    )

    wick_x: list[pd.Timestamp | None] = []
    wick_y: list[float | None] = []
    for ts, high, low in zip(data.index, high_values, low_values):
        wick_x.extend([ts, ts, None])
        wick_y.extend([low, high, None])

    fig.add_trace(
        go.Scatter(
            x=wick_x,
            y=wick_y,
            mode="lines",
            line=dict(color="#616161", width=1.0),
            showlegend=False,
            hoverinfo="skip",
            name="Wicks",
        )
    )

    shading_windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    orb_tf_value = None
    if "orb_base_timeframe" in data.columns:
        non_null = data["orb_base_timeframe"].dropna().unique()
        if len(non_null):
            orb_tf_value = str(non_null[0])
    orb_delta = _timeframe_to_timedelta(orb_tf_value)
    chart_tf_value = timeframe

    if show_sessions:
        session_ids = [col for col in data.columns if col.startswith("session_id_")]
        for sid in session_ids:
            name = sid[len("session_id_"):]
            if selected_sessions is not None and name not in selected_sessions:
                continue
            if not visibility.get(name, True):
                continue
            high_col = f"orb_high_{name}"
            low_col = f"orb_low_{name}"
            mid_col = f"orb_mid_{name}"
            if high_col not in data.columns or low_col not in data.columns:
                continue
            base_color = _session_color(name)
            for chunk in _iter_session_chunks(data, name):
                fig.add_trace(
                    go.Scatter(
                        x=chunk.index,
                        y=chunk[high_col],
                        mode="lines",
                        name=f"{name.upper()} ORB",
                        line=dict(color=base_color, width=1.6),
                        opacity=0.9,
                        showlegend=False,
                        hovertemplate=f"{name.upper()} High: %{{y:.2f}}<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=chunk.index,
                        y=chunk[low_col],
                        mode="lines",
                        name=f"{name.upper()} ORB Low",
                        line=dict(color=base_color, width=1.2, dash="dot"),
                        opacity=0.65,
                        showlegend=False,
                        hovertemplate=f"{name.upper()} Low: %{{y:.2f}}<extra></extra>",
                    )
                )
                if mid_col in chunk.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chunk.index,
                            y=chunk[mid_col],
                            mode="lines",
                            name=f"{name.upper()} ORB Mid",
                            line=dict(color=base_color, width=1.2, dash="dash"),
                            opacity=0.65,
                            showlegend=False,
                            hovertemplate=f"{name.upper()} Mid: %{{y:.2f}}<extra></extra>",
                        )
                    )
                if (
                    orb_delta is not None
                    and chart_tf_value
                    and orb_tf_value
                    and orb_tf_value != chart_tf_value
                ):
                    is_orb_col = f"is_orb_{name}"
                    if is_orb_col in chunk.columns:
                        orb_rows = chunk.index[chunk[is_orb_col].fillna(False)]
                        if len(orb_rows):
                            start = orb_rows[0]
                            end = start + orb_delta
                            shading_windows.append((start, end, base_color))

    for left, right, color in shading_windows:
        fig.add_vrect(
            x0=left,
            x1=right,
            fillcolor=_hex_to_rgba(color, 0.2),
            layer="below",
            line_width=0,
        )

    volume_traces_added = False
    if show_buy_volume and "volume_buy" in data.columns:
        buy_volume = data["volume_buy"].astype(float).fillna(0.0)
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=buy_volume,
                customdata=buy_volume.to_numpy(),
                name="Buy Volume",
                marker=dict(color="rgba(46, 125, 50, 0.55)"),
                opacity=0.6,
                yaxis="y2",
                showlegend=False,
                hovertemplate="Buy Volume: %{customdata:,.0f}<extra></extra>",
            )
        )
        volume_traces_added = True

    if show_sell_volume and "volume_sell" in data.columns:
        sell_volume = data["volume_sell"].astype(float).fillna(0.0)
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=-sell_volume,
                customdata=sell_volume.to_numpy(),
                name="Sell Volume",
                marker=dict(color="rgba(198, 40, 40, 0.55)"),
                opacity=0.6,
                yaxis="y2",
                showlegend=False,
                hovertemplate="Sell Volume: %{customdata:,.0f}<extra></extra>",
            )
        )
        volume_traces_added = True

    if volume_traces_added:
        fig.update_layout(
            barmode="relative",
            yaxis2=dict(
                title=dict(text="Volume", font=dict(color="#212121", size=12)),
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=True,
                zerolinecolor="#b0b0b0",
                tickfont=dict(color="#424242"),
            ),
        )

    title = f"{title_prefix}: {symbol}"
    if timeframe:
        title = f"{title_prefix}: {symbol} ({timeframe})"
    fig.update_layout(
        title=dict(text=title, font=dict(color="#212121", size=16)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1f1f1f", size=12),
        xaxis=dict(
            title=dict(text="Time", font=dict(color="#212121", size=12)),
            showgrid=False,
            zeroline=False,
            linecolor="#b0b0b0",
            tickfont=dict(color="#424242"),
            title_standoff=24,
        ),
        yaxis=dict(
            title=dict(text="Price", font=dict(color="#212121", size=12)),
            showgrid=False,
            zeroline=False,
            linecolor="#b0b0b0",
            tickfont=dict(color="#424242"),
            title_standoff=24,
        ),
        margin=dict(t=70, l=60, r=220, b=110),
        xaxis_rangeslider_visible=False,
        hovermode="closest",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    if x_range is not None:
        left = _coerce_timestamp(x_range[0])
        right = _coerce_timestamp(x_range[1])
        if left is not None and right is not None:
            if left > right:
                left, right = right, left
            fig.update_xaxes(range=[left, right])
    return fig


__all__ = ["make_orb_figure"]
