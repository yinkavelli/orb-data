from __future__ import annotations

import re
from typing import Iterable, List, Tuple

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
) -> go.Figure:
    data = _select_symbol_frame(frame, symbol).copy()
    if data.empty:
        return go.Figure()

    data = _ensure_datetime_index(data).sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
            increasing_line_color="#2e7d32",
            increasing_fillcolor="rgba(46, 125, 50, 0.55)",
            decreasing_line_color="#c62828",
            decreasing_fillcolor="rgba(198, 40, 40, 0.55)",
            whiskerwidth=0.4,
        )
    )

    if "ema_5" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["ema_5"],
                mode="lines",
                name="EMA 5",
                line=dict(color="#ff6f61", width=1.6),
                hovertemplate="EMA 5: %{y:.2f}<extra></extra>",
            )
        )
    if "ema_13" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["ema_13"],
                mode="lines",
                name="EMA 13",
                line=dict(color="#00acc1", width=1.4),
                hovertemplate="EMA 13: %{y:.2f}<extra></extra>",
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
            if sessions is not None and name not in sessions:
                continue
            high_col = f"orb_high_{name}"
            low_col = f"orb_low_{name}"
            mid_col = f"orb_mid_{name}"
            if high_col not in data.columns or low_col not in data.columns:
                continue
            base_color = _session_color(name)
            show_leg = True
            for chunk in _iter_session_chunks(data, name):
                fig.add_trace(
                    go.Scatter(
                        x=chunk.index,
                        y=chunk[high_col],
                        mode="lines",
                        name=f"{name.upper()} ORB",
                        line=dict(color=base_color, width=1.6),
                        opacity=0.9,
                        legendgroup=f"session_{name}",
                        legendgrouptitle=dict(text=name.upper()) if show_leg else None,
                        showlegend=show_leg,
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
                        legendgroup=f"session_{name}",
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
                            legendgroup=f"session_{name}",
                            showlegend=False,
                            hovertemplate=f"{name.upper()} Mid: %{{y:.2f}}<extra></extra>",
                        )
                    )
                for level_col, label_suffix, dash, opacity in (
                    (f"L1_bull_{name}", "L1+", "dash", 0.55),
                    (f"L2_bull_{name}", "L2+", "dot", 0.45),
                    (f"L3_bull_{name}", "L3+", "dashdot", 0.35),
                    (f"L1_bear_{name}", "L1-", "dash", 0.55),
                    (f"L2_bear_{name}", "L2-", "dot", 0.45),
                    (f"L3_bear_{name}", "L3-", "dashdot", 0.35),
                ):
                    if level_col in chunk.columns and chunk[level_col].notna().any():
                        fig.add_trace(
                            go.Scatter(
                                x=chunk.index,
                                y=chunk[level_col],
                                mode="lines",
                                name=f"{name.upper()} {label_suffix}",
                                line=dict(color=base_color, width=1, dash=dash),
                                opacity=opacity,
                                legendgroup=f"session_{name}",
                                showlegend=show_leg,
                                hovertemplate=f"{name.upper()} {label_suffix}: %{{y:.2f}}<extra></extra>",
                            )
                        )
                        show_leg = False
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
                show_leg = False

    for left, right, color in shading_windows:
        fig.add_vrect(
            x0=left,
            x1=right,
            fillcolor=_hex_to_rgba(color, 0.2),
            layer="below",
            line_width=0,
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
            showgrid=True,
            gridcolor="#ececec",
            gridwidth=1,
            zeroline=False,
            linecolor="#b0b0b0",
            tickfont=dict(color="#424242"),
            title_standoff=24,
        ),
        yaxis=dict(
            title=dict(text="Price", font=dict(color="#212121", size=12)),
            showgrid=True,
            gridcolor="#ececec",
            gridwidth=1,
            zeroline=False,
            linecolor="#b0b0b0",
            tickfont=dict(color="#424242"),
            title_standoff=24,
        ),
        margin=dict(t=70, l=60, r=220, b=110),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        ),
    )
    return fig


__all__ = ["make_orb_figure"]
