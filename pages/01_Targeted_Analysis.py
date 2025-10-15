from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from bokeh.models import ColumnDataSource, HoverTool

from orb_analysis import (
    SESSIONS,
    DOWN_TARGETS,
    UP_TARGETS,
    compute_orb_outcomes,
    compute_pullback_trades,
    feature_lift_summary,
    identify_best_target,
    summarise_target_hits,
    summarise_target_vs_stop,
)
from orb_data import ChartBackendError, make_bokeh_candlestick, prepare_candlestick_frame
from orb_data.figure import make_orb_figure

SESSION_NAMES = SESSIONS.copy()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _require_dataset() -> pd.DataFrame:
    if "orb_df" not in st.session_state:
        st.info("Fetch data from the main ORB Data Viewer page before running the analysis.")
        st.stop()
    return st.session_state["orb_df"]


def _dataset_signature() -> object:
    return st.session_state.get("orb_dataset_key")


def _ensure_utc(ts: object) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    parsed = pd.to_datetime(ts, errors="coerce")
    if parsed is None or pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def _ensure_naive(ts: object) -> Optional[pd.Timestamp]:
    parsed = _ensure_utc(ts)
    if parsed is None:
        return None
    return parsed.tz_convert("UTC").tz_localize(None)


def _available_symbols(frame: pd.DataFrame) -> List[str]:
    if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        return sorted(frame.index.get_level_values("symbol").unique())
    if "symbol" in frame.columns:
        return sorted(frame["symbol"].unique())
    return []


def _slice_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.index, pd.MultiIndex) and "symbol" in data.index.names:
        try:
            data = data.xs(symbol, level="symbol")
        except KeyError:
            return pd.DataFrame()
    elif "symbol" in data.columns:
        data = data[data["symbol"] == symbol]
    return data.copy()


def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = pd.DatetimeIndex(idx, tz="UTC")
        df.index = idx
    else:
        idx = idx.tz_convert("UTC")
        df.index = idx
    if "time_utc" not in df.columns:
        df["time_utc"] = idx
    if "time_utc_plus4" in df.columns:
        local_series = pd.to_datetime(df["time_utc_plus4"])
        if getattr(local_series.dtype, "tz", None) is None:
            local_series = local_series.dt.tz_localize("Etc/GMT-4")
        else:
            local_series = local_series.dt.tz_convert("Etc/GMT-4")
    else:
        local_series = idx.tz_convert("Etc/GMT-4")
    df["time_utc_plus4"] = local_series
    df["time_local"] = local_series
    return df


def _filter_day(frame: pd.DataFrame, symbol: str, day: date) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.index, pd.MultiIndex):
        dup_levels = [lvl for lvl in data.index.names if lvl in data.columns]
        if dup_levels:
            data = data.drop(columns=dup_levels)
        data = data.reset_index()
    else:
        idx_name = data.index.name or "time"
        if idx_name in data.columns:
            data = data.drop(columns=[idx_name])
        data = data.reset_index().rename(columns={idx_name: "time"})

    data["time"] = pd.to_datetime(data["time"])
    if "symbol" not in data.columns:
        if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
            data["symbol"] = frame.index.get_level_values("symbol")
        elif frame.index.name == "symbol":
            data["symbol"] = frame.index
        else:
            data["symbol"] = symbol

    if symbol and symbol != "(all)" and "symbol" in data.columns:
        data = data[data["symbol"] == symbol]
    if data.empty:
        return data

    if "time_utc_plus4" in data.columns:
        local_series = pd.to_datetime(data["time_utc_plus4"]).dt.tz_convert("Etc/GMT-4")
    else:
        if pd.api.types.is_datetime64tz_dtype(data["time"]):
            local_series = data["time"].dt.tz_convert("Etc/GMT-4")
        else:
            local_series = data["time"].dt.tz_localize("UTC").dt.tz_convert("Etc/GMT-4")

    start = pd.Timestamp(day, tz="Etc/GMT-4")
    end = start + pd.Timedelta(days=1)
    mask = (local_series >= start) & (local_series < end)
    data = data.loc[mask]
    if data.empty:
        return data

    data = data.sort_values("time")
    if "time_utc_plus4" in data.columns:
        data.set_index("time_utc_plus4", inplace=True)
    else:
        data.set_index("time", inplace=True)
    return data


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def _format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    formatted = summary.copy()
    formatted["hit_rate"] = (formatted["hit_rate"] * 100.0).round(1)
    if "timeout_rate" in formatted.columns:
        formatted["timeout_rate"] = (formatted["timeout_rate"] * 100.0).round(1)
    for col in ("ttt_p25", "ttt_p50", "ttt_p75"):
        if col in formatted.columns:
            formatted[col] = formatted[col].astype(float).round(1)
    for col in ("pm_p25", "pm_p50", "pm_p75"):
        if col in formatted.columns:
            formatted[col] = (formatted[col].astype(float) * 100.0).round(2)
    formatted["bullish_close_pct"] = (formatted["bullish_close_pct"] * 100.0).round(1)
    formatted["close_above_mid_pct"] = (formatted["close_above_mid_pct"] * 100.0).round(1)
    formatted.rename(
        columns={
            "target": "Target",
            "direction": "Direction",
            "opportunities": "Opportunities",
            "hits": "Hits",
            "hit_rate": "Hit rate (%)",
            "timeout_rate": "Timeout rate (%)",
            "ttt_p25": "TTT p25 (min)",
            "ttt_p50": "TTT median (min)",
            "ttt_p75": "TTT p75 (min)",
            "bullish_close_pct": "Bullish close (%)",
            "close_above_mid_pct": "Close ≥ mid (%)",
            "avg_body_ratio": "Avg body ratio",
            "avg_upper_wick_ratio": "Avg upper wick ratio",
            "avg_lower_wick_ratio": "Avg lower wick ratio",
            "pm_p25": "%-Move p25 (%)",
            "pm_p50": "%-Move median (%)",
            "pm_p75": "%-Move p75 (%)",
        },
        inplace=True,
    )
    return formatted


def _baseline_hit_table(outcomes: pd.DataFrame, session: str) -> pd.DataFrame:
    data = outcomes[outcomes["session"] == session]
    return _format_summary_table(summarise_target_hits(data))


def _available_mask(outcomes: pd.DataFrame, target: str) -> pd.Series:
    col = f"available_{target}"
    if col not in outcomes.columns:
        return pd.Series(False, index=outcomes.index)
    return outcomes[col].fillna(False)


def _hit_mask(outcomes: pd.DataFrame, target: str, direction: str) -> pd.Series:
    col = f"tt_{'up' if direction=='up' else 'down'}_{target}"
    return outcomes[col].notna() if col in outcomes.columns else pd.Series(False, index=outcomes.index)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _add_marker_layer(fig, points: List[Dict[str, object]], glyph: str, color: str) -> None:
    if not points:
        return
    times: List[pd.Timestamp] = []
    prices: List[float] = []
    sessions: List[str] = []
    minutes: List[str] = []
    for point in points:
        naive = _ensure_naive(point.get("time"))
        if naive is None:
            continue
        times.append(naive)
        prices.append(float(point.get("price", 0.0)))
        sessions.append(str(point.get("session_id") or ""))
        mins = point.get("minutes")
        minutes.append("" if mins is None or pd.isna(mins) else f"{float(mins):.1f}")
    if not times:
        return
    source = ColumnDataSource(data={"time": times, "price": prices, "session": sessions, "minutes": minutes})
    renderer = fig.scatter(
        x="time",
        y="price",
        source=source,
        marker=glyph,
        size=8,
        fill_color=color,
        fill_alpha=0.95,
        line_color="#212121",
        line_width=0.6,
    )
    fig.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("Time", "@time{%Y-%m-%d %H:%M:%S}"),
                ("Price", "@price{0,0.00000}"),
                ("Session ID", "@session"),
                ("Minutes", "@minutes"),
            ],
            formatters={"@time": "datetime"},
            mode="mouse",
        )
    )


def _render_best_strategy_chart(frame: pd.DataFrame, session: str, best_target: Optional[Dict[str, object]]) -> None:
    if best_target is None:
        st.info("Run the analysis to identify the best-performing target.")
        return

    symbols = _available_symbols(frame)
    if not symbols:
        st.info("No symbols available for the best-target view.")
        return

    symbol = st.selectbox("Symbol", symbols, index=0, key="best_symbol")
    day_series = None
    if "time_utc_plus4" in frame.columns:
        day_series = pd.to_datetime(frame["time_utc_plus4"]).dt.normalize()
    elif isinstance(frame.index, pd.MultiIndex) and "time" in frame.index.names:
        day_series = pd.to_datetime(frame.index.get_level_values("time")).tz_convert("UTC").tz_localize(None)
    else:
        idx = pd.to_datetime(frame.index)
        if getattr(idx, "tz", None) is None:
            day_series = idx.tz_localize("UTC").tz_localize(None)
        else:
            day_series = idx.tz_convert("UTC").tz_localize(None)

    if day_series is None or day_series.empty:
        st.info("Unable to determine available days for the dataset.")
        return

    available_days = sorted(day_series.unique())
    day_choice = st.selectbox(
        "Session day (UTC+4)",
        options=available_days,
        format_func=lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d"),
        key="best_day",
    )

    filtered = _filter_day(frame, symbol, pd.Timestamp(day_choice).date())
    if filtered.empty:
        st.info("No data matches the selected day.")
        return

    fig = make_orb_figure(
        filtered,
        symbol=symbol,
        timeframe=st.session_state.get("orb_chart_tf"),
        sessions=[session],
        session_visibility={session: True},
        show_buy_volume=False,
        show_sell_volume=False,
    )

    scalar_targets = {"prev_day_high", "prev_day_low", "prev_week_high", "prev_week_low"}
    column_name = best_target["target"] if best_target["target"] in scalar_targets else f"{best_target['target']}_{session}"
    if column_name in filtered.columns:
        level_series = filtered[column_name].dropna()
        if not level_series.empty:
            level = float(level_series.iloc[-1])
            fig.add_hline(
                y=level,
                line=dict(color="#E53935", width=2, dash="dash"),
                annotation_text=f"Best target: {best_target['target']}",
                annotation_position="top left",
                annotation_font=dict(color="#E53935"),
            )

    st.plotly_chart(fig, use_container_width=True)


def _build_outcome_table(
    outcomes: pd.DataFrame,
    *,
    session: Optional[str],
    symbol: Optional[str],
    target: str,
    direction: str,
) -> pd.DataFrame:
    data = outcomes.copy()
    if session:
        data = data[data["session"] == session]
    if symbol and "symbol" in data.columns:
        data = data[data["symbol"] == symbol]
    if data.empty:
        return pd.DataFrame()
    prefix = "up" if direction == "up" else "down"
    cols = [
        "symbol",
        "session",
        "session_id",
        "orb_time",
        "entry_time",
        "entry_price",
        "orb_high_value",
        "orb_low_value",
        "volume_bin",
        "spread_bin",
        "volume_spread_profile",
        f"target_price_{prefix}_{target}",
        f"available_{target}",
        f"hit_{prefix}_{target}",
        f"hit_time_{prefix}_{target}",
        f"hit_price_{prefix}_{target}",
        f"tt_{prefix}_{target}",
        f"pm_{prefix}_{target}",
        f"first_outcome_{prefix}_{target}",
        f"first_minutes_{prefix}_{target}",
        f"first_return_{prefix}_{target}",
        f"first_event_time_{prefix}_{target}",
        f"first_event_price_{prefix}_{target}",
    ]
    present_cols = [col for col in cols if col in data.columns]
    if not present_cols:
        return pd.DataFrame()
    table = data[present_cols].copy()
    rename = {
        f"target_price_{prefix}_{target}": "target_price",
        f"available_{target}": "target_available",
        f"hit_{prefix}_{target}": "target_hit",
        f"hit_time_{prefix}_{target}": "target_hit_time",
        f"hit_price_{prefix}_{target}": "target_hit_price",
        f"tt_{prefix}_{target}": "time_to_target_min",
        f"pm_{prefix}_{target}": "target_move_pct",
        f"first_outcome_{prefix}_{target}": "first_event",
        f"first_minutes_{prefix}_{target}": "first_event_minutes",
        f"first_return_{prefix}_{target}": "first_event_return_pct",
        f"first_event_time_{prefix}_{target}": "first_event_time",
        f"first_event_price_{prefix}_{target}": "first_event_price",
    }
    table.rename(columns={k: v for k, v in rename.items() if k in table.columns}, inplace=True)
    sort_keys = [col for col in ["symbol", "session", "orb_time"] if col in table.columns]
    if sort_keys:
        table.sort_values(sort_keys, inplace=True, ignore_index=True)
    return table


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Targeted ORB Analysis")

    frame = _require_dataset()
    signature = _dataset_signature()
    cached = st.session_state.get("targeted_analysis_results")
    previous_mode = cached.get("entry_mode") if cached else None

    entry_options = ["orb_close", "first_outside_close"]
    default_index = entry_options.index(previous_mode) if previous_mode in entry_options else 0
    entry_mode = st.radio(
        "Entry mode",
        options=entry_options,
        index=default_index,
        format_func=lambda s: "ORB close" if s == "orb_close" else "First candle outside ORB (close)",
        horizontal=True,
    )

    if st.button("Run targeted analysis", type="primary"):
        with st.spinner("Processing session outcomes..."):
            outcomes = compute_orb_outcomes(frame, entry_mode=entry_mode)
        if outcomes.empty:
            st.warning("Unable to derive ORB session outcomes from the current dataset.")
            st.session_state.pop("targeted_analysis_results", None)
            st.stop()
        st.session_state["targeted_analysis_results"] = {
            "outcomes": outcomes,
            "summary": summarise_target_hits(outcomes),
            "signature": signature,
            "entry_mode": entry_mode,
        }
        cached = st.session_state["targeted_analysis_results"]

    if not cached or cached.get("signature") != signature:
        st.info("Execute the analysis to view hit statistics and visualisations.")
        st.stop()

    outcomes: pd.DataFrame = cached["outcomes"]

    session_options = sorted(outcomes["session"].dropna().unique())
    if not session_options:
        st.warning("No ORB sessions found in the cached dataset.")
        st.stop()

    selected_session = st.selectbox("Session", session_options, index=0)
    st.markdown("---")

    st.subheader(f"Baseline hit rates ({selected_session.upper()} session)")
    st.dataframe(_baseline_hit_table(outcomes, selected_session), use_container_width=True)

    st.markdown("---")
    st.subheader("Visualize target outcomes on the candlestick chart")

    viz_cols = st.columns(2)
    with viz_cols[0]:
        viz_session = st.selectbox(
            "Session for markers",
            session_options,
            index=session_options.index(selected_session),
            key="viz_session_select",
        )
    scenario_map: Dict[str, Tuple[str, str]] = {}
    for target in UP_TARGETS:
        scenario_map[f"{target} (long)"] = (target, "up")
    for target in DOWN_TARGETS:
        scenario_map[f"{target} (short)"] = (target, "down")
    scenario_labels = list(scenario_map.keys())
    default_label = "L1_bull (long)" if "L1_bull (long)" in scenario_labels else scenario_labels[0]
    with viz_cols[1]:
        scenario_label = st.selectbox(
            "Target & direction",
            scenario_labels,
            index=scenario_labels.index(default_label),
            key="viz_target_direction",
        )
    selected_target, selected_direction = scenario_map[scenario_label]

    symbol_choices = st.session_state.get("orb_symbols") or _available_symbols(frame)
    if not symbol_choices:
        st.info("No symbols available for the current dataset.")
        st.stop()
    symbol_choice = st.selectbox("Symbol", symbol_choices, index=0, key="viz_symbol_select")

    scenario_df = outcomes[outcomes["session"] == viz_session].copy()
    if "symbol" in scenario_df.columns:
        scenario_df = scenario_df[scenario_df["symbol"] == symbol_choice]
    available_mask = _available_mask(scenario_df, selected_target)
    scenario_df = scenario_df[available_mask]
    if scenario_df.empty:
        st.info("No opportunities available for the chosen combination.")
    else:
        orb_times = pd.to_datetime(scenario_df["orb_time"], utc=True)
        scenario_df = scenario_df.assign(_orb_time_utc=orb_times)
        orb_dates = orb_times.dt.date
        min_date = orb_dates.min()
        max_date = orb_dates.max()
        date_input = st.date_input(
            "ORB date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="viz_date_range",
        )
        if isinstance(date_input, tuple) and len(date_input) == 2:
            start_date, end_date = date_input
        else:
            st.info("Select both start and end dates to render the chart.")
            st.stop()
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        mask_dates = (orb_dates >= start_date) & (orb_dates <= end_date)
        scenario_df = scenario_df.loc[mask_dates].copy()
        if scenario_df.empty:
            st.info("No opportunities fall inside the selected window.")
        else:
            start_ts = pd.Timestamp(start_date).tz_localize("UTC")
            end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize("UTC")
            symbol_frame = _slice_symbol(frame, symbol_choice)
            if symbol_frame.empty:
                st.info("No candles available for the selected symbol.")
            else:
                idx = pd.to_datetime(symbol_frame.index)
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize("UTC")
                else:
                    idx = idx.tz_convert("UTC")
                symbol_frame.index = idx
                window_frame = symbol_frame[(symbol_frame.index >= start_ts) & (symbol_frame.index <= end_ts)].copy()
                if window_frame.empty:
                    st.info("No candles fall inside the selected window.")
                else:
                    show_buy = st.checkbox("Show buy volume", value=False, key="viz_buy_volume_toggle")
                    show_sell = st.checkbox("Show sell volume", value=False, key="viz_sell_volume_toggle")
                    session_visibility = {name: name == viz_session for name in SESSIONS}
                    try:
                        candle_frame = prepare_candlestick_frame(window_frame)
                        bokeh_fig = make_bokeh_candlestick(
                            candle_frame,
                            title=f"ORB Levels: {symbol_choice} ({st.session_state.get('orb_chart_tf')})",
                            timeframe=st.session_state.get("orb_chart_tf"),
                            sessions=[viz_session],
                            session_visibility=session_visibility,
                            show_sessions=True,
                            show_prev_levels=True,
                            show_buy_volume=show_buy,
                            show_sell_volume=show_sell,
                            show_day_boundaries=True,
                            x_range=(start_ts, end_ts),
                        )
                    except ChartBackendError as exc:
                        st.error(str(exc))
                        bokeh_fig = None

                    if bokeh_fig is not None:
                        prefix = "up" if selected_direction == "up" else "down"
                        outcome_col = f"first_outcome_{prefix}_{selected_target}"
                        event_time_col = f"first_event_time_{prefix}_{selected_target}"
                        event_price_col = f"first_event_price_{prefix}_{selected_target}"
                        first_minutes_col = f"first_minutes_{prefix}_{selected_target}"
                        target_price_col = f"target_price_{prefix}_{selected_target}"

                        target_points: List[Dict[str, object]] = []
                        stop_points: List[Dict[str, object]] = []
                        neutral_points: List[Dict[str, object]] = []

                        for _, row in scenario_df.iterrows():
                            outcome = row.get(outcome_col)
                            event_time = _ensure_utc(row.get(event_time_col))
                            if event_time is None:
                                event_time = _ensure_utc(row.get("_orb_time_utc"))
                            if event_time is None:
                                continue

                            if outcome == "target":
                                price = row.get(event_price_col)
                                if price is None or pd.isna(price):
                                    price = row.get(target_price_col)
                                if price is None or pd.isna(price):
                                    price = row.get("orb_high_value") if selected_direction == "up" else row.get("orb_low_value")
                                if price is None or pd.isna(price):
                                    continue
                                target_points.append(
                                    {
                                        "time": event_time,
                                        "price": float(price),
                                        "session_id": row.get("session_id"),
                                        "minutes": row.get(first_minutes_col),
                                    }
                                )
                            elif outcome == "stop":
                                price = row.get(event_price_col)
                                if price is None or pd.isna(price):
                                    price = row.get("orb_low_value") if selected_direction == "up" else row.get("orb_high_value")
                                if price is None or pd.isna(price):
                                    continue
                                stop_points.append(
                                    {
                                        "time": event_time,
                                        "price": float(price),
                                        "session_id": row.get("session_id"),
                                        "minutes": row.get(first_minutes_col),
                                    }
                                )
                            else:
                                entry_time = _ensure_utc(row.get("entry_time"))
                                if entry_time is None:
                                    entry_time = _ensure_utc(row.get("_orb_time_utc"))
                                price = row.get("entry_price")
                                if price is None or pd.isna(price):
                                    price = row.get("orb_high_value") if selected_direction == "up" else row.get("orb_low_value")
                                if price is None or pd.isna(price):
                                    continue
                                neutral_points.append(
                                    {
                                        "time": entry_time,
                                        "price": float(price),
                                        "session_id": row.get("session_id"),
                                        "minutes": row.get(first_minutes_col),
                                    }
                                )

                        summary_text = (
                            f"Opportunities: {len(scenario_df)} | "
                            f"Target-first: {len(target_points)} | "
                            f"Stop-first: {len(stop_points)} | "
                            f"Neither: {len(neutral_points)}"
                        )

                        _add_marker_layer(bokeh_fig, target_points, "triangle", "#2E7D32")
                        _add_marker_layer(bokeh_fig, stop_points, "inverted_triangle", "#C62828")
                        _add_marker_layer(bokeh_fig, neutral_points, "circle", "#757575")

                        st.caption(summary_text)
                        st.bokeh_chart(bokeh_fig, use_container_width=True)

    st.markdown("---")
    with st.expander("Inspect raw ORB outcome rows", expanded=False):
        session_filter_options = ["(all)"] + session_options
    session_filter = st.selectbox(
        "Session filter",
        session_filter_options,
        index=session_filter_options.index(selected_session),
        key="raw_session_filter",
    )
    symbol_filter_options = ["(all)"] + _available_symbols(outcomes)
    symbol_filter = st.selectbox("Symbol filter", symbol_filter_options, index=0, key="raw_symbol_filter")

    direction_label = st.radio(
        "Direction",
        options=["Long (up targets)", "Short (down targets)"],
        horizontal=True,
        key="raw_direction_radio",
    )
    if direction_label.startswith("Long"):
        direction = "up"
        target_choices = UP_TARGETS
        default_target = "L1_bull" if "L1_bull" in target_choices else target_choices[0]
    else:
        direction = "down"
        target_choices = DOWN_TARGETS
        default_target = "L1_bear" if "L1_bear" in target_choices else target_choices[0]
    target_selection = st.selectbox("Target", target_choices, index=target_choices.index(default_target), key="raw_target_select")

    detail_df = _build_outcome_table(
        outcomes,
        session=None if session_filter == "(all)" else session_filter,
        symbol=None if symbol_filter == "(all)" else symbol_filter,
        target=target_selection,
        direction=direction,
    )
    if detail_df.empty:
        st.info("No outcome rows available for the selected combination.")
    else:
        st.dataframe(detail_df, use_container_width=True)
        csv_bytes = detail_df.to_csv(index=False).encode()
        st.download_button(
            "Download filtered rows",
            data=csv_bytes,
            file_name=f"orb_outcomes_{direction}_{target_selection}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Target-first vs Stop-first and simulated returns")
    session_choice = st.selectbox(
        "Session for simulation",
        options=sorted(outcomes["session"].dropna().unique()),
        index=0,
        key="sim_session_select",
    )
    target_options = [
        ("L1_bull", "up"), ("L2_bull", "up"), ("L3_bull", "up"), ("prev_day_high", "up"), ("prev_week_high", "up"),
        ("L1_bear", "down"), ("L2_bear", "down"), ("L3_bear", "down"), ("prev_day_low", "down"), ("prev_week_low", "down"),
    ]
    target_labels = [f"{name} ({direction})" for name, direction in target_options]
    target_label = st.selectbox("Target (direction)", target_labels, index=0, key="sim_target_select")
    sim_target, sim_direction = target_options[target_labels.index(target_label)]
    st.caption("Entry assumed at ORB close; stop at opposite ORB extremity.")

    stats = summarise_target_vs_stop(outcomes, session=session_choice, target=sim_target, direction=sim_direction)
    if stats["opportunities"] == 0:
        st.info("No opportunities available for the chosen combination.")
    else:
        col_metrics = st.columns(6)
        col_metrics[0].metric("Opportunities", stats["opportunities"])
        col_metrics[1].metric("Target-first", f"{stats['target_first']} ({stats['target_first_pct']*100:.1f}%)")
        col_metrics[2].metric("Stop-first", f"{stats['stop_first']} ({stats['stop_first_pct']*100:.1f}%)")
        col_metrics[3].metric("Neither", f"{stats['neither']} ({stats['neither_pct']*100:.1f}%)")
        col_metrics[4].metric("Avg return", f"{stats['avg_return_pct']*100:.2f}%")
        col_metrics[5].metric("Cumulative", f"{stats['cumulative_return_pct']*100:.2f}%")

        eq = stats["series"]
        if isinstance(eq, pd.DataFrame) and not eq.empty:
            eq_sorted = eq.sort_values("orb_time").copy()
            if "first_return" in eq_sorted.columns:
                if "cum_sum_return" not in eq_sorted.columns:
                    eq_sorted["cum_sum_return"] = eq_sorted["first_return"].cumsum()
                if "cum_comp_return" not in eq_sorted.columns:
                    eq_sorted["cum_comp_return"] = (1.0 + eq_sorted["first_return"]).cumprod() - 1.0
            if not {"cum_sum_return", "cum_comp_return"}.issubset(eq_sorted.columns):
                st.info("Insufficient data to build the equity curve.")
            else:
                eq_melt = eq_sorted.melt(
                    id_vars="orb_time",
                    value_vars=["cum_sum_return", "cum_comp_return"],
                    var_name="series",
                    value_name="value",
                )
                eq_fig = px.line(
                    eq_melt,
                    x="orb_time",
                    y="value",
                    color="series",
                    labels={"value": "Return", "orb_time": "Time", "series": "Series"},
                )
                eq_fig.update_yaxes(tickformat=".2%")
                eq_fig.update_traces(mode="lines")
                st.plotly_chart(eq_fig, use_container_width=True)

    st.markdown("---")
    with st.expander("Inspect raw ORB outcome rows", expanded=False):
        session_filter_options = ["(all)"] + session_options
        session_filter = st.selectbox("Session filter", session_filter_options, index=session_filter_options.index(selected_session))
        symbol_filter_options = ["(all)"] + _available_symbols(outcomes)
        symbol_filter = st.selectbox("Symbol filter", symbol_filter_options, index=0)

        direction_label = st.radio("Direction", options=["Long (up targets)", "Short (down targets)"], horizontal=True)
        if direction_label.startswith("Long"):
            direction = "up"
            target_choices = UP_TARGETS
            default_target = "L1_bull" if "L1_bull" in target_choices else target_choices[0]
        else:
            direction = "down"
            target_choices = DOWN_TARGETS
            default_target = "L1_bear" if "L1_bear" in target_choices else target_choices[0]
        target_selection = st.selectbox("Target", target_choices, index=target_choices.index(default_target))

        detail_df = _build_outcome_table(
            outcomes,
            session=None if session_filter == "(all)" else session_filter,
            symbol=None if symbol_filter == "(all)" else symbol_filter,
            target=target_selection,
            direction=direction,
        )
    if detail_df.empty:
        st.info("No outcome rows available for the selected combination.")
    else:
        st.dataframe(detail_df, use_container_width=True)
        csv_bytes = detail_df.to_csv(index=False).encode()
        st.download_button(
            "Download filtered rows",
            data=csv_bytes,
            file_name=f"orb_outcomes_{direction}_{target_selection}.csv",
            mime="text/csv",
            use_container_width=True,
            key="raw_download_button",
        )

    st.markdown("---")
    st.subheader("Pullback strategy explorer")
    if not session_options:
        st.info("No sessions available for pullback analysis.")
    else:
        pullback_cols = st.columns(3)
        pullback_session = pullback_cols[0].selectbox(
            "Session",
            session_options,
            index=session_options.index(selected_session) if selected_session in session_options else 0,
            key="pullback_session_select",
        )
        target_labels = {
            "ORB high/low": "orb",
            "L1 extension": "L1",
            "L2 extension": "L2",
            "L3 extension": "L3",
        }
        pullback_target_label = pullback_cols[1].selectbox(
            "Target level",
            list(target_labels.keys()),
            index=0,
            key="pullback_target_level_select",
        )
        pullback_df = compute_pullback_trades(
            frame,
            session=pullback_session,
            target_level=target_labels[pullback_target_label],
        )
        if pullback_df.empty:
            st.info("No qualifying pullback trades detected for the selected session.")
        else:
            pullback_summary = (
                pullback_df.groupby("bias", dropna=False)["outcome"]
                .agg(
                    trades="count",
                    target_hits=lambda s: int((s == "target").sum()),
                    stop_hits=lambda s: int((s == "stop").sum()),
                    pending=lambda s: int((s == "none").sum()),
                )
                .reset_index()
            )
            pullback_summary["hit_rate"] = pullback_summary["target_hits"] / pullback_summary["trades"]
            st.dataframe(pullback_summary, use_container_width=True)

            symbols_available = sorted([sym for sym in pullback_df["symbol"].dropna().unique()])
            if not symbols_available:
                st.info("No symbol data available for charting these trades.")
            else:
                pullback_symbol = pullback_cols[2].selectbox(
                    "Symbol",
                    symbols_available,
                    index=0,
                    key="pullback_symbol_select",
                )
                pullback_subset = pullback_df[pullback_df["symbol"] == pullback_symbol].copy()
                if pullback_subset.empty:
                    st.info("No pullback trades for the selected symbol.")
                else:
                    outcomes_list = ["All", "target", "stop", "none"]
                    outcome_choice = st.selectbox(
                        "Outcome filter",
                        outcomes_list,
                        index=0,
                        key="pullback_outcome_filter",
                    )
                    if outcome_choice != "All":
                        pullback_subset = pullback_subset[pullback_subset["outcome"] == outcome_choice].copy()
                    if pullback_subset.empty:
                        st.info("No pullback trades match the selected filters.")
                    else:
                        entry_times = pd.to_datetime(pullback_subset["entry_time"], utc=True)
                        date_default = (entry_times.dt.date.min(), entry_times.dt.date.max())
                        date_range = st.date_input(
                            "Entry date range",
                            value=date_default if date_default[0] is not None else None,
                            min_value=date_default[0],
                            max_value=date_default[1],
                            key="pullback_date_range",
                        )
                        if isinstance(date_range, tuple) and len(date_range) == 2 and date_range[0] and date_range[1]:
                            start_date, end_date = date_range
                            if start_date > end_date:
                                start_date, end_date = end_date, start_date
                            mask_dates = (entry_times.dt.date >= start_date) & (entry_times.dt.date <= end_date)
                            pullback_subset = pullback_subset.loc[mask_dates].copy()
                            entry_times = pd.to_datetime(pullback_subset["entry_time"], utc=True)
                        if pullback_subset.empty:
                            st.info("No trades remain after applying the date filter.")
                        else:
                            volume_cols = st.columns(2)
                            show_buy_volume = volume_cols[0].checkbox(
                                "Show buy volume (pullback)",
                                value=False,
                                key="pullback_show_buy_volume_toggle",
                            )
                            show_sell_volume = volume_cols[1].checkbox(
                                "Show sell volume (pullback)",
                                value=False,
                                key="pullback_show_sell_volume_toggle",
                            )
                            target_points: List[Dict[str, object]] = []
                            stop_points: List[Dict[str, object]] = []
                            pending_points: List[Dict[str, object]] = []
                            for _, trade_row in pullback_subset.iterrows():
                                point = {
                                    "time": trade_row["entry_time"],
                                    "price": trade_row["entry_price"],
                                    "session_id": trade_row.get("session_id"),
                                    "minutes": trade_row.get("minutes_to_outcome"),
                                }
                                if trade_row["outcome"] == "target":
                                    target_points.append(point)
                                elif trade_row["outcome"] == "stop":
                                    stop_points.append(point)
                                else:
                                    pending_points.append(point)

                            symbol_frame = _slice_symbol(frame, pullback_symbol)
                            symbol_frame = _ensure_time_columns(symbol_frame)
                            candle_frame = prepare_candlestick_frame(symbol_frame)
                            chart_tf_value = st.session_state.get("orb_chart_tf", "15m")
                            bias_session = [pullback_session]
                            bokeh_fig = make_bokeh_candlestick(
                                candle_frame,
                                title=f"{pullback_symbol} Pullback Entries",
                                timeframe=chart_tf_value,
                                sessions=bias_session,
                                session_visibility={name: name in bias_session for name in SESSION_NAMES},
                                show_sessions=True,
                                show_prev_levels=True,
                                show_buy_volume=show_buy_volume,
                                show_sell_volume=show_sell_volume,
                                show_day_boundaries=True,
                                x_range=(entry_times.min(), entry_times.max() + pd.Timedelta(hours=6)) if not entry_times.empty else None,
                            )
                            _add_marker_layer(bokeh_fig, target_points, "triangle", "#2E7D32")
                            _add_marker_layer(bokeh_fig, stop_points, "inverted_triangle", "#C62828")
                            _add_marker_layer(bokeh_fig, pending_points, "circle", "#616161")
                            st.bokeh_chart(bokeh_fig, use_container_width=True)

                            trade_csv = pullback_subset.to_csv(index=False).encode()
                            st.download_button(
                                "Download pullback trades",
                                data=trade_csv,
                                file_name=f"pullback_trades_{pullback_session}_{pullback_symbol}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

    st.markdown("---")
    st.subheader("Feature dominance explorer")
    feature_session = st.selectbox(
        "Session for feature analysis",
        session_options,
        index=session_options.index(selected_session) if selected_session in session_options else 0,
        key="feature_session_select",
    )
    feature_direction_label = st.radio(
        "Direction",
        options=["Long (up targets)", "Short (down targets)"],
        horizontal=True,
        key="feature_direction_radio",
    )
    if feature_direction_label.startswith("Long"):
        feature_direction = "up"
        feature_targets = UP_TARGETS
        feature_default = "L1_bull" if "L1_bull" in feature_targets else feature_targets[0]
    else:
        feature_direction = "down"
        feature_targets = DOWN_TARGETS
        feature_default = "L1_bear" if "L1_bear" in feature_targets else feature_targets[0]
    feature_target = st.selectbox(
        "Target",
        feature_targets,
        index=feature_targets.index(feature_default),
        key="feature_target_select",
    )

    feature_result = feature_lift_summary(
        outcomes,
        session=feature_session,
        target=feature_target,
        direction=feature_direction,
    )
    if not feature_result:
        st.info("No opportunities available to compute feature dominance for the selected configuration.")
    else:
        meta_cols = st.columns(3)
        meta_cols[0].metric("Opportunities", feature_result["total_opportunities"])
        meta_cols[1].metric("Hits", feature_result["hits"])
        meta_cols[2].metric("Hit rate", f"{feature_result['hit_rate']*100:.1f}%")

        bool_df = feature_result["boolean"]
        if isinstance(bool_df, pd.DataFrame) and not bool_df.empty:
            st.markdown("**Boolean feature lift**")
            st.dataframe(bool_df, use_container_width=True)

        cat_tables = feature_result["categorical"]
        if cat_tables:
            st.markdown("**Top categorical buckets**")
            for label, table in cat_tables.items():
                if table.empty:
                    continue
                st.caption(label.replace("_", " "))
                display = table.copy()
                display["hit_rate"] = (display["hit_rate"] * 100).round(1)
                st.dataframe(display, use_container_width=True)

        quant_df = feature_result["quantitative"]
        if isinstance(quant_df, pd.DataFrame) and not quant_df.empty:
            st.markdown("**Quantitative feature summary**")
            st.dataframe(quant_df, use_container_width=True)

        feature_csv = pd.concat(
            [df.assign(table=name) for name, df in cat_tables.items() if isinstance(df, pd.DataFrame) and not df.empty],
            ignore_index=True,
        ) if cat_tables else pd.DataFrame()
        export_payload = {
            "boolean": bool_df.to_csv(index=False) if isinstance(bool_df, pd.DataFrame) and not bool_df.empty else "",
            "quantitative": quant_df.to_csv(index=False) if isinstance(quant_df, pd.DataFrame) and not quant_df.empty else "",
            "categorical": feature_csv.to_csv(index=False) if not feature_csv.empty else "",
        }
        combined_bytes = "\n\n".join(
            f"# {name}\n{content}" for name, content in export_payload.items() if content
        ).encode()
        st.download_button(
            "Download feature tables",
            data=combined_bytes,
            file_name=f"feature_dominance_{feature_session}_{feature_target}_{feature_direction}.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
