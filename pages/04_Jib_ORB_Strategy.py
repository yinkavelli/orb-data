from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from bokeh.models import ColumnDataSource, Span

from orb_analysis import SESSIONS
from orb_data import ChartBackendError, make_bokeh_candlestick, prepare_candlestick_frame

ENTRY_TYPES: Tuple[str, ...] = ("Direct (Candle #2)", "Pullback")


def _require_dataset() -> pd.DataFrame:
    if "orb_df" not in st.session_state:
        st.info("Fetch data from the main ORB Data Viewer page before running this analysis.")
        st.stop()
    return st.session_state["orb_df"]


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "candle_range" not in data.columns:
        data["candle_range"] = data["high"] - data["low"]
    if "upper_wick" not in data.columns:
        data["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
    if "lower_wick" not in data.columns:
        data["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
    safe_range = data["candle_range"].replace(0, np.nan)
    if "close_position_ratio" not in data.columns:
        data["close_position_ratio"] = (data["close"] - data["low"]) / safe_range
    if "upper_wick_ratio" not in data.columns:
        data["upper_wick_ratio"] = data["upper_wick"] / safe_range
    if "lower_wick_ratio" not in data.columns:
        data["lower_wick_ratio"] = data["lower_wick"] / safe_range
    data[["close_position_ratio", "upper_wick_ratio", "lower_wick_ratio"]] = data[
        ["close_position_ratio", "upper_wick_ratio", "lower_wick_ratio"]
    ].fillna(0.0).clip(lower=0.0)
    return data


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.index, pd.MultiIndex):
        duplicate_levels = [name for name in data.index.names if name in data.columns]
        if duplicate_levels:
            data = data.drop(columns=duplicate_levels)
        data = data.reset_index(drop=False)
    else:
        idx_name = data.index.name or "time"
        if idx_name in data.columns:
            data = data.drop(columns=[idx_name])
        data = data.reset_index().rename(columns={idx_name: "time"})

    if "symbol" not in data.columns:
        symbols = st.session_state.get("orb_symbols")
        fallback = ["_aggregate_"] * len(data)
        data["symbol"] = symbols if symbols else fallback

    if "time" not in data.columns:
        data["time"] = pd.to_datetime(data.index, errors="coerce")
    else:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")

    data = data.sort_values(["symbol", "time"]).set_index("time", drop=False)
    return data


@dataclass(frozen=True)
class StrategyConfig:
    min_close_strength: float = 0.6  # 0-1 based on close_position_ratio
    pullback_tolerance: float = 0.2  # fraction of ORB range overshoot allowed


@dataclass
class TradeResult:
    symbol: str
    session: str
    session_id: str
    session_start: pd.Timestamp
    session_end: pd.Timestamp
    entry_type: str
    direction: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target1: float
    target2: float
    exit_time: pd.Timestamp
    exit_price: float
    outcome: str
    r_multiple: float


def _get_session_groups(frame: pd.DataFrame, session: str) -> Dict[str, pd.DataFrame]:
    session_col = f"session_id_{session}"
    groups: Dict[str, pd.DataFrame] = {}
    if session_col not in frame.columns:
        return groups
    for sid, group in frame.groupby(session_col, sort=False):
        if pd.isna(sid) or group.empty:
            continue
        groups[str(sid)] = group.sort_index()
    return groups


def _evaluate_exit(
    future: pd.DataFrame,
    direction: Literal["long", "short"],
    entry_price: float,
    stop_price: float,
    target1: float,
    target2: float,
) -> Tuple[str, pd.Timestamp, float, float]:
    if future.empty:
        return ("NO_HIT", future.index[-1] if len(future.index) else pd.NaT, entry_price, 0.0)
    risk = entry_price - stop_price if direction == "long" else stop_price - entry_price
    risk = float(risk)
    if risk <= 0 or not np.isfinite(risk):
        return ("INVALID", future.index[-1], entry_price, 0.0)

    for idx, row in future.iterrows():
        high = float(row.get("high", np.nan))
        low = float(row.get("low", np.nan))
        if direction == "long":
            if not np.isnan(low) and low <= stop_price:
                return ("LOSS", idx, stop_price, -1.0)
            if not np.isnan(high) and high >= target2:
                r = (target2 - entry_price) / risk
                return ("WIN_T2", idx, target2, r)
            if not np.isnan(high) and high >= target1:
                r = (target1 - entry_price) / risk
                return ("WIN_T1", idx, target1, r)
        else:
            if not np.isnan(high) and high >= stop_price:
                return ("LOSS", idx, stop_price, -1.0)
            if not np.isnan(low) and low <= target2:
                r = (entry_price - target2) / risk
                return ("WIN_T2", idx, target2, r)
            if not np.isnan(low) and low <= target1:
                r = (entry_price - target1) / risk
                return ("WIN_T1", idx, target1, r)

    last_idx = future.index[-1]
    last_close = float(future["close"].iloc[-1])
    if direction == "long":
        r = (last_close - entry_price) / risk
    else:
        r = (entry_price - last_close) / risk
    return ("NO_HIT", last_idx, last_close, r)


def _compute_direct_entries(
    session_df: pd.DataFrame,
    orb_idx: int,
    orb_candle: pd.Series,
    session: str,
    config: StrategyConfig,
    symbol: str,
    session_id: str,
    session_bounds: Tuple[pd.Timestamp, pd.Timestamp],
) -> List[TradeResult]:
    trades: List[TradeResult] = []
    post_df = session_df.iloc[orb_idx + 1 :]
    if post_df.empty:
        return trades

    second = post_df.iloc[0]
    indices = post_df.index
    orb_high = float(orb_candle.get(f"orb_high_{session}", np.nan))
    orb_low = float(orb_candle.get(f"orb_low_{session}", np.nan))
    if np.isnan(orb_high) or np.isnan(orb_low):
        return trades
    orb_range = orb_high - orb_low
    if not np.isfinite(orb_range) or orb_range <= 0:
        return trades

    entry_candidates: List[Tuple[Literal["long", "short"], pd.Series]] = []
    close_ratio = float(second.get("close_position_ratio", np.nan))
    if np.isfinite(second.get("close", np.nan)):
        if second.get("close", np.nan) >= orb_high and close_ratio >= config.min_close_strength:
            entry_candidates.append(("long", second))
        if second.get("close", np.nan) <= orb_low and close_ratio <= (1.0 - config.min_close_strength):
            entry_candidates.append(("short", second))

    for direction, entry_row in entry_candidates:
        entry_time = indices[0]
        entry_price = float(entry_row.get("close", np.nan))
        if not np.isfinite(entry_price):
            continue
        if direction == "long":
            stop = orb_low
            target1 = orb_high + 0.5 * orb_range
            target2 = orb_high + 1.0 * orb_range
            risk = entry_price - stop
        else:
            stop = orb_high
            target1 = orb_low - 0.5 * orb_range
            target2 = orb_low - 1.0 * orb_range
            risk = stop - entry_price

        if risk <= 0 or not np.isfinite(risk):
            continue

        future = post_df.iloc[1:]
        outcome, exit_time, exit_price, r = _evaluate_exit(
            future, direction, entry_price, stop, target1, target2
        )
        trades.append(
            TradeResult(
                symbol=symbol,
                session=session,
                session_id=session_id,
                session_start=session_bounds[0],
                session_end=session_bounds[1],
                entry_type="Direct (Candle #2)",
                direction=direction,
                entry_time=entry_time,
                entry_price=entry_price,
                stop_price=stop,
                target1=target1,
                target2=target2,
                exit_time=exit_time,
                exit_price=exit_price,
                outcome=outcome,
                r_multiple=r,
            )
        )
    return trades


def _compute_pullback_entries(
    session_df: pd.DataFrame,
    orb_idx: int,
    session: str,
    config: StrategyConfig,
    symbol: str,
    session_id: str,
    session_bounds: Tuple[pd.Timestamp, pd.Timestamp],
) -> List[TradeResult]:
    trades: List[TradeResult] = []
    post_df = session_df.iloc[orb_idx + 1 :]
    if len(post_df) < 3:
        return trades
    orb_row = session_df.iloc[orb_idx]

    orb_high = float(orb_row.get(f"orb_high_{session}", np.nan))
    orb_low = float(orb_row.get(f"orb_low_{session}", np.nan))
    if np.isnan(orb_high) or np.isnan(orb_low):
        return trades
    orb_range = orb_high - orb_low
    if orb_range <= 0 or not np.isfinite(orb_range):
        return trades

    breakout_state = {"long": False, "short": False}
    idx_list = list(post_df.index)
    for i in range(len(post_df) - 1):
        current = post_df.iloc[i]
        next_row = post_df.iloc[i + 1]

        high = float(current.get("high", np.nan))
        low = float(current.get("low", np.nan))
        close = float(current.get("close", np.nan))
        open_price = float(current.get("open", np.nan))

        if not np.isnan(close) and close >= orb_high:
            breakout_state["long"] = True
        if not np.isnan(close) and close <= orb_low:
            breakout_state["short"] = True

        tolerance_val = config.pullback_tolerance * orb_range

        # Long pullback
        if breakout_state["long"]:
            overshoot = orb_high - low if not np.isnan(low) else np.nan
            if (
                np.isfinite(overshoot)
                and overshoot >= 0
                and overshoot <= tolerance_val
                and close >= orb_high
                and close > open_price
            ):
                entry_price = float(next_row.get("open", np.nan))
                if not np.isfinite(entry_price):
                    continue
                stop = orb_low
                target1 = orb_high + 0.5 * orb_range
                target2 = orb_high + 1.0 * orb_range
                risk = entry_price - stop
                if risk <= 0 or not np.isfinite(risk):
                    continue
                future = post_df.iloc[i + 2 :]
                outcome, exit_time, exit_price, r = _evaluate_exit(
                    future, "long", entry_price, stop, target1, target2
                )
                trades.append(
                    TradeResult(
                        symbol=symbol,
                        session=session,
                        session_id=session_id,
                        session_start=session_bounds[0],
                        session_end=session_bounds[1],
                        entry_type="Pullback",
                        direction="long",
                        entry_time=idx_list[i + 1],
                        entry_price=entry_price,
                        stop_price=stop,
                        target1=target1,
                        target2=target2,
                        exit_time=exit_time,
                        exit_price=exit_price,
                        outcome=outcome,
                        r_multiple=r,
                )
                )
                breakout_state["long"] = False

        # Short pullback
        if breakout_state["short"]:
            overshoot = low - orb_low if not np.isnan(low) else np.nan
            if (
                np.isfinite(overshoot)
                and overshoot <= 0
                and abs(overshoot) <= tolerance_val
                and close <= orb_low
                and close < open_price
            ):
                entry_price = float(next_row.get("open", np.nan))
                if not np.isfinite(entry_price):
                    continue
                stop = orb_high
                target1 = orb_low - 0.5 * orb_range
                target2 = orb_low - 1.0 * orb_range
                risk = stop - entry_price
                if risk <= 0 or not np.isfinite(risk):
                    continue
                future = post_df.iloc[i + 2 :]
                outcome, exit_time, exit_price, r = _evaluate_exit(
                    future, "short", entry_price, stop, target1, target2
                )
                trades.append(
                    TradeResult(
                        symbol=symbol,
                        session=session,
                        session_id=session_id,
                        session_start=session_bounds[0],
                        session_end=session_bounds[1],
                        entry_type="Pullback",
                        direction="short",
                        entry_time=idx_list[i + 1],
                        entry_price=entry_price,
                        stop_price=stop,
                        target1=target1,
                        target2=target2,
                        exit_time=exit_time,
                        exit_price=exit_price,
                        outcome=outcome,
                        r_multiple=r,
                    )
                )
                breakout_state["short"] = False

    return trades


def compute_trades(frame: pd.DataFrame, session: str, entry_types: List[str], config: StrategyConfig) -> List[TradeResult]:
    trades: List[TradeResult] = []
    session_flag = f"is_orb_{session}"
    frame = frame.copy()

    if session_flag not in frame.columns:
        st.warning(f"The dataset is missing ORB markers for session '{session}'.")
        return trades

    for symbol, symbol_df in frame.groupby("symbol", sort=False):
        symbol_df = symbol_df.sort_index()
        session_groups = _get_session_groups(symbol_df, session)
        for session_id, session_df in session_groups.items():
            if session_df.empty:
                continue
            orb_rows = session_df[session_df[session_flag]]
            if orb_rows.empty:
                continue
            orb_candle = orb_rows.iloc[-1]
            matches = np.where(session_df.index == orb_candle.name)[0]
            if len(matches) == 0:
                continue
            orb_loc = int(matches[-1])
            bounds = (session_df.index.min(), session_df.index.max())

            if "Direct (Candle #2)" in entry_types:
                trades.extend(
                    _compute_direct_entries(
                        session_df,
                        orb_loc,
                        orb_candle,
                        session,
                        config,
                        symbol,
                        session_id,
                        bounds,
                    )
                )
            if "Pullback" in entry_types:
                trades.extend(
                    _compute_pullback_entries(
                        session_df,
                        orb_loc,
                        session,
                        config,
                        symbol,
                        session_id,
                        bounds,
                    )
                )
    return trades


def trades_to_dataframe(trades: List[TradeResult]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    records = [trade.__dict__ for trade in trades]
    df = pd.DataFrame(records)
    df = df.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    return df


def summary_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    pivot = (
        trades.pivot_table(
            index=["entry_type", "direction"],
            columns="outcome",
            values="symbol",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["Total"] = pivot.filter(regex="WIN|LOSS|NO_HIT").sum(axis=1)
    win_cols = [col for col in pivot.columns if col.startswith("WIN")]
    pivot["Win %"] = (
        pivot[win_cols].sum(axis=1) / pivot["Total"] * 100.0
    ).round(2)
    pivot = pivot.sort_values(["entry_type", "direction"])
    return pivot


def _as_naive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts is None or pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def plot_trade(trade: TradeResult, frame: pd.DataFrame, session: str) -> None:
    session_col = f"session_id_{session}"
    symbol_df = frame[frame["symbol"] == trade.symbol]
    session_df = symbol_df[symbol_df[session_col] == trade.session_id]
    if session_df.empty:
        st.info("Session data for the selected trade is unavailable.")
        return
    session_df = session_df.sort_index()

    try:
        candle_frame = prepare_candlestick_frame(session_df)
        fig = make_bokeh_candlestick(
            candle_frame,
            title=f"{trade.symbol} | {trade.entry_type} | {trade.outcome}",
            timeframe=st.session_state.get("orb_chart_tf"),
            sessions=[session],
            session_visibility={name: name == session for name in SESSIONS},
            show_sessions=True,
            show_prev_levels=True,
            show_buy_volume=False,
            show_sell_volume=False,
            show_day_boundaries=True,
        )
    except ChartBackendError as exc:
        st.error(str(exc))
        return

    entry_time = _as_naive(trade.entry_time)
    stop = trade.stop_price
    target1 = trade.target1
    target2 = trade.target2
    color = "#2E7D32" if trade.outcome.startswith("WIN") else "#C62828" if trade.outcome == "LOSS" else "#616161"

    entry_source = ColumnDataSource(
        data={"time": [entry_time], "price": [trade.entry_price], "label": [trade.outcome]}
    )
    fig.scatter("time", "price", source=entry_source, marker="triangle", size=14, color=color, legend_label="Entry")

    stop_span = Span(location=stop, dimension="width", line_color="#C62828", line_dash="dotted", line_width=2)
    t1_span = Span(location=target1, dimension="width", line_color="#388E3C", line_dash="dashed", line_width=1)
    t2_span = Span(location=target2, dimension="width", line_color="#1B5E20", line_dash="dashed", line_width=1)
    fig.add_layout(stop_span)
    fig.add_layout(t1_span)
    fig.add_layout(t2_span)

    st.bokeh_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Jib ORB Strategy Backtest")

    raw = _require_dataset()
    frame = _ensure_features(_prepare_frame(raw))
    session = st.selectbox("Session", SESSIONS, index=SESSIONS.index("us") if "us" in SESSIONS else 0)

    entry_types = st.multiselect(
        "Entry types",
        ENTRY_TYPES,
        default=list(ENTRY_TYPES),
    )
    if not entry_types:
        st.info("Select at least one entry type to evaluate.")
        st.stop()

    min_strength = st.slider(
        "Minimum close strength for direct entries",
        min_value=0.4,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )
    pullback_tol = st.slider(
        "Pullback tolerance (fraction of ORB range)",
        min_value=0.05,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )

    config = StrategyConfig(min_close_strength=min_strength, pullback_tolerance=pullback_tol)

    trades = compute_trades(frame, session=session, entry_types=entry_types, config=config)
    trades_df = trades_to_dataframe(trades)

    if trades_df.empty:
        st.warning("No trades matched the selected criteria.")
        st.stop()

    st.subheader("Summary")
    summary = summary_table(trades_df)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)

    st.subheader("Trades")
    st.dataframe(trades_df, use_container_width=True, height=400)
    st.download_button(
        "Download trades (CSV)",
        data=trades_df.to_csv(index=False).encode(),
        file_name="jib_orb_strategy_trades.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown("---")
    st.subheader("Visualise a Trade")
    trade_options = {
        f"{row.symbol} | {row.entry_type} | {row.entry_time.strftime('%Y-%m-%d %H:%M')} | {row.outcome}": idx
        for idx, row in trades_df.iterrows()
    }
    selected_label = st.selectbox("Select trade", list(trade_options.keys()))
    selected_trade = trades[trade_options[selected_label]]
    plot_trade(selected_trade, frame, session)


if __name__ == "__main__":
    main()
