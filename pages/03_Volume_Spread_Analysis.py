from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from bokeh.models import ColumnDataSource

from orb_data import ChartBackendError, make_bokeh_candlestick, prepare_candlestick_frame


def _require_dataset() -> pd.DataFrame:
    if "orb_df" not in st.session_state:
        st.info("Fetch data from the main ORB Data Viewer page before running this analysis.")
        st.stop()
    return st.session_state["orb_df"]


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

    if "open" not in data.columns or "close" not in data.columns:
        raise ValueError("Dataset is missing required price columns.")

    data = data.sort_values(["symbol", "time"]).reset_index(drop=True)
    return data


def _summary_for_column(
    data: pd.DataFrame,
    column: str,
    *,
    min_samples: int,
    sort_by: str,
    ascending: bool,
) -> pd.DataFrame:
    subset = data[data[column].notna()].copy()
    if subset.empty:
        return pd.DataFrame()

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        samples = len(group)
        bullish_count = int(group["bullish"].sum())
        bearish_count = samples - bullish_count
        bullish_pct = (bullish_count / samples) * 100.0 if samples else np.nan
        bearish_pct = (bearish_count / samples) * 100.0 if samples else np.nan
        dominant = "Bullish" if bullish_pct >= bearish_pct else "Bearish"
        dominant_pct = max(bullish_pct, bearish_pct)

        bull_returns = group.loc[group["bullish"], "return_pct"].dropna()
        bear_returns = group.loc[~group["bullish"], "return_pct"].dropna()

        total_return = bull_returns.sum() * 100.0 if samples else np.nan
        total_loss = bear_returns.sum() * 100.0 if samples else np.nan
        if dominant == "Bullish":
            dominant_pnl = total_return
            opposition_pnl = -total_loss
        else:
            dominant_pnl = -total_loss
            opposition_pnl = total_return
        net_return = dominant_pnl - opposition_pnl

        return pd.Series(
            {
                "Samples": samples,
                "Dominant": dominant,
                "Dominant count": bullish_count if dominant == "Bullish" else bearish_count,
                "Dominant %": dominant_pct,
                "Dominant P&L": dominant_pnl,
                "Opposition P&L": opposition_pnl,
                "Net Return": net_return,
                "Avg bull return %": bull_returns.mean() * 100.0 if not bull_returns.empty else np.nan,
                "Avg bear return %": bear_returns.mean() * 100.0 if not bear_returns.empty else np.nan,
            }
        )

    summary = subset.groupby(column, dropna=False).apply(_aggregate).reset_index()
    summary.rename(columns={column: "Pattern"}, inplace=True)
    summary = summary[summary["Samples"] >= min_samples]
    if summary.empty:
        return summary

    if sort_by in summary.columns:
        summary = summary.sort_values(by=sort_by, ascending=ascending)
    else:
        summary = summary.sort_values(by="Samples", ascending=False)

    numeric_cols = ["Dominant %", "Dominant P&L", "Opposition P&L", "Net Return", "Avg bull return %", "Avg bear return %"]
    summary[numeric_cols] = summary[numeric_cols].round(2)
    summary.reset_index(drop=True, inplace=True)
    summary = summary[
        [
            "Pattern",
            "Dominant",
            "Dominant count",
            "Samples",
            "Dominant %",
            "Dominant P&L",
            "Opposition P&L",
            "Net Return",
            "Avg bull return %",
            "Avg bear return %",
        ]
    ]
    return summary


def _filter_symbols(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    if "symbol" not in data.columns or not symbols:
        return data
    if "All symbols" in symbols:
        return data
    return data[data["symbol"].isin(symbols)].copy()


def main() -> None:
    st.title("Volume-Spread Analysis")

    raw = _require_dataset()
    frame = _prepare_frame(raw)

    open_prices = pd.to_numeric(frame["open"], errors="coerce")
    close_prices = pd.to_numeric(frame["close"], errors="coerce")
    frame["return_pct"] = (close_prices - open_prices) / open_prices.replace(0, np.nan)
    frame["bullish"] = frame["return_pct"] >= 0.0

    if "volume_spread_profile" not in frame.columns:
        raise ValueError("Dataset does not contain 'volume_spread_profile'. Please run the ORB pipeline with volume bins.")

    grouped = frame.groupby("symbol", sort=False)
    frame["vsp_prev1"] = grouped["volume_spread_profile"].shift(1)
    frame["vsp_prev2"] = grouped["volume_spread_profile"].shift(2)
    frame["bullish_prev1"] = grouped["bullish"].shift(1)
    frame["bullish_prev2"] = grouped["bullish"].shift(2)
    combo = pd.Series(pd.NA, index=frame.index, dtype="string")
    valid = frame["vsp_prev1"].notna() & frame["vsp_prev2"].notna()
    prev1_label = pd.Series(pd.NA, index=frame.index, dtype="string")
    prev2_label = pd.Series(pd.NA, index=frame.index, dtype="string")
    prev1_label.loc[frame["vsp_prev1"].notna()] = (
        frame.loc[frame["vsp_prev1"].notna(), "vsp_prev1"].astype("string").str.strip()
        + " ("
        + np.where(
            frame.loc[frame["vsp_prev1"].notna(), "bullish_prev1"].fillna(False),
            "Bullish",
            "Bearish",
        )
        + ")"
    )
    prev2_label.loc[frame["vsp_prev2"].notna()] = (
        frame.loc[frame["vsp_prev2"].notna(), "vsp_prev2"].astype("string").str.strip()
        + " ("
        + np.where(
            frame.loc[frame["vsp_prev2"].notna(), "bullish_prev2"].fillna(False),
            "Bullish",
            "Bearish",
        )
        + ")"
    )
    combo.loc[valid] = prev2_label.loc[valid] + " -> " + prev1_label.loc[valid]
    frame["vsp_prev1_label"] = prev1_label.replace("", pd.NA)
    frame["vsp_prev2_label"] = prev2_label.replace("", pd.NA)
    frame["vsp_combo_prev2_label"] = combo.replace("", pd.NA)

    unique_symbols = sorted(frame["symbol"].dropna().astype("string").unique())
    symbol_options = ["All symbols"] + unique_symbols
    selected_symbols = st.multiselect("Symbols", symbol_options, default=["All symbols"])

    filtered = _filter_symbols(frame, selected_symbols)
    if filtered.empty:
        st.info("No candles remain after applying the symbol filter.")
        st.stop()

    min_samples = st.number_input("Minimum samples per pattern", min_value=5, max_value=500, value=25, step=5)
    sort_choice = st.selectbox("Sort results by", options=["Samples", "Dominant %", "Net Return"], index=2)
    sort_order = st.radio("Sort order", options=["Descending", "Ascending"], index=0, horizontal=True)
    ascending = sort_order == "Ascending"

    top_n = st.slider("Rows to display", min_value=5, max_value=200, value=100, step=5)

    scenarios = {
        "Scenario 1 - prior candle profile": ("vsp_prev1_label", "volume_spread_scenario1.csv"),
        "Scenario 2 - prior two-candle combo": ("vsp_combo_prev2_label", "volume_spread_scenario2.csv"),
    }
    scenario_label = st.selectbox("Scenario", list(scenarios.keys()), index=1)
    column_name, file_name = scenarios[scenario_label]

    summary = _summary_for_column(
        filtered,
        column_name,
        min_samples=int(min_samples),
        sort_by=sort_choice,
        ascending=ascending,
    )
    if summary.empty:
        st.info("No qualifying rows for the selected scenario after applying filters.")
        st.stop()

    summary = summary.head(top_n)
    st.dataframe(summary, use_container_width=True)
    st.download_button(
        "Download table",
        data=summary.to_csv(index=False).encode(),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Visualise pattern occurrences")

    patterns = summary["Pattern"].tolist()
    if not patterns:
        st.info("No patterns available to visualise.")
        st.stop()
    selected_pattern = st.selectbox("Pattern", patterns, index=0)

    pattern_matches = filtered[filtered[column_name] == selected_pattern].copy()
    if pattern_matches.empty:
        st.info("No candles matched the selected pattern within the filtered dataset.")
        st.stop()

    dominant_direction = summary.loc[summary["Pattern"] == selected_pattern, "Dominant"].iloc[0]

    symbol_choice = st.selectbox(
        "Symbol",
        sorted(pattern_matches["symbol"].unique()),
        index=0,
        key="vsp_symbol_select",
    )
    symbol_df = filtered[filtered["symbol"] == symbol_choice].copy()
    symbol_df = symbol_df.sort_values("time").set_index("time")

    try:
        candle_frame = prepare_candlestick_frame(symbol_df)
        fig = make_bokeh_candlestick(
            candle_frame,
            title=f"{symbol_choice} | {selected_pattern}",
            timeframe=st.session_state.get("orb_chart_tf"),
            sessions=[],
            session_visibility={},
            show_sessions=False,
            show_prev_levels=False,
            show_buy_volume=False,
            show_sell_volume=False,
            show_day_boundaries=True,
        )
    except ChartBackendError as exc:
        st.error(str(exc))
        st.stop()

    pattern_symbol_matches = pattern_matches[pattern_matches["symbol"] == symbol_choice].copy()
    pattern_symbol_matches["time"] = pd.to_datetime(pattern_symbol_matches["time"])
    pattern_symbol_matches = pattern_symbol_matches.dropna(subset=["time"])
    pattern_symbol_matches["close"] = pd.to_numeric(pattern_symbol_matches["close"], errors="coerce")
    pattern_symbol_matches = pattern_symbol_matches.dropna(subset=["close"])

    if not pattern_symbol_matches.empty:
        offset_basis = pd.to_numeric(symbol_df["close"], errors="coerce").abs().median()
        price_offset = float(offset_basis) * 0.002 if np.isfinite(offset_basis) else 1.0

        wins_mask = pattern_symbol_matches["bullish"] if dominant_direction == "Bullish" else ~pattern_symbol_matches["bullish"]
        wins_df = pattern_symbol_matches[wins_mask].copy()
        losses_df = pattern_symbol_matches[~wins_mask].copy()

        if dominant_direction == "Bullish":
            wins_df["marker_y"] = pd.to_numeric(wins_df["low"], errors="coerce") - price_offset
            wins_marker = "triangle"
            losses_df["marker_y"] = pd.to_numeric(losses_df["high"], errors="coerce") + price_offset
        else:
            wins_df["marker_y"] = pd.to_numeric(wins_df["high"], errors="coerce") + price_offset
            wins_marker = "inverted_triangle"
            losses_df["marker_y"] = pd.to_numeric(losses_df["low"], errors="coerce") - price_offset

        wins_df = wins_df.dropna(subset=["marker_y"])
        losses_df = losses_df.dropna(subset=["marker_y"])

        if not wins_df.empty:
            win_source = ColumnDataSource({"time": wins_df["time"], "price": wins_df["marker_y"]})
            fig.scatter("time", "price", marker=wins_marker, size=14, color="#2E7D32", source=win_source, legend_label="Dominant win")

        if not losses_df.empty:
            loss_source = ColumnDataSource({"time": losses_df["time"], "price": losses_df["marker_y"]})
            fig.scatter("time", "price", marker="x", size=12, color="#C62828", source=loss_source, legend_label="Loss")

    st.bokeh_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
