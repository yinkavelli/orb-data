from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import streamlit as st


def _require_dataset() -> pd.DataFrame:
    if "orb_df" not in st.session_state:
        st.info("Fetch data from the main ORB Data Viewer page before running this analysis.")
        st.stop()
    return st.session_state["orb_df"]


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    if isinstance(data.index, pd.MultiIndex):
        index_names = list(data.index.names)
        if "symbol" not in data.columns and "symbol" in index_names:
            data = data.assign(symbol=data.index.get_level_values("symbol"))
        if "time" not in data.columns:
            time_level = index_names[-1] if index_names else None
            times = data.index.get_level_values(time_level) if time_level else data.index.get_level_values(0)
            data = data.assign(time=pd.to_datetime(times, errors="coerce"))
        data = data.reset_index(drop=True)
    else:
        if "symbol" not in data.columns:
            symbols = st.session_state.get("orb_symbols")
            fallback = ["_aggregate_"] * len(data)
            data = data.assign(symbol=symbols if symbols else fallback)
        if "time" not in data.columns:
            data = data.assign(time=pd.to_datetime(data.index, errors="coerce"))
        data = data.reset_index(drop=True)

    data = data.sort_values(["symbol", "time"]).reset_index(drop=True)

    for column in ("open", "close"):
        if column not in data.columns:
            raise ValueError(f"Dataset is missing required column '{column}'.")

    open_prices = pd.to_numeric(data["open"], errors="coerce")
    close_prices = pd.to_numeric(data["close"], errors="coerce")
    data["return_pct"] = (close_prices - open_prices) / open_prices.replace(0, np.nan)
    data["bullish"] = data["return_pct"] >= 0.0

    if "volume_spread_profile" not in data.columns:
        raise ValueError("Dataset does not contain 'volume_spread_profile'. Ensure volume/spread bins were computed upstream.")

    grouped = data.groupby("symbol", sort=False)
    data["vsp_prev1"] = grouped["volume_spread_profile"].shift(1)
    data["vsp_prev2"] = grouped["volume_spread_profile"].shift(2)

    combo = pd.Series(pd.NA, index=data.index, dtype="string")
    valid = data["vsp_prev1"].notna() & data["vsp_prev2"].notna()
    combo.loc[valid] = (
        data.loc[valid, "vsp_prev2"].astype("string").str.strip()
        + " -> "
        + data.loc[valid, "vsp_prev1"].astype("string").str.strip()
    )
    data["vsp_combo_prev2"] = combo.replace("", pd.NA)

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

        returns = group["return_pct"].dropna()
        mean_return = returns.mean() * 100.0 if not returns.empty else np.nan
        median_return = returns.median() * 100.0 if not returns.empty else np.nan

        bull_returns = group.loc[group["bullish"], "return_pct"].dropna()
        bear_returns = group.loc[~group["bullish"], "return_pct"].dropna()

        total_return = (group.loc[group["bullish"], "return_pct"].sum()) * 100.0 if samples else np.nan
        total_loss = (group.loc[~group["bullish"], "return_pct"].sum()) * 100.0 if samples else np.nan

        return pd.Series(
            {
                "Samples": samples,
                "Dominant": dominant,
                "Dominant count": bullish_count if dominant == "Bullish" else bearish_count,
                "Dominant %": dominant_pct,
                "Total Return": total_return,
                "Total Loss": total_loss,
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

    numeric_cols = [
        "Dominant %",
        "Total Return",
        "Total Loss",
        "Avg bull return %",
        "Avg bear return %",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(2)
    summary.reset_index(drop=True, inplace=True)
    summary = summary[
        [
            "Pattern",
            "Dominant",
            "Dominant count",
            "Samples",
            "Dominant %",
            "Total Return",
            "Total Loss",
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

    try:
        frame = _prepare_frame(_require_dataset())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    unique_symbols = sorted(frame["symbol"].dropna().astype("string").unique())
    symbol_options = ["All symbols"] + unique_symbols
    selected_symbols = st.multiselect("Symbols", symbol_options, default=["All symbols"])

    filtered = _filter_symbols(frame, selected_symbols)
    if filtered.empty:
        st.info("No candles remain after applying the symbol filter.")
        st.stop()

    min_samples = st.number_input("Minimum samples per pattern", min_value=5, max_value=500, value=25, step=5)
    sort_choice = st.selectbox(
        "Sort results by",
        options=["Samples", "Dominant %"],
        index=1,
    )
    sort_order = st.radio("Sort order", options=["Descending", "Ascending"], index=0, horizontal=True)
    ascending = sort_order == "Ascending"

    top_n = st.slider("Rows to display", min_value=5, max_value=200, value=100, step=5)

    scenarios = {
        "Scenario 1 - prior candle profile": ("vsp_prev1", "volume_spread_scenario1.csv"),
        "Scenario 2 - prior two-candle combo": ("vsp_combo_prev2", "volume_spread_scenario2.csv"),
    }
    scenario_label = st.selectbox("Scenario", list(scenarios.keys()), index=0)
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
    else:
        st.dataframe(summary.head(top_n), use_container_width=True)
        st.download_button(
            "Download table",
            data=summary.to_csv(index=False).encode(),
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
