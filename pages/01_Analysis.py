from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

SESSIONS = ["asia", "europe", "us", "overnight"]
UP_TARGETS = ["orb_high", "L1_bull", "L2_bull", "L3_bull", "prev_day_high", "prev_week_high"]
DOWN_TARGETS = ["orb_low", "L1_bear", "L2_bear", "L3_bear", "prev_day_low", "prev_week_low"]


def compute_orb_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"])
    symbol_groups = [(None, data)]
    if "symbol" in data.columns:
        symbol_groups = list(data.groupby("symbol"))

    records: list[dict] = []

    for symbol, symbol_df in symbol_groups:
        for session in SESSIONS:
            sid_col = f"session_id_{session}"
            is_orb_col = f"is_orb_{session}"
            high_col = f"orb_high_{session}"
            low_col = f"orb_low_{session}"
            if sid_col not in symbol_df.columns:
                continue
            session_df = symbol_df[symbol_df[sid_col].notna()].copy()
            if session_df.empty:
                continue
            for sid, grp in session_df.groupby(sid_col):
                grp = grp.sort_values("time")
                orb_window = grp[grp[is_orb_col].fillna(False)]
                if orb_window.empty:
                    continue
                orb_candle = orb_window.iloc[-1]
                post_orb = grp[~grp.index.isin(orb_window.index)].sort_values("time")
                if post_orb.empty:
                    continue
                orb_high = orb_candle.get(high_col)
                orb_low = orb_candle.get(low_col)
                if not (np.isfinite(orb_high) and np.isfinite(orb_low)):
                    continue

                features: dict[str, object] = {
                    "symbol": symbol,
                    "session": session,
                    "session_id": sid,
                    "orb_time": orb_candle.get("time"),
                    "candle_direction": int(orb_candle.get("candle_direction", 0)),
                    "body_ratio": orb_candle.get("body_ratio", np.nan),
                    "upper_wick_ratio": orb_candle.get("upper_wick_ratio", np.nan),
                    "lower_wick_ratio": orb_candle.get("lower_wick_ratio", np.nan),
                    "is_doji": bool(orb_candle.get("is_doji", False)),
                    "is_marubozu": bool(orb_candle.get("is_marubozu", False)),
                    "is_hammer": bool(orb_candle.get("is_hammer", False)),
                    "is_inverted_hammer": bool(orb_candle.get("is_inverted_hammer", False)),
                }

                def _valid_target(value: object) -> float | None:
                    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                        return float(value)
                    return None

                targets_up = {"orb_high": _valid_target(orb_high)}
                targets_down = {"orb_low": _valid_target(orb_low)}
                for idx in range(1, 4):
                    targets_up[f"L{idx}_bull"] = _valid_target(orb_candle.get(f"L{idx}_bull_{session}"))
                    targets_down[f"L{idx}_bear"] = _valid_target(orb_candle.get(f"L{idx}_bear_{session}"))
                targets_up["prev_day_high"] = _valid_target(orb_candle.get("prev_day_high"))
                targets_down["prev_day_low"] = _valid_target(orb_candle.get("prev_day_low"))
                targets_up["prev_week_high"] = _valid_target(orb_candle.get("prev_week_high"))
                targets_down["prev_week_low"] = _valid_target(orb_candle.get("prev_week_low"))

                first_touch_direction: str | None = None
                first_touch_target: str | None = None
                first_touch_time = None

                for name, value in targets_up.items():
                    col = f"tt_up_{name}"
                    if value is None:
                        features[col] = np.nan
                        continue
                    hits = post_orb[post_orb["high"] >= value]
                    if hits.empty:
                        features[col] = np.nan
                    else:
                        hit_time = hits.iloc[0]["time"]
                        delta = (hit_time - orb_candle["time"]).total_seconds() / 60.0
                        features[col] = delta
                        if first_touch_time is None or hit_time < first_touch_time:
                            first_touch_time = hit_time
                            first_touch_direction = "up"
                            first_touch_target = name

                for name, value in targets_down.items():
                    col = f"tt_down_{name}"
                    if value is None:
                        features[col] = np.nan
                        continue
                    hits = post_orb[post_orb["low"] <= value]
                    if hits.empty:
                        features[col] = np.nan
                    else:
                        hit_time = hits.iloc[0]["time"]
                        delta = (hit_time - orb_candle["time"]).total_seconds() / 60.0
                        features[col] = delta
                        if first_touch_time is None or hit_time < first_touch_time:
                            first_touch_time = hit_time
                            first_touch_direction = "down"
                            first_touch_target = name

                features["first_touch_direction"] = first_touch_direction
                features["first_touch_target"] = first_touch_target
                records.append(features)

    outcomes = pd.DataFrame(records)
    return outcomes


def format_percent(df: pd.DataFrame) -> pd.DataFrame:
    return (df * 100).round(1)


st.title("ORB Analysis")

if "orb_df" not in st.session_state:
    st.info("Run a data fetch from the main ORB Data Viewer page to populate the cache.")
    st.stop()

frame = st.session_state["orb_df"]
cache_key = st.session_state.get("orb_cache_key")

if (
    "analysis_outcomes" not in st.session_state
    or st.session_state.get("analysis_cache_key") != cache_key
):
    with st.spinner("Processing ORB statistics..."):
        outcomes = compute_orb_outcomes(frame)
    st.session_state["analysis_outcomes"] = outcomes
    st.session_state["analysis_cache_key"] = cache_key
else:
    outcomes = st.session_state["analysis_outcomes"]

if outcomes.empty:
    st.warning("No ORB sessions found in the cached dataset.")
    st.stop()

st.caption(
    "Statistics below are derived from the cached dataset currently loaded on the main page. "
    "Each session is evaluated from the final ORB candle onward."
)

up_cols = [f"tt_up_{t}" for t in UP_TARGETS]
down_cols = [f"tt_down_{t}" for t in DOWN_TARGETS]
for col in up_cols:
    if col not in outcomes.columns:
        outcomes[col] = np.nan
for col in down_cols:
    if col not in outcomes.columns:
        outcomes[col] = np.nan

for t in UP_TARGETS:
    outcomes[f"hit_up_{t}"] = outcomes[f"tt_up_{t}"] .notna()
for t in DOWN_TARGETS:
    outcomes[f"hit_down_{t}"] = outcomes[f"tt_down_{t}"] .notna()

outcomes["body_bucket"] = pd.cut(
    outcomes["body_ratio"],
    bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
    labels=["small", "medium", "large", "very_large"],
)
outcomes["upper_dominant"] = outcomes["upper_wick_ratio"] > outcomes["lower_wick_ratio"]

# Session + direction summary
direction_cols = {
    "hit_up_orb_high": "Hit ORB High %",
    "hit_up_L1_bull": "Hit L1 Bull %",
    "hit_up_L2_bull": "Hit L2 Bull %",
    "hit_up_prev_day_high": "Hit Prev Day High %",
    "hit_down_orb_low": "Hit ORB Low %",
    "hit_down_L1_bear": "Hit L1 Bear %",
    "hit_down_L2_bear": "Hit L2 Bear %",
    "hit_down_prev_day_low": "Hit Prev Day Low %",
}

direction_summary = (
    outcomes.groupby(["session", "candle_direction"])[list(direction_cols.keys())]
    .mean()
    .rename(columns=direction_cols)
    .pipe(format_percent)
    .reset_index()
)
direction_summary["candle_direction"] = direction_summary["candle_direction"].map(
    {1: "Bullish close", -1: "Bearish close", 0: "Flat"}
)

st.subheader("Target hit probability by session & ORB candle direction")
st.dataframe(direction_summary, use_container_width=True)

# Body-size summary
body_summary = (
    outcomes.groupby("body_bucket")[
        [
            "hit_up_orb_high",
            "hit_up_L1_bull",
            "hit_down_orb_low",
            "hit_down_L1_bear",
        ]
    ]
    .mean()
    .rename(
        columns={
            "hit_up_orb_high": "Hit ORB High %",
            "hit_up_L1_bull": "Hit L1 Bull %",
            "hit_down_orb_low": "Hit ORB Low %",
            "hit_down_L1_bear": "Hit L1 Bear %",
        }
    )
    .pipe(format_percent)
    .reset_index()
)

st.subheader("Influence of ORB body size")
st.dataframe(body_summary, use_container_width=True)

# Wick dominance summary
wick_summary = (
    outcomes.groupby("upper_dominant")[
        ["hit_up_orb_high", "hit_down_orb_low"]
    ]
    .mean()
    .rename(
        columns={
            "hit_up_orb_high": "Hit ORB High %",
            "hit_down_orb_low": "Hit ORB Low %",
        }
    )
    .pipe(format_percent)
    .reset_index()
)
wick_summary["upper_dominant"] = wick_summary["upper_dominant"].map(
    {True: "Upper wick dominates", False: "Lower wick dominates"}
)

st.subheader("Wick dominance – first breakout bias")
st.dataframe(wick_summary, use_container_width=True)

# First-touch distribution
first_touch_counts = outcomes["first_touch_target"].value_counts(dropna=False)
first_touch_table = first_touch_counts.rename_axis("Target").reset_index(name="Count")
st.subheader("First breakout target distribution")
st.dataframe(first_touch_table, use_container_width=True)

ft_direction = (
    outcomes.groupby("candle_direction")["first_touch_direction"].value_counts(normalize=True)
    .unstack(fill_value=0)
    .pipe(format_percent)
    .reset_index()
)
ft_direction["candle_direction"] = ft_direction["candle_direction"].map(
    {1: "Bullish close", -1: "Bearish close", 0: "Flat"}
)
ft_direction = ft_direction.rename(columns={"down": "Down %", "up": "Up %"})

st.dataframe(ft_direction, use_container_width=True)

# Session drill-down
session_choice = st.selectbox(
    "Inspect raw outcomes for session",
    options=sorted(outcomes["session"].unique()),
)
filtered = outcomes[outcomes["session"] == session_choice]
show_cols = [
    "symbol",
    "session_id",
    "candle_direction",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "first_touch_direction",
    "first_touch_target",
    "tt_up_orb_high",
    "tt_down_orb_low",
    "tt_up_prev_day_high",
    "tt_down_prev_day_low",
]
existing_cols = [c for c in show_cols if c in filtered.columns]
st.dataframe(filtered[existing_cols].head(50), use_container_width=True)

st.caption(
    "Tip: use the table above to sanity check individual sessions or export the results "
    "to CSV for deeper offline analysis."
)
