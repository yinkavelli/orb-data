from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from orb_analysis import SESSIONS, compute_orb_outcomes


# ---------------------------------------------------------------------------
# Dataset hooks
# ---------------------------------------------------------------------------


def _require_dataset() -> pd.DataFrame:
    if "orb_df" not in st.session_state:
        st.info("Fetch data from the main ORB Data Viewer page before running this analysis.")
        st.stop()
    return st.session_state["orb_df"]


def _dataset_signature() -> object:
    return st.session_state.get("orb_dataset_key")


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    label: str
    mask: pd.Series


UP_TARGETS: Tuple[str, ...] = (
    "orb_high",
    "L1_bull",
    "L2_bull",
    "L3_bull",
    "prev_day_high",
    "prev_week_high",
)
DOWN_TARGETS: Tuple[str, ...] = (
    "orb_low",
    "L1_bear",
    "L2_bear",
    "L3_bear",
    "prev_day_low",
    "prev_week_low",
)

TARGET_DEFINITIONS: List[Tuple[str, str, str]] = [
    ("orb_high", "ORB high", "up"),
    ("L1_bull", "L1", "up"),
    ("L2_bull", "L2", "up"),
    ("L3_bull", "L3", "up"),
    ("prev_day_high", "Prev day high", "up"),
    ("prev_week_high", "Prev week high", "up"),
    ("orb_low", "ORB low", "down"),
    ("L1_bear", "L1", "down"),
    ("L2_bear", "L2", "down"),
    ("L3_bear", "L3", "down"),
    ("prev_day_low", "Prev day low", "down"),
    ("prev_week_low", "Prev week low", "down"),
]


def _boolean_series(series: pd.Series, default: bool = False) -> pd.Series:
    return series.fillna(default).astype(bool)


def _body_quartiles(df: pd.DataFrame) -> tuple[float | None, float | None]:
    body = pd.to_numeric(df.get("body_ratio"), errors="coerce").dropna()
    if body.empty:
        return None, None
    return float(body.quantile(0.25)), float(body.quantile(0.75))


def _volume_spread_conditions(
    df: pd.DataFrame,
    *,
    field: str,
    label_prefix: str,
    max_items: int,
    min_samples: int,
) -> List[Condition]:
    if field not in df.columns:
        return []
    series = df[field].astype("string").fillna("<NA>")
    counts = series.value_counts()
    selected: List[str] = []
    for value, freq in counts.items():
        if freq < min_samples:
            continue
        selected.append(value)
        if len(selected) >= max_items:
            break
    conditions: List[Condition] = []
    for value in selected:
        mask = series == value
        conditions.append(Condition(f"{label_prefix}: {value}", mask))
    return conditions


def _percentile_condition(
    df: pd.DataFrame,
    *,
    column: str,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> Condition | None:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    if series.empty:
        return None
    if lower is not None:
        mask = series >= lower
    elif upper is not None:
        mask = series <= upper
    else:
        return None
    return Condition(label, mask.fillna(False))


def _long_conditions(
    df: pd.DataFrame,
    *,
    min_samples: int,
    max_volume_entries: int,
) -> List[Condition]:
    bullish = _boolean_series(df.get("candle_direction") == 1)
    close_above = _boolean_series(df.get("close_above_mid"))
    doji = _boolean_series(df.get("is_doji"))
    hammer = _boolean_series(df.get("is_hammer"))
    body_q1, _ = _body_quartiles(df)
    bearish = _boolean_series(df.get("candle_direction") == -1)

    body = pd.to_numeric(df.get("body_ratio"), errors="coerce")
    lower_wick = pd.to_numeric(df.get("lower_wick_ratio"), errors="coerce")
    upper_wick = pd.to_numeric(df.get("upper_wick_ratio"), errors="coerce")

    conditions: List[Condition] = [
        Condition("Any ORB candle", pd.Series(True, index=df.index)),
        Condition("Bullish close & close ≥ mid", bullish & close_above),
        Condition("Doji close ≥ mid", doji & close_above),
        Condition("Hammer", hammer),
    ]

    if body_q1 is not None:
        dominance = (lower_wick > upper_wick).fillna(False)
        tiny_body = body.le(body_q1).fillna(False)
        conditions.append(Condition("Tiny body Q1 + lower-wick dominance", tiny_body & dominance))

    conditions.append(Condition("Bearish close below mid (avoid)", bearish & ~close_above))

    vol_high = _percentile_condition(
        df,
        column="volume_percentile",
        label="Volume percentile ≥ 75%",
        lower=0.75,
    )
    vol_low = _percentile_condition(
        df,
        column="volume_percentile",
        label="Volume percentile ≤ 25%",
        upper=0.25,
    )
    spread_wide = _percentile_condition(
        df,
        column="spread_percentile",
        label="Spread percentile ≥ 75%",
        lower=0.75,
    )
    spread_narrow = _percentile_condition(
        df,
        column="spread_percentile",
        label="Spread percentile ≤ 25%",
        upper=0.25,
    )
    for extra in (vol_high, vol_low, spread_wide, spread_narrow):
        if extra is not None:
            conditions.append(extra)

    conditions.extend(
        _volume_spread_conditions(
            df,
            field="volume_bin",
            label_prefix="Volume bin",
            max_items=max_volume_entries,
            min_samples=min_samples,
        )
    )
    conditions.extend(
        _volume_spread_conditions(
            df,
            field="spread_bin",
            label_prefix="Spread bin",
            max_items=max_volume_entries,
            min_samples=min_samples,
        )
    )
    conditions.extend(
        _volume_spread_conditions(
            df,
            field="volume_spread_profile",
            label_prefix="Volume-Spread",
            max_items=max_volume_entries,
            min_samples=min_samples,
        )
    )
    return conditions


def _short_conditions(
    df: pd.DataFrame,
    *,
    min_samples: int,
    max_volume_entries: int,
) -> List[Condition]:
    bearish = _boolean_series(df.get("candle_direction") == -1)
    bullish = _boolean_series(df.get("candle_direction") == 1)
    close_above = _boolean_series(df.get("close_above_mid"))
    doji = _boolean_series(df.get("is_doji"))
    upper_wick = pd.to_numeric(df.get("upper_wick_ratio"), errors="coerce")
    lower_wick = pd.to_numeric(df.get("lower_wick_ratio"), errors="coerce")
    _, body_q3 = _body_quartiles(df)

    body = pd.to_numeric(df.get("body_ratio"), errors="coerce")

    dominance_upper = (upper_wick > lower_wick).fillna(False)

    conditions: List[Condition] = [
        Condition("Any ORB candle", pd.Series(True, index=df.index)),
        Condition("Bearish close & close < mid", bearish & ~close_above),
        Condition("Bearish with upper-wick dominance", bearish & dominance_upper),
        Condition("Bearish doji below mid", doji & ~close_above),
    ]

    if body_q3 is not None:
        large_body = body.ge(body_q3).fillna(False) & bearish
        conditions.append(Condition("Large body Q4 bearish candle", large_body))

    conditions.append(Condition("Bullish close above mid (avoid)", bullish & close_above))

    vol_high = _percentile_condition(
        df,
        column="volume_percentile",
        label="Volume percentile ≥ 75%",
        lower=0.75,
    )
    vol_low = _percentile_condition(
        df,
        column="volume_percentile",
        label="Volume percentile ≤ 25%",
        upper=0.25,
    )
    spread_wide = _percentile_condition(
        df,
        column="spread_percentile",
        label="Spread percentile ≥ 75%",
        lower=0.75,
    )
    spread_narrow = _percentile_condition(
        df,
        column="spread_percentile",
        label="Spread percentile ≤ 25%",
        upper=0.25,
    )
    for extra in (vol_high, vol_low, spread_wide, spread_narrow):
        if extra is not None:
            conditions.append(extra)

    conditions.extend(
        _volume_spread_conditions(
            df,
            field="volume_bin",
            label_prefix="Volume bin",
            max_items=max_volume_entries,
            min_samples=min_samples,
        )
    )
    conditions.extend(
        _volume_spread_conditions(
            df,
            field="spread_bin",
            label_prefix="Spread bin",
            max_items=max_volume_entries,
            min_samples=min_samples,
        )
    )
    conditions.extend(
        _volume_spread_conditions(
            df,
            field="volume_spread_profile",
            label_prefix="Volume-Spread",
            max_items=max_volume_entries,
            min_samples=min_samples,
        )
    )
    return conditions


def _stats_for_condition(
    base: pd.DataFrame,
    mask: pd.Series,
    *,
    target: str,
    direction: str,
) -> Dict[str, object]:
    avail = _boolean_series(base.get(f"available_{target}"))
    aligned_mask = mask.reindex(base.index, fill_value=False)
    subset = base[avail & aligned_mask].copy()
    opportunities = int(len(subset))
    if opportunities == 0:
        return {
            "condition": "",
            "hit_rate": np.nan,
            "hit_rate_pct": np.nan,
            "hits": 0,
            "opportunities": 0,
        }

    tt_col = f"tt_{direction}_{target}"
    hit_mask = subset[tt_col].notna() if tt_col in subset.columns else pd.Series(False, index=subset.index)
    hits = int(hit_mask.sum())
    hit_rate = hits / opportunities if opportunities else np.nan

    first_col = f"first_outcome_{direction}_{target}"
    target_first = subset[first_col].eq("target").sum() if first_col in subset.columns else hits

    return {
        "condition": "",
        "hit_rate": hit_rate,
        "hit_rate_pct": hit_rate * 100.0 if np.isfinite(hit_rate) else np.nan,
        "hits": hits,
        "target_first": int(target_first),
        "opportunities": opportunities,
    }


def _build_condition_table(
    df: pd.DataFrame,
    *,
    session: str,
    target: str,
    direction: str,
    condition_builder: Callable[[pd.DataFrame, int, int], List[Condition]],
    min_samples: int,
    max_volume_entries: int,
) -> pd.DataFrame:
    session_df = df[df["session"] == session].copy()
    if session_df.empty:
        return pd.DataFrame()

    avail_col = f"available_{target}"
    if avail_col not in session_df.columns:
        return pd.DataFrame()

    session_df = session_df[_boolean_series(session_df[avail_col])].copy()
    if session_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    conditions = condition_builder(
        session_df,
        min_samples=min_samples,
        max_volume_entries=max_volume_entries,
    )
    for cond in conditions:
        stats = _stats_for_condition(session_df, cond.mask, target=target, direction=direction)
        stats["condition"] = cond.label
        rows.append(stats)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table[["condition", "opportunities", "hits", "target_first", "hit_rate_pct", "hit_rate"]]
    table.rename(
        columns={
            "condition": "Condition",
            "opportunities": "Samples",
            "hits": "Hits",
            "target_first": "Target-first",
            "hit_rate_pct": "Hit %",
            "hit_rate": "Hit rate",
        },
        inplace=True,
    )
    table["Hit %"] = table["Hit %"].round(1)
    table["Hit rate"] = table["Hit rate"].round(4)
    return table


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("ORB Signature Edge Explorer")

    frame = _require_dataset()
    signature = _dataset_signature()

    store_key = "signature_edge_analysis"
    cached = st.session_state.get(store_key)
    cached_signature = cached.get("signature") if cached else None

    if st.button("Compute signature edges", type="primary"):
        with st.spinner("Deriving ORB outcomes..."):
            outcomes = compute_orb_outcomes(frame, entry_mode="orb_close")
        st.session_state[store_key] = {
            "outcomes": outcomes,
            "signature": signature,
        }
        cached = st.session_state[store_key]
        cached_signature = signature

    if not cached or cached_signature != signature:
        st.info('Click "Compute signature edges" to populate the tables for the current dataset.')
        st.stop()

    outcomes: pd.DataFrame = cached["outcomes"]
    session_list = [sess for sess in SESSIONS if sess in outcomes["session"].unique()]
    if not session_list:
        st.warning("No ORB sessions detected in the derived outcomes.")
        st.stop()

    session_choice = st.selectbox(
        "Session",
        session_list,
        index=session_list.index("us") if "us" in session_list else 0,
    )

    target_label_map: Dict[str, Tuple[str, str]] = {
        f"{target_id} ({'long' if direction == 'up' else 'short'})": (target_id, direction)
        for target_id, _, direction in TARGET_DEFINITIONS
    }
    target_labels = list(target_label_map.keys())
    default_label = "L1_bull (long)" if "L1_bull (long)" in target_labels else target_labels[0]
    target_display = st.selectbox("Target level", target_labels, index=target_labels.index(default_label))
    target_choice, target_direction = target_label_map[target_display]

    min_samples = st.number_input(
        "Minimum samples for volume/spread buckets",
        min_value=5,
        max_value=500,
        value=25,
        step=5,
    )
    max_bins = st.slider(
        "Top volume/spread buckets to display",
        min_value=1,
        max_value=8,
        value=5,
    )

    direction_word = "long" if target_direction == "up" else "short"
    st.subheader(f"Signature stats: {target_choice} ({direction_word})")

    builder = _long_conditions if target_direction == "up" else _short_conditions
    table = _build_condition_table(
        outcomes,
        session=session_choice,
        target=target_choice,
        direction=target_direction,
        condition_builder=builder,
        min_samples=int(min_samples),
        max_volume_entries=int(max_bins),
    )
    if table.empty:
        st.info("No data available for the selected configuration.")
    else:
        st.dataframe(table, use_container_width=True)
        st.download_button(
            "Download table",
            data=table.to_csv(index=False).encode(),
            file_name=f"signature_edges_{target_choice}_{session_choice}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    with st.expander("Session comparison", expanded=False):
        tables: List[pd.DataFrame] = []
        for sess in session_list:
            tbl = _build_condition_table(
                outcomes,
                session=sess,
                target=target_choice,
                direction=target_direction,
                condition_builder=builder,
                min_samples=int(min_samples),
                max_volume_entries=int(max_bins),
            )
            if not tbl.empty:
                tbl.insert(0, "Session", sess)
                tables.append(tbl)
        if tables:
            combined = pd.concat(tables, ignore_index=True)
            st.dataframe(combined, use_container_width=True)
        else:
            st.info("No data available for the selected configuration.")

    st.markdown("---")
    with st.expander("Quick scan: all targets (current session)", expanded=False):
        quick_tables: List[pd.DataFrame] = []
        for target_id, _, direction in TARGET_DEFINITIONS:
            builder_fn = _long_conditions if direction == "up" else _short_conditions
            tbl = _build_condition_table(
                outcomes,
                session=session_choice,
                target=target_id,
                direction=direction,
                condition_builder=builder_fn,
                min_samples=int(min_samples),
                max_volume_entries=int(max_bins),
            )
            if not tbl.empty:
                tbl.insert(0, "Target", target_id)
                tbl.insert(1, "Direction", "long" if direction == "up" else "short")
                quick_tables.append(tbl)
        if quick_tables:
            aggregate = pd.concat(quick_tables, ignore_index=True)
            st.dataframe(aggregate, use_container_width=True)
        else:
            st.info("No data available for the selected configuration.")


if __name__ == "__main__":
    main()
