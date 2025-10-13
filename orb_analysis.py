from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

SESSIONS: List[str] = ["asia", "europe", "us", "overnight"]
UP_TARGETS: List[str] = ["orb_high", "L1_bull", "L2_bull", "L3_bull", "prev_day_high", "prev_week_high"]
DOWN_TARGETS: List[str] = ["orb_low", "L1_bear", "L2_bear", "L3_bear", "prev_day_low", "prev_week_low"]


def _prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with explicit `time` and `symbol` columns."""
    data = frame.copy()
    if isinstance(data.index, pd.MultiIndex):
        duplicate_levels = [name for name in data.index.names if name in data.columns]
        if duplicate_levels:
            data = data.drop(columns=duplicate_levels)
        data = data.reset_index()
    else:
        index_name = data.index.name or "time"
        if index_name in data.columns:
            data = data.drop(columns=[index_name])
        data = data.reset_index().rename(columns={index_name: "time"})

    if "time" not in data.columns:
        data["time"] = pd.to_datetime(data.index)
    else:
        data["time"] = pd.to_datetime(data["time"])

    if "symbol" not in data.columns:
        if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
            data["symbol"] = frame.index.get_level_values("symbol").values
        elif frame.index.name == "symbol":
            data["symbol"] = frame.index.values

    return data


def _valid_target(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    return None


def compute_orb_outcomes(frame: pd.DataFrame, *, entry_mode: str = "orb_close") -> pd.DataFrame:
    """Compute per-session ORB outcomes and time-to-target statistics.

    Parameters
    ----------
    frame : pd.DataFrame
        Input dataset from pipeline.
    entry_mode : str
        One of:
          - 'orb_close': enter at the close of the last ORB candle (default)
          - 'first_outside_close': enter at the close of the first candle outside the ORB window,
            and only evaluate target/stop hits from the following candle onward (conservative).
    """
    data = _prepare_dataframe(frame)

    if "symbol" in data.columns:
        symbol_groups: Iterable[tuple[str | None, pd.DataFrame]] = data.groupby("symbol", sort=False)
    else:
        symbol_groups = [(None, data)]

    records: List[Dict[str, object]] = []

    for symbol, symbol_df in symbol_groups:
        for session in SESSIONS:
            sid_col = f"session_id_{session}"
            is_orb_col = f"is_orb_{session}"
            high_col = f"orb_high_{session}"
            low_col = f"orb_low_{session}"

            if sid_col not in symbol_df.columns or is_orb_col not in symbol_df.columns:
                continue

            session_df = symbol_df[symbol_df[sid_col].notna()].copy()
            if session_df.empty:
                continue

            session_df["time"] = pd.to_datetime(session_df["time"])

            for sid, grp in session_df.groupby(sid_col):
                grp = grp.sort_values("time")
                orb_window = grp[grp[is_orb_col].fillna(False)]
                if orb_window.empty:
                    continue

                orb_candle = orb_window.iloc[-1]
                post_orb = grp.loc[~grp.index.isin(orb_window.index)].sort_values("time")
                if post_orb.empty:
                    continue

                orb_high = _valid_target(orb_candle.get(high_col))
                orb_low = _valid_target(orb_candle.get(low_col))
                if orb_high is None or orb_low is None:
                    continue

                # Base features captured on the ORB candle
                features: Dict[str, object] = {
                    "symbol": symbol,
                    "session": session,
                    "session_id": sid,
                    "orb_time": pd.to_datetime(orb_candle.get("time")),
                    "candle_direction": int(orb_candle.get("candle_direction", 0)),
                    "body_ratio": float(orb_candle.get("body_ratio", np.nan)),
                    "upper_wick_ratio": float(orb_candle.get("upper_wick_ratio", np.nan)),
                    "lower_wick_ratio": float(orb_candle.get("lower_wick_ratio", np.nan)),
                    "is_doji": bool(orb_candle.get("is_doji", False)),
                    "is_marubozu": bool(orb_candle.get("is_marubozu", False)),
                    "is_hammer": bool(orb_candle.get("is_hammer", False)),
                    "is_inverted_hammer": bool(orb_candle.get("is_inverted_hammer", False)),
                    "is_shooting_star": bool(orb_candle.get("is_shooting_star", False)),
                }

                orb_mid = _valid_target(orb_candle.get(f"orb_mid_{session}"))
                close_price = _valid_target(orb_candle.get("close"))
                if orb_mid is not None and close_price is not None:
                    features["close_above_mid"] = close_price >= orb_mid
                else:
                    features["close_above_mid"] = False

                # Add volume/spread signature bins if present
                for key in ("volume_bin", "spread_bin", "volume_spread_profile"):
                    if key in orb_candle:
                        features[key] = orb_candle.get(key)

                # Entry price/time and evaluation window depend on entry_mode
                if entry_mode == "first_outside_close":
                    first_out = post_orb.iloc[0] if len(post_orb) > 0 else None
                    entry_close = _valid_target(first_out.get("close")) if first_out is not None else None
                    entry_time = pd.to_datetime(first_out.get("time")) if first_out is not None else pd.NaT
                    eval_df = post_orb.iloc[1:]  # evaluate from the candle AFTER the first outside close
                else:
                    entry_close = _valid_target(orb_candle.get("close"))
                    entry_time = pd.to_datetime(orb_candle.get("time"))
                    eval_df = post_orb
                features["entry_mode"] = entry_mode
                features["entry_price"] = float(entry_close) if entry_close is not None else np.nan
                features["entry_time"] = entry_time
                features["orb_high_value"] = float(orb_high) if orb_high is not None else np.nan
                features["orb_low_value"] = float(orb_low) if orb_low is not None else np.nan

                def _first_event(df: pd.DataFrame, condition: pd.Series) -> tuple[float | None, pd.Timestamp, pd.Series | None]:
                    """Return minutes-to-event, event timestamp, and the first matching row."""
                    if entry_time is pd.NaT or pd.isna(entry_time):
                        return None, pd.NaT, None
                    hits = df.loc[condition]
                    if hits.empty:
                        return None, pd.NaT, None
                    first_row = hits.iloc[0]
                    event_time = pd.to_datetime(first_row["time"])
                    minutes = (event_time - entry_time).total_seconds() / 60.0
                    return minutes, event_time, first_row

                # Upward targets
                up_targets: Dict[str, float | None] = {
                    "orb_high": orb_high,
                    "L1_bull": _valid_target(orb_candle.get(f"L1_bull_{session}")),
                    "L2_bull": _valid_target(orb_candle.get(f"L2_bull_{session}")),
                    "L3_bull": _valid_target(orb_candle.get(f"L3_bull_{session}")),
                    "prev_day_high": _valid_target(orb_candle.get("prev_day_high")),
                    "prev_week_high": _valid_target(orb_candle.get("prev_week_high")),
                }

                up_hit_meta: Dict[str, tuple[float | None, pd.Timestamp, pd.Series | None]] = {}

                for name, target_value in up_targets.items():
                    features[f"available_{name}"] = target_value is not None
                    features[f"target_price_up_{name}"] = float(target_value) if target_value is not None else np.nan
                    if target_value is None:
                        features[f"tt_up_{name}"] = np.nan
                        features[f"pm_up_{name}"] = np.nan
                        features[f"hit_up_{name}"] = False
                        features[f"hit_time_up_{name}"] = pd.NaT
                        features[f"hit_price_up_{name}"] = np.nan
                        continue
                    cond = eval_df["high"] >= float(target_value)
                    minutes, event_time, event_row = _first_event(eval_df, cond)
                    up_hit_meta[name] = (minutes, event_time, event_row)
                    features[f"hit_up_{name}"] = minutes is not None
                    features[f"hit_time_up_{name}"] = event_time
                    features[f"hit_price_up_{name}"] = (
                        float(event_row["high"]) if event_row is not None and "high" in event_row else float(target_value)
                    )
                    if minutes is None:
                        features[f"tt_up_{name}"] = np.nan
                        features[f"pm_up_{name}"] = (
                            (float(target_value) - float(entry_close)) / float(entry_close) if entry_close is not None else np.nan
                        )
                    else:
                        features[f"tt_up_{name}"] = float(minutes)
                        features[f"pm_up_{name}"] = (
                            (float(target_value) - float(entry_close)) / float(entry_close) if entry_close is not None else np.nan
                        )

                down_targets: Dict[str, float | None] = {
                    "orb_low": orb_low,
                    "L1_bear": _valid_target(orb_candle.get(f"L1_bear_{session}")),
                    "L2_bear": _valid_target(orb_candle.get(f"L2_bear_{session}")),
                    "L3_bear": _valid_target(orb_candle.get(f"L3_bear_{session}")),
                    "prev_day_low": _valid_target(orb_candle.get("prev_day_low")),
                    "prev_week_low": _valid_target(orb_candle.get("prev_week_low")),
                }

                down_hit_meta: Dict[str, tuple[float | None, pd.Timestamp, pd.Series | None]] = {}

                for name, target_value in down_targets.items():
                    features[f"available_{name}"] = target_value is not None
                    features[f"target_price_down_{name}"] = float(target_value) if target_value is not None else np.nan
                    if target_value is None:
                        features[f"tt_down_{name}"] = np.nan
                        features[f"pm_down_{name}"] = np.nan
                        features[f"hit_down_{name}"] = False
                        features[f"hit_time_down_{name}"] = pd.NaT
                        features[f"hit_price_down_{name}"] = np.nan
                        continue
                    cond = eval_df["low"] <= float(target_value)
                    minutes, event_time, event_row = _first_event(eval_df, cond)
                    down_hit_meta[name] = (minutes, event_time, event_row)
                    features[f"hit_down_{name}"] = minutes is not None
                    features[f"hit_time_down_{name}"] = event_time
                    features[f"hit_price_down_{name}"] = (
                        float(event_row["low"]) if event_row is not None and "low" in event_row else float(target_value)
                    )
                    if minutes is None:
                        features[f"tt_down_{name}"] = np.nan
                        features[f"pm_down_{name}"] = (
                            (float(target_value) - float(entry_close)) / float(entry_close) if entry_close is not None else np.nan
                        )
                    else:
                        features[f"tt_down_{name}"] = float(minutes)
                        features[f"pm_down_{name}"] = (
                            (float(target_value) - float(entry_close)) / float(entry_close) if entry_close is not None else np.nan
                        )

                # Next-candle retrace to ORB high/low
                next_candle = post_orb.iloc[0] if len(post_orb) > 0 else None
                if next_candle is not None:
                    features["retrace_next_up"] = bool(
                        _valid_target(next_candle.get("high")) is not None
                        and orb_high is not None
                        and float(next_candle.get("high")) >= orb_high
                    )
                    features["retrace_next_down"] = bool(
                        _valid_target(next_candle.get("low")) is not None
                        and orb_low is not None
                        and float(next_candle.get("low")) <= orb_low
                    )
                else:
                    features["retrace_next_up"] = False
                    features["retrace_next_down"] = False

                # ---------------------------------------------------------
                # Target vs Stop (first-touch) simulation from ORB close
                # Assumptions:
                # - Long trades (up targets) stop at ORB low; Short trades (down targets) stop at ORB high
                # - If target and stop are touched on the same bar, we assume stop triggers first (conservative)
                # - Return is computed as percent P&L relative to entry_close
                # ---------------------------------------------------------
                # Pre-compute stop hit times relative to orb_time (minutes)
                # Long stop: breach of ORB low
                if orb_low is not None:
                    long_stop_minutes, long_stop_time, long_stop_row = _first_event(
                        eval_df, eval_df["low"] <= float(orb_low)
                    )
                else:
                    long_stop_minutes, long_stop_time, long_stop_row = (None, pd.NaT, None)
                # Short stop: breach of ORB high
                if orb_high is not None:
                    short_stop_minutes, short_stop_time, short_stop_row = _first_event(
                        eval_df, eval_df["high"] >= float(orb_high)
                    )
                else:
                    short_stop_minutes, short_stop_time, short_stop_row = (None, pd.NaT, None)

                # Stop returns (negative)
                if entry_close is not None and orb_low is not None:
                    ret_stop_long = (float(orb_low) - float(entry_close)) / float(entry_close)
                else:
                    ret_stop_long = np.nan
                if entry_close is not None and orb_high is not None:
                    ret_stop_short = (float(entry_close) - float(orb_high)) / float(entry_close)
                else:
                    ret_stop_short = np.nan

                features["stop_hit_long"] = long_stop_minutes is not None
                features["stop_time_long"] = long_stop_time
                features["stop_price_long"] = (
                    float(long_stop_row["low"]) if long_stop_row is not None and "low" in long_stop_row else (float(orb_low) if orb_low is not None else np.nan)
                )
                features["stop_hit_short"] = short_stop_minutes is not None
                features["stop_time_short"] = short_stop_time
                features["stop_price_short"] = (
                    float(short_stop_row["high"]) if short_stop_row is not None and "high" in short_stop_row else (float(orb_high) if orb_high is not None else np.nan)
                )

                # For each up-target: determine which comes first: target or long stop
                for name, target_value in up_targets.items():
                    # Availability already recorded above
                    if target_value is None or entry_close is None:
                        features[f"first_outcome_up_{name}"] = None
                        features[f"first_minutes_up_{name}"] = np.nan
                        features[f"first_return_up_{name}"] = np.nan
                        features[f"first_event_time_up_{name}"] = pd.NaT
                        features[f"first_event_price_up_{name}"] = np.nan
                        continue
                    hit_minutes, hit_time, _ = up_hit_meta.get(name, (None, pd.NaT, None))
                    # Compare with stop minutes
                    # tie -> stop wins (conservative)
                    if hit_minutes is not None and (long_stop_minutes is None or hit_minutes < long_stop_minutes):
                        features[f"first_outcome_up_{name}"] = "target"
                        features[f"first_minutes_up_{name}"] = float(hit_minutes)
                        features[f"first_return_up_{name}"] = (float(target_value) - float(entry_close)) / float(entry_close)
                        features[f"first_event_time_up_{name}"] = hit_time
                        features[f"first_event_price_up_{name}"] = float(target_value)
                    elif long_stop_minutes is not None:
                        features[f"first_outcome_up_{name}"] = "stop"
                        features[f"first_minutes_up_{name}"] = float(long_stop_minutes)
                        features[f"first_return_up_{name}"] = float(ret_stop_long)
                        features[f"first_event_time_up_{name}"] = long_stop_time
                        features[f"first_event_price_up_{name}"] = float(orb_low) if orb_low is not None else np.nan
                    else:
                        features[f"first_outcome_up_{name}"] = None
                        features[f"first_minutes_up_{name}"] = np.nan
                        features[f"first_return_up_{name}"] = 0.0  # assume flat if neither hit
                        features[f"first_event_time_up_{name}"] = pd.NaT
                        features[f"first_event_price_up_{name}"] = np.nan

                # For each down-target: determine which comes first: target or short stop
                for name, target_value in down_targets.items():
                    if target_value is None or entry_close is None:
                        features[f"first_outcome_down_{name}"] = None
                        features[f"first_minutes_down_{name}"] = np.nan
                        features[f"first_return_down_{name}"] = np.nan
                        features[f"first_event_time_down_{name}"] = pd.NaT
                        features[f"first_event_price_down_{name}"] = np.nan
                        continue
                    hit_minutes, hit_time, _ = down_hit_meta.get(name, (None, pd.NaT, None))
                    if hit_minutes is not None and (short_stop_minutes is None or hit_minutes < short_stop_minutes):
                        features[f"first_outcome_down_{name}"] = "target"
                        features[f"first_minutes_down_{name}"] = float(hit_minutes)
                        # Short profit: entry - target
                        features[f"first_return_down_{name}"] = (float(entry_close) - float(target_value)) / float(entry_close)
                        features[f"first_event_time_down_{name}"] = hit_time
                        features[f"first_event_price_down_{name}"] = float(target_value)
                    elif short_stop_minutes is not None:
                        features[f"first_outcome_down_{name}"] = "stop"
                        features[f"first_minutes_down_{name}"] = float(short_stop_minutes)
                        features[f"first_return_down_{name}"] = float(ret_stop_short)
                        features[f"first_event_time_down_{name}"] = short_stop_time
                        features[f"first_event_price_down_{name}"] = float(orb_high) if orb_high is not None else np.nan
                    else:
                        features[f"first_outcome_down_{name}"] = None
                        features[f"first_minutes_down_{name}"] = np.nan
                        features[f"first_return_down_{name}"] = 0.0
                        features[f"first_event_time_down_{name}"] = pd.NaT
                        features[f"first_event_price_down_{name}"] = np.nan

                records.append(features)

    if not records:
        return pd.DataFrame()

    outcomes = pd.DataFrame(records)
    return outcomes


def summarise_target_vs_stop(
    outcomes: pd.DataFrame,
    *,
    session: str | None = None,
    target: str,
    direction: str,
) -> Dict[str, object]:
    """Summarise first-touch results (target vs stop) and returns.

    Returns a dict with:
      - opportunities: count of available trades
      - target_first: count
      - stop_first: count
      - neither: count
      - target_first_pct / stop_first_pct / neither_pct
      - avg_return_pct: mean of first returns across opportunities (neither counted as 0)
      - cumulative_return_pct: sum of first returns across opportunities
      - series: DataFrame with columns [orb_time, first_return] sorted by orb_time
    """
    data = outcomes.copy()
    if session is not None:
        data = data[data["session"] == session]
    if data.empty:
        return {
            "opportunities": 0,
            "target_first": 0,
            "stop_first": 0,
            "neither": 0,
            "target_first_pct": np.nan,
            "stop_first_pct": np.nan,
            "neither_pct": np.nan,
            "avg_return_pct": np.nan,
            "cumulative_return_pct": np.nan,
            "series": pd.DataFrame(columns=["orb_time", "first_return"]),
        }

    dir_prefix = "up" if direction == "up" else "down"
    outcome_col = f"first_outcome_{dir_prefix}_{target}"
    return_col = f"first_return_{dir_prefix}_{target}"
    available_col = f"available_{target}"

    if outcome_col not in data.columns or return_col not in data.columns or available_col not in data.columns:
        return {
            "opportunities": 0,
            "target_first": 0,
            "stop_first": 0,
            "neither": 0,
            "target_first_pct": np.nan,
            "stop_first_pct": np.nan,
            "neither_pct": np.nan,
            "avg_return_pct": np.nan,
            "cumulative_return_pct": np.nan,
            "series": pd.DataFrame(columns=["orb_time", "first_return"]),
        }

    avail = data[available_col].fillna(False)
    subset = data.loc[avail, ["orb_time", outcome_col, return_col]].copy()
    if subset.empty:
        return {
            "opportunities": 0,
            "target_first": 0,
            "stop_first": 0,
            "neither": 0,
            "target_first_pct": np.nan,
            "stop_first_pct": np.nan,
            "neither_pct": np.nan,
            "avg_return_pct": np.nan,
            "cumulative_return_pct": np.nan,
            "series": pd.DataFrame(columns=["orb_time", "first_return"]),
        }

    # Fill missing returns as 0 for neither-hit cases
    subset[return_col] = pd.to_numeric(subset[return_col], errors="coerce").fillna(0.0)
    subset.sort_values("orb_time", inplace=True)

    target_first = int((subset[outcome_col] == "target").sum())
    stop_first = int((subset[outcome_col] == "stop").sum())
    neither = int((subset[outcome_col].isna()).sum())
    opp = int(len(subset))

    avg_return = float(subset[return_col].mean()) if opp else float("nan")
    cum_return = float(subset[return_col].sum()) if opp else float("nan")

    series = subset.rename(columns={return_col: "first_return"})[["orb_time", "first_return"]].copy()

    def pct(x: int) -> float:
        return (x / opp) if opp else float("nan")

    return {
        "opportunities": opp,
        "target_first": target_first,
        "stop_first": stop_first,
        "neither": neither,
        "target_first_pct": pct(target_first),
        "stop_first_pct": pct(stop_first),
        "neither_pct": pct(neither),
        "avg_return_pct": avg_return,
        "cumulative_return_pct": cum_return,
        "series": series,
    }


def summarise_target_hits(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated hit counts and feature statistics per target."""
    rows: List[Dict[str, object]] = []
    total_sessions = len(outcomes)

    for target in UP_TARGETS + DOWN_TARGETS:
        direction = "up" if target in UP_TARGETS else "down"
        time_col = f"tt_{direction}_{target}"
        available_col = f"available_{target}"

        if time_col not in outcomes.columns or available_col not in outcomes.columns:
            continue

        available_mask = outcomes[available_col].fillna(False)
        opportunities = int(available_mask.sum())
        hits_mask = outcomes[time_col].notna()
        hits = int(hits_mask.sum())
        time_vals = outcomes.loc[hits_mask, time_col]

        if opportunities == 0:
            hit_rate = np.nan
            hit_subset = outcomes.iloc[0:0]
        else:
            hit_rate = hits / opportunities if opportunities else np.nan
            hit_subset = outcomes.loc[hits_mask]
        timeout_rate = ((opportunities - hits) / opportunities) if opportunities else np.nan

        # Percentiles for time-to-target (minutes)
        def _pct(s: pd.Series, q: float) -> float:
            return float(s.quantile(q)) if not s.empty else float("nan")

        ttt_p25 = _pct(time_vals, 0.25) if hits > 0 else np.nan
        ttt_p50 = _pct(time_vals, 0.50) if hits > 0 else np.nan
        ttt_p75 = _pct(time_vals, 0.75) if hits > 0 else np.nan

        bullish_pct = hit_subset["candle_direction"].eq(1).mean() if not hit_subset.empty else np.nan
        close_above_pct = hit_subset["close_above_mid"].mean() if not hit_subset.empty else np.nan
        avg_body_ratio = hit_subset["body_ratio"].mean() if not hit_subset.empty else np.nan
        avg_upper_wick = hit_subset["upper_wick_ratio"].mean() if not hit_subset.empty else np.nan
        avg_lower_wick = hit_subset["lower_wick_ratio"].mean() if not hit_subset.empty else np.nan

        # Percent move percentiles (signed), use appropriate column
        pm_col = f"pm_{direction}_{target}"
        pm_vals = outcomes.loc[hits_mask, pm_col] if pm_col in outcomes.columns else pd.Series(dtype=float)
        pm_p25 = _pct(pm_vals.dropna(), 0.25) if not pm_vals.empty else np.nan
        pm_p50 = _pct(pm_vals.dropna(), 0.50) if not pm_vals.empty else np.nan
        pm_p75 = _pct(pm_vals.dropna(), 0.75) if not pm_vals.empty else np.nan

        rows.append(
            {
                "target": target,
                "direction": direction,
                "opportunities": opportunities,
                "hits": hits,
                "hit_rate": hit_rate,
                "timeout_rate": timeout_rate,
                "ttt_p25": ttt_p25,
                "ttt_p50": ttt_p50,
                "ttt_p75": ttt_p75,
                "bullish_close_pct": bullish_pct,
                "close_above_mid_pct": close_above_pct,
                "avg_body_ratio": avg_body_ratio,
                "avg_upper_wick_ratio": avg_upper_wick,
                "avg_lower_wick_ratio": avg_lower_wick,
                "pm_p25": pm_p25,
                "pm_p50": pm_p50,
                "pm_p75": pm_p75,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.sort_values("hit_rate", ascending=False, inplace=True)
    return summary


def identify_best_target(summary: pd.DataFrame) -> Dict[str, object] | None:
    if summary.empty:
        return None
    ranked = summary.sort_values(["hit_rate", "hits"], ascending=[False, False])
    return ranked.iloc[0].to_dict()

