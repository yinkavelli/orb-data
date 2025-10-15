from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from orb_analysis import DOWN_TARGETS, UP_TARGETS, compute_orb_outcomes
from orb_data import DEFAULT_SESSIONS, OrbDataPipeline


def _format_rate(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _median(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.median())


def _quantile(series: pd.Series, q: float) -> float | None:
    if series.empty:
        return None
    return float(series.quantile(q))


def _summarise_boolean_features(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hits = data["__hit_mask"]
    for column in columns:
        if column not in data.columns:
            continue
        series = data[column].fillna(False).astype(bool)
        hit_share = series.loc[hits].mean() if hits.any() else float("nan")
        miss_mask = ~hits
        miss_share = series.loc[miss_mask].mean() if miss_mask.any() else float("nan")
        lift = hit_share - miss_share if not (math.isnan(hit_share) or math.isnan(miss_share)) else float("nan")
        rows.append(
            {
                "feature": column,
                "hit_share": hit_share,
                "miss_share": miss_share,
                "lift": lift,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values(by="lift", ascending=False, inplace=True, na_position="last")
    return result


def _summarise_categorical_feature(
    data: pd.DataFrame,
    column: str,
    *,
    min_count: int,
) -> pd.DataFrame:
    if column not in data.columns:
        return pd.DataFrame()
    tmp = (
        data.groupby(column, dropna=False)["__hit_mask"]
        .agg(hit_rate="mean", hits="sum", opportunities="count")
        .reset_index()
        .rename(columns={column: "bucket"})
    )
    tmp = tmp[tmp["opportunities"] >= min_count]
    tmp.sort_values(by=["hit_rate", "hits"], ascending=[False, False], inplace=True)
    return tmp


def _summarise_quantitative_features(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hits = data["__hit_mask"]
    misses = ~hits
    for column in columns:
        if column not in data.columns:
            continue
        series = pd.to_numeric(data[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "hit_p25": _quantile(series.loc[hits].dropna(), 0.25),
                "hit_median": _median(series.loc[hits].dropna()),
                "hit_p75": _quantile(series.loc[hits].dropna(), 0.75),
                "miss_median": _median(series.loc[misses].dropna()),
            }
        )
    return pd.DataFrame(rows)


def analyse_session_features(
    *,
    session: str,
    symbols: list[str],
    chart_timeframe: str,
    orb_timeframe: str,
    start: str,
    end: str | None,
    volume_window: int = 20,
    percentile_bins: int = 5,
) -> dict[str, dict[str, object]]:
    pipeline = OrbDataPipeline(
        symbols=symbols,
        chart_timeframe=chart_timeframe,
        orb_timeframe=orb_timeframe,
        start=start,
        end=end,
        sessions=DEFAULT_SESSIONS,
        volume_percentile_window=volume_window,
        percentile_bins=percentile_bins,
    )
    frame = pipeline.run()
    outcomes = compute_orb_outcomes(frame, entry_mode="orb_close")
    session_df = outcomes[outcomes["session"] == session].copy()
    session_df["bullish_candle"] = session_df["candle_direction"] == 1
    summaries: dict[str, dict[str, object]] = {}

    targets = list(UP_TARGETS + DOWN_TARGETS)

    for target in targets:
        direction = "up" if target in UP_TARGETS else "down"
        available_col = f"available_{target}"
        time_col = f"tt_{direction}_{target}"
        if available_col not in session_df.columns or time_col not in session_df.columns:
            continue
        subset = session_df.loc[session_df[available_col].fillna(False)].copy()
        if subset.empty:
            continue

        subset["__hit_mask"] = subset[time_col].notna()
        subset["__tt_minutes"] = pd.to_numeric(subset[time_col], errors="coerce")

        min_bucket = max(3, int(0.05 * len(subset)))
        bool_df = _summarise_boolean_features(
            subset,
            [
                "bullish_candle",
                "close_above_mid",
                "is_doji",
                "is_long_legged_doji",
                "is_spinning_top",
                "is_marubozu",
                "is_hammer",
                "is_inverted_hammer",
                "is_shooting_star",
            ],
        )
        cat_map = {
            "volume_bin": _summarise_categorical_feature(subset, "volume_bin", min_count=min_bucket),
            "spread_bin": _summarise_categorical_feature(subset, "spread_bin", min_count=min_bucket),
            "volume_spread_profile": _summarise_categorical_feature(subset, "volume_spread_profile", min_count=min_bucket),
        }
        quant_df = _summarise_quantitative_features(
            subset,
            [
                "body_ratio",
                "upper_wick_ratio",
                "lower_wick_ratio",
                "volume_percentile",
                "spread_percentile",
            ],
        )
        first_outcome_col = f"first_outcome_{direction}_{target}"
        first_minutes_col = f"first_minutes_{direction}_{target}"
        first_return_col = f"first_return_{direction}_{target}"

        summary: dict[str, object] = {
            "direction": direction,
            "opportunities": int(len(subset)),
            "hits": int(subset["__hit_mask"].sum()),
            "hit_rate": float(subset["__hit_mask"].mean()),
            "ttt_median": _median(subset.loc[subset["__hit_mask"], "__tt_minutes"].dropna()),
            "ttt_p25": _quantile(subset.loc[subset["__hit_mask"], "__tt_minutes"].dropna(), 0.25),
            "ttt_p75": _quantile(subset.loc[subset["__hit_mask"], "__tt_minutes"].dropna(), 0.75),
            "first_outcome_counts": (
                subset[first_outcome_col].value_counts(dropna=False).to_dict()
                if first_outcome_col in subset.columns
                else {}
            ),
            "first_minutes_median": (
                _median(pd.to_numeric(subset[first_minutes_col], errors="coerce"))
                if first_minutes_col in subset.columns
                else None
            ),
            "first_return_median": (
                _median(pd.to_numeric(subset[first_return_col], errors="coerce"))
                if first_return_col in subset.columns
                else None
            ),
            "boolean_feature_lift": bool_df,
            "categorical_feature_tables": cat_map,
            "quant_feature_summary": quant_df,
        }
        summaries[target] = summary

    return summaries


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)

    config = {
        "session": "asia",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "chart_timeframe": "15m",
        "orb_timeframe": "30m",
        "start": "2025-09-01",
        "end": "2025-10-31",
        "volume_window": 20,
        "percentile_bins": 5,
    }

    print("Running OrbDataPipeline with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    summaries = analyse_session_features(**config)

    output_dir = Path("analysis_reports")
    output_dir.mkdir(exist_ok=True)

    for target, summary in summaries.items():
        path = output_dir / f"feature_summary_{config['session']}_{target}.txt"
        lines: list[str] = []
        lines.append(f"Target: {target} ({summary['direction']})")
        lines.append(f"Opportunities: {summary['opportunities']}")
        lines.append(f"Hits: {summary['hits']}")
        lines.append(f"Hit rate: {_format_rate(summary['hit_rate'])}")
        lines.append(
            "Time-to-target (minutes): "
            f"p25={summary['ttt_p25']}, median={summary['ttt_median']}, p75={summary['ttt_p75']}"
        )
        if summary["first_outcome_counts"]:
            lines.append("First outcome counts:")
            for outcome, count in summary["first_outcome_counts"].items():
                outcome_label = "target" if outcome == "target" else ("stop" if outcome == "stop" else str(outcome))
                lines.append(f"  {outcome_label}: {count}")
        bool_df = summary["boolean_feature_lift"]
        if isinstance(bool_df, pd.DataFrame) and not bool_df.empty:
            lines.append("")
            lines.append("Boolean feature lift (Top 6 by positive lift):")
            top_bool = bool_df.head(6).copy()
            top_bool[["hit_share", "miss_share", "lift"]] = top_bool[
                ["hit_share", "miss_share", "lift"]
            ].applymap(lambda x: None if (x is None or math.isnan(x)) else round(float(x), 3))
            lines.append(top_bool.to_string(index=False))

        cat_tables = summary["categorical_feature_tables"]
        for label, df in cat_tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                lines.append("")
                lines.append(f"Top {label} buckets (min count applied):")
                tmp = df.head(6).copy()
                tmp["hit_rate"] = tmp["hit_rate"].apply(lambda x: None if math.isnan(x) else round(float(x) * 100, 1))
                lines.append(tmp.to_string(index=False))

        quant_df = summary["quant_feature_summary"]
        if isinstance(quant_df, pd.DataFrame) and not quant_df.empty:
            lines.append("")
            lines.append("Body / wick / percentile summary (hit vs miss):")
            tmp = quant_df.copy()
            for col in ["hit_p25", "hit_median", "hit_p75", "miss_median"]:
                tmp[col] = tmp[col].apply(lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3))
            lines.append(tmp.to_string(index=False))

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote summary for {target} -> {path}")


if __name__ == "__main__":
    main()
