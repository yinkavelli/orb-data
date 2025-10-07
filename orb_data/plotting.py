from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandlePatternThresholds:
    doji_body_ratio: float = 0.1
    spinning_top_body_ratio: float = 0.3
    spinning_top_wick_ratio: float = 0.3
    marubozu_body_ratio: float = 0.9
    marubozu_wick_ratio: float = 0.05
    hammer_wick_ratio: float = 0.6
    hammer_body_ratio: float = 0.4
    hammer_upper_limit: float = 0.1
    inverted_hammer_wick_ratio: float = 0.6
    inverted_hammer_body_ratio: float = 0.4
    inverted_hammer_lower_limit: float = 0.1
    shooting_star_upper_ratio: float = 0.6
    shooting_star_lower_limit: float = 0.1
    shooting_star_body_ratio: float = 0.4


PATTERN_THRESHOLDS = CandlePatternThresholds()


def _safe_division(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan)


def add_candle_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"open", "high", "low", "close"}.issubset(frame.columns):
        return frame

    stats = frame.copy()
    stats["candle_range"] = stats["high"] - stats["low"]
    stats["candle_body"] = (stats["close"] - stats["open"]).abs()
    stats["candle_direction"] = np.where(stats["close"] >= stats["open"], 1, -1)
    stats["body_ratio"] = _safe_division(stats["candle_body"], stats["candle_range"])

    upper_wick = stats["high"] - stats[["open", "close"]].max(axis=1)
    lower_wick = stats[["open", "close"]].min(axis=1) - stats["low"]
    stats["upper_wick"] = upper_wick
    stats["lower_wick"] = lower_wick
    stats["upper_wick_ratio"] = _safe_division(upper_wick, stats["candle_range"])
    stats["lower_wick_ratio"] = _safe_division(lower_wick, stats["candle_range"])

    stats["close_position_ratio"] = _safe_division(stats["close"] - stats["low"], stats["candle_range"])
    stats["open_position_ratio"] = _safe_division(stats["open"] - stats["low"], stats["candle_range"])

    thresholds = PATTERN_THRESHOLDS

    stats["is_doji"] = stats["body_ratio"] <= thresholds.doji_body_ratio
    stats["is_spinning_top"] = (
        (stats["body_ratio"] <= thresholds.spinning_top_body_ratio)
        & (stats["upper_wick_ratio"] >= thresholds.spinning_top_wick_ratio)
        & (stats["lower_wick_ratio"] >= thresholds.spinning_top_wick_ratio)
    )
    stats["is_long_legged_doji"] = stats["is_doji"] & (
        (stats["upper_wick_ratio"] >= thresholds.spinning_top_wick_ratio)
        & (stats["lower_wick_ratio"] >= thresholds.spinning_top_wick_ratio)
    )
    stats["is_marubozu"] = (
        (stats["body_ratio"] >= thresholds.marubozu_body_ratio)
        & (stats["upper_wick_ratio"] <= thresholds.marubozu_wick_ratio)
        & (stats["lower_wick_ratio"] <= thresholds.marubozu_wick_ratio)
    )

    stats["is_hammer"] = (
        (stats["lower_wick_ratio"] >= thresholds.hammer_wick_ratio)
        & (stats["upper_wick_ratio"] <= thresholds.hammer_upper_limit)
        & (stats["body_ratio"] <= thresholds.hammer_body_ratio)
        & (stats["close_position_ratio"] >= 0.6)
    )
    stats["is_inverted_hammer"] = (
        (stats["upper_wick_ratio"] >= thresholds.inverted_hammer_wick_ratio)
        & (stats["lower_wick_ratio"] <= thresholds.inverted_hammer_lower_limit)
        & (stats["body_ratio"] <= thresholds.inverted_hammer_body_ratio)
        & (stats["open_position_ratio"] <= 0.4)
    )
    stats["is_shooting_star"] = (
        (stats["upper_wick_ratio"] >= thresholds.shooting_star_upper_ratio)
        & (stats["lower_wick_ratio"] <= thresholds.shooting_star_lower_limit)
        & (stats["body_ratio"] <= thresholds.shooting_star_body_ratio)
        & (stats["close_position_ratio"] <= 0.4)
    )

    for column in (
        "is_doji",
        "is_spinning_top",
        "is_long_legged_doji",
        "is_marubozu",
        "is_hammer",
        "is_inverted_hammer",
        "is_shooting_star",
    ):
        stats[column] = stats[column].fillna(False)

    return stats


VOLUME_BIN_LABELS = {
    3: ["Low Volume", "Average Volume", "High Volume"],
    5: [
        "Very Low Volume",
        "Low Volume",
        "Average Volume",
        "High Volume",
        "Very High Volume",
    ],
}

SPREAD_BIN_LABELS = {
    3: ["Narrow Spread", "Average Spread", "Wide Spread"],
    5: [
        "Very Narrow Spread",
        "Narrow Spread",
        "Average Spread",
        "Wide Spread",
        "Very Wide Spread",
    ],
}

VOLUME_SPREAD_COLOR_3 = {
    ("High Volume", "Wide Spread"): "#1E88E5",
    ("High Volume", "Narrow Spread"): "#FFFFFF",
    ("Low Volume", "Wide Spread"): "#FB8C00",
    ("Low Volume", "Narrow Spread"): "#FFEB3B",
}

DEFAULT_COLOR_3 = "#B0BEC5"

VOLUME_SPREAD_COLOR_5 = {
    "Very Low Volume": {
        "Very Narrow Spread": "#DDEFFD",
        "Narrow Spread": "#9ED1F9",
        "Average Spread": "#64B5F6",
        "Wide Spread": "#2A99F3",
        "Very Wide Spread": "#0C78CF",
    },
    "Low Volume": {
        "Very Narrow Spread": "#6EDED3",
        "Narrow Spread": "#38D1C3",
        "Average Spread": "#26A69A",
        "Wide Spread": "#1B746C",
        "Very Wide Spread": "#0E3E3A",
    },
    "Average Volume": {
        "Very Narrow Spread": "#FAFAFB",
        "Narrow Spread": "#D3DBDF",
        "Average Spread": "#B0BEC5",
        "Wide Spread": "#8DA1AB",
        "Very Wide Spread": "#68818E",
    },
    "High Volume": {
        "Very Narrow Spread": "#FFDBA5",
        "Narrow Spread": "#FFC063",
        "Average Spread": "#FFA726",
        "Wide Spread": "#E88A00",
        "Very Wide Spread": "#A56200",
    },
    "Very High Volume": {
        "Very Narrow Spread": "#F5C6FA",
        "Narrow Spread": "#E58AF2",
        "Average Spread": "#D05CE6",
        "Wide Spread": "#B13AC7",
        "Very Wide Spread": "#7F2A90",
    },
}

DEFAULT_COLOR_5 = "#90A4AE"


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")

    def percentile(values: np.ndarray) -> float:
        if len(values) == 0:
            return np.nan
        current = values[-1]
        if not np.isfinite(current):
            return np.nan
        less = np.sum(values < current)
        equal = np.sum(values == current)
        return (less + 0.5 * equal) / len(values)

    return series.rolling(window=window, min_periods=window).apply(percentile, raw=True)


def _assign_bins(percentile: pd.Series, bins: int, labels: List[str]) -> pd.Series:
    if labels is None:
        raise ValueError(f"Unsupported bin configuration for {bins} bins")
    if bins == 3:
        thresholds = [1 / 3, 2 / 3]
    elif bins == 5:
        thresholds = [0.2, 0.4, 0.6, 0.8]
    else:
        raise ValueError(f"Unsupported bin count: {bins}")

    result = pd.Series(pd.NA, index=percentile.index, dtype="object")
    valid = percentile.notna()
    values = percentile[valid].clip(lower=0.0, upper=1.0)
    bin_indices = np.digitize(values, thresholds, right=False)
    bin_indices = np.clip(bin_indices, 0, len(labels) - 1)
    mapped = [labels[idx] for idx in bin_indices]
    result.loc[valid] = mapped
    return result


def add_volume_spread_bins(frame: pd.DataFrame, window: int, bins: int) -> pd.DataFrame:
    if frame.empty:
        return frame

    enriched = frame.copy()
    if "volume" in enriched.columns:
        vol_pct = _rolling_percentile(enriched["volume"], window)
        enriched["volume_percentile"] = vol_pct
        vol_labels = VOLUME_BIN_LABELS.get(bins)
        enriched["volume_bin"] = _assign_bins(vol_pct, bins, vol_labels)

    if "candle_range" not in enriched.columns and {"high", "low"}.issubset(enriched.columns):
        enriched["candle_range"] = enriched["high"] - enriched["low"]

    if "candle_range" in enriched.columns:
        spread_pct = _rolling_percentile(enriched["candle_range"], window)
        enriched["spread_percentile"] = spread_pct
        spread_labels = SPREAD_BIN_LABELS.get(bins)
        enriched["spread_bin"] = _assign_bins(spread_pct, bins, spread_labels)

    if "volume_bin" in enriched.columns and "spread_bin" in enriched.columns:
        volume_short = enriched["volume_bin"].astype("string").str.replace(" Volume", "", regex=False)
        spread_short = enriched["spread_bin"].astype("string").str.replace(" Spread", "", regex=False)
        combined = (volume_short.fillna("") + " | " + spread_short.fillna("")).str.strip(" |")
        combined = combined.replace("", pd.NA)
        enriched["volume_spread_profile"] = combined

        def _resolve_color(vol_label: str | None, spread_label: str | None) -> str:
            vol_label = vol_label or ""
            spread_label = spread_label or ""
            if bins == 3:
                return VOLUME_SPREAD_COLOR_3.get((vol_label, spread_label), DEFAULT_COLOR_3)
            vol_map = VOLUME_SPREAD_COLOR_5.get(vol_label)
            if vol_map:
                color = vol_map.get(spread_label)
                if color:
                    return color
            return DEFAULT_COLOR_5

        enriched["volume_spread_color"] = enriched.apply(
            lambda row: _resolve_color(row.get("volume_bin"), row.get("spread_bin"))
            if pd.notna(row.get("volume_bin")) and pd.notna(row.get("spread_bin"))
            else (DEFAULT_COLOR_3 if bins == 3 else DEFAULT_COLOR_5),
            axis=1,
        )

    return enriched


__all__ = [
    "add_candle_statistics",
    "add_volume_spread_bins",
    "PATTERN_THRESHOLDS",
]
