from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionConfig:
    name: str
    start_utc: str
    end_utc: str


DEFAULT_SESSIONS: Sequence[SessionConfig] = (
    SessionConfig(name="asia", start_utc="00:00", end_utc="08:00"),
    SessionConfig(name="europe", start_utc="08:00", end_utc="13:00"),
    SessionConfig(name="us", start_utc="13:00", end_utc="21:00"),
    SessionConfig(name="overnight", start_utc="21:00", end_utc="24:00"),
)


def annotate_daily_orb(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate the daily opening range breakout levels."""
    if df.empty:
        return df

    out = df.copy()
    idx = pd.to_datetime(out.index, utc=True)
    out.index = idx
    out["__date"] = idx.normalize().date

    first_candle_index = out.groupby("__date", sort=False).head(1).index
    out["is_orb_daily"] = out.index.isin(first_candle_index)

    orb_high = out.loc[first_candle_index, "high"].rename("orb_high_daily")
    orb_low = out.loc[first_candle_index, "low"].rename("orb_low_daily")
    per_day = pd.DataFrame({"orb_high_daily": orb_high, "orb_low_daily": orb_low})
    per_day["orb_mid_daily"] = (per_day["orb_high_daily"] + per_day["orb_low_daily"]) / 2.0
    range_ = per_day["orb_high_daily"] - per_day["orb_low_daily"]

    per_day["orb_range_daily"] = range_

    extensions = {
        "L1_bull_daily": per_day["orb_high_daily"] + 0.5 * range_,
        "L2_bull_daily": per_day["orb_high_daily"] + 1.0 * range_,
        "L3_bull_daily": per_day["orb_high_daily"] + 2.0 * range_,
        "L1_bear_daily": per_day["orb_low_daily"] - 0.5 * range_,
        "L2_bear_daily": per_day["orb_low_daily"] - 1.0 * range_,
        "L3_bear_daily": per_day["orb_low_daily"] - 2.0 * range_,
    }
    for column, series in extensions.items():
        per_day[column] = series

    per_day["__date"] = per_day.index.tz_convert("UTC").normalize().date
    merged = per_day.reset_index()
    if "index" in merged.columns:
        merged = merged.drop(columns=["index"])
    out = out.merge(merged, on="__date", how="left")
    out = out.set_index(idx).drop(columns=["__date"])
    return out


def _hhmm_to_offset(hhmm: str) -> pd.Timedelta:
    hours, minutes = map(int, hhmm.split(":"))
    return pd.Timedelta(hours=hours, minutes=minutes)


def annotate_session_orb(df: pd.DataFrame, sessions: Sequence[SessionConfig]) -> pd.DataFrame:
    """Add session-level opening range derived from the first candle of each session."""
    if df.empty:
        return df

    out = df.copy()
    idx_utc = pd.to_datetime(out.index, utc=True)
    out.index = idx_utc
    base_day = idx_utc.normalize()

    for session in sessions:
        name = session.name
        start_offset = _hhmm_to_offset(session.start_utc)
        end_offset = _hhmm_to_offset(session.end_utc)

        session_start = base_day + start_offset
        session_end = base_day + end_offset
        wraps_next_day = session_end <= session_start
        session_end = session_end.where(~wraps_next_day, session_end + pd.Timedelta(days=1))

        in_session = (idx_utc >= session_start) & (idx_utc < session_end)
        session_id = (base_day + start_offset).strftime("%Y-%m-%d") + f"_{name}"
        out[f"session_id_{name}"] = np.where(in_session, np.asarray(session_id), pd.NA)

        first_candles = out.loc[in_session].groupby(f"session_id_{name}", dropna=True).head(1).index
        out[f"is_orb_{name}"] = False
        out.loc[first_candles, f"is_orb_{name}"] = True

        if in_session.any():
            grouped = out.loc[in_session].groupby(f"session_id_{name}", dropna=True)
            orb_high = grouped["high"].transform("first")
            orb_low = grouped["low"].transform("first")
            orb_mid = (orb_high + orb_low) / 2.0
            range_ = orb_high - orb_low

            extensions = {
                f"orb_high_{name}": orb_high,
                f"orb_low_{name}": orb_low,
                f"orb_mid_{name}": orb_mid,
                f"orb_range_{name}": range_,
                f"L1_bull_{name}": orb_high + 0.5 * range_,
                f"L2_bull_{name}": orb_high + 1.0 * range_,
                f"L3_bull_{name}": orb_high + 2.0 * range_,
                f"L1_bear_{name}": orb_low - 0.5 * range_,
                f"L2_bear_{name}": orb_low - 1.0 * range_,
                f"L3_bear_{name}": orb_low - 2.0 * range_,
            }

            for column, series in extensions.items():
                if column not in out.columns:
                    out[column] = np.nan
                out.loc[in_session, column] = series
        else:
            for column in (
                f"orb_high_{name}",
                f"orb_low_{name}",
                f"orb_mid_{name}",
                f"orb_range_{name}",
                f"L1_bull_{name}",
                f"L2_bull_{name}",
                f"L3_bull_{name}",
                f"L1_bear_{name}",
                f"L2_bear_{name}",
                f"L3_bear_{name}",
            ):
                if column not in out.columns:
                    out[column] = np.nan

    return out


__all__ = [
    "SessionConfig",
    "DEFAULT_SESSIONS",
    "annotate_daily_orb",
    "annotate_session_orb",
]
