from __future__ import annotations


from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from .client import BinanceClient, filter_usdt_symbols
from .orb_levels import (
    DEFAULT_SESSIONS,
    SessionConfig,
    annotate_daily_orb,
    annotate_session_orb,
)


LOCAL_TIMEZONE = "Etc/GMT-4"

def _coerce_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _prepare_price_df(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "n_trades",
        "taker_buy_base_vol",
        "taker_buy_quote_vol",
        "close_time",
    ]
    existing = [col for col in columns if col in frame.columns]
    df = frame[existing].copy()
    if "quote_volume" in df.columns and "volume" in df.columns and "close" in frame.columns:
        df["volume_usdt"] = df["quote_volume"].where(df["quote_volume"] > 0, frame["close"] * df["volume"])
    return df


def _parse_hhmm(value: str) -> pd.Timedelta:
    hours, minutes = map(int, value.split(':'))
    return pd.Timedelta(hours=hours, minutes=minutes)


def _session_active_mask(index: pd.Index, session: SessionConfig) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    else:
        idx = idx.tz_convert('UTC')

    base = idx.normalize()
    start_offset = _parse_hhmm(session.start_utc)
    end_offset = _parse_hhmm(session.end_utc)

    session_start = base + start_offset
    session_end = base + end_offset
    wraps = session_end <= session_start
    if wraps.any():
        session_end = session_end + pd.to_timedelta(wraps.astype(int), unit='D')

    mask = (idx >= session_start) & (idx < session_end)
    return pd.Series(mask, index=index)

@dataclass
class OrbDataPipeline:
    """Fetch OHLCV data and attach daily/session ORB levels."""

    symbols: Sequence[str]
    chart_timeframe: str
    start: datetime | str
    end: datetime | str | None = None
    orb_timeframe: str | None = None
    sessions: Sequence[SessionConfig] = DEFAULT_SESSIONS
    client: BinanceClient = field(default_factory=BinanceClient)
    usdt_only: bool = True
    sort_by_symbol: bool = True

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        symbols = list(self.symbols)
        if self.usdt_only:
            symbols = filter_usdt_symbols(symbols)
        if not symbols:
            raise ValueError("No symbols remain after USDT filtering")
        self.symbols = symbols

        self.start_dt = _coerce_datetime(self.start)
        self.end_dt = _coerce_datetime(self.end) if self.end is not None else None
        self.orb_timeframe = self.orb_timeframe or self.chart_timeframe

    def run(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        symbols: list[str] = []

        for symbol in self.symbols:
            price_raw = self.client.fetch_ohlcv(
                symbol,
                self.chart_timeframe,
                since=self.start_dt,
                until=self.end_dt,
            )
            if price_raw.empty:
                continue
            price_df = _prepare_price_df(price_raw)

            orb_raw = price_raw
            if self.orb_timeframe != self.chart_timeframe:
                orb_raw = self.client.fetch_ohlcv(
                    symbol,
                    self.orb_timeframe,
                    since=self.start_dt,
                    until=self.end_dt,
                )

            orb_columns: list[str] = []
            if not orb_raw.empty:
                orb_df = orb_raw[[col for col in ["open", "high", "low"] if col in orb_raw.columns]].copy()
                orb_df = annotate_daily_orb(orb_df)
                orb_df = annotate_session_orb(orb_df, self.sessions)
                orb_columns = [
                    col
                    for col in orb_df.columns
                    if col.startswith(("orb_", "L1_", "L2_", "L3_", "session_id_", "is_orb"))
                ]
                if orb_columns:
                    aligned_orb = orb_df[orb_columns].reindex(price_df.index, method="ffill")
                    price_df = price_df.join(aligned_orb, how="left")

            for column in orb_columns:
                if column.startswith("is_orb"):
                    price_df[column] = price_df[column].fillna(False).astype(bool)
                else:
                    price_df[column] = price_df[column].ffill()

            if orb_columns:
                for session in self.sessions:
                    name = session.name
                    mask = _session_active_mask(price_df.index, session)
                    sid_col = f"session_id_{name}"
                    if sid_col in price_df.columns:
                        price_df.loc[~mask, sid_col] = pd.NA

                    level_cols = [
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
                    ]
                    for col in level_cols:
                        if col in price_df.columns:
                            price_df.loc[~mask, col] = np.nan

            if "close" in price_df.columns:
                price_df["ema_5"] = price_df["close"].ewm(span=5, adjust=False).mean()
                price_df["ema_13"] = price_df["close"].ewm(span=13, adjust=False).mean()

            index_utc = price_df.index
            if getattr(index_utc, "tz", None) is None:
                index_utc = pd.DatetimeIndex(index_utc, tz="UTC")
                price_df.index = index_utc
            else:
                index_utc = index_utc.tz_convert("UTC")
                price_df.index = index_utc
            price_df["time_utc"] = index_utc
            price_df["time_utc_plus4"] = index_utc.tz_convert(LOCAL_TIMEZONE)
            price_df["symbol"] = symbol
            price_df["orb_base_timeframe"] = self.orb_timeframe
            frames.append(price_df)
            symbols.append(symbol)

        if not frames:
            return pd.DataFrame()

        if self.sort_by_symbol:
            combined = pd.concat(frames, keys=symbols, names=["symbol", "time"])
            combined["symbol"] = combined.index.get_level_values("symbol")
            return combined
        combined = pd.concat(frames).sort_index()
        return combined


__all__ = ["OrbDataPipeline"]

