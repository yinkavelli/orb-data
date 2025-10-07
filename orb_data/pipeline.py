from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from .candles import add_candle_statistics, add_volume_spread_bins
from .client import BinanceClient, filter_usdt_symbols
from .orb_levels import (
    DEFAULT_SESSIONS,
    SessionConfig,
    annotate_daily_orb,
    annotate_session_orb,
)


LOCAL_TIMEZONE = "Etc/GMT-4"


_TIMEFRAME_PATTERN = re.compile(r"^(\d+)([mhdw])$")


def _add_previous_extrema(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"high", "low"}.issubset(frame.columns):
        return frame

    enriched = frame.copy()
    idx = pd.DatetimeIndex(enriched.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    enriched.index = idx
    enriched = enriched.sort_index()

    idx_local = idx.tz_convert(LOCAL_TIMEZONE)

    enriched["__day"] = idx_local.normalize()
    daily = enriched.groupby("__day", sort=True).agg({"high": "max", "low": "min"})
    daily = daily.sort_index()

    daily_extrema = pd.DataFrame(index=daily.index)
    daily_extrema["prev_day_high"] = daily["high"].shift(1)
    daily_extrema["prev_day_low"] = daily["low"].shift(1)

    week_start = daily.index - pd.to_timedelta(daily.index.dayofweek, unit="D")
    week_start = week_start.normalize()

    weekly = daily.groupby(week_start, sort=True).agg({"high": "max", "low": "min"})
    weekly = weekly.sort_index()
    prev_weekly = weekly.shift(1)

    daily_extrema["prev_week_high"] = prev_weekly["high"].reindex(week_start).to_numpy()
    daily_extrema["prev_week_low"] = prev_weekly["low"].reindex(week_start).to_numpy()

    result = enriched.join(daily_extrema, on="__day")
    return result.drop(columns=["__day"])


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
    hours, minutes = map(int, value.split(":"))
    return pd.Timedelta(hours=hours, minutes=minutes)


def _session_active_mask(index: pd.Index, session: SessionConfig) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    base = idx.normalize()
    start_offset = _parse_hhmm(session.start_utc)
    end_offset = _parse_hhmm(session.end_utc)

    session_start = base + start_offset
    session_end = base + end_offset
    wraps = session_end <= session_start
    if wraps.any():
        session_end = session_end + pd.to_timedelta(wraps.astype(int), unit="D")

    mask = (idx >= session_start) & (idx < session_end)
    return pd.Series(mask, index=index)


def _timeframe_to_timedelta(value: str | None) -> pd.Timedelta:
    if not value:
        return pd.Timedelta(minutes=1)
    match = _TIMEFRAME_PATTERN.match(value.strip())
    if not match:
        return pd.Timedelta(minutes=1)
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    if unit == "w":
        return pd.Timedelta(weeks=amount)
    return pd.Timedelta(minutes=1)


@dataclass
class OrbDataPipeline:
    symbols: Sequence[str]
    chart_timeframe: str
    start: datetime | str
    end: datetime | str | None = None
    orb_timeframe: str | None = None
    sessions: Sequence[SessionConfig] = DEFAULT_SESSIONS
    client: BinanceClient = field(default_factory=BinanceClient)
    usdt_only: bool = True
    sort_by_symbol: bool = True
    volume_percentile_window: int = 20
    percentile_bins: int = 3

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
        if self.volume_percentile_window < 1:
            raise ValueError("volume_percentile_window must be >= 1")
        if self.percentile_bins not in (3, 5):
            raise ValueError("percentile_bins must be 3 or 5")

    def run(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        symbols: list[str] = []

        chart_delta = _timeframe_to_timedelta(self.chart_timeframe)
        orb_delta = _timeframe_to_timedelta(self.orb_timeframe)
        base_delta = chart_delta if chart_delta >= orb_delta else orb_delta
        percentile_buffer = base_delta * max(self.volume_percentile_window, 1)
        lookback = max(pd.Timedelta(days=7), percentile_buffer)
        fetch_since = self.start_dt - lookback

        for symbol in self.symbols:
            price_raw = self.client.fetch_ohlcv(
                symbol,
                self.chart_timeframe,
                since=fetch_since,
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
                    since=fetch_since,
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

            price_df = add_candle_statistics(price_df)
            price_df = add_volume_spread_bins(
                price_df,
                window=self.volume_percentile_window,
                bins=self.percentile_bins,
            )

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

            if "volume" in price_df.columns:
                buy_volume = price_df.get(
                    "taker_buy_base_vol",
                    pd.Series(0.0, index=price_df.index, dtype=float),
                ).astype(float).fillna(0.0)
                price_df["volume_buy"] = buy_volume
                price_df["volume_sell"] = (price_df["volume"] - buy_volume).clip(lower=0.0)
                total_vol = price_df["volume"].replace(0, np.nan)
                price_df["volume_delta"] = price_df["volume_buy"] - price_df["volume_sell"]
                price_df["volume_buy_share"] = (price_df["volume_buy"] / total_vol).clip(upper=1.0)
                price_df["volume_sell_share"] = (price_df["volume_sell"] / total_vol).clip(upper=1.0)
                price_df[["volume_buy_share", "volume_sell_share"]] = price_df[["volume_buy_share", "volume_sell_share"]].fillna(0.0)

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
            price_df = _add_previous_extrema(price_df)
            price_df = price_df[price_df.index >= self.start_dt]
            if self.end_dt is not None:
                price_df = price_df[price_df.index <= self.end_dt]
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
