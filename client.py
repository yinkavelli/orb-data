from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable, List

import ccxt
import pandas as pd


def _to_millis(value: datetime | int | float) -> int:
    """Convert a datetime-like value to an integer UTC millisecond timestamp."""
    if isinstance(value, (int, float)):
        return int(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


class BinanceClient:
    """Thin wrapper around ccxt.binance focused on historical OHLCV retrieval."""

    def __init__(
        self,
        *,
        rate_limit_ms: int = 1200,
        spot: bool = True,
        enable_rate_limit: bool = True,
    ) -> None:
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": enable_rate_limit,
                "rateLimit": rate_limit_ms,
                "options": {"defaultType": "spot" if spot else "future"},
            }
        )
        # Pre-load markets so ccxt knows about symbol metadata (id vs unified symbol)
        self.exchange.load_markets()

    def available_symbols(self, *, usdt_only: bool = True) -> List[str]:
        markets = self.exchange.markets
        symbols: List[str] = [symbol for symbol, meta in markets.items() if meta.get("active")]
        if usdt_only:
            symbols = [s for s in symbols if s.endswith("/USDT")]
        return symbols

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        since: datetime | int | float,
        until: datetime | int | float | None = None,
        batch_limit: int = 1000,
        sleep_ms: int = 0,
    ) -> pd.DataFrame:
        """Fetch historical klines in successive batches and return a DataFrame.

        The DataFrame index is the candle opening time (UTC) and includes the
        default Binance kline payload columns.
        """
        market = self.exchange.market(symbol)
        symbol_id = market["id"]

        since_ms = _to_millis(since)
        if until is None:
            until_ms = _to_millis(datetime.now(timezone.utc))
        else:
            until_ms = _to_millis(until)

        rows: List[List[object]] = []
        next_ms = since_ms

        while True:
            payload = {
                "symbol": symbol_id,
                "interval": timeframe,
                "limit": batch_limit,
                "startTime": next_ms,
            }
            batch = self.exchange.publicGetKlines(payload)
            if not batch:
                break

            rows.extend(batch)
            last_close = int(batch[-1][6])
            next_ms = last_close + 1
            if next_ms >= until_ms:
                break

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        return self._format_klines(rows)

    @staticmethod
    def _format_klines(raw: Iterable[Iterable[object]]) -> pd.DataFrame:
        rows = list(raw)
        if not rows:
            return pd.DataFrame(
                columns=[
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
            )

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "n_trades",
            "taker_buy_base_vol",
            "taker_buy_quote_vol",
            "ignore",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base_vol",
            "taker_buy_quote_vol",
        ]
        frame[numeric_cols] = frame[numeric_cols].astype(float)
        frame["n_trades"] = frame["n_trades"].astype(int)
        frame["open_time"] = pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"].astype("int64"), unit="ms", utc=True)
        frame = frame.set_index("open_time").sort_index()
        return frame.drop(columns=["ignore"])


def filter_usdt_symbols(symbols: Iterable[str]) -> List[str]:
    return [symbol for symbol in symbols if symbol.endswith("/USDT")]
