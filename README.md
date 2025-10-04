# ORB Data

Lightweight utilities for downloading Binance spot OHLCV data and annotating daily/session opening range breakout (ORB) levels. The project keeps only the data-focused pieces of the original ORB explorer so you can script data pulls or plug the annotated candles into your own analytics pipeline.

## Highlights
- ccxt-powered Binance client with batched OHLCV downloads.
- Daily ORB levels (high/low/mid, range and +/-1/+/-2 range extensions).
- Session ORB levels for configurable UTC sessions (Asia, Europe, US, Overnight by default).
- Adds `time_utc` and `time_utc_plus4` columns so you can line up local (UTC+4) sessions.
- Simple pipeline wrapper that returns a tidy DataFrame for one or many symbols.
- Optional Streamlit app with Plotly candlestick visualization, day/session filters, honor local session gaps, and ORB base shading when chart/orb timeframes differ.

## Installation
```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```python
from orb_data import OrbDataPipeline, DEFAULT_SESSIONS

pipeline = OrbDataPipeline(
    symbols=["BTC/USDT", "ETH/USDT"],
    chart_timeframe="15m",
    start="2024-01-01",
)
frame = pipeline.run()
print(frame.tail())
```

`OrbDataPipeline` fetches the requested candles, aligns ORB levels, and returns a pandas DataFrame indexed by `symbol` and candle start time when `sort_by_symbol=True` (default). ORB columns are forward-filled so you always have the latest active range for each symbol.

## Streamlit app
```bash
streamlit run app.py
```
This launches an interactive UI that fetches data via `OrbDataPipeline` and draws Plotly candlesticks (indexed by UTC+4) with session ORB overlays. Use the sidebar to pick symbols, date range, and timeframes. Navigate day-by-day with the Prev/Next controls, swap between session combinations, and download whatever slice you are viewing. Session lines reset cleanly at session boundaries and, when the ORB timeframe is higher than the chart timeframe, the ORB base candle window is shaded on the chart.

### Lower-level access
If you need more control:
- `BinanceClient.fetch_ohlcv(...)` returns raw OHLCV candles.
- `annotate_daily_orb(df)` adds daily ORB columns to an OHLCV DataFrame.
- `annotate_session_orb(df, sessions)` annotates session-level ranges using `SessionConfig` definitions.

## Custom Sessions Example
```python
from orb_data import BinanceClient, SessionConfig, annotate_session_orb

client = BinanceClient()
sessions = (
    SessionConfig(name="london", start_utc="07:00", end_utc="11:00"),
    SessionConfig(name="ny", start_utc="12:00", end_utc="20:00"),
)
raw = client.fetch_ohlcv("BTC/USDT", "15m", since="2024-06-01")
with_sessions = annotate_session_orb(raw[["open", "high", "low"]], sessions)
```

## Disclaimer
For research/education only. Binance/ccxt rules and rate limits apply. No trading advice.
