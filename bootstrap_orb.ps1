<#!
Usage (on your Windows machine after copying this file to C:\Users\<you>\Projects\ORB):
  1. Open PowerShell in the target empty folder.
  2. Run:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
  3. Run:  .\bootstrap_orb.ps1
Optional switches:
  -NoVenv          (skip creating virtual environment)
  -NoRun           (skip launching streamlit after setup)
  -Force           (overwrite existing files)
!>
param(
    [switch]$NoVenv,
    [switch]$NoRun,
    [switch]$Force
)

function Write-Section($t){ Write-Host "`n==== $t ====\n" -ForegroundColor Cyan }

$Files = @{
  'requirements.txt' = @'
ccxt==4.3.71
pandas>=2.2.0
numpy>=1.26.0
bokeh>=3.4.0
streamlit>=1.37.0
pyarrow>=15.0.0
'@;
  'pyproject.toml' = @'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "orb-explorer"
version = "0.1.0"
description = "ORB multi-session crypto explorer"
authors = [{name="Your Name"}]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "ccxt>=4.0.0",
  "pandas>=2.2.0",
  "numpy>=1.26.0",
  "bokeh>=3.4.0",
  "streamlit>=1.37.0",
  "pyarrow>=15.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "black"]

[tool.setuptools.packages.find]
where = ["."]
include = ["orb_pipeline*"]
'@;
  '.streamlit/config.toml' = @'
[server]
headless = true
enableCORS = false
port = 8501
'@;
  'README.md' = @'# ORB Explorer\n\nOpening Range Breakout (multi-session) Streamlit app using Binance (ccxt), screening modes, indicators & Bokeh.\n\n## Quick Start\n```powershell\npython -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npip install --upgrade pip\npip install -r requirements.txt\nstreamlit run app.py\n```\n'@;
  'orb_pipeline/\n' = '';
  'orb_pipeline/__init__.py' = @'
from .pipeline import (
    CryptoDataPipeline,
    ScreeningMode,
    screen_symbols,
    DEFAULT_SESSIONS,
)
__all__ = [
    "CryptoDataPipeline",
    "ScreeningMode",
    "screen_symbols",
    "DEFAULT_SESSIONS",
]
'@;
  'orb_pipeline/pipeline.py' = @'
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import ccxt
class ScreeningMode(str, Enum):
    NONE = "none"
    VOLUME_24H_THRESHOLD = "24h_volume_threshold"
    TWO_HOUR_VOLUME_SURGE = "2h_volume_surge"
    PREVDAY_VOLUME_GAIN = "prevday_volume_gain"
# --- utility ---

def _to_millis(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _ensure_usdt(symbols: List[str]) -> List[str]:
    return [s for s in symbols if s.endswith("/USDT")]
@dataclass
class ScreenResult:
    available: List[str]
    used: List[str]
    note: str = ""
class BinanceData:
    def __init__(self, rate_limit_ms: int = 1200, spot: bool = True):
        self.exchange = ccxt.binance({"enableRateLimit": True, "rateLimit": rate_limit_ms, "options": {"defaultType": "spot" if spot else "future"}})
        self.exchange.load_markets()
    def _raw_klines(self, symbol: str, interval: str, start_ms: int, limit: int = 1000) -> List[List[Any]]:
        market = self.exchange.market(symbol)
        params = {"symbol": market["id"], "interval": interval, "limit": limit, "startTime": start_ms}
        return self.exchange.publicGetKlines(params)
    def fetch_ohlcv_batched(self, symbol: str, timeframe: str, since_ms: int, until_ms: Optional[int] = None, step_limit: int = 1000, sleep_ms: int = 0) -> pd.DataFrame:
        all_rows: List[List[Any]] = []
        next_ms = int(since_ms)
        if until_ms is None:
            until_ms = _to_millis(datetime.now(timezone.utc))
        else:
            until_ms = int(until_ms)
        while True:
            rows = self._raw_klines(symbol, timeframe, next_ms, limit=step_limit)
            if not rows:
                break
            all_rows.extend(rows)
            last_close = int(rows[-1][6])
            next_ms = last_close + 1
            if next_ms >= until_ms:
                break
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)
        if not all_rows:
            return pd.DataFrame(columns=["open","high","low","close","volume","quote_volume","n_trades","taker_buy_base_vol","taker_buy_quote_vol","close_time"])
        cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","n_trades","taker_buy_base_vol","taker_buy_quote_vol","_ignore"]
        df = pd.DataFrame(all_rows, columns=cols)
        num = ["open","high","low","close","volume","quote_volume","taker_buy_base_vol","taker_buy_quote_vol"]
        df[num] = df[num].astype(float)
        df["n_trades"] = df["n_trades"].astype(int)
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
        return df.set_index("open_time").sort_index().drop(columns=["_ignore"])
    def _load_market_symbols(self, usdt_only: bool = True) -> List[str]:
        markets = self.exchange.load_markets()
        symbols = [s for s, m in markets.items() if m.get("active")]
        if usdt_only:
            symbols = [s for s in symbols if s.endswith("/USDT")]
        return symbols
    def screen_24h_volume_threshold(self, min_usd: float) -> List[str]:
        symbols = self._load_market_symbols(usdt_only=True)
        tickers = self.exchange.fetch_tickers(symbols)
        selected = [s for s in symbols if tickers.get(s, {}).get("quoteVolume") is not None and float(tickers[s]["quoteVolume"]) >= float(min_usd)]
        selected.sort(key=lambda s: tickers.get(s, {}).get("quoteVolume", 0.0), reverse=True)
        return selected
    def screen_prevday_volume_gain(self, top_n: int = 10, lookback_days: int = 5) -> List[str]:
        symbols = self._load_market_symbols(usdt_only=True)
        out: List[Tuple[str, float]] = []
        now = datetime.now(timezone.utc)
        since_ms = _to_millis(now - timedelta(days=lookback_days+2))
        for s in symbols:
            try:
                daily = self.fetch_ohlcv_batched(s, "1d", since_ms=since_ms)
                if len(daily) < 3:
                    continue
                vol = daily["quote_volume"].dropna()
                gain = (vol.iloc[-2] - vol.iloc[-3]) / max(vol.iloc[-3], 1e-9)
                out.append((s, gain))
            except Exception:
                continue
        out.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in out[:top_n]]
    def screen_two_hour_volume_surge(self, lookback_hours: int = 6, surge_multiple: float = 3.0, base_timeframe: str = "5m") -> List[str]:
        symbols = self._load_market_symbols(usdt_only=True)
        now = datetime.now(timezone.utc)
        since_ms = _to_millis(now - timedelta(hours=lookback_hours))
        selected: List[str] = []
        for s in symbols:
            try:
                ohlcv = self.fetch_ohlcv_batched(s, base_timeframe, since_ms=since_ms)
                if len(ohlcv) < 30:
                    continue
                vol = ohlcv["quote_volume"].astype(float)
                window_ms = 2 * 60 * 60 * 1000
                cutoff = vol.index[-1] - pd.Timedelta(milliseconds=window_ms)
                recent = vol[vol.index >= cutoff]
                prev = vol[vol.index < cutoff]
                if recent.empty or prev.empty:
                    continue
                if recent.mean() >= surge_multiple * prev.mean():
                    selected.append(s)
            except Exception:
                continue
        return selected
# indicators

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(0.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def mark_orb_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy(); idx = pd.to_datetime(out.index, utc=True); out.index = idx
    dates = idx.normalize().date; out["__date"] = dates
    first_idx = out.groupby("__date", sort=False).head(1).index
    out["is_orb"] = out.index.isin(first_idx)
    orb_high = out.loc[first_idx, "high"].rename("orb_high"); orb_low  = out.loc[first_idx, "low"].rename("orb_low")
    per_day = pd.DataFrame({"orb_high": orb_high, "orb_low": orb_low})
    per_day["orb_mid"] = (per_day["orb_high"] + per_day["orb_low"]) / 2.0
    rng = (per_day["orb_high"] - per_day["orb_low"])
    per_day["L1_bull"] = per_day["orb_high"] + 0.5 * rng
    per_day["L2_bull"] = per_day["L1_bull"] + 0.5 * (per_day["L1_bull"] - per_day["orb_high"])
    per_day["L1_bear"] = per_day["orb_low"] - 0.5 * rng
    per_day["L2_bear"] = per_day["L1_bear"] - 0.5 * (per_day["orb_low"] - per_day["L1_bear"])
    per_day["__date"] = per_day.index.tz_convert("UTC").normalize().date
    out = out.merge(per_day.reset_index()[["__date","orb_high","orb_low","orb_mid","L1_bull","L2_bull","L1_bear","L2_bear"]], on="__date", how="left").set_index(idx)
    return out.drop(columns=["__date"])

def _hhmm_to_offset(hhmm: str) -> pd.Timedelta:
    h, m = map(int, hhmm.split(":")); return pd.Timedelta(hours=h, minutes=m)
DEFAULT_SESSIONS = [
    {"name": "asia",      "start_utc": "00:00", "end_utc": "08:00"},
    {"name": "europe",    "start_utc": "08:00", "end_utc": "13:00"},
    {"name": "us",        "start_utc": "13:00", "end_utc": "21:00"},
    {"name": "overnight", "start_utc": "21:00", "end_utc": "24:00"},
]

def add_sessions_and_orbs(df: pd.DataFrame, sessions: List[Dict[str, str]]) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy(); idx_utc = pd.to_datetime(out.index, utc=True); out.index = idx_utc; base = idx_utc.normalize()
    for s in sessions:
        nm = s["name"]
        start_off = _hhmm_to_offset(s["start_utc"]); end_off = _hhmm_to_offset(s["end_utc"])
        start_ts = base + start_off; end_ts = base + end_off
        wrap = end_ts <= start_ts; end_ts = end_ts.where(~wrap, end_ts + pd.Timedelta(days=1))
        in_session = (idx_utc >= start_ts) & (idx_utc < end_ts)
        sess_id = (base + start_off).strftime("%Y-%m-%d") + "_" + nm
        sess_id_arr = np.where(in_session, np.asarray(sess_id), pd.NA)
        out[f"session_id_{nm}"] = sess_id_arr
        if in_session.any():
            first_idx = out.loc[in_session].groupby(f"session_id_{nm}", dropna=True).head(1).index
        else:
            first_idx = out.index[[]]
        out[f"is_orb_{nm}"] = False; out.loc[first_idx, f"is_orb_{nm}"] = True
        if in_session.any():
            g = out.loc[in_session].groupby(f"session_id_{nm}", dropna=True)
            orb_high = g["high"].transform("first"); orb_low  = g["low"].transform("first"); mid = (orb_high + orb_low) / 2.0; rng = (orb_high - orb_low)
            L1_bull = orb_high + 0.5 * rng; L2_bull = L1_bull + 0.5 * (L1_bull - orb_high)
            L1_bear = orb_low  - 0.5 * rng; L2_bear = L1_bear - 0.5 * (orb_low - L1_bear)
            for col, series in [
                (f"orb_high_{nm}", orb_high),(f"orb_low_{nm}", orb_low),(f"orb_mid_{nm}", mid),(f"L1_bull_{nm}", L1_bull),(f"L2_bull_{nm}", L2_bull),(f"L1_bear_{nm}", L1_bear),(f"L2_bear_{nm}", L2_bear)]:
                out[col] = pd.NA; out.loc[in_session, col] = series
        else:
            for col in [f"orb_high_{nm}", f"orb_low_{nm}", f"orb_mid_{nm}", f"L1_bull_{nm}", f"L2_bull_{nm}", f"L1_bear_{nm}", f"L2_bear_{nm}"]:
                out[col] = pd.NA
    return out

def candle_stats(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    body = (c - o).abs(); rng = (h - l).replace(0, np.nan)
    upper_wick = (h - np.maximum(c, o)); lower_wick = (np.minimum(c, o) - l)
    stats = pd.DataFrame(index=df.index)
    stats["candle_color"] = np.where(c > o, "bull", np.where(c < o, "bear", "doji"))
    stats["body"] = body; stats["range"] = (h - l)
    stats["upper_wick"] = upper_wick; stats["lower_wick"] = lower_wick
    stats["body_pct_of_range"] = (body / rng).fillna(0.0)
    stats["upper_wick_pct_of_range"] = (upper_wick / rng).fillna(0.0); stats["lower_wick_pct_of_range"] = (lower_wick / rng).fillna(0.0)
    stats["is_doji"] = stats["body_pct_of_range"] < 0.1
    stats["is_marubozu"] = (stats["upper_wick_pct_of_range"] < 0.05) & (stats["lower_wick_pct_of_range"] < 0.05)
    return stats

def screen_symbols(exchange: BinanceData, mode: ScreeningMode, *, min_24h_volume: float = 10_000_000.0, top_n_prevday_gain: int = 20, surge_multiple: float = 3.0) -> ScreenResult:
    if mode == ScreeningMode.NONE:
        syms = exchange._load_market_symbols(usdt_only=True); return ScreenResult(available=syms, used=syms, note="All active */USDT symbols")
    if mode == ScreeningMode.VOLUME_24H_THRESHOLD:
        syms = exchange.screen_24h_volume_threshold(min_24h_volume); return ScreenResult(available=syms, used=syms, note=f"24h quoteVolume >= {min_24h_volume}")
    if mode == ScreeningMode.PREVDAY_VOLUME_GAIN:
        syms = exchange.screen_prevday_volume_gain(top_n=top_n_prevday_gain); return ScreenResult(available=syms, used=syms, note=f"Top {top_n_prevday_gain} prev-day volume gain")
    if mode == ScreeningMode.TWO_HOUR_VOLUME_SURGE:
        syms = exchange.screen_two_hour_volume_surge(surge_multiple=surge_multiple); return ScreenResult(available=syms, used=syms, note=f"2h mean >= {surge_multiple}× previous mean")
    raise ValueError(f"Unsupported screening mode: {mode}")
class CryptoDataPipeline:
    def __init__(self, timeframe: str = "15m", start_date: str = "2024-01-01", symbols: Optional[List[str]] = None, screening_mode: ScreeningMode = ScreeningMode.VOLUME_24H_THRESHOLD, min_24h_usdt_volume: float = 10_000_000.0, top_n_prevday_gain: int = 20, surge_multiple: float = 3.0, sessions: Optional[List[Dict[str, str]]] = None, rate_limit_ms: int = 1200, sort_mode: str = "symbol_then_time", custom_symbols: Optional[List[str]] = None):
        self.timeframe = timeframe
        self.start_ms = _to_millis(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        self.exchange = BinanceData(rate_limit_ms=rate_limit_ms)
        self.sessions = sessions if sessions is not None else DEFAULT_SESSIONS
        self.sort_mode = sort_mode
        if symbols is not None:
            sel = _ensure_usdt(symbols); note = "Provided symbols"
        else:
            screen_res = screen_symbols(self.exchange, screening_mode, min_24h_volume=min_24h_usdt_volume, top_n_prevday_gain=top_n_prevday_gain, surge_multiple=surge_multiple)
            sel = screen_res.used; note = screen_res.note
        if custom_symbols:
            custom_usdt = _ensure_usdt(custom_symbols); sel = sorted(set(sel).union(custom_usdt)); note += f" + {len(custom_usdt)} custom"
        if not sel: raise ValueError("No symbols resolved for pipeline.")
        self.symbols = sel; self.selection_note = note
    def _build_dataframe_for_symbol(self, symbol: str) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv_batched(symbol, self.timeframe, since_ms=self.start_ms, step_limit=1000, sleep_ms=0)
        if raw.empty: return raw
        df = raw.copy(); df = df[["open","high","low","close","volume","quote_volume","n_trades","taker_buy_base_vol","taker_buy_quote_vol","close_time"]]
        df["volume_usdt"] = df["quote_volume"].where(df["quote_volume"] > 0, df["close"] * df["volume"])
        df["dow"] = df.index.dayofweek
        for span in (9, 20, 100, 200): df[f"ema_{span}"] = ema(df["close"], span)
        macd_line, macd_signal, macd_hist = macd(df["close"]); df["macd_line"] = macd_line; df["macd_signal"] = macd_signal; df["macd_hist"] = macd_hist
        df["rsi_14"] = rsi(df["close"], 14)
        df = mark_orb_daily(df); df = add_sessions_and_orbs(df, self.sessions)
        vol = df["volume_usdt"].astype(float)
        p20 = vol.rolling(50, min_periods=5).quantile(0.2); p50 = vol.rolling(50, min_periods=5).quantile(0.5); p80 = vol.rolling(50, min_periods=5).quantile(0.8)
        df["vol_p20"] = p20; df["vol_p50"] = p50; df["vol_p80"] = p80
        stats = candle_stats(df); df = pd.concat([df, stats], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            df["taker_buy_share"] = (df["taker_buy_base_vol"] / df["volume"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["bull_bear_vol_ratio"] = df["taker_buy_share"] / (1 - df["taker_buy_share"] + 1e-9)
        df["symbol"] = symbol
        return df
    def run(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []; syms: List[str] = []
        for s in self.symbols:
            try:
                d = self._build_dataframe_for_symbol(s)
                if not d.empty: frames.append(d); syms.append(s)
            except Exception as e: print(f"[WARN] {s} failed: {e}")
        if not frames: return pd.DataFrame()
        if self.sort_mode == "symbol_then_time":
            out = pd.concat(frames, keys=syms, names=["symbol", "time"]); out["symbol"] = out.index.get_level_values("symbol"); return out
        else: return pd.concat(frames).sort_index()
'@;
  'orb_pipeline/plotting.py' = @'
from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Span, BoxAnnotation, Label, NumeralTickFormatter, DatetimeTickFormatter
SESSION_LABEL_DEFAULT: Dict[str, str] = {"asia": "ASIA", "europe": "EUROPE", "us": "US", "overnight": "OVERNIGHT"}

def make_orb_figure(df: pd.DataFrame, symbol: str, sessions: List[str], timeframe: str, title_prefix: str = "ORB Multi-Session", session_label: Dict[str, str] | None = None):
    if session_label is None: session_label = SESSION_LABEL_DEFAULT
    if "symbol" in df.columns: data = df[df["symbol"] == symbol].copy()
    else:
        try: data = df.xs(symbol, level="symbol").copy()
        except Exception: data = df.copy()
    if data.empty: return figure(height=300, title=f"{symbol} (no data)")
    data = data.sort_index(); days = data.index.normalize().unique()
    if len(days):
        last_day = days[-1]; next_day = last_day + pd.Timedelta(days=1)
        day_slice = data[(data.index >= last_day) & (data.index < next_day)]
        if not day_slice.empty: data = day_slice
    data["time_i"] = data.index; data["body_top"] = np.maximum(data["open"], data["close"]); data["body_bot"] = np.minimum(data["open"], data["close"])
    data["color"] = np.where(data["close"] >= data["open"], "#2ca02c", "#d62728")
    if len(data) > 1:
        deltas = np.diff(data.index.values).astype("timedelta64[ms]").astype(int); step_ms = int(np.median(deltas))
    else: step_ms = 60_000
    bar_width = int(step_ms * 0.8); src = ColumnDataSource(data)
    p = figure(x_axis_type="datetime", width=1100, height=500, toolbar_location="right"); p.title.text = f"{title_prefix}: {symbol} ({timeframe})"
    p.yaxis.formatter = NumeralTickFormatter(format="0,0"); p.xaxis.formatter = DatetimeTickFormatter(hours="%H:%M")
    p.segment("time_i", "high", "time_i", "low", color="black", line_width=1, source=src)
    body = p.vbar(x="time_i", top="body_top", bottom="body_bot", width=bar_width, fill_color="color", line_color="black", source=src)
    p.add_tools(HoverTool(renderers=[body], mode="vline", tooltips=[("Time", "@time_i{%F %T}"),("O", "@open{0,0.000}"),("H", "@high{0,0.000}"),("L", "@low{0,0.000}"),("C", "@close{0,0.000}")], formatters={"@time_i": "datetime"}))
    for s in sessions:
        sid_col = f"session_id_{s}"; hi_col = f"orb_high_{s}"; lo_col = f"orb_low_{s}"; mid_col = f"orb_mid_{s}"
        if not all(c in data.columns for c in [sid_col, hi_col, lo_col, mid_col]): continue
        sess_rows = data.dropna(subset=[sid_col])
        if sess_rows.empty: continue
        t0, t1 = sess_rows.index.min(), sess_rows.index.max(); orb_row = sess_rows.iloc[0]
        try: orb_high = float(orb_row[hi_col]); orb_low = float(orb_row[lo_col]); orb_mid = float(orb_row[mid_col])
        except (TypeError, ValueError): continue
        band = BoxAnnotation(left=t0, right=t1, bottom=orb_low, top=orb_high, fill_alpha=0.07, fill_color="#2196F3"); p.add_layout(band)
        p.add_layout(Span(location=t0.timestamp()*1000, dimension="height", line_color="#bdbdbd", line_dash="dashed", line_width=1))
        p.add_layout(Span(location=t1.timestamp()*1000, dimension="height", line_color="#bdbdbd", line_dash="dashed", line_width=1))
        for y, lbl, col in [(orb_high, f"{session_label.get(s,s)} H", "#1565c0"),(orb_low, f"{session_label.get(s,s)} L", "#1565c0"),(orb_mid, f"{session_label.get(s,s)} M", "#8e24aa")]:
            p.line([t0, t1], [y, y], color=col, line_width=1); p.add_layout(Label(x=t1, y=y, x_offset=6, text=lbl, text_color=col, text_font_size="8pt"))
    return p
'@;
  'app.py' = @'
from __future__ import annotations
from datetime import datetime, date
from typing import List
import streamlit as st
import pandas as pd
from orb_pipeline.pipeline import CryptoDataPipeline, ScreeningMode, screen_symbols
from orb_pipeline.plotting import make_orb_figure, SESSION_LABEL_DEFAULT
st.set_page_config(page_title="ORB Multi-Session Explorer", layout="wide")
st.title("📊 ORB Multi-Session Explorer")
with st.sidebar:
    st.header("Fetch Parameters")
    timeframe = st.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h"], index=2)
    start_date_val = st.date_input("Start Date", value=date(2024,1,1))
    start_date = start_date_val.strftime("%Y-%m-%d")
    screening_mode = st.selectbox("Screening Mode", [("None (all active USDT)", ScreeningMode.NONE),("24h Volume Threshold", ScreeningMode.VOLUME_24H_THRESHOLD),("Prev-day Volume Gain (Top N)", ScreeningMode.PREVDAY_VOLUME_GAIN),("2h Volume Surge", ScreeningMode.TWO_HOUR_VOLUME_SURGE)], format_func=lambda x: x[0])[1]
    min_24h = st.number_input("Min 24h Quote Volume (USDT)", 1_000_000.0, 5_000_000_000.0, 10_000_000.0, step=1_000_000.0)
    top_n_prevday = st.slider("Top N Prev-day Gain", 5, 100, 20)
    surge_multiple = st.slider("2h Surge Multiple", 1.5, 10.0, 3.0, 0.1)
    custom_syms_text = st.text_area("Custom Symbols (comma separated)", placeholder="BTC/USDT,ETH/USDT")
    custom_list: List[str] = []
    if custom_syms_text.strip(): custom_list = [s.strip().upper() for s in custom_syms_text.split(",") if s.strip()]
    run_btn = st.button("Run Fetch", type="primary")
try:
    tmp_exchange = __import__('orb_pipeline.pipeline', fromlist=['BinanceData']).BinanceData()
    screen_res = screen_symbols(tmp_exchange, screening_mode, min_24h_volume=min_24h, top_n_prevday_gain=top_n_prevday, surge_multiple=surge_multiple)
    available_syms = screen_res.available; screen_note = screen_res.note
except Exception as e:
    available_syms = []; screen_note = f"Screen preview error: {e}"
st.markdown("### Symbol Screening Preview"); st.caption(screen_note)
select_all = st.checkbox("Select ALL screened symbols", value=True, help="Override below multi-select")
selected_syms = available_syms if select_all else st.multiselect("Choose symbols", available_syms, available_syms[: min(25, len(available_syms))])
merged_syms = sorted(set(selected_syms).union([s for s in custom_list if s.endswith('/USDT')]))
st.write(f"Total planned symbols: {len(merged_syms)}")
if run_btn:
    if not merged_syms and not custom_list: st.error("No symbols selected.")
    else:
        with st.status("Fetching & Processing...", expanded=True) as status:
            st.write("Initializing pipeline...")
            pipe = CryptoDataPipeline(timeframe=timeframe, start_date=start_date, symbols=merged_syms if merged_syms else None, screening_mode=screening_mode, min_24h_usdt_volume=min_24h, top_n_prevday_gain=top_n_prevday, surge_multiple=surge_multiple, custom_symbols=custom_list)
            st.write(f"Resolved {len(pipe.symbols)} symbols. (Note: {pipe.selection_note})")
            df = pipe.run()
            if df.empty: st.warning("No data returned.")
            else:
                csv_name = f"crypto_data_{timeframe}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"; df.to_csv(csv_name)
                st.success(f"DataFrame shape: {df.shape}. Saved CSV: {csv_name}")
                st.download_button("Download CSV", data=df.to_csv().encode(), file_name=csv_name, mime="text/csv")
                st.session_state["data_df"] = df; st.session_state["symbols"] = pipe.symbols; st.session_state["timeframe"] = timeframe
        st.toast("Fetch complete", icon="✅")
else: st.info("Adjust parameters, then click Run Fetch.")
if "data_df" in st.session_state:
    df = st.session_state["data_df"]; symbols = st.session_state["symbols"]; tf = st.session_state.get("timeframe", timeframe)
    st.markdown("---"); st.subheader("Visualization")
    sym_pick = st.selectbox("Symbol", symbols)
    fig = make_orb_figure(df, symbol=sym_pick, sessions=["asia","europe","us","overnight"], timeframe=tf, title_prefix="ORB Multi-Session", session_label=SESSION_LABEL_DEFAULT)
    st.bokeh_chart(fig, use_container_width=True)
    with st.expander("Latest Snapshot Table", expanded=False):
        latest = df.groupby("symbol").tail(1)[["close","volume_usdt","rsi_14","macd_hist"]].reset_index(); st.dataframe(latest, use_container_width=True)
st.markdown("---"); st.caption("ORB Explorer • Session ORB levels, screening & technical snapshot.")
'@;
}

# --- File creation ---
Write-Section "Creating files"
foreach ($path in $Files.Keys) {
  $target = Join-Path (Get-Location) $path
  $isDir = $path.EndsWith('/') -or $path.EndsWith('\n')
  if ($isDir) {
    $dir = $path.TrimEnd('/','\n')
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    continue
  }
  if ((Test-Path $target) -and -not $Force) {
    Write-Host "SKIP (exists) $path (use -Force to overwrite)" -ForegroundColor Yellow
    continue
  }
  $parent = Split-Path $target -Parent
  if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
  $Files[$path] | Set-Content -NoNewline -Path $target
  Write-Host "Wrote $path" -ForegroundColor Green
}

# --- Virtual environment & dependencies ---
if (-not $NoVenv) {
  Write-Section "Virtual environment"
  if (-not (Test-Path .venv)) { python -m venv .venv }
  & .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip || exit 1
  pip install -r requirements.txt || exit 1
  Write-Host "Dependencies installed" -ForegroundColor Green
} else {
  Write-Host "Skipping venv creation (-NoVenv)" -ForegroundColor Yellow
}

# Smoke test
Write-Section "Import smoke test"
try {
  python - <<'PY'
from orb_pipeline.pipeline import CryptoDataPipeline, ScreeningMode
print('Imports OK')
PY
} catch {
  Write-Host "Import test failed" -ForegroundColor Red
}

if (-not $NoRun) {
  Write-Section "Launching Streamlit"
  streamlit run app.py
} else {
  Write-Host "Skipping app launch (-NoRun)." -ForegroundColor Yellow
}
