from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from ccxt.base.errors import ExchangeError, NetworkError

from orb_data import (
    DEFAULT_SESSIONS,
    OrbDataPipeline,
    prepare_candlestick_frame,
    make_bokeh_candlestick,
    ChartBackendError,
)
from orb_data.figure import make_orb_figure

st.set_page_config(page_title="ORB Data Viewer", layout="wide")
st.title("ORB Data Viewer")

DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT"
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h"]
SESSION_NAMES = [session.name for session in DEFAULT_SESSIONS]
SESSION_PRESETS: Dict[str, List[str]] = {
    "Asia + Europe": ["asia", "europe"],
    "Asia + US": ["asia", "us"],
    "Europe + US": ["europe", "us"],
    "Europe + Overnight": ["europe", "overnight"],
    "Asia + Overnight": ["asia", "overnight"],
    "US + Overnight": ["us", "overnight"],
}
LOCAL_TZ = "Etc/GMT-4"
LOCAL_TZ_LABEL = "UTC+4"

SESSION_LABEL_MARKUP = {
    "asia": ":blue[ASIA]",
    "europe": ":orange[EUROPE]",
    "us": ":green[US]",
    "overnight": ":violet[OVERNIGHT]",
}
VOLUME_LABEL_MARKUP = {
    "buy": ":green[BUY]",
    "sell": ":red[SELL]",
}


@st.cache_data(show_spinner="Fetching data...")
def run_pipeline_cached(
    symbols: Tuple[str, ...],
    chart_timeframe: str,
    orb_timeframe: str,
    start_iso: str,
    end_iso: Optional[str],
    volume_window: int,
    percentile_bins: int,
):
    pipeline = OrbDataPipeline(
        symbols=list(symbols),
        chart_timeframe=chart_timeframe,
        start=start_iso,
        end=end_iso,
        orb_timeframe=orb_timeframe,
        sessions=DEFAULT_SESSIONS,
        volume_percentile_window=volume_window,
        percentile_bins=percentile_bins,
    )
    frame = pipeline.run()
    exchange_name = getattr(pipeline.client, "exchange_id", None)
    return frame, tuple(pipeline.symbols), exchange_name


def _extract_symbol_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        try:
            sub = frame.xs(symbol, level="symbol")
        except KeyError:
            sub = frame.loc[(symbol, slice(None))]
    elif "symbol" in frame.columns:
        sub = frame[frame["symbol"] == symbol]
    else:
        sub = frame
    sub = sub.copy()
    idx = pd.to_datetime(sub.index)
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        sub.index = idx
    return sub.sort_index()


def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = pd.DatetimeIndex(idx, tz="UTC")
        df.index = idx
    else:
        idx = idx.tz_convert("UTC")
        df.index = idx
    if "time_utc" not in df.columns:
        df["time_utc"] = idx
    if "time_utc_plus4" in df.columns:
        local_series = pd.to_datetime(df["time_utc_plus4"])
        if getattr(local_series.dtype, "tz", None) is None:
            local_series = local_series.dt.tz_localize(LOCAL_TZ)
        else:
            local_series = local_series.dt.tz_convert(LOCAL_TZ)
    else:
        local_series = idx.tz_convert(LOCAL_TZ)
    df["time_utc_plus4"] = local_series
    df["time_local"] = local_series
    return df


with st.sidebar:
    st.header("Parameters")
    symbols_text = st.text_input("Symbols", value=DEFAULT_SYMBOLS)
    chart_tf = st.selectbox("Chart timeframe", TIMEFRAMES, index=0)
    orb_tf = st.selectbox("ORB timeframe", TIMEFRAMES, index=3, help="Timeframe used for ORB levels")
    start_dt = st.date_input("Start date", value=date(2025, 9, 30))
    use_end = st.checkbox("Use end date", value=False)
    end_dt = None
    if use_end:
        end_dt = st.date_input("End date", value=date.today())
    volume_window = st.number_input(
        "Percentile lookback (candles)",
        min_value=5,
        max_value=500,
        value=20,
        step=1,
        help="Window used to compute volume/spread percentiles.",
    )
    bin_labels = {
        "3 bins (Low / Average / High)": 3,
        "5 bins (Very Low → Very High)": 5,
    }
    bin_choice = st.selectbox(
        "Percentile bins",
        options=list(bin_labels.keys()),
        index=0,
        help="Number of percentile buckets for candle volume and spread.",
    )
    percentile_bins = bin_labels[bin_choice]
    view_options = {
        "Last 1 day": "1d",
        "Last N candles": "ncandles",
        "Auto (full data)": "auto",
    }
    view_choice_label = st.selectbox(
        "Initial chart window",
        options=list(view_options.keys()),
        index=0,
        help="Controls the default zoom when the chart first renders.",
    )
    view_choice = view_options[view_choice_label]
    candle_window = 200
    if view_choice == "ncandles":
        candle_window = st.number_input(
            "Candles in initial view",
            min_value=10,
            max_value=5000,
            value=200,
            step=10,
            help="The chart will open showing the most recent N candles.",
        )
    run_btn = st.button("Fetch data", type="primary")
    if st.button("Clear cached data", type="secondary"):
        run_pipeline_cached.clear()
        for key in (
            "orb_df",
            "orb_symbols",
            "orb_chart_tf",
            "orb_orb_tf",
            "orb_volume_window",
            "orb_percentile_bins",
            "orb_exchange",
            "orb_dataset_key",
            "targeted_analysis_results",
            "targeted_analysis_signature",
        ):
            st.session_state.pop(key, None)
        st.success("Pipeline cache cleared. Next fetch will hit the data source.")

symbols: List[str] = []
if symbols_text.strip():
    symbols = [item.strip().upper() for item in symbols_text.split(",") if item.strip()]

if run_btn:
    if not symbols:
        st.error("Please enter at least one symbol (e.g. BTC/USDT).")
    else:
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat() if end_dt else None
        symbols_tuple = tuple(symbols)
        with st.status("Fetching data...", expanded=True) as status:
            status.write(f"Running pipeline for {len(symbols_tuple)} symbol(s)...")
            try:
                frame, resolved_symbols, exchange_name = run_pipeline_cached(
                    symbols_tuple,
                    chart_tf,
                    orb_tf,
                    start_iso,
                    end_iso,
                    int(volume_window),
                    int(percentile_bins),
                )
            except (NetworkError, ExchangeError) as exc:
                status.update(label="Fetch failed", state="error")
                st.error(
                    "Pipeline error: Binance API request failed. This usually means a temporary network issue or "
                    "rate-limit response. Try reducing the symbol list or widening the date window, then retry.\n\n"
                    f"Details: {exc}"
                )
                frame = pd.DataFrame()
                exchange_name = None
                resolved_symbols = symbols_tuple
            except Exception as exc:
                status.update(label="Fetch failed", state="error")
                st.error(f"Pipeline error: {exc}")
                frame = pd.DataFrame()
                exchange_name = None
                resolved_symbols = symbols_tuple
            else:
                if frame.empty:
                    status.update(label="No data returned", state="warning")
                    st.warning("No data returned for the selected configuration.")
                else:
                    status.update(label="Fetch complete", state="complete")
                    st.success(
                        f"Fetched {frame.shape[0]} rows across {len(resolved_symbols)} symbol(s)."
                    )
                    if exchange_name:
                        st.caption(f"Data source: ccxt.{exchange_name}")
                    st.session_state["orb_df"] = frame
                    st.session_state["orb_symbols"] = list(resolved_symbols)
                    st.session_state["orb_chart_tf"] = chart_tf
                    st.session_state["orb_orb_tf"] = orb_tf
                    st.session_state["orb_volume_window"] = int(volume_window)
                    st.session_state["orb_percentile_bins"] = int(percentile_bins)
                    if exchange_name:
                        st.session_state["orb_exchange"] = exchange_name
                    dataset_key = (
                        tuple(resolved_symbols),
                        chart_tf,
                        orb_tf,
                        start_iso,
                        end_iso,
                        int(volume_window),
                        int(percentile_bins),
                    )
                    st.session_state["orb_dataset_key"] = dataset_key
                    st.session_state.pop("targeted_analysis_results", None)
                    st.session_state.pop("targeted_analysis_signature", None)
        st.toast("Fetch finished")

if "orb_df" not in st.session_state:
    st.info("Enter parameters and click Fetch data to load candles and ORB levels.")
    st.stop()

frame: pd.DataFrame = st.session_state["orb_df"]
symbols = st.session_state["orb_symbols"]
chart_tf = st.session_state["orb_chart_tf"]
orb_tf = st.session_state["orb_orb_tf"]
exchange_name = st.session_state.get("orb_exchange")
volume_window = st.session_state.get("orb_volume_window", int(volume_window))
percentile_bins = st.session_state.get("orb_percentile_bins", int(percentile_bins))

st.markdown("---")

symbol_choice = st.selectbox("Symbol", symbols, key="symbol_selector")

st.subheader("Visualization")
st.caption(f"Chart timeframe: {chart_tf} | ORB timeframe: {orb_tf}")
if exchange_name:
    st.caption(f"Exchange source: ccxt.{exchange_name}")
st.caption(f"Percentile window: {volume_window} | Bins: {percentile_bins}")
chart_backend = st.selectbox(
    "Chart backend",
    ("Bokeh", "Plotly"),
    index=0,
    help="Bokeh offers seamless wheel-zoom and panning; Plotly remains available for reference.",
)

symbol_df = _extract_symbol_frame(frame, symbol_choice)
if symbol_df.empty:
    st.warning("No data available for the selected symbol.")
    st.stop()

symbol_df = _ensure_time_columns(symbol_df)
local_series = pd.to_datetime(symbol_df["time_utc_plus4"])
if local_series.empty:
    st.warning("No timestamps found for the selected symbol.")
    st.stop()

selected_sessions: List[str] = SESSION_NAMES.copy()

session_visibility: Dict[str, bool] = {
    name: st.session_state.get(f"session_toggle_{name}", True) for name in SESSION_NAMES
}
ordered_sessions = [name for name in SESSION_NAMES if name in selected_sessions]
toggle_cols = st.columns(len(ordered_sessions) + 2)
for idx, name in enumerate(ordered_sessions):
    key = f"session_toggle_{name}"
    label = SESSION_LABEL_MARKUP.get(name, name.upper())
    session_visibility[name] = toggle_cols[idx].checkbox(
        label,
        value=session_visibility[name],
        key=key,
    )

buy_key = "volume_toggle_buy"
if buy_key not in st.session_state:
    st.session_state[buy_key] = False
show_buy_volume = toggle_cols[-2].checkbox(
    VOLUME_LABEL_MARKUP["buy"],
    value=st.session_state[buy_key],
    key=buy_key,
)

sell_key = "volume_toggle_sell"
if sell_key not in st.session_state:
    st.session_state[sell_key] = False
show_sell_volume = toggle_cols[-1].checkbox(
    VOLUME_LABEL_MARKUP["sell"],
    value=st.session_state[sell_key],
    key=sell_key,
)

filtered_df = symbol_df.copy()

if not filtered_df.empty and selected_sessions != SESSION_NAMES:
    session_cols = [
        f"session_id_{name}"
        for name in selected_sessions
        if f"session_id_{name}" in filtered_df.columns
    ]
    if session_cols:
        session_mask = pd.Series(False, index=filtered_df.index)
        for col in session_cols:
            session_mask |= filtered_df[col].notna()
        filtered_df = filtered_df.loc[session_mask]

if filtered_df.empty:
    st.warning("No candles available after applying filters.")
    st.stop()

filtered_df.sort_values("time_utc_plus4", inplace=True)
plot_df = filtered_df.set_index("time_utc_plus4")

coverage_start = local_series.min()
coverage_end = local_series.max()
if pd.isna(coverage_start) or pd.isna(coverage_end):
    coverage_text = f"Data coverage ({LOCAL_TZ_LABEL}): unavailable"
else:
    coverage_text = (
        f"Data coverage ({LOCAL_TZ_LABEL}): "
        f"{coverage_start.strftime('%Y-%m-%d %H:%M')} \u2192 {coverage_end.strftime('%Y-%m-%d %H:%M')}"
    )
caption_sessions = ", ".join(s.upper() for s in selected_sessions)
caption_text = f"{coverage_text} | Sessions: {caption_sessions}"

initial_range = None
if not plot_df.empty:
    idx_range = plot_df.index.sort_values()
    end_ts = idx_range.max()
    start_ts = None
    if view_choice == "1d":
        start_candidate = end_ts - pd.Timedelta(days=1)
        min_ts = idx_range.min()
        start_ts = start_candidate if start_candidate > min_ts else min_ts
    elif view_choice == "ncandles":
        n = int(candle_window)
        if n > 0:
            if n < len(idx_range):
                start_ts = idx_range[-n]
            else:
                start_ts = idx_range.min()
    if start_ts is not None and end_ts is not None:
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
        initial_range = (start_ts, end_ts)

st.caption(caption_text)
chart_title = f"{symbol_choice} ({chart_tf})"

try:
    if chart_backend == "Plotly":
        fig = make_orb_figure(
            plot_df,
            symbol=symbol_choice,
            timeframe=chart_tf,
            title_prefix="ORB Levels",
            sessions=selected_sessions,
            session_visibility=session_visibility,
            show_buy_volume=show_buy_volume,
            show_sell_volume=show_sell_volume,
            show_day_boundaries=True,
            show_prev_levels=True,
            show_rangeslider=True,
            initial_x_range=initial_range,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        candle_frame = prepare_candlestick_frame(plot_df)
        bokeh_fig = make_bokeh_candlestick(
            candle_frame,
            title=chart_title,
            timeframe=chart_tf,
            sessions=selected_sessions,
            session_visibility=session_visibility,
            show_sessions=True,
            show_prev_levels=True,
            show_buy_volume=show_buy_volume,
            show_sell_volume=show_sell_volume,
            show_day_boundaries=True,
            x_range=initial_range,
        )
        st.bokeh_chart(bokeh_fig, use_container_width=True)
except ChartBackendError as exc:
    st.error(str(exc))

with st.expander("Full pipeline output (first 200 rows)", expanded=False):
    st.dataframe(frame.head(200), use_container_width=True)

full_csv_bytes = frame.to_csv().encode()
st.download_button(
    "Download full dataset",
    data=full_csv_bytes,
    file_name=f"orb_full_{chart_tf.replace('/', '-')}.csv",
    mime="text/csv",
    use_container_width=True,
)
