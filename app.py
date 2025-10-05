from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd
import streamlit as st

from orb_data import DEFAULT_SESSIONS, OrbDataPipeline
from orb_data.plotting import make_orb_figure

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
    run_btn = st.button("Fetch data", type="primary")

symbols: List[str] = []
if symbols_text.strip():
    symbols = [item.strip().upper() for item in symbols_text.split(",") if item.strip()]

if run_btn:
    if not symbols:
        st.error("Please enter at least one symbol (e.g. BTC/USDT).")
    else:
        with st.status("Fetching data...", expanded=True) as status:
            start_iso = start_dt.isoformat()
            end_iso = end_dt.isoformat() if end_dt else None
            status.write(f"Running pipeline for {len(symbols)} symbol(s)...")
            try:
                pipeline = OrbDataPipeline(
                    symbols=symbols,
                    chart_timeframe=chart_tf,
                    start=start_iso,
                    end=end_iso,
                    orb_timeframe=orb_tf,
                    volume_percentile_window=int(volume_window),
                    percentile_bins=int(percentile_bins),
                )
                frame = pipeline.run()
            except Exception as exc:
                status.update(label="Fetch failed", state="error")
                st.error(f"Pipeline error: {exc}")
                frame = pd.DataFrame()
            else:
                if frame.empty:
                    status.update(label="No data returned", state="warning")
                    st.warning("No data returned for the selected configuration.")
                else:
                    status.update(label="Fetch complete", state="complete")
                    st.success(f"Fetched {frame.shape[0]} rows across {len(pipeline.symbols)} symbol(s).")
                    exchange_name = getattr(pipeline.client, "exchange_id", None)
                    if exchange_name:
                        st.caption(f"Data source: ccxt.{exchange_name}")
                    st.session_state["orb_df"] = frame
                    st.session_state["orb_symbols"] = list(pipeline.symbols)
                    st.session_state["orb_chart_tf"] = chart_tf
                    st.session_state["orb_orb_tf"] = orb_tf
                    st.session_state["orb_volume_window"] = int(volume_window)
                    st.session_state["orb_percentile_bins"] = int(percentile_bins)
                    if exchange_name:
                        st.session_state["orb_exchange"] = exchange_name
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

session_cols = [col for col in frame.columns if col.startswith("session_id_")]
if session_cols:
    session_names = [col.replace("session_id_", "") for col in session_cols]
    color_map = {
        "asia": "#1f77b4",
        "europe": "#ff7f0e",
        "us": "#2ca02c",
        "overnight": "#9467bd",
    }
    cards = st.columns(len(session_names))
    for col, name in zip(cards, session_names):
        with col:
            st.markdown(
                f"<div style='background:{color_map.get(name, '#cccccc')}; color:white; padding:6px 10px; border-radius:6px; font-size:0.75rem; text-align:center;'>"
                f"{name.upper()}" "</div>",
                unsafe_allow_html=True,
            )
symbol_df = _extract_symbol_frame(frame, symbol_choice)
if symbol_df.empty:
    st.warning("No data available for the selected symbol.")
    st.stop()

symbol_df = _ensure_time_columns(symbol_df)
local_series = symbol_df["time_utc_plus4"]
available_dates = sorted(local_series.dt.normalize().unique())
if not available_dates:
    st.warning("No timestamps found for the selected symbol.")
    st.stop()

date_state_key = "orb_selected_dates"
if date_state_key not in st.session_state:
    st.session_state[date_state_key] = {}
if symbol_choice not in st.session_state[date_state_key] or st.session_state[date_state_key][symbol_choice] not in available_dates:
    st.session_state[date_state_key][symbol_choice] = available_dates[-1]

index_lookup = {ts: idx for idx, ts in enumerate(available_dates)}
current_date = st.session_state[date_state_key][symbol_choice]

session_picker_key = f"date_picker_{symbol_choice}"
current_idx = index_lookup[current_date]
picked = st.selectbox(
    f"Session day ({LOCAL_TZ_LABEL})",
    options=available_dates,
    index=current_idx,
    format_func=lambda ts: ts.strftime("%Y-%m-%d"),
    key=session_picker_key,
)
if picked != current_date:
    st.session_state[date_state_key][symbol_choice] = picked
    current_date = picked

prev_col, next_col = st.columns([1, 1])
with prev_col:
    disable_prev = index_lookup[current_date] == 0
    if st.button("\u2190 Prev Day", use_container_width=True, disabled=disable_prev):
        idx = index_lookup[current_date]
        if idx > 0:
            new_date = available_dates[idx - 1]
            st.session_state[date_state_key][symbol_choice] = new_date
            st.session_state[session_picker_key] = new_date
            current_date = new_date
with next_col:
    disable_next = index_lookup[current_date] == len(available_dates) - 1
    if st.button("Next Day \u2192", use_container_width=True, disabled=disable_next):
        idx = index_lookup[current_date]
        if idx < len(available_dates) - 1:
            new_date = available_dates[idx + 1]
            st.session_state[date_state_key][symbol_choice] = new_date
            st.session_state[session_picker_key] = new_date
            current_date = new_date

current_date = st.session_state[date_state_key][symbol_choice]
current_idx = index_lookup[current_date]

session_mode = st.radio("Session filter", ["All sessions", "Custom"], horizontal=True, key="session_mode")
session_select_key = "session_multiselect"
if session_select_key not in st.session_state:
    st.session_state[session_select_key] = SESSION_NAMES.copy()

selected_sessions: List[str]
if session_mode == "All sessions":
    selected_sessions = SESSION_NAMES.copy()
else:
    preset_choice = st.selectbox(
        "Quick presets",
        options=["Keep current"] + list(SESSION_PRESETS.keys()),
        help="Apply a pre-defined session combination, then fine-tune below.",
        key="session_preset",
    )
    if preset_choice != "Keep current":
        st.session_state[session_select_key] = SESSION_PRESETS[preset_choice]
    selected_sessions = st.multiselect(
        "Visible sessions",
        SESSION_NAMES,
        key=session_select_key,
        help="Pick one or more sessions to overlay.",
    )
    if not selected_sessions:
        selected_sessions = SESSION_NAMES.copy()

start_ts = current_date
end_ts = current_date + pd.Timedelta(days=1)
mask = (local_series >= start_ts) & (local_series < end_ts)
day_df = symbol_df.loc[mask].copy()

if not day_df.empty and selected_sessions != SESSION_NAMES:
    session_cols = [f"session_id_{name}" for name in selected_sessions if f"session_id_{name}" in day_df.columns]
    if session_cols:
        session_mask = pd.Series(False, index=day_df.index)
        for col in session_cols:
            session_mask |= day_df[col].notna()
        day_df = day_df.loc[session_mask]

if day_df.empty:
    st.warning("No candles match the selected day/session filters.")
    st.stop()

day_df.sort_values("time_utc_plus4", inplace=True)
plot_df = day_df.set_index("time_utc_plus4")
fig = make_orb_figure(
    plot_df,
    symbol=symbol_choice,
    timeframe=chart_tf,
    title_prefix="ORB Levels",
    sessions=selected_sessions,
)
caption_day = current_date.strftime("%Y-%m-%d")
caption_sessions = ", ".join(s.upper() for s in selected_sessions)
st.caption(f"Day: {caption_day} {LOCAL_TZ_LABEL} | Sessions: {caption_sessions}")
st.plotly_chart(fig, use_container_width=True)

st.markdown("<h4 style='text-align:center;'>Filtered rows (latest 10)</h4>", unsafe_allow_html=True)
table_df = day_df.copy()
local_col = f"time_{LOCAL_TZ_LABEL}"
table_df.insert(0, local_col, table_df["time_utc_plus4"].dt.strftime("%Y-%m-%d %H:%M"))
if "time_utc" in table_df.columns:
    table_df["time_utc"] = table_df["time_utc"].dt.strftime("%Y-%m-%d %H:%M")
preferred_order = [
    local_col,
    "time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "volume_usdt",
    "symbol",
]
ordered_cols = [col for col in preferred_order if col in table_df.columns]
ordered_cols += [col for col in table_df.columns if col not in ordered_cols]
st.dataframe(table_df[ordered_cols].tail(10), use_container_width=True)

csv_bytes = day_df.to_csv().encode()
btn_cols = st.columns([1, 1, 1])
with btn_cols[1]:
    st.download_button(
        "Download visible slice",
        data=csv_bytes,
        file_name=f"orb_{symbol_choice.replace('/', '_')}_{caption_day}.csv",
        mime="text/csv",
    )

with st.expander("Full pipeline output (first 200 rows)", expanded=False):
    st.dataframe(frame.head(200), use_container_width=True)
