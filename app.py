from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from orb_data import DEFAULT_SESSIONS, OrbDataPipeline
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
    run_btn = st.button("Fetch data", type="primary")
    if st.button("Clear cached data", type="secondary"):
        run_pipeline_cached.clear()
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
        cache_key = (
            symbols_tuple,
            chart_tf,
            orb_tf,
            start_iso,
            end_iso,
            int(volume_window),
            int(percentile_bins),
        )
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
                    st.session_state["orb_cache_key"] = cache_key
                    st.session_state.pop("analysis_cache_key", None)
                    st.session_state.pop("analysis_outcomes", None)
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
view_mode = st.radio(
    "Chart mode",
    ["Continuous", "Daily"],
    horizontal=True,
    key="orb_view_mode",
)

current_date = st.session_state[date_state_key][symbol_choice]
current_idx = index_lookup[current_date]

if view_mode == "Daily":
    session_picker_key = f"date_picker_{symbol_choice}"
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
        current_idx = index_lookup[current_date]

prev_col, next_col = st.columns([1, 1])
with prev_col:
    disable_prev = index_lookup[current_date] == 0
    if st.button("\u2190 Prev Day", use_container_width=True, disabled=disable_prev):
        idx = index_lookup[current_date]
        if idx > 0:
            new_date = available_dates[idx - 1]
            st.session_state[date_state_key][symbol_choice] = new_date
            current_date = new_date
            current_idx = index_lookup[current_date]
with next_col:
    disable_next = index_lookup[current_date] == len(available_dates) - 1
    if st.button("Next Day \u2192", use_container_width=True, disabled=disable_next):
        idx = index_lookup[current_date]
        if idx < len(available_dates) - 1:
            new_date = available_dates[idx + 1]
            st.session_state[date_state_key][symbol_choice] = new_date
            current_date = new_date
            current_idx = index_lookup[current_date]

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

caption_sessions = ", ".join(s.upper() for s in selected_sessions)
focus_start = current_date
focus_end = current_date + pd.Timedelta(days=1)
x_range = (focus_start, focus_end)

mask = (local_series >= focus_start) & (local_series < focus_end)
filtered_df = symbol_df.loc[mask].copy()

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
    st.warning("No candles match the selected day/session filters.")
    st.stop()

filtered_df.sort_values("time_utc_plus4", inplace=True)
plot_df = filtered_df.set_index("time_utc_plus4")

caption_day = current_date.strftime("%Y-%m-%d")
if view_mode == "Daily":
    caption_text = f"Day: {caption_day} {LOCAL_TZ_LABEL} | Sessions: {caption_sessions}"
else:
    full_start = symbol_df["time_utc_plus4"].min()
    full_end = symbol_df["time_utc_plus4"].max()
    if pd.isna(full_start) or pd.isna(full_end):
        caption_text = f"Focus day: {caption_day} | Sessions: {caption_sessions}"
    else:
        caption_text = (
            f"Continuous range: {full_start.strftime('%Y-%m-%d %H:%M')} → "
            f"{full_end.strftime('%Y-%m-%d %H:%M')} {LOCAL_TZ_LABEL} | Focus day: {caption_day} | Sessions: {caption_sessions}"
        )

fig = make_orb_figure(
    plot_df,
    symbol=symbol_choice,
    timeframe=chart_tf,
    title_prefix="ORB Levels",
    sessions=selected_sessions,
    x_range=x_range,
    session_visibility=session_visibility,
    show_buy_volume=show_buy_volume,
    show_sell_volume=show_sell_volume,
)
st.caption(caption_text)
st.plotly_chart(fig, use_container_width=True)

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
