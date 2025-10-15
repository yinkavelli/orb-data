# ORB Project – Context & Handoff Doc

_Last updated: 09 Oct 2025 (Asia/Kolkata)_

## TL;DR
- **Project goal:** Build an ORB (Opening Range Breakout) research + charting tool for intraday scalping, with session-aware ORB levels, stats, and a Streamlit UI.
- **Code layout:** Modern `orb_data/*` package (✅ use this) coexists with legacy top‑level modules (⚠️ avoid mixing). Streamlit app lives in `app.py` with analysis pages under `pages/`.
- **UI state:** We reverted to a simple, always‑scrollable chart (no Daily/Continuous toggle). Day start markers (dotted vlines) are shown. Prev/Next day buttons sit **under** the chart.
- **Caching:** Data/figures are cached in `st.session_state`; recomputation happens **only** on explicit button clicks (Fetch or Run Analysis). We did earlier experiments with figure caching; we rolled back to the checkpoint branch for performance stability.
- **Analysis to date (30‑min data, BTC & ETH, 2020‑01‑01 → 2025‑10‑09):**
  - **US session:** `L1_bull` ~63.9% base hit; conditional edges up to **~77%** when the ORB candle is small/doji/hammer, closes ≥ mid, and shows **lower‑wick dominance**. Symmetric results for downside (`L1_bear` ~63.2% base; ~69% with bearish close < mid and **upper‑wick dominance**).
  - **Asia session:** Same patterns hold with slightly lower base rates (~59–60%).
  - L2/L3 and previous day/week levels drop markedly in hit rate (L2 ~50%, L3 ~30%).
- **Next analytics requested:** time‑to‑target & % move (entry = ORB close), test **ORB high/low retrace on next candle**, and add **volume–spread signatures** into the ORB signature search; report probabilities **bifurcated by ORB direction**.

---

## Repo & Branch State
- **Checkpoint branch:** `checkpoint/volume-spread-candles` (commit `dcd971c`) — stable baseline. Running `streamlit run app.py` here reflects “earlier work.”
- **Feature branch:** `feat/next-phase` had an extra commit adding an analysis dashboard; those edits were **stashed** as `stash@{0}: temp before checkpoint`.
- **Verify stash:**
  - `git status -sb` → clean tree
  - `git stash list` → shows the `temp before checkpoint` entry

### Git snippets
```bash
# Save local changes
git stash push -u -m "temp before checkpoint"

# Switch to checkpoint
git checkout checkpoint/volume-spread-candles
# (optional) ensure remote is current
git pull

# Later: return and reapply
git checkout feat/next-phase
git stash pop   # or: git stash apply
```

---

## Data Pipeline & Package Map

### High‑level flow
- `orb_data/pipeline.py` → **OrbDataPipeline**: resolves symbol list, expands fetch window, downloads OHLCV (ccxt) for chart/ORB/daily/weekly frames, computes ORB levels & technicals, returns a **multi‑indexed DataFrame** (symbol × time) with TZ‑aware timestamps.
- `orb_data/candles.py` → candlestick **pattern flags** (e.g., hammer), rolling **volume/spread percentiles**, **label bins**, and a `volume_spread_color` used by charts.
- `orb_data/orb_levels.py` → daily/session **ORB ranges** (high/low/mid, L1–L3 extensions, session ids/flags). Sessions: `asia`, `europe`, `us`, `overnight` via `SessionConfig`.
- `orb_data/client.py` → ccxt wrapper with rate‑limit & fallback (binance/binanceus); symbol filtering; batched `fetch_ohlcv` with normalized numerics and timestamps.
- `orb_data/figure.py` → Plotly candlestick: body/wick, session overlays, daily/weekly prior levels, EMA overlays, optional buy/sell volume, day‑boundary **dotted vlines**; session colors via `SESSION_COLOR_MAP`.
- `orb_data/__init__.py` → exports package surface; `pyproject.toml` registers `orb-data` deps: **ccxt, pandas, numpy, plotly, streamlit**.

### Streamlit UI
- `app.py` sidebar → parameters drive cached pipeline (`run_pipeline_cached`), persist in `st.session_state`, configure chart (session buttons, prev/next day nav below chart). **Rangeslider** for horizontal scrolling.
- `pages/01_Targeted_Analysis.py` → on-demand analysis with "Run targeted analysis" button; tabulates hit rates and shows bar/overlay charts.

### Data & artifacts
- `orb_full_5m.csv` → example pipeline output with ORB + candle stats and splits.
- `hourly_orb_outcomes.csv` → precomputed session outcomes.
- Legacy notebooks: `ORB.ipynb`, `Multi ORB.ipynb`.
- Legacy scaffolding script: `bootstrap_orb.ps1` (older layout under `orb_pipeline/*`).

**Note:** Legacy top‑level modules (`client.py`, `orb_levels.py`, `pipeline.py`, `plotting.py`) overlap names with `orb_data/*` but lack newer features (weekly levels, percentiles). Ensure imports resolve to `orb_data.*`.

---

## Charting & UI Decisions
- **Removed chart mode** (Daily/Continuous) — the chart is always scrollable with Plotly’s rangeslider.
- **Day boundaries:** grey dotted vlines at UTC‑converted day starts.
- **Prev/Next day** buttons positioned **under** the chart.
- **Previous-day/week levels:** plotted as **stepwise time‑series**, not flat global lines.
- **Default windowing:** the initial zoom scales with timeframe (smaller tf → tighter view; larger tf → broader window) while retaining full-history scrolling via the rangeslider.

---

## Caching & Rerun Behavior
- **Goal:** No recomputation when flipping pages; recompute only on explicit user actions.
- **Mechanisms used over iterations:**
  - `st.session_state` for storing: parameters, the fetched `orb_df`, a figure cache, and (optionally) an **analysis cache** keyed by a dataset signature.
  - **Dataset signature** (symbol/timeframe/date/session set) so caches invalidated **only** when inputs actually change.
  - A **Clear cached data** button to drop dataset/figure/analysis caches.

If you see recomputation on page changes, confirm: (1) controls are wrapped in non‑eager widgets, (2) derived state isn’t recomputed at import time, (3) figure construction is gated behind cache checks.

---

## Targeted Analysis – Current Findings (30‑min, BTC+ETH)

### Baseline hit rates (US session)
- `L1_bull` **63.9%**; `L1_bear` **63.2%**
- `L2_bull/bear` ~**50%**
- `L3_bull/bear` ~**30%**
- `prev_day_high/low` ~**39% / 33%**
- `prev_week_high/low` ~**30% / 19%**

### Long‑side edges (target: `L1_bull`)
- **Any ORB:** 63.9%
- **Bullish close & close ≥ mid:** ~**70%**
- **Doji close ≥ mid:** ~**77%** _(smaller sample)_
- **Hammer (is_hammer):** ~**71%** _(smaller sample)_
- **Tiny body (Q1) + lower‑wick dominance:** ~**73%**
- **Avoid:** bearish close < mid → ~**55%**

### Short‑side edges (target: `L1_bear`)
- **Any ORB:** 63.2%
- **Bearish close & close < mid:** ~**69%**
- **Upper‑wick dominance:** ~**69%**
- **Bearish doji < mid:** ~**69%**
- **Large body bearish:** ~**68%**
- **Avoid:** bullish close ≥ mid → ~**56%**

### Asia session echoes
- Same signatures work with slightly lower base rates (~59–60% → mid‑60s with bullish ≥ mid + lower‑wick dominance for longs).

### Definitions (as used)
- **Close ≥ mid:** `close >= (orb_high + orb_low)/2`.
- **Body ratio:** `abs(close - open) / (high - low)`.
- **Upper/lower wick ratio:** `[high - max(open, close)] / (high - low)` and `[min(open, close) - low] / (high - low)`.
- **Wick dominance:** compare upper vs lower wick ratio.
- **Pattern flags:** e.g., `is_doji`, `is_hammer` from `orb_data/candles.py`.

---

## Requested Next Analyses (to implement)
1. **Time‑to‑Target (TTT):**
   - For each ORB signature → measure bars/minutes from ORB close to first touch of each target (`orb_high/low`, `L1–L3`, `PDH/PDL`, `PWH/PWL`).
   - Report **median/percentiles (p25/p75)** and **timeout rate** (not hit within session/day).

2. **% Move to Target:**
   - Entry at **ORB close**; compute `(target_price - entry)/entry` (signed) until first touch or timeout; summarize distribution per signature.

3. **ORB High/Low Retrace (Next‑Candle scalp):**
   - Despite being set by the ORB candle itself, test if **next candle revisits** the ORB high/low.
   - Conditioned on ORB direction and wick dominance. Output: probability, TTT, average excursion.

4. **Volume–Spread Signatures:**
   - Use rolling **volume percentiles** and **spread percentiles** from `candles.py` to bin ORB candles (e.g., vol ≥ 75th & spread ≥ 75th = “wide & heavy”).
   - Cross with direction & close vs mid to find edges (e.g., **wide+heavy bullish ≥ mid** → L1_bull in X% with Y‑minute median TTT).

5. **Direction‑Bifurcated Reporting:**
   - Every signature’s probability & TTT/% move reported separately for **bullish ORB** and **bearish ORB**.

6. **Visualization hooks:**
   - Add overlay of **best‑performing signature** on the chart (e.g., shaded bars where conditions met) and annotate realized touches (✅) vs timeouts (✖︎).

---

## Known Issues & Fixes Log
- **`NameError: current_date`** → Introduced explicit view controls; later removed when chart mode was dropped.
- **`TypeError: Cannot localize tz‑aware Timestamp`** → Avoided re‑localizing tz‑aware series; use `tz_convert` for aware timestamps.
- **`ValueError: cannot insert symbol, already exists`** → When resetting index on MultiIndex frames, **drop columns matching index level names** first; rehydrate symbol only if missing.
- **Page reruns & performance** → Prefer cached dataset/figure keyed by a **dataset signature**; compute analysis only on button press; avoid heavy work at import time.

---

## Open Questions / To Decide
- Should we **retire** legacy modules to prevent import collisions? At least add a README note or delete if safe.
- Confirm whether `orb_target_outcomes.csv` is still needed (referenced by IDE but missing in the repo).
- Default session timezone: pipeline uses `Etc/GMT‑4` while the UI labels **UTC+4**; verify this is intentional and consistent.

---

## Quick Start (new run from checkpoint)
1. `git checkout checkpoint/volume-spread-candles && git pull`
2. `python -m compileall app.py orb_data` (sanity check)
3. `streamlit run app.py`
4. Set sidebar params → **Fetch data**
5. Navigate to **01_Targeted_Analysis** → **Run targeted analysis**

---

## Appendix – Suggested Metrics Schema
- **Per target T:**
  - `hit` (bool), `bars_to_hit`, `minutes_to_hit`, `%move_to_hit`, `timeout` (bool), `MAE/MFE` until hit/timeout.
- **Per signature S:**
  - counts, hit‑rate, median/p25/p75 TTT, median/p25/p75 %move, avg/median MFE/MAE, win/lose expectancy with simple stop placements (e.g., ±X% of ORB range).
- **Signature knobs:**
  - `orb_dir ∈ {bull, bear}`; `close_vs_mid ∈ {≥, <}`; `body_bin ∈ {Q1..Q4}`; `wick_dom ∈ {upper, lower, none}`; `pattern_flags`; `vol_bin`, `spread_bin`; `session`.

