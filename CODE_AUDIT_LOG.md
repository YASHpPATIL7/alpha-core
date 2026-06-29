# Code Audit Log — Alpha-Core

**Repo:** `alpha-core`
**Latest entries:** 2026-06-28 → 2026-06-29
**Format:** each entry = What was wrong · Why it matters · The fix · Files · How to explain it

This file documents the regime-detection and cointegration changes made in this
work block. The companion dashboard changes are logged in
`quant-dashboard/CHANGELOG.md`.

---

## Summary of this work block

Triggered by external review (the live dashboard showed a **BULL** regime and a
**−2.46% CVaR** that never moved, and the cointegration pairs looked spurious).
Three real problems were found and fixed:

1. **Stale data** — the dashboard read CSVs from GitHub `main` that were frozen
   in 2019/2022. Fixed by regenerating + pushing fresh data (see entries below
   for the model changes that the regenerated data now reflects).
2. **Regime mislabelled as Bull** — the HMM could not tell a slow grind-down
   bear from a steady bull, so it over-called Bull. Fixed with a new feature.
3. **Cointegration fitting noise** — pairs were selected on raw prices (shared
   market beta) with no economic rationale. Fixed with factor-neutralisation +
   a sector screen.

Net result: current regime now reads **Sideways** (was Bull), regime validation
went **2/4 → 3/4**, and the 3 spurious cross-sector pairs were removed.

---

## Entry #1 — Regime: honest state probabilities (no more single brittle label)

**What was wrong**
`regime_labels.csv` stored only a hard label (`regime_name`). A single word like
"BULL" is brittle and easy to call wrong, and hides how (un)confident the model is.

**Why it matters**
A visibly wrong hard label is an instant credibility hit. Probabilities are
honest and let the dashboard show "Bull 12% · Sideways 30% · Bear 58%".

**The fix**
`run_hmm_pipeline()` now writes the **filtered** posterior `P(state_t | x_1..x_t)`
(past-only, no look-ahead) as `prob_bull` / `prob_sideways` / `prob_bear`, plus
`regime_confidence` (the max posterior). Columns are named by regime, not by the
arbitrary HMM state integer, so consumers never re-derive the mapping.

**Files**
`alpha_core/hmm_regime.py` — `run_hmm_pipeline()` (output build), live-readout log.

**How to explain it**
"I expose the filtered state posterior, not just an argmax label, so the regime
read-out is honest about its own uncertainty."

---

## Entry #2 — Regime: 63-day trend feature (the real fix for the Bull bias)

**What was wrong**
The HMM features were daily return, India VIX, and a 20-day momentum-Sharpe.
Over any 20-day window a slow grind-down bear (e.g. the H1-2022 rate-hike
selloff) looks mildly positive — indistinguishable from a steady bull. The model
therefore **assigned H1-2022 days to the Bull state itself** (90% of days), and
the Oct-2023 election rally to the flat state. No relabelling could fix this —
the days were in the wrong state to begin with.

**Why it matters**
This is why the live read-out skewed Bull and why H1-2022 / Oct-2023 failed the
historical validation. The bug was in the **feature set**, one layer below the
label logic.

**The fix**
Added a 4th feature: `trend_63d = mkt.rolling(63).mean()` — the average daily
drift over ≈ a quarter. It is negative through sustained declines and positive
through sustained rallies, giving the HMM the persistence signal it lacked.
Offline separation (annualised 63-day trend): COVID −110%, 2021 +21%,
H1-2022 −20%, Oct-2023 +17% — clean.

**Result**
- Current regime flipped **BULL (100%) → SIDEWAYS (100%)** — matches reality
  (late-Apr-2026 has a negative quarter trend).
- Validation **2/4 → 3/4**: COVID ✓, 2021 ✓, **Oct-2023 fixed ✓**,
  H1-2022 now Sideways (was Bull).
- States are now economically clean: Bear = crash (−33%/yr, 44% vol),
  Sideways = chop/mild drift (+2%/yr), Bull = steady uptrend (+8%/yr, 10% vol).

**Files**
`alpha_core/hmm_regime.py` — `load_features()` (new feature), `label_states()`.

**How to explain it**
"A 20-day momentum can't separate a slow bear from a bull — both look flat over
a month. I added a quarter-scale trend feature; that's what lets the model put
the 2022 rate-hike selloff somewhere other than 'Bull'."

---

## Entry #3 — Regime: return-aware state labelling

**What was wrong**
`label_states()` mapped Bull = max momentum and Bear = **max VIX only**. Defining
Bear by volatility alone means a low-volatility decline is never tagged Bear.

**Why it matters**
It baked a "Bear = high-vol crisis only" definition into the labels, compounding
the feature problem above.

**The fix**
States are ranked by a balanced score
`bull_score = z(return) + z(momentum) + z(trend) − z(VIX)`;
top = Bull, bottom = Bear, middle = Sideways. This rewards trend/momentum and
penalises turbulence, catching both crisis bears (high VIX) and — together with
the new trend feature — sustained declines.

**Files**
`alpha_core/hmm_regime.py` — `label_states()` (n==3 branch).

**How to explain it**
"Labels are assigned by a return/trend/VIX composite, not by volatility alone, so
the naming follows the economics of each state."

---

## Entry #4 — Cointegration: factor-neutralisation (stop fitting market beta)

**What was wrong**
Pairs were Johansen-tested on **raw reconstructed prices**. Two stocks can look
cointegrated simply because they both load on the market (and SMB/HML/RMW/CMA) —
that is shared beta, not a genuine pair relationship. The pre-fix "tradeable"
output (TCS/BAJFINANCE, HDFCBANK/INFY, RELIANCE/ITC) was exactly this artefact:
all cross-sector, no economic anchor.

**Why it matters**
This is the textbook "fitting to noise" trap a quant reviewer probes first.

**The fix**
Reconstruct **idiosyncratic** prices from the FF5 residual returns
(`factor_residuals.csv`) and run Johansen on those. A spread that survives on
residuals reflects a real stock-specific link, not common-factor co-movement.
Each pair now reports both raw and residual test statistics for transparency.

**Result**
3 raw-price pairs were exposed as spurious (died on residuals):
TCS/BAJFINANCE, HDFCBANK/INFY, RELIANCE/HINDUNILVR.

**Files**
`alpha_core/cointegration.py` — `load_residual_prices()`, `_eval_pair()`,
`scan_all_pairs()`, `run_cointegration_scanner()`, signal generation on residual prices.

**How to explain it**
"I test cointegration on FF5 residuals, not raw prices — otherwise you're just
rediscovering that both names track the market."

---

## Entry #5 — Cointegration: economic-mechanism (sector) screen

**What was wrong**
No requirement that a statistical pair have a *reason* to co-move.

**Why it matters**
A spread with no economic mechanism is a coincidence; it has no reason to persist
out-of-sample.

**The fix**
Added a sector map + `pair_mechanism()`; a pair must be **same-sector** (with a
stated mechanism, e.g. "shared rate cycle, credit growth & deposit dynamics") to
be tradeable. Final gate = factor-neutral cointegration **and** same sector
**and** half-life in 5–60d.

**Result**
0 pairs clear the full bar — an honest outcome among 14 diversified large-caps.
The closest same-sector candidates are surfaced as a **monitored watchlist**
(`cointegration_watchlist.csv`): INFY/TCS, AXISBANK/BAJFINANCE, ICICIBANK/BAJFINANCE,
RELIANCE/ONGC, HDFCBANK/BAJFINANCE — they pass the trace test but not the stricter
max-eigenvalue leg.

**Files**
`alpha_core/cointegration.py` — `SECTOR_MAP`, `SECTOR_MECHANISM`, `pair_mechanism()`,
watchlist build + save.

**How to explain it**
"Statistical significance isn't enough — I require a same-sector economic
mechanism, and I'd rather show zero tradeable pairs than trade a coincidence."

---

## Known limitations / next steps (not yet done)

- **H1-2022 reads Sideways, not Bear.** Defensible — it was a ~−15% slow drift,
  not a crash, and the Bear state is now reserved for crises. Getting it as a
  distinct mild-bear state would require **K=4** (Crash / Bear / Sideways / Bull).
  BIC actually prefers K≥4; we force K=3 for the 3-regime downstream contract
  (Kelly sizing, dashboard colours). K=4 is a deliberate next-session change.
- **Data ends 2026-04-30.** "Today's" regime needs the market-data pipeline
  refreshed through the current date.
- Chasing 4/4 on the four hand-picked validation windows was deliberately avoided
  — that would be curve-fitting the validation set.

---

## Reproduce

```bash
cd alpha-core && source venv/bin/activate
python -m alpha_core.hmm_regime        # regime_labels.csv (+ prob_* + trend_63d)
python -m alpha_core.cointegration     # *_all_pairs / _tradeable / _watchlist
```
