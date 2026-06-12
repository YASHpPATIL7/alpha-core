# Methodology & Referee Defense Checklist
## "Regime-Conditional Factor Investing in Indian Equities"

For Antigravity: implement these fixes in priority order before running Table 3.
Each item lists the exact code change needed.

---

## PRIORITY 1 — FATAL if unfixed (kill Panel B)

### 1. HMM Look-Ahead Bias

**Problem:** Full-sample Viterbi smoothing embeds future returns into regime labels.
Every date in your "Bear" bucket knows what happens after it.
Conditioning a causality test on a state estimated from the outcome variable
= manufactured significance.

**Fix A (correct, preferred):** Use *filtered* regime probabilities only.
In your HMM script, replace smoothed labels with an explicit forward-pass recursion:
```python
# WRONG (uses future data):
regime_labels = model.predict(returns_matrix)  # Viterbi, full-sample
# ALSO WRONG (uses forward-backward smoothing):
# filtered_probs = model.predict_proba(returns_matrix)

# RIGHT (online, causal):
filtered_probs = filtered_state_probs(model, returns_matrix)  # explicit forward-only recursion
```

**Fix B (acceptable for monthly rebalancing):** Walk-forward HMM.
Re-estimate HMM on rolling 756-day window (3 years), label only the next day.
More expensive but fully defensible.

**Fix C (minimum viable):** Lag regime label by 1 day.
Use `regime(t-1)` when conditioning tests at time `t`.
This doesn't fix smoothing bias but removes simultaneity.
State clearly in footnote: "Results condition on prior-day regime to avoid
contemporaneous look-ahead."

**Implement Fix C now, Fix A before submission.**

---

### 2. Multiple Testing — Benjamini-Hochberg FDR

**Problem:** Panel B is 15 tests (5 factors × 3 regimes). Finding RMW in Bear
at p<0.01 among 15 tests is expected by chance ~15% of the time.
Any referee will ask for FDR correction before accepting Panel B.

**Fix:** Add this block after collecting all Panel B p-values:

```python
from statsmodels.stats.multitest import multipletests

# Collect all p-values from the 15-cell grid
all_pvals = panel_b_df["p_value"].values
reject, pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")
panel_b_df["p_fdr"] = pvals_corrected
panel_b_df["sig_fdr"] = panel_b_df["p_fdr"].apply(sig_stars)
```

Report both columns in the table. If significant cells survive FDR: gold.
If they don't: still publishable as "FII contains market-level but not
factor-level incremental information after multiple testing correction."
Either way you're honest.

---

### 3. Factor Data — Use IIMA Instead of Rolling Own

**Problem:** RMW and CMA from yfinance require clean historical book equity
and operating profitability for NSE stocks. yfinance does not provide
point-in-time fundamentals. Survivorship-free construction is essentially
impossible with public data. A referee will ask and you cannot defend it.

**Fix:** Download IIM Ahmedabad Indian Factor Data (Agarwalla, Jacob, Varma).
Publicly available, peer-reviewed, updated through recent years.

URL: https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/

Files available: Monthly factor returns for MKT, SMB, HML, WML (momentum).
Note: IIMA data is monthly. Two options:

**Option A (preferred):** Use IIMA monthly factors for Table 1/2 statistical
tests. Interpolate to daily for Table 3 Granger tests. State clearly:
"Monthly IIMA factors are linearly interpolated to daily frequency for
flow-factor Granger tests; robustness using own-constructed daily factors
reported in Appendix."

**Option B:** Keep own-constructed factors, add IIMA as robustness check.
Table footnote: "Primary results use own-constructed daily factors from NSE
universe; Table A1 replicates using monthly IIM-Ahmedabad factors (Agarwalla,
Jacob, and Varma, 2014)."

Option B is minimum viable. Option A is cleaner.

**Citation to add:**
Agarwalla, S.K., Jacob, J., and Varma, J.R. (2014).
"A Four-Factor Model in Indian Equities Market."
IIM Ahmedabad Working Paper No. 2014-09-035.

---

## PRIORITY 2 — Referee will flag, not fatal

### 4. COVID Robustness Window

**Problem:** 2015–2026 has one dominant Bear episode (Mar–May 2020). Panel B's
Bear-regime results may be entirely driven by this single event.

**Fix:** Add `exclude_covid` parameter to all regime-conditional tests.

```python
COVID_START = "2020-02-01"
COVID_END   = "2020-06-30"

def mask_covid(df):
    return df[(df.index < COVID_START) | (df.index > COVID_END)]
```

Run every panel with `exclude_covid=False` (main table) and
`exclude_covid=True` (robustness appendix). Report both.

If the finding survives ex-COVID: state it as a strength in the paper.
"The factor-selective predictability of FII flows in Bear regimes persists
after excluding the COVID-19 crash window, suggesting it is not a one-time
structural event."

If it doesn't survive: acknowledge honestly in limitations. Still publishable;
you've correctly identified the mechanism.

---

### 5. Newey-West / HAC Standard Errors

**Problem:** Daily returns and daily FII flows are both autocorrelated and
heteroskedastic. Vanilla Granger F-statistics with OLS standard errors
overstate significance. This is a standard methodological gap referees flag.

**Fix:** In `run_granger()`, after estimating the VAR, use HAC covariance:

```python
# In statsmodels, grangercausalitytests uses internal OLS.
# For HAC-corrected version, estimate manually:

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# For each lag k, build the restricted and unrestricted models,
# compute F-stat using HAC covariance matrix.
# Or: use the 'params_ftest' with cov_type='HAC' in VAR estimation.
```

Simpler alternative: use `statsmodels.tsa.vector_ar.var_model.VAR` with
`cov_type='HAC'` and `cov_kwds={'maxlags': 5, 'use_correction': True}`.

State in paper: "All F-statistics use Newey-West heteroskedasticity and
autocorrelation consistent (HAC) standard errors with bandwidth = 5."

---

### 6. Panel C — Replace HMM Onset with Objective Criterion

**Problem:** "FII predicts Bear onset 3–5 days before HMM detects it" is
circular if HMM onset is defined by smoothed labels (HMM knows the future).
Also: AUC looks fine when classes are imbalanced but precision can be poor.

**Fix:** Define Bear onset using an objective forward-looking drawdown criterion:

```python
# Bear onset = Nifty drawdown > 8% over next 20 trading days
rolling_max = nifty_returns.cumsum().rolling(20).max()
drawdown_fwd = rolling_max - nifty_returns.cumsum().shift(-20)
bear_onset_objective = (drawdown_fwd > 0.08).astype(int)
```

This is:
1. Not derived from the same HMM → no circularity
2. Objective and replicable
3. Operationally meaningful (a PM cares about drawdown, not HMM states)

Also report precision-recall curve (not just AUC) since Bear onset is rare.
Add DeLong confidence intervals around AUC.

---

## PRIORITY 3 — Language / Framing (1 hour, high ROI)

### 7. Mechanism Language

**Never say:** "FII herds into quality stocks and exits together under stress"

**Say instead:** "Results are consistent with institutional quality-herding
during market stress. We do not directly observe stock-level FII positions,
but the systematic compression of the profitability premium (RMW) coincident
with large FII outflows in Bear regimes is consistent with this mechanism.
Quarterly NSDL FPI stock-level holdings data could provide direct verification."

### 8. Survivorship Bias — Standard Acknowledgment

Add to Data section:
"Our NSE universe includes all constituents of the Nifty 200 index as of
[date]. We acknowledge survivorship bias: firms delisted or significantly
restructured over 2015–2026 (e.g., YES Bank, DHFL, IL&FS subsidiaries)
are excluded from historical returns. This bias tends to inflate value and
profitability premia. Results should be interpreted with this limitation."

### 9. Always Report n Per Regime

Every regime-conditional table: include a column for n (number of observations).
A "null" in Bear with n=80 has low power. A "null" with n=800 is a real null.
These have different implications for the paper.

---

## ROBUSTNESS TABLES TO INCLUDE (instruct Antigravity)

| Table | Contents |
|-------|----------|
| Table A1 | Panel B replicated with IIMA monthly factors |
| Table A2 | Panel B replicated excluding COVID window (Feb–Jun 2020) |
| Table A3 | Panel B with FDR-adjusted p-values (BH correction) |
| Table A4 | Panel A with HAC standard errors vs. OLS standard errors |
| Figure A1 | Precision-Recall curve for Panel C early warning model |

---

## ONE-LINE DEFENSES FOR EACH ATTACK

| Attack | Defense |
|--------|---------|
| "HMM look-ahead" | "Regime labels lagged one day; filtered (causal) probabilities used as robustness." |
| "Multiple testing" | "BH-FDR correction applied across all 15 cells; [X] cells survive at q<0.05." |
| "Factor construction" | "Primary results replicated using IIM-Ahmedabad factors (Table A1)." |
| "COVID dominance" | "Results hold excluding the COVID-19 window (Table A2)." |
| "Herding claim" | "Framed as consistent-with, not direct evidence; aggregate flow data does not support stock-level attribution." |
| "HAC errors" | "All F-statistics computed with Newey-West HAC, bandwidth=5 (Table A4)." |
| "Survivorship bias" | "Acknowledged; bias direction is upward for RMW/HML, making rejection of factor significance conservative." |
| "Simultaneity" | "Strictly lagged predictors (t-1 to t-5); all Granger tests use past information only." |
