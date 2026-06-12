"""
DCC-Based Stress Detection as a Leading Indicator in Indian Equities
====================================================================
Paper: "Realized Correlation Dynamics and Regime Stress in Indian Equities:
        DCC Evidence, Factor Premium Collapse, and the VIX Lead"

Methodology:
  Step 1 — DCC Time Series Construction
    (a) GARCH(1,1) on each of 14 NSE stocks → standardised residuals eps_t
    (b) DCC recursion:
          Q_t = (1-a-b)*Q_bar + a*(e_{t-1} @ e_{t-1}.T) + b*Q_{t-1}
          R_t = diag(Q_t)^{-0.5} @ Q_t @ diag(Q_t)^{-0.5}
    (c) dcc_avg(t) = mean(upper triangle of R_t) — one scalar per day

  Panel A — DCC Granger-leads India VIX
    Bivariate VAR: does Δdcc_avg_{t-k} predict Δlog_vix_t after
    controlling for own lags? HAC F-stats, AIC lag selection.

  Panel B — High DCC predicts factor Sharpe collapse
    Quintile sort on dcc_avg(t) → forward 21-day factor Sharpe by quintile.
    Monotone decline in factor diversification benefit = confirmation.

  Panel C — AUC comparison: DCC vs VIX vs FII as Bear onset predictor
    Logit models using 5-day lagged predictors; AUC-ROC for each.
    Bear onset = HMM Bear within next 5 days.

  Panel D — Walk-forward Granger stability (rolling 500-day windows)
    Tests whether the DCC→VIX lead holds out-of-sample across sub-periods.
    Reports fraction of windows where DCC Granger-causes VIX at 10% level.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
KUBER_DIR  = BASE_DIR.parent / "ml-portfolio-optimizer" / "data"
VAJRA_DIR  = BASE_DIR.parent / "indian-risk-engine" / "data"
OUT_DIR    = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

MAX_LAG    = 5
ROLL_WIN   = 500      # days for walk-forward Granger
FWD_DAYS   = 21      # forward window for factor Sharpe collapse test
BEAR_FWD   = 5       # days ahead for Bear onset definition


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1 — LOADING DATA")
print("=" * 70)

# Stock returns
ret = pd.read_csv(DATA_DIR / "vajra_returns.csv", index_col=0, parse_dates=True)
stocks = ret.columns.tolist()
print(f"Returns: {ret.shape} | {ret.index[0].date()} → {ret.index[-1].date()}")
print(f"Stocks ({len(stocks)}): {', '.join(stocks)}")

# India VIX
vix_raw = pd.read_csv(DATA_DIR / "india_vix_history.csv", index_col=0, parse_dates=True)
vix_raw.columns = ["vix"]
print(f"VIX: {len(vix_raw)} days | {vix_raw.index[0].date()} → {vix_raw.index[-1].date()}")

# Factor returns
fr = pd.read_csv(DATA_DIR / "factor_returns.csv", index_col=0, parse_dates=True)
factors = [c for c in ["MKT","SMB","HML","RMW","CMA"] if c in fr.columns]
fr = fr[factors]
print(f"Factors: {factors} | {len(fr)} days")

# FII flows
fii = pd.read_csv(DATA_DIR / "fii_nsdl_direct.csv", index_col=0, parse_dates=True)
if "net_crore" not in fii.columns:
    fii.columns = [c.strip().lower().replace(" ","_") for c in fii.columns]
    net_col = [c for c in fii.columns if "net" in c][0]
    fii = fii[[net_col]].rename(columns={net_col: "net_crore"})
else:
    fii = fii[["net_crore"]]
print(f"FII: {len(fii)} days")

# Regime labels
rl = pd.read_csv(DATA_DIR / "regime_labels.csv", index_col=0, parse_dates=True)
print(f"Regimes: {rl['regime_name'].value_counts().to_dict()}")

# DCC params and Q_bar
dcc_params = pd.read_csv(VAJRA_DIR / "vajra_dcc_params.csv")
a_dcc = float(dcc_params["a"].iloc[0])
b_dcc = float(dcc_params["b"].iloc[0])
print(f"\nDCC params: a={a_dcc:.4f}  b={b_dcc:.4f}  persistence={a_dcc+b_dcc:.4f}")

Q_bar_df = pd.read_csv(KUBER_DIR / "dcc_correlation.csv", index_col=0)
Q_bar_df = Q_bar_df.loc[stocks, stocks]
Q_bar = Q_bar_df.values.astype(float)
print(f"Q_bar loaded: {Q_bar.shape} — avg off-diagonal = {np.mean(Q_bar[np.triu_indices(len(stocks), k=1)]):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — GARCH STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2 — GARCH(1,1) STANDARDISATION")
print("=" * 70)

eps_dict = {}
for s in stocks:
    r = ret[s].dropna() * 100          # arch wants % scale
    am = arch_model(r, vol="GARCH", p=1, q=1, dist="Normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    std_res = res.resid / res.conditional_volatility
    eps_dict[s] = std_res
    print(f"  {s:<14} omega={res.params['omega']:.4f}  "
          f"alpha={res.params['alpha[1]']:.4f}  beta={res.params['beta[1]']:.4f}")

eps = pd.DataFrame(eps_dict).dropna()
print(f"\nStandardised residuals: {eps.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DCC FORWARD PASS → daily avg pairwise correlation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 — DCC FORWARD PASS")
print("=" * 70)

n = len(stocks)
T = len(eps)
Q_t = Q_bar.copy()
avg_corr_vals = np.full(T, np.nan)
triu_idx = np.triu_indices(n, k=1)

for t in range(1, T):
    e_prev = eps.iloc[t-1].values.reshape(-1, 1)
    outer  = e_prev @ e_prev.T
    Q_t    = (1 - a_dcc - b_dcc) * Q_bar + a_dcc * outer + b_dcc * Q_t

    # Enforce positive-definiteness numerically
    d     = np.sqrt(np.maximum(np.diag(Q_t), 1e-8))
    D_inv = np.diag(1.0 / d)
    R_t   = D_inv @ Q_t @ D_inv
    # Clip to [-1, 1]
    np.fill_diagonal(R_t, 1.0)
    R_t = np.clip(R_t, -1.0, 1.0)

    avg_corr_vals[t] = R_t[triu_idx].mean()

dcc_avg = pd.Series(avg_corr_vals, index=eps.index, name="dcc_avg_corr")
dcc_avg = dcc_avg.dropna()

print(f"DCC series: {len(dcc_avg)} days | {dcc_avg.index[0].date()} → {dcc_avg.index[-1].date()}")
print(f"  Full-sample mean corr  = {dcc_avg.mean():.4f}")
print(f"  Full-sample range      = [{dcc_avg.min():.4f}, {dcc_avg.max():.4f}]")
print(f"  Percentile 95          = {dcc_avg.quantile(0.95):.4f}")

# Save the time series
dcc_avg.to_csv(OUT_DIR / "dcc_avg_corr.csv", header=True)
print(f"Saved: outputs/dcc_avg_corr.csv")

# Spot-check: COVID peak
covid_window = dcc_avg["2020-03-01":"2020-04-30"]
if len(covid_window) > 0:
    peak_date = covid_window.idxmax()
    print(f"\n  COVID peak DCC: {dcc_avg[peak_date]:.4f} on {peak_date.date()}")

# Check stationarity
adf_lev = adfuller(dcc_avg, maxlag=10, autolag="AIC")
adf_dif = adfuller(dcc_avg.diff().dropna(), maxlag=10, autolag="AIC")
print(f"\n  ADF (levels): stat={adf_lev[0]:.3f}  p={adf_lev[1]:.3f}  "
      f"{'stationary' if adf_lev[1] < 0.05 else 'non-stationary (use diffs)'}")
print(f"  ADF (diffs):  stat={adf_dif[0]:.3f}  p={adf_dif[1]:.3f}  "
      f"{'stationary ✓' if adf_dif[1] < 0.05 else 'still non-stationary'}")

# Use levels if stationary, diffs otherwise
USE_DIFF_DCC = adf_lev[1] >= 0.05
dcc_signal = dcc_avg.diff().dropna() if USE_DIFF_DCC else dcc_avg
dcc_label  = "Δdcc_avg_corr" if USE_DIFF_DCC else "dcc_avg_corr"
print(f"\n  Using {dcc_label} for Granger tests.")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hac_granger_f(y, x_lags, lag):
    """
    Bivariate Granger F-test with HAC (Newey-West) standard errors.
    H0: coefficients on x_lags[:lag] are jointly zero in the
        regression y ~ y_lags + x_lags.
    Returns (F_stat, p_value, n_obs).
    """
    df = pd.DataFrame({"y": y})
    for k in range(1, lag + 1):
        df[f"y_lag{k}"] = y.shift(k)
        df[f"x_lag{k}"] = x_lags.shift(k)
    df = df.dropna()
    n  = len(df)
    if n < 50:
        return np.nan, np.nan, n

    y_  = df["y"].values
    X_u = sm.add_constant(df[[f"y_lag{k}" for k in range(1, lag+1)] +
                              [f"x_lag{k}" for k in range(1, lag+1)]].values)
    X_r = sm.add_constant(df[[f"y_lag{k}" for k in range(1, lag+1)]].values)

    res_u = sm.OLS(y_, X_u).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": 5, "use_correction": True})
    res_r = sm.OLS(y_, X_r).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": 5, "use_correction": True})

    # Wald test on x coefficients (last `lag` params of unrestricted model)
    # Use F-test via R matrix
    R = np.zeros((lag, X_u.shape[1]))
    for i in range(lag):
        R[i, -(lag - i)] = 1
    wald = res_u.wald_test(R, use_f=True)
    F    = float(np.squeeze(wald.statistic))
    p    = float(np.squeeze(wald.pvalue))
    return F, p, n


def aic_best_lag(y, x, max_lag=MAX_LAG):
    """AIC-select best lag for bivariate Granger."""
    best_aic, best_lag = np.inf, 1
    for lag in range(1, max_lag + 1):
        df = pd.DataFrame({"y": y})
        for k in range(1, lag + 1):
            df[f"y_lag{k}"] = y.shift(k)
            df[f"x_lag{k}"] = x.shift(k)
        df = df.dropna()
        if len(df) < 30:
            continue
        y_  = df["y"].values
        X_u = sm.add_constant(df[[c for c in df.columns if c != "y"]].values)
        try:
            res = sm.OLS(y_, X_u).fit()
            if res.aic < best_aic:
                best_aic, best_lag = res.aic, lag
        except Exception:
            pass
    return best_lag


def sig_stars(p):
    if pd.isna(p):   return ""
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    if p < 0.10:     return "†"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — DCC Granger-leads India VIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PANEL A — DCC Granger-leads India VIX")
print("=" * 70)

# Align
vix_adf  = adfuller(np.log(vix_raw["vix"]).dropna(), maxlag=10, autolag="AIC")
USE_DIFF_VIX = vix_adf[1] >= 0.05
vix_signal   = np.log(vix_raw["vix"]).diff() if USE_DIFF_VIX else np.log(vix_raw["vix"])
vix_label    = "Δlog_VIX" if USE_DIFF_VIX else "log_VIX"
vix_signal   = vix_signal.dropna()

common = dcc_signal.index.intersection(vix_signal.index)
dcc_a  = dcc_signal.loc[common]
vix_a  = vix_signal.loc[common]
print(f"Aligned: {len(common)} days | {common[0].date()} → {common[-1].date()}")
print(f"Testing: {dcc_label} → {vix_label}")

panel_a_rows = []

# A1: DCC → VIX (main test)
best_lag = aic_best_lag(vix_a, dcc_a)
F, p, n  = hac_granger_f(vix_a, dcc_a, best_lag)
print(f"\n[DCC → VIX]  lag={best_lag}  F={F:.3f}  p={p:.4f}  n={n}  {sig_stars(p)}")
panel_a_rows.append({"direction": "DCC→VIX", "lag": best_lag, "F": F, "p": p,
                     "n": n, "sig": sig_stars(p)})

# Also test each lag 1–5 individually
print(f"  Lag-by-lag:")
for lag in range(1, MAX_LAG + 1):
    F_, p_, n_ = hac_granger_f(vix_a, dcc_a, lag)
    print(f"    lag={lag}  F={F_:.3f}  p={p_:.4f}  {sig_stars(p_)}")

# A2: VIX → DCC (reverse, for asymmetry claim)
best_lag_r = aic_best_lag(dcc_a, vix_a)
F_r, p_r, n_r = hac_granger_f(dcc_a, vix_a, best_lag_r)
print(f"\n[VIX → DCC]  lag={best_lag_r}  F={F_r:.3f}  p={p_r:.4f}  n={n_r}  {sig_stars(p_r)}")
panel_a_rows.append({"direction": "VIX→DCC", "lag": best_lag_r, "F": F_r, "p": p_r,
                     "n": n_r, "sig": sig_stars(p_r)})

# A3: DCC → Nifty returns (do correlations predict market direction?)
mkt_signal = fr["MKT"] if "MKT" in fr.columns else None
if mkt_signal is not None:
    common_m = dcc_signal.index.intersection(mkt_signal.index)
    dcc_m    = dcc_signal.loc[common_m]
    mkt_m    = mkt_signal.loc[common_m]
    best_lag_m = aic_best_lag(mkt_m, dcc_m)
    F_m, p_m, n_m = hac_granger_f(mkt_m, dcc_m, best_lag_m)
    print(f"\n[DCC → MKT]  lag={best_lag_m}  F={F_m:.3f}  p={p_m:.4f}  n={n_m}  {sig_stars(p_m)}")
    panel_a_rows.append({"direction": "DCC→MKT", "lag": best_lag_m, "F": F_m,
                         "p": p_m, "n": n_m, "sig": sig_stars(p_m)})

panel_a_df = pd.DataFrame(panel_a_rows)
panel_a_df.to_csv(OUT_DIR / "dcc_panel_a.csv", index=False)
print(f"\n→ Saved: dcc_panel_a.csv")

# ── Key result summary ──
dcc_vix_result = panel_a_df[panel_a_df["direction"] == "DCC→VIX"].iloc[0]
vix_dcc_result = panel_a_df[panel_a_df["direction"] == "VIX→DCC"].iloc[0]
print(f"\nKEY RESULT:")
print(f"  DCC→VIX: F={dcc_vix_result['F']:.3f}  p={dcc_vix_result['p']:.4f}  {sig_stars(dcc_vix_result['p'])}")
print(f"  VIX→DCC: F={vix_dcc_result['F']:.3f}  p={vix_dcc_result['p']:.4f}  {sig_stars(vix_dcc_result['p'])}")
if dcc_vix_result['p'] < 0.10 and vix_dcc_result['p'] >= 0.10:
    print(f"  → DCC Granger-leads VIX (unidirectional). FINDING.")
elif dcc_vix_result['p'] < 0.10 and vix_dcc_result['p'] < 0.10:
    print(f"  → Bidirectional (contemporaneous relationship, not clean lead). WEAKER.")
else:
    print(f"  → Null in both directions.")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — DCC level predicts factor Sharpe collapse
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PANEL B — DCC Quintile → Forward Factor Sharpe")
print("=" * 70)
print("(Does high realized correlation predict factor diversification failure?)")

# Align DCC with factor returns
common_b = dcc_avg.index.intersection(fr.index)
dcc_b    = dcc_avg.loc[common_b]
fr_b     = fr.loc[common_b]

# Quintile sort on dcc_avg level (not diff — levels tell us the current correlation regime)
dcc_q = pd.qcut(dcc_b, q=5, labels=["Q1\n(lowest)","Q2","Q3","Q4","Q5\n(highest)"])

panel_b_rows = []
print(f"\n{'Factor':<8}", end="")
for q in ["Q1\n(lowest)","Q2","Q3","Q4","Q5\n(highest)"]:
    label = q.replace("\n","=")
    print(f"  {label:>12}", end="")
print(f"  {'Q5-Q1 spread':>14}  {'monotone?':>10}")
print("-" * 90)

for factor in factors:
    sharpe_by_q = {}
    n_by_q = {}
    for q_label in ["Q1\n(lowest)","Q2","Q3","Q4","Q5\n(highest)"]:
        mask = dcc_q == q_label
        # Forward Sharpe: at time t, use factor returns t+1 to t+FWD_DAYS
        fwd_ret = fr_b.loc[mask, factor].shift(-FWD_DAYS)
        fwd_ret = fwd_ret.dropna()
        if len(fwd_ret) < 20:
            sharpe_by_q[q_label] = np.nan
            n_by_q[q_label] = len(fwd_ret)
            continue
        ann_ret = fwd_ret.mean() * 252
        ann_vol = fwd_ret.std() * np.sqrt(252)
        sharpe_by_q[q_label] = ann_ret / ann_vol if ann_vol > 0 else np.nan
        n_by_q[q_label] = int(mask.sum())

    q_labels = ["Q1\n(lowest)","Q2","Q3","Q4","Q5\n(highest)"]
    vals = [sharpe_by_q[q] for q in q_labels]
    spread = (vals[4] - vals[0]) if not (np.isnan(vals[0]) or np.isnan(vals[4])) else np.nan
    non_nan = [v for v in vals if not np.isnan(v)]
    monotone = all(non_nan[i] >= non_nan[i+1] for i in range(len(non_nan)-1))

    print(f"{factor:<8}", end="")
    for v in vals:
        print(f"  {v:>12.3f}" if not np.isnan(v) else f"  {'NaN':>12}", end="")
    print(f"  {spread:>14.3f}  {'YES ✓' if monotone else 'no':>10}")

    for q_label in q_labels:
        panel_b_rows.append({
            "factor": factor,
            "quintile": q_label.replace("\n",""),
            "fwd_21d_sharpe": sharpe_by_q[q_label],
            "n_obs": n_by_q.get(q_label, 0)
        })

panel_b_df = pd.DataFrame(panel_b_rows)
panel_b_df.to_csv(OUT_DIR / "dcc_panel_b.csv", index=False)
print(f"\n→ Saved: dcc_panel_b.csv")
print("  Interpretation: negative Q5-Q1 spread = diversification collapses under high correlation.")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — AUC Comparison: DCC vs VIX vs FII as Bear onset predictor
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PANEL C — AUC Comparison: DCC vs VIX vs FII as Bear Onset Predictor")
print("=" * 70)

# Bear onset: HMM Bear state within next BEAR_FWD days
bear_ind = (rl["regime_name"] == "Bear").astype(int)
bear_fwd = bear_ind.rolling(BEAR_FWD).max().shift(-BEAR_FWD + 1)  # forward roll
# Alternative: direct HMM Bear = 1 on the day
bear_onset = bear_fwd.dropna().astype(int)

print(f"Bear onset events ({BEAR_FWD}-day): {bear_onset.sum()} / {len(bear_onset)}")

# FII normalisation
fii_roll252 = fii["net_crore"].abs().rolling(252, min_periods=60).mean()
fii_norm    = fii["net_crore"] / fii_roll252.replace(0, np.nan)

def logit_auc(signal_series, target_series, name, max_lag=MAX_LAG):
    """Logit with lags 1-max_lag as features; returns AUC, McFadden R², AIC."""
    df = pd.DataFrame({"target": target_series, "signal": signal_series})
    for k in range(1, max_lag + 1):
        df[f"lag{k}"] = df["signal"].shift(k)
    df = df.dropna()
    if df["target"].sum() < 10:
        print(f"  {name:<15}: insufficient Bear events")
        return np.nan, np.nan, np.nan
    X = sm.add_constant(df[[f"lag{k}" for k in range(1, max_lag+1)]].values)
    y = df["target"].values
    try:
        logit = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
        prob  = logit.predict(X)
        auc   = roc_auc_score(y, prob)
        llf   = logit.llf
        llnull= logit.llnull
        mcf   = 1 - llf / llnull if llnull != 0 else np.nan
        aic   = logit.aic
        print(f"  {name:<15}: AUC={auc:.4f}  McFadden R²={mcf:.4f}  AIC={aic:.1f}")
        return auc, mcf, aic
    except Exception as e:
        print(f"  {name:<15}: ERROR — {e}")
        return np.nan, np.nan, np.nan

panel_c_rows = []
for name, signal in [("DCC_avg_corr", dcc_avg),
                      ("log_VIX",      np.log(vix_raw["vix"])),
                      ("FII_net_norm", fii_norm),
                      ("DCC+VIX joint", None)]:

    if name == "DCC+VIX joint":
        # Joint model: DCC lags + VIX lags
        common_j = dcc_avg.index.intersection(vix_raw.index).intersection(bear_onset.index)
        df_j = pd.DataFrame({
            "target": bear_onset,
            "dcc": dcc_avg,
            "vix": np.log(vix_raw["vix"])
        }).loc[common_j].dropna()
        for k in range(1, MAX_LAG + 1):
            df_j[f"dcc_lag{k}"] = df_j["dcc"].shift(k)
            df_j[f"vix_lag{k}"] = df_j["vix"].shift(k)
        df_j = df_j.dropna()
        X_j = sm.add_constant(
            df_j[[f"dcc_lag{k}" for k in range(1, MAX_LAG+1)] +
                  [f"vix_lag{k}" for k in range(1, MAX_LAG+1)]].values
        )
        y_j = df_j["target"].values
        try:
            logit_j = sm.Logit(y_j, X_j).fit(disp=0, method="bfgs", maxiter=200)
            prob_j  = logit_j.predict(X_j)
            auc_j   = roc_auc_score(y_j, prob_j)
            mcf_j   = 1 - logit_j.llf / logit_j.llnull
            print(f"  {'DCC+VIX joint':<15}: AUC={auc_j:.4f}  McFadden R²={mcf_j:.4f}  AIC={logit_j.aic:.1f}")
            panel_c_rows.append({"predictor": name, "auc": auc_j,
                                  "mcfadden_r2": mcf_j, "aic": logit_j.aic})
        except Exception as e:
            print(f"  DCC+VIX joint: ERROR — {e}")
        continue

    auc, mcf, aic = logit_auc(signal, bear_onset, name)
    panel_c_rows.append({"predictor": name, "auc": auc, "mcfadden_r2": mcf, "aic": aic})

panel_c_df = pd.DataFrame(panel_c_rows)
panel_c_df.to_csv(OUT_DIR / "dcc_panel_c.csv", index=False)
print(f"\n→ Saved: dcc_panel_c.csv")

best_predictor = panel_c_df.sort_values("auc", ascending=False).iloc[0]
print(f"\nBest predictor: {best_predictor['predictor']} (AUC={best_predictor['auc']:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — Walk-forward Granger stability (DCC → VIX)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PANEL D — Walk-Forward Granger Stability (DCC → VIX)")
print("=" * 70)
print(f"Rolling window: {ROLL_WIN} days | Testing DCC→VIX at best lag from Panel A")

common_d  = dcc_signal.index.intersection(vix_signal.index)
dcc_d     = dcc_signal.loc[common_d]
vix_d     = vix_signal.loc[common_d]
n_windows = len(dcc_d) - ROLL_WIN
STEP      = 21  # test every 21 days (approx monthly)

wf_rows   = []
for start in range(0, n_windows, STEP):
    end = start + ROLL_WIN
    if end > len(dcc_d):
        break
    dcc_w = dcc_d.iloc[start:end]
    vix_w = vix_d.iloc[start:end]
    lag_w = aic_best_lag(vix_w, dcc_w, max_lag=3)  # max 3 for speed
    F_w, p_w, n_w = hac_granger_f(vix_w, dcc_w, lag_w)
    wf_rows.append({
        "window_start": dcc_d.index[start].date(),
        "window_end":   dcc_d.index[end-1].date(),
        "lag": lag_w,
        "F": F_w,
        "p": p_w,
        "sig_10": int(p_w < 0.10) if not np.isnan(p_w) else 0,
        "sig_05": int(p_w < 0.05) if not np.isnan(p_w) else 0,
    })

wf_df = pd.DataFrame(wf_rows)
wf_df.to_csv(OUT_DIR / "dcc_panel_d.csv", index=False)

n_windows_run = len(wf_df.dropna(subset=["F"]))
n_sig10 = wf_df["sig_10"].sum()
n_sig05 = wf_df["sig_05"].sum()
pct_10  = n_sig10 / n_windows_run * 100 if n_windows_run > 0 else 0
pct_05  = n_sig05 / n_windows_run * 100 if n_windows_run > 0 else 0

print(f"\n  Windows run: {n_windows_run}")
print(f"  Significant at p<0.10: {n_sig10}/{n_windows_run} ({pct_10:.0f}%)")
print(f"  Significant at p<0.05: {n_sig05}/{n_windows_run} ({pct_05:.0f}%)")
print(f"\n  Median F across windows: {wf_df['F'].median():.3f}")
print(f"  Median p across windows: {wf_df['p'].median():.4f}")
print(f"\n→ Saved: dcc_panel_d.csv")

if pct_10 >= 60:
    print(f"  → DCC→VIX lead is STABLE: holds in {pct_10:.0f}% of rolling windows.")
elif pct_10 >= 40:
    print(f"  → Lead is EPISODIC: holds in {pct_10:.0f}% of windows — stress regime-dependent.")
else:
    print(f"  → Lead is FRAGILE: only {pct_10:.0f}% of windows significant.")


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — DCC Regime-conditional statistics (for paper context)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BONUS — DCC by Regime (sanity / paper context)")
print("=" * 70)

common_r = dcc_avg.index.intersection(rl.index)
dcc_r    = dcc_avg.loc[common_r]
rl_r     = rl.loc[common_r, "regime_name"]

for regime in ["Bull", "Sideways", "Bear"]:
    mask    = rl_r == regime
    vals    = dcc_r[mask]
    n       = mask.sum()
    mean    = vals.mean()
    p95     = vals.quantile(0.95)
    print(f"  {regime:<10} n={n:>4}  mean_corr={mean:.4f}  p95={p95:.4f}")

# Does DCC level differ significantly across regimes?
bear_vals  = dcc_r[rl_r == "Bear"]
bull_vals  = dcc_r[rl_r == "Bull"]
t_stat, t_p = stats.ttest_ind(bear_vals.dropna(), bull_vals.dropna())
print(f"\n  Bear vs Bull DCC t-test: t={t_stat:.3f}  p={t_p:.4f}  "
      f"{'Bear DCC > Bull DCC ✓' if t_stat > 0 and t_p < 0.05 else 'not significant'}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
dcc_vix = panel_a_df[panel_a_df["direction"]=="DCC→VIX"].iloc[0]
vix_dcc = panel_a_df[panel_a_df["direction"]=="VIX→DCC"].iloc[0]
print(f"\nPanel A — DCC Granger-leads VIX:")
print(f"  DCC→VIX: F={dcc_vix['F']:.3f}  p={dcc_vix['p']:.4f}  lag={dcc_vix['lag']}  {sig_stars(dcc_vix['p'])}")
print(f"  VIX→DCC: F={vix_dcc['F']:.3f}  p={vix_dcc['p']:.4f}  lag={vix_dcc['lag']}  {sig_stars(vix_dcc['p'])}")

print(f"\nPanel B — Factor Sharpe by DCC Quintile (Q5-Q1 spread):")
for factor in factors:
    sub = panel_b_df[panel_b_df["factor"]==factor]
    q5  = sub[sub["quintile"]=="Q5(highest)"]["fwd_21d_sharpe"].values
    q1  = sub[sub["quintile"]=="Q1(lowest)"]["fwd_21d_sharpe"].values
    if len(q5) and len(q1):
        spread = q5[0] - q1[0]
        print(f"  {factor}: spread={spread:+.3f}  "
              f"{'factor collapses ✓' if spread < -0.10 else 'flat' if abs(spread) < 0.10 else 'strengthens'}")

print(f"\nPanel C — AUC comparison:")
for _, row in panel_c_df.dropna(subset=["auc"]).sort_values("auc", ascending=False).iterrows():
    print(f"  {row['predictor']:<20}: AUC={row['auc']:.4f}  McFadden R²={row['mcfadden_r2']:.4f}")

print(f"\nPanel D — Walk-forward stability:")
print(f"  DCC→VIX significant in {pct_10:.0f}% of rolling {ROLL_WIN}-day windows (p<0.10)")
print(f"  DCC→VIX significant in {pct_05:.0f}% of rolling {ROLL_WIN}-day windows (p<0.05)")

print("\n" + "=" * 70)
print("ALL OUTPUTS in paper_analysis/outputs/:")
print("  dcc_avg_corr.csv    — daily DCC average pairwise correlation series")
print("  dcc_panel_a.csv     — Granger: DCC→VIX, VIX→DCC, DCC→MKT")
print("  dcc_panel_b.csv     — Factor Sharpe by DCC quintile")
print("  dcc_panel_c.csv     — AUC: DCC vs VIX vs FII as Bear onset predictor")
print("  dcc_panel_d.csv     — Walk-forward Granger stability")
print("=" * 70)
print("\nNext: python paper_analysis/table_dcc_stress.py")
print("Then: paste results → update paper headline and re-submit SSRN")
