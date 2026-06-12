"""
Table 3 — FII Flow as Leading Indicator of Regime Transitions
==============================================================

Paper: "Regime-Conditional Factor Investing in Indian Equities"

Research Question:
  Does daily net FII (Foreign Institutional Investor) equity flow in India
  predict regime transitions BEFORE they manifest in price returns?

  Formally: Does FII_flow(t-k) Granger-cause Regime_transition(t)?
  Where k ∈ {1, 2, 3, 5} trading days.

Why this is novel:
  - Almost all FII/factor research uses MONTHLY SEBI data (published with lag).
  - Daily NSDL FII flow is publicly available but rarely used in academic work.
  - If FII outflow LEADS the Bear regime by 2-3 days, you get early warning
    to reduce factor exposure before the regime flips — practically valuable
    for any Indian AMC running systematic strategies.

Two statistical tests:
  1. Granger Causality: Does FII flow add predictive information about
     regime transitions beyond what regime history already provides?
     Uses statsmodels.tsa.stattools.grangercausalitytests.
     H0: FII flow does NOT Granger-cause regime transitions.
     Reject H0 (p < 0.05) → FII flow is a leading indicator.

  2. Logistic Regression: P(Bear_transition_t=1 | FII_flow_lags)
     Allows us to estimate: "1 std dev net FII outflow → X% higher P(Bear)"
     This is more interpretable than Granger (which is binary accept/reject).

Outputs:
  table3_granger_results.csv      — Granger p-values by lag
  table3_logistic_results.csv     — Logistic regression coefficients
  table3_fii_summary.csv          — FII flow descriptive stats by regime
  table3_fii_leading_indicator.tex — LaTeX table for paper

FII Data Format Expected (NSE download):
  CSV with columns: Date, Buy (₹ Cr), Sell (₹ Cr), Net (₹ Cr)
  The script normalises Net by a 252-day rolling AUM proxy (cumulative buy+sell)
  to make the signal comparable across time. Raw ₹ crore figures are
  inflated in later years vs earlier years.

If FII data not yet downloaded:
  → Run: python paper_analysis/download_fii_data.py
  → Or manually download from NSE:
     nseindia.com → Reports → Historical Data → FII/DII Activity
  → Save as: alpha-core/data/fii_dii_daily.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Run: pip install statsmodels")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FII_PATH = DATA_DIR / "fii_dii_daily.csv"
MAX_LAG   = 5   # test lags 1-5 trading days


# ── Load and validate FII data ────────────────────────────────────────────────
def load_fii_data() -> pd.DataFrame:
    """
    Load FII daily data. Expected columns (case-insensitive, flexible):
      Date | FII_Buy | FII_Sell | FII_Net
      OR
      Date | Buy | Sell | Net

    NSE CSV format typically looks like:
      Date,Buy Value (₹ Cr),Sell Value (₹ Cr),Net Value (₹ Cr)

    Returns DataFrame with DatetimeIndex and column 'fii_net' (₹ Cr).
    """
    if not FII_PATH.exists():
        raise FileNotFoundError(
            f"\nFII data not found at: {FII_PATH}\n"
            "Download steps:\n"
            "  1. Go to: https://www.nseindia.com/reports-indices-historical-fii-dii\n"
            "  2. Select Date Range: 01-Jan-2019 to today\n"
            "  3. Select: FII / FPI (Equity)\n"
            "  4. Download CSV → save as alpha-core/data/fii_dii_daily.csv\n"
            "\nAlternative: Use the NSE bulk download or the provided"
            " download_fii_data.py script."
        )

    # Read the CSV — handle NSE's various header formats
    df = pd.read_csv(FII_PATH)
    print(f"FII raw file: {df.shape} | columns: {df.columns.tolist()}")

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    print(f"Normalised columns: {df.columns.tolist()}")

    # Find the date column
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        raise ValueError("No 'date' column found in FII CSV.")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    df = df.sort_index()

    # Find the net FII column
    net_col = next((c for c in df.columns if "net" in c and ("fii" in c or "fpi" in c or c == "net_value___cr_" or "net" in c)), None)
    if net_col is None:
        # Try any column with 'net'
        net_col = next((c for c in df.columns if "net" in c), None)
    if net_col is None:
        raise ValueError(f"Cannot find net FII column. Columns: {df.columns.tolist()}")

    fii = df[[net_col]].copy()
    fii.columns = ["fii_net"]

    # Clean: remove commas, convert to numeric
    fii["fii_net"] = pd.to_numeric(
        fii["fii_net"].astype(str).str.replace(",", "").str.replace("(", "-").str.replace(")", ""),
        errors="coerce"
    )
    fii = fii.dropna()

    print(f"FII net flow loaded: {len(fii)} days | {fii.index[0].date()} → {fii.index[-1].date()}")
    print(f"Mean: ₹{fii['fii_net'].mean():.0f} Cr | Std: ₹{fii['fii_net'].std():.0f} Cr")
    print(f"Negative (outflow) days: {(fii['fii_net'] < 0).sum()} / {len(fii)}")

    return fii


# ── Normalise FII flow ────────────────────────────────────────────────────────
def normalise_fii(fii: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise raw ₹ crore FII flow by 252-day rolling average of |flow|.

    Why normalise?
      Raw FII flow in 2019: ₹500-2000 Cr/day typical.
      Raw FII flow in 2025: ₹2000-8000 Cr/day typical (market grown 3×).
      Without normalisation, recent flows dominate the regression and
      Granger tests are biased. Normalised flow is comparable across time.

    Result: fii_net_norm ~ N(0, ~1) approximately — a z-score-like signal.
    """
    roll_scale = fii["fii_net"].abs().rolling(252, min_periods=60).mean()
    fii["fii_net_norm"] = fii["fii_net"] / (roll_scale + 1e-9)
    fii["fii_net_norm"] = fii["fii_net_norm"].clip(-10, 10)  # remove extreme outliers
    return fii


# ── Build regime transition indicator ────────────────────────────────────────
def build_transition_indicator(rl: pd.DataFrame) -> pd.Series:
    """
    Create a binary indicator: did the market ENTER a Bear regime today?
    bear_entry(t) = 1 if regime(t)=Bear AND regime(t-1) != Bear, else 0.

    Why "entry" not "is Bear"?
      We want to predict TRANSITIONS, not levels.
      If Granger-causality held for "is Bear", it would just mean FII sells
      during bear markets — a tautology. The entry indicator tests whether
      FII predicts the ONSET of bear periods.

    Also create: bear_state(t) = 1 if regime = Bear (for logistic regression)
    """
    regime = rl["regime_name"]
    bear_state  = (regime == "Bear").astype(int)
    bear_entry  = ((regime == "Bear") & (regime.shift(1) != "Bear")).astype(int)

    result = pd.DataFrame({
        "bear_state" : bear_state,
        "bear_entry" : bear_entry,
        "regime_int" : rl["regime_int"],
        "regime_name": rl["regime_name"],
    })
    print(f"\nBear entry events: {bear_entry.sum()} episodes over {len(bear_entry)} days")
    return result


# ── Stationarity check ────────────────────────────────────────────────────────
def check_stationarity(series: pd.Series, name: str) -> bool:
    """ADF test. Granger causality requires stationarity of all variables."""
    if not HAS_STATSMODELS:
        return True
    result = adfuller(series.dropna(), autolag="AIC")
    p = result[1]
    stationary = p < 0.05
    print(f"  ADF({name}): p={p:.4f} → {'STATIONARY ✓' if stationary else 'NON-STATIONARY ⚠'}")
    return stationary


# ── Granger causality test ────────────────────────────────────────────────────
def run_granger_test(fii_aligned: pd.DataFrame,
                     transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether FII flow Granger-causes regime transitions.

    Granger causality framework:
      Bivariate VAR(p):
        X(t) = a1*X(t-1) + ... + ap*X(t-p) + b1*Y(t-1) + ... + bp*Y(t-p) + ε
      Y does NOT Granger-cause X if all b_i = 0.
      Test: F-test of restricted vs unrestricted VAR.

    We test X = bear_state, Y = fii_net_norm
    At lags 1, 2, 3, 4, 5 (one week of trading).

    Interpretation:
      p < 0.05 → FII flow has statistically significant predictive power
                  for regime state beyond regime's own history.
      This is NOT the same as "FII causes Bear markets" — Granger causality
      is predictive, not structural/causal. Documented in paper.
    """
    if not HAS_STATSMODELS:
        print("Skipping Granger test — statsmodels not available")
        return pd.DataFrame()

    # Align data
    combined = pd.concat([
        transitions["bear_state"],
        fii_aligned["fii_net_norm"]
    ], axis=1).dropna()
    combined.columns = ["bear_state", "fii_net_norm"]

    print(f"\nGranger test dataset: {len(combined)} days")

    # Stationarity checks
    print("Stationarity checks (required for Granger):")
    check_stationarity(combined["bear_state"],   "bear_state")
    check_stationarity(combined["fii_net_norm"], "fii_net_norm")

    print(f"\nGranger Causality: FII flow → Bear regime state")
    print(f"H0: FII flow does NOT Granger-cause bear_state")
    print(f"{'Lag':<6} {'F-stat':>10} {'p-value':>10} {'Result':>20}")
    print("-" * 50)

    rows = []
    try:
        gc_results = grangercausalitytests(
            combined[["bear_state", "fii_net_norm"]],
            maxlag=MAX_LAG,
            verbose=False
        )
        for lag in range(1, MAX_LAG + 1):
            test = gc_results[lag][0]["ssr_ftest"]
            f_stat, p_val, df_denom, df_num = test
            sig = "*** REJECT H0" if p_val < 0.001 else "** REJECT H0" if p_val < 0.01 else "* REJECT H0" if p_val < 0.05 else "fail to reject"
            print(f"{lag:<6} {f_stat:>10.3f} {p_val:>10.4f} {sig:>20}")
            rows.append({"lag_days": lag, "f_stat": round(f_stat, 3),
                         "p_value": round(p_val, 4), "result": sig})
    except Exception as e:
        print(f"Granger test failed: {e}")

    return pd.DataFrame(rows)


# ── Logistic Regression: P(Bear entry | FII lags) ────────────────────────────
def run_logistic_regression(fii_aligned: pd.DataFrame,
                             transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic regression: P(bear_entry_t = 1 | fii_lag_1, fii_lag_2, fii_lag_3)

    Why logistic not OLS?
      bear_entry is binary (0/1). OLS would predict probabilities outside [0,1].
      Logistic regression gives calibrated probabilities and interpretable
      log-odds coefficients.

    Coefficient interpretation:
      β_lag1 < 0 → 1 unit increase in normalised FII flow DECREASES
                    probability of Bear entry (outflow = negative flow → more Bear)
      β_lag1 > 0 → unexpected: more buying predicts Bear — would need explanation.

    Economic expectation:
      FII net flow = Buy - Sell.
      Negative flow (outflow) → we expect positive coefficient on (-flow) or
      negative coefficient on (flow) for predicting Bear.
    """
    if not HAS_STATSMODELS:
        print("Skipping logistic regression — statsmodels not available")
        return pd.DataFrame()

    # Build lagged features
    data = pd.DataFrame(index=fii_aligned.index)
    data["bear_entry"] = transitions["bear_entry"]
    for lag in [1, 2, 3, 5]:
        data[f"fii_lag{lag}"] = fii_aligned["fii_net_norm"].shift(lag)

    # Also add a DII_net if available (column dii_net_norm in fii_aligned)
    if "dii_net_norm" in fii_aligned.columns:
        for lag in [1, 2]:
            data[f"dii_lag{lag}"] = fii_aligned["dii_net_norm"].shift(lag)

    data = data.dropna()
    print(f"\nLogistic regression dataset: {len(data)} days | Bear entries: {data['bear_entry'].sum()}")

    y = data["bear_entry"]
    X_cols = [c for c in data.columns if c.startswith("fii_lag") or c.startswith("dii_lag")]
    X = sm.add_constant(data[X_cols])

    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        print("\nLogistic Regression: P(Bear Entry | FII lags)")
        print(model.summary2().tables[1].to_string())

        # Extract results
        results_df = pd.DataFrame({
            "variable"   : model.params.index,
            "coefficient": model.params.values.round(4),
            "std_err"    : model.bse.values.round(4),
            "z_stat"     : model.tvalues.values.round(3),
            "p_value"    : model.pvalues.values.round(4),
            "odds_ratio" : np.exp(model.params.values).round(4),
        })

        print(f"\nPseudo R² (McFadden): {model.prsquared:.4f}")
        print(f"AIC: {model.aic:.1f} | BIC: {model.bic:.1f}")
        print(f"Correctly classified: {model.pred_table()}")

        return results_df

    except Exception as e:
        print(f"Logistic regression failed: {e}")
        return pd.DataFrame()


# ── FII descriptive stats by regime ──────────────────────────────────────────
def fii_by_regime(fii_aligned: pd.DataFrame,
                  transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive statistics of FII flow within each regime.
    This is the "smell test" before formal Granger tests:
    If FII is genuinely a leading indicator, you'd expect:
      Bull  regime: mean FII flow > 0 (inflows during bull)
      Bear  regime: mean FII flow < 0 (outflows during bear)
      The key question is WHEN they happen — before or during the regime.
    """
    merged = pd.concat([fii_aligned["fii_net"], transitions["regime_name"]], axis=1).dropna()
    merged.columns = ["fii_net", "regime"]

    summary = merged.groupby("regime")["fii_net"].agg(
        n_days="count",
        mean_flow="mean",
        std_flow="std",
        pct_outflow=lambda x: (x < 0).mean() * 100,
        median_flow="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).round(1)

    print("\nFII Net Flow by Regime (₹ Crore):")
    print(summary.to_string())
    return summary


# ── LaTeX for Table 3 ─────────────────────────────────────────────────────────
def to_latex(granger_df: pd.DataFrame, logit_df: pd.DataFrame,
             fii_regime_df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{FII Daily Flow as Leading Indicator of Regime Transitions}")
    lines.append(r"\label{tab:fii_leading_indicator}")
    lines.append(r"\begin{subtable}[t]{0.45\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\caption{Panel A: Granger Causality Tests}")
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\toprule")
    lines.append(r"Lag (days) & $F$-stat & $p$-value & Decision \\")
    lines.append(r"\midrule")

    if not granger_df.empty:
        for _, row in granger_df.iterrows():
            sig_str = ("$^{***}$" if row["p_value"] < 0.001
                       else "$^{**}$" if row["p_value"] < 0.01
                       else "$^{*}$" if row["p_value"] < 0.05
                       else "")
            decision = r"\textbf{Reject H\textsubscript{0}}" if row["p_value"] < 0.05 else "Fail to reject"
            lines.append(f"{int(row['lag_days'])} & {row['f_stat']:.3f} & {row['p_value']:.4f}{sig_str} & {decision} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{subtable}")
    lines.append(r"\hfill")
    lines.append(r"\begin{subtable}[t]{0.50\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\caption{Panel B: FII Flow by Regime (₹ Cr)}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Regime & Mean & Median & \% Outflow \\")
    lines.append(r"\midrule")

    if not fii_regime_df.empty:
        for regime in ["Bull", "Sideways", "Bear"]:
            if regime in fii_regime_df.index:
                r = fii_regime_df.loc[regime]
                lines.append(f"{regime} & {r['mean_flow']:.0f} & {r['median_flow']:.0f} & {r['pct_outflow']:.1f}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{subtable}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item \textit{Note:} Panel A reports Granger causality tests at lags 1--5 trading days. "
                 r"FII flow is normalised by 252-day rolling mean of $|$flow$|$. "
                 r"H\textsubscript{0}: FII flow does not Granger-cause bear regime state. "
                 r"Panel B reports raw FII net equity flow (₹ crore) by HMM regime. "
                 r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("TABLE 3 — FII FLOW AS LEADING INDICATOR OF REGIME TRANSITIONS")
    print("Paper: Regime-Conditional Factor Investing in Indian Equities")
    print("=" * 80)

    # Load regime labels
    rl = pd.read_csv(DATA_DIR / "regime_labels.csv", index_col=0, parse_dates=True)

    # Load FII data
    try:
        fii = load_fii_data()
    except FileNotFoundError as e:
        print(e)
        print("\nRun download_fii_data.py first or manually download the CSV.")
        exit(1)

    # Normalise FII flow
    fii = normalise_fii(fii)

    # Align FII with regime labels
    common = rl.index.intersection(fii.index)
    fii_aligned  = fii.loc[common]
    rl_aligned   = rl.loc[common]
    print(f"\nAligned: {len(common)} days | {common[0].date()} → {common[-1].date()}")

    # Build regime transition indicator
    transitions = build_transition_indicator(rl_aligned)

    # FII descriptive stats by regime
    fii_regime_df = fii_by_regime(fii_aligned, transitions)

    # Granger causality
    granger_df = run_granger_test(fii_aligned, transitions)

    # Logistic regression
    logit_df = run_logistic_regression(fii_aligned, transitions)

    # LaTeX output
    latex = to_latex(granger_df, logit_df, fii_regime_df)
    (OUT_DIR / "table3_fii_leading_indicator.tex").write_text(latex)

    # Save CSV outputs
    if not granger_df.empty:
        granger_df.to_csv(OUT_DIR / "table3_granger_results.csv", index=False)
    if not logit_df.empty:
        logit_df.to_csv(OUT_DIR / "table3_logistic_results.csv", index=False)
    fii_regime_df.to_csv(OUT_DIR / "table3_fii_by_regime.csv")

    print(f"\nSaved:")
    print(f"  {OUT_DIR}/table3_granger_results.csv")
    print(f"  {OUT_DIR}/table3_logistic_results.csv")
    print(f"  {OUT_DIR}/table3_fii_by_regime.csv")
    print(f"  {OUT_DIR}/table3_fii_leading_indicator.tex")
