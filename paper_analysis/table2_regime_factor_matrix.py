"""
Table 2 — Regime-Conditional Factor Premia Matrix
===================================================

Paper: "Regime-Conditional Factor Investing in Indian Equities"

What this script produces:
  For each (Regime × Factor) cell, compute:
    - Mean annualised return
    - t-statistic (H0: mean = 0)
    - Sharpe ratio (annualised, zero risk-free rate for factor long-short portfolios)
    - Number of days in regime

  Output: regime_factor_matrix.csv  ← Table 2 in the paper
          regime_factor_matrix.tex  ← LaTeX-ready table

Inputs (already exist in alpha-core/data/):
  regime_labels.csv   — HMM output (Bull / Bear / Sideways per day)
  factor_returns.csv  — FF5F daily factor returns (MKT, SMB, HML, RMW, CMA)

Why this is the paper's core claim:
  Full-sample factor Sharpes (Table 1) say "does this factor work in India?"
  Regime-conditional Sharpes (Table 2) say "does this factor work in THIS regime?"
  The hypothesis: momentum crashes in Bear, value works in Sideways, market
  factor dominates in Bull. If confirmed, this justifies regime-adaptive allocation.

Interpretation notes for the paper:
  - Factor long-short returns don't have a cash component → R_f = 0 for Sharpe.
  - MKT is excess return (already Nifty - R_f), so treat consistently.
  - t-stat significance: |t| > 1.96 = 5%, |t| > 2.58 = 1%, |t| > 3.29 = 0.1%
  - Annualisation: mean * 252, vol * sqrt(252)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# t-test implemented via numpy (no scipy needed)
def ttest_1samp(a, popmean=0.0):
    """One-sample t-test: H0: mean(a) == popmean. Returns (t_stat, p_value)."""
    n = len(a)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(a)
    se   = np.std(a, ddof=1) / np.sqrt(n)
    t    = (mean - popmean) / se if se > 0 else np.nan
    # p-value via t-distribution CDF (two-tailed), approximated via normal for large n
    # For n > 30, t ≈ z (normal approximation is fine for academic purposes)
    # Full: use scipy.stats.t.sf — but we avoid the dependency here.
    # For the paper, we report t-stats directly; p-values via normal approximation.
    if np.isnan(t):
        return np.nan, np.nan
    # Two-tailed p-value using normal approximation (valid for n > 30)
    import math
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    p = 2 * (1 - norm_cdf(abs(t)))
    return t, p

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FACTORS   = ["MKT", "SMB", "HML", "RMW", "CMA"]
REGIMES   = ["Bull", "Sideways", "Bear"]
TRADING_DAYS = 252


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rl = pd.read_csv(DATA_DIR / "regime_labels.csv", index_col=0, parse_dates=True)
    fr = pd.read_csv(DATA_DIR / "factor_returns.csv", index_col=0, parse_dates=True)

    # Align on common dates
    common = rl.index.intersection(fr.index)
    rl = rl.loc[common]
    fr = fr.loc[common]

    print(f"Aligned dataset: {len(common)} days | {common[0].date()} → {common[-1].date()}")
    print(f"Regime distribution:\n{rl['regime_name'].value_counts()}\n")
    return rl, fr


# ── Compute stats for one (regime, factor) cell ───────────────────────────────
def regime_factor_stats(returns: pd.Series, regime_name: str) -> dict:
    """
    Given a Series of daily factor returns for a specific regime,
    compute annualised mean, vol, Sharpe, t-stat, n_days, p-value.

    t-stat uses a one-sample t-test against H0: mean = 0.
    This tests whether the factor premium is statistically non-zero
    within this regime — the fundamental hypothesis of Table 2.
    """
    n = len(returns)
    if n < 20:
        # Too few observations for meaningful inference
        return {
            "regime": regime_name, "n_days": n,
            "mean_ann": np.nan, "vol_ann": np.nan,
            "sharpe": np.nan, "t_stat": np.nan, "p_value": np.nan,
            "sig": "—"
        }

    mean_daily = returns.mean()
    std_daily  = returns.std(ddof=1)

    mean_ann = mean_daily * TRADING_DAYS
    vol_ann  = std_daily * np.sqrt(TRADING_DAYS)
    sharpe   = mean_ann / vol_ann if vol_ann > 0 else np.nan

    # One-sample t-test: H0: μ = 0
    t_stat, p_value = ttest_1samp(returns.dropna().values, 0.0)

    # Significance stars
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = ""

    return {
        "regime"  : regime_name,
        "n_days"  : n,
        "mean_ann": round(mean_ann * 100, 2),   # in percent
        "vol_ann" : round(vol_ann  * 100, 2),
        "sharpe"  : round(sharpe,   3),
        "t_stat"  : round(t_stat,   2),
        "p_value" : round(p_value,  4),
        "sig"     : sig,
    }


# ── Build the full matrix ─────────────────────────────────────────────────────
def build_regime_factor_matrix(rl: pd.DataFrame,
                                fr: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Regime, Factor) pair compute regime_factor_stats().
    Returns a long-form DataFrame that's easy to pivot for display.
    """
    rows = []
    for regime in REGIMES:
        mask = rl["regime_name"] == regime
        dates_in_regime = rl.index[mask]
        regime_factor_returns = fr.loc[fr.index.isin(dates_in_regime)]

        for factor in FACTORS:
            if factor not in regime_factor_returns.columns:
                continue
            stats_dict = regime_factor_stats(regime_factor_returns[factor], regime)
            stats_dict["factor"] = factor
            rows.append(stats_dict)

    df = pd.DataFrame(rows)
    return df


# ── Full-sample baseline (Table 1 equivalent) ─────────────────────────────────
def full_sample_stats(fr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute full-sample factor statistics for comparison.
    This becomes Table 1 in the paper.
    """
    rows = []
    for factor in FACTORS:
        s = regime_factor_stats(fr[factor], "Full Sample")
        s["factor"] = factor
        rows.append(s)
    return pd.DataFrame(rows)


# ── Print nicely ──────────────────────────────────────────────────────────────
def print_matrix(matrix_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("TABLE 2 — REGIME-CONDITIONAL FACTOR PREMIA (Annualised %)")
    print("=" * 80)
    print(f"{'Factor':<8}", end="")
    for regime in REGIMES:
        n = matrix_df[matrix_df["regime"] == regime]["n_days"].iloc[0] if not matrix_df[matrix_df["regime"] == regime].empty else 0
        print(f"  {'':>4}{regime:^22}(N={n})", end="")
    print()

    header = f"{'':8}" + "".join([f"  {'Ret%':>7} {'Sharpe':>7} {'t-stat':>7} {'Sig':>4}" for _ in REGIMES])
    print(header)
    print("-" * 80)

    for factor in FACTORS:
        row_str = f"{factor:<8}"
        for regime in REGIMES:
            cell = matrix_df[(matrix_df["factor"] == factor) & (matrix_df["regime"] == regime)]
            if cell.empty:
                row_str += f"  {'—':>7} {'—':>7} {'—':>7} {'':>4}"
            else:
                c = cell.iloc[0]
                row_str += f"  {c['mean_ann']:>7.2f} {c['sharpe']:>7.3f} {c['t_stat']:>7.2f} {c['sig']:>4}"
        print(row_str)

    print("\n* p<0.05  ** p<0.01  *** p<0.001")
    print("Annualised return (%) and Sharpe assume zero risk-free rate for long-short factor portfolios.")
    print("MKT is excess return (Nifty50 - RBI repo/252).")


# ── LaTeX output ──────────────────────────────────────────────────────────────
def to_latex(matrix_df: pd.DataFrame, full_sample_df: pd.DataFrame) -> str:
    """
    Generates LaTeX for Table 2.
    Format: Factor | Full Sample | Bull | Sideways | Bear
    Each block shows: Ann.Ret% / Sharpe / t-stat
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Regime-Conditional Factor Premia in Indian Equities (2019--2026)}")
    lines.append(r"\label{tab:regime_factor_matrix}")
    lines.append(r"\begin{tabular}{lrrrrrrrrrrrr}")
    lines.append(r"\toprule")

    # Column headers
    cols = ["Full Sample", "Bull", "Sideways", "Bear"]
    lines.append(r" & \multicolumn{3}{c}{Full Sample} & \multicolumn{3}{c}{Bull} "
                 r"& \multicolumn{3}{c}{Sideways} & \multicolumn{3}{c}{Bear} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
    lines.append(r"Factor & Ret\% & SR & $t$ & Ret\% & SR & $t$ & Ret\% & SR & $t$ & Ret\% & SR & $t$ \\")
    lines.append(r"\midrule")

    for factor in FACTORS:
        row = [factor]
        # Full sample
        fs = full_sample_df[full_sample_df["factor"] == factor].iloc[0]
        sig = fs["sig"]
        row.append(f"{fs['mean_ann']:.2f}{sig}")
        row.append(f"{fs['sharpe']:.3f}")
        row.append(f"{fs['t_stat']:.2f}")

        # By regime
        for regime in ["Bull", "Sideways", "Bear"]:
            cell = matrix_df[(matrix_df["factor"] == factor) & (matrix_df["regime"] == regime)]
            if cell.empty:
                row.extend(["—", "—", "—"])
            else:
                c = cell.iloc[0]
                sig = c["sig"]
                row.append(f"{c['mean_ann']:.2f}{sig}")
                row.append(f"{c['sharpe']:.3f}")
                row.append(f"{c['t_stat']:.2f}")

        lines.append(" & ".join(row) + r" \\")

    # N days row
    lines.append(r"\midrule")
    n_row = [r"\textit{N (days)}"]
    total_n = len(full_sample_df)
    fs_n = full_sample_df["n_days"].iloc[0] if len(full_sample_df) > 0 else "—"
    # Approximate N per regime from matrix
    for c_label in ["Full Sample", "Bull", "Sideways", "Bear"]:
        if c_label == "Full Sample":
            n_val = matrix_df.groupby("regime")["n_days"].first().sum()
        else:
            subset = matrix_df[matrix_df["regime"] == c_label]
            n_val = subset["n_days"].iloc[0] if not subset.empty else "—"
        n_row.extend([str(n_val), "", ""])
    lines.append(" & ".join(n_row[:13]) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item \textit{Note:} Factor returns are annualised. "
                 r"Sharpe ratios assume zero risk-free rate for long-short factor portfolios. "
                 r"MKT is Nifty~50 excess return over RBI repo rate. "
                 r"HMM regimes (3-state Gaussian, BIC-selected) are estimated on daily "
                 r"returns, realised volatility, and 20-day momentum. "
                 r"$^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$ (two-tailed $t$-test, H\textsubscript{0}: mean = 0).")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("TABLE 2 — REGIME-CONDITIONAL FACTOR PREMIA MATRIX")
    print("Paper: Regime-Conditional Factor Investing in Indian Equities")
    print("=" * 80)

    rl, fr = load_data()

    # Full-sample baseline
    full_df = full_sample_stats(fr)
    print("\nTABLE 1 — FULL-SAMPLE FACTOR PREMIA:")
    print(full_df[["factor", "n_days", "mean_ann", "vol_ann", "sharpe", "t_stat", "sig"]].to_string(index=False))

    # Regime-conditional matrix
    matrix_df = build_regime_factor_matrix(rl, fr)
    print_matrix(matrix_df)

    # Key findings — print for paper narrative
    print("\n" + "=" * 80)
    print("KEY FINDINGS (for paper narrative):")
    print("=" * 80)
    for factor in FACTORS:
        sharpes = {}
        for regime in REGIMES:
            cell = matrix_df[(matrix_df["factor"] == factor) & (matrix_df["regime"] == regime)]
            if not cell.empty:
                sharpes[regime] = cell.iloc[0]["sharpe"]
        if sharpes:
            best  = max(sharpes, key=sharpes.get)
            worst = min(sharpes, key=sharpes.get)
            print(f"  {factor}: best in {best} (SR={sharpes[best]:.3f}), "
                  f"worst in {worst} (SR={sharpes[worst]:.3f}), "
                  f"spread={sharpes[best]-sharpes[worst]:.3f}")

    # Save outputs
    matrix_df.to_csv(OUT_DIR / "table2_regime_factor_matrix.csv", index=False)
    full_df.to_csv(OUT_DIR / "table1_full_sample_premia.csv", index=False)

    latex = to_latex(matrix_df, full_df)
    (OUT_DIR / "table2_regime_factor_matrix.tex").write_text(latex)

    print(f"\nSaved:")
    print(f"  {OUT_DIR}/table1_full_sample_premia.csv")
    print(f"  {OUT_DIR}/table2_regime_factor_matrix.csv")
    print(f"  {OUT_DIR}/table2_regime_factor_matrix.tex")
