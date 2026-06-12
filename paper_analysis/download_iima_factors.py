"""
Download IIMA Fama-French + Momentum daily factors.
Saves to data/iima_factors.csv and runs regime-conditional Table A1.

Source: Agarwalla, Jacob & Varma (2013)
URL: https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

URL = ("https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/DATA/"
       "2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv")

# ── Download ──────────────────────────────────────────────────────────────────
print("Downloading IIMA daily factors...")
save_path = DATA_DIR / "iima_factors_raw.csv"
urllib.request.urlretrieve(URL, save_path)
print(f"Saved raw file: {save_path}")

# ── Parse ─────────────────────────────────────────────────────────────────────
raw = pd.read_csv(save_path, index_col=0, parse_dates=True, na_values=["NA", ""])
print(f"Raw shape: {raw.shape}  |  {raw.index[0].date()} → {raw.index[-1].date()}")
print(f"Columns: {raw.columns.tolist()}")

# Rename to standard labels
# IIMA: SMB, HML, WML (momentum), MF (market factor), RF (risk-free)
iima = raw.rename(columns={
    "MF":  "MKT",   # market excess return
    "WML": "MOM",   # momentum
    "SMB": "SMB",
    "HML": "HML",
    "RF":  "RF",
})
iima = iima[["MKT", "SMB", "HML", "MOM", "RF"]]

# Values are in % → convert to decimals
iima = iima / 100.0

# Restrict to our sample window and drop NA rows
iima = iima.loc["2019-01-01":].dropna(subset=["MKT", "SMB", "HML"])
print(f"\nCleaned IIMA (2019–): {len(iima)} days")
print(f"  MKT  mean={iima['MKT'].mean()*252*100:.2f}% ann  "
      f"vol={iima['MKT'].std()*np.sqrt(252)*100:.2f}%")
print(f"  SMB  mean={iima['SMB'].mean()*252*100:.2f}% ann")
print(f"  HML  mean={iima['HML'].mean()*252*100:.2f}% ann")
print(f"  MOM  mean={iima['MOM'].mean()*252*100:.2f}% ann")

iima.to_csv(DATA_DIR / "iima_factors.csv")
print(f"\nSaved: data/iima_factors.csv ({len(iima)} rows)")

# ── Table A1: Regime-conditional IIMA factor premia ───────────────────────────
print("\n" + "=" * 70)
print("TABLE A1 — REGIME-CONDITIONAL IIMA FACTOR PREMIA (Robustness)")
print("Replicates Table 2 using Agarwalla-Jacob-Varma (2013) factors")
print("=" * 70)

rl = pd.read_csv(DATA_DIR / "regime_labels.csv", index_col=0, parse_dates=True)

# Factors in both datasets: MKT, SMB, HML, MOM
iima_factors = ["MKT", "SMB", "HML", "MOM"]

common = iima.index.intersection(rl.index)
iima_a = iima.loc[common]
rl_a   = rl.loc[common]

print(f"\nAligned: {len(common)} days | "
      f"{common[0].date()} → {common[-1].date()}")
print(f"Regimes: {rl_a['regime_name'].value_counts().to_dict()}")

def sig_stars(p):
    if pd.isna(p):  return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    if p < 0.10:    return "†"
    return ""

rows = []

# Full sample
print(f"\n{'Factor':<8}  {'n':>5}  {'ann_ret%':>9}  {'vol%':>8}  "
      f"{'sharpe':>7}  {'t_stat':>7}  {'sig':>4}")
print("-" * 58)
for f in iima_factors:
    ret  = iima_a[f]
    n    = len(ret)
    ann  = ret.mean() * 252 * 100
    vol  = ret.std() * np.sqrt(252) * 100
    sr   = ann / vol if vol > 0 else np.nan
    t    = ret.mean() / (ret.std() / np.sqrt(n)) if ret.std() > 0 else np.nan
    from scipy.stats import t as t_dist
    p    = 2 * t_dist.sf(abs(t), df=n-1) if not np.isnan(t) else np.nan
    print(f"{f:<8}  {n:>5}  {ann:>9.2f}  {vol:>8.2f}  "
          f"{sr:>7.3f}  {t:>7.2f}  {sig_stars(p):>4}")
    rows.append({"regime": "Full", "factor": f, "n": n,
                 "ann_ret": ann, "vol": vol, "sharpe": sr, "t": t, "p": p,
                 "sig": sig_stars(p)})

# Regime-conditional
print(f"\n{'Factor':<8}  {'Regime':<10}  {'n':>5}  {'ann_ret%':>9}  "
      f"{'vol%':>8}  {'sharpe':>7}  {'t_stat':>7}  {'sig':>4}")
print("-" * 70)

from scipy.stats import t as t_dist

for regime in ["Bull", "Sideways", "Bear"]:
    mask = rl_a["regime_name"] == regime
    n_r  = mask.sum()
    for f in iima_factors:
        ret = iima_a.loc[mask, f]
        n   = len(ret)
        ann = ret.mean() * 252 * 100
        vol = ret.std() * np.sqrt(252) * 100
        sr  = ann / vol if vol > 0 else np.nan
        t   = ret.mean() / (ret.std() / np.sqrt(n)) if ret.std() > 0 else np.nan
        p   = 2 * t_dist.sf(abs(t), df=n-1) if not np.isnan(t) else np.nan
        print(f"{f:<8}  {regime:<10}  {n:>5}  {ann:>9.2f}  "
              f"{vol:>8.2f}  {sr:>7.3f}  {t:>7.2f}  {sig_stars(p):>4}")
        rows.append({"regime": regime, "factor": f, "n": n,
                     "ann_ret": ann, "vol": vol, "sharpe": sr,
                     "t": t, "p": p, "sig": sig_stars(p)})
    print()

a1_df = pd.DataFrame(rows)
a1_df.to_csv(OUT_DIR / "table_a1_iima_factors.csv", index=False)

# Generate LaTeX
print("\n" + "─" * 70)
print("LATEX (Table A1 regime matrix):")
print("─" * 70)
print(r"\begin{table}[h]")
print(r"\caption{Table A1: Regime-Conditional Factor Premia — IIMA Factors (Robustness)}")
print(r"\small")
print(r"\begin{tabular}{llrrrrr}")
print(r"\hline")
print(r"Factor & Regime & N & Ann.Ret (\%) & Ann.Vol (\%) & Sharpe & t-stat \\")
print(r"\hline")
for _, row in a1_df[a1_df["regime"] != "Full"].iterrows():
    sig = row["sig"] if row["sig"] else ""
    print(f"{row['factor']} & {row['regime']} & {row['n']} & "
          f"{row['ann_ret']:.2f} & {row['vol']:.2f} & "
          f"{row['sharpe']:.3f} & {row['t']:.2f}{sig} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\begin{tablenotes}\footnotesize")
print(r"\item Factors from Agarwalla, Jacob \& Varma (2013) IIM Ahmedabad.")
print(r"\item MOM = WML momentum factor (substitute for RMW/CMA not in IIMA library).")
print(r"\item $^\dagger p<0.10$, $^* p<0.05$, $^{**} p<0.01$, $^{***} p<0.001$")
print(r"\end{tablenotes}")
print(r"\end{table}")

# Save LaTeX
latex_lines = [
    r"\begin{table}[h]",
    r"\caption{Table A1: Regime-Conditional Factor Premia --- IIMA Factors (Robustness)}",
    r"\small",
    r"\begin{tabular}{llrrrrr}",
    r"\hline",
    r"Factor & Regime & N & Ann.Ret (\%) & Ann.Vol (\%) & Sharpe & t-stat \\",
    r"\hline",
]
for _, row in a1_df[a1_df["regime"] != "Full"].iterrows():
    sig = row["sig"] if row["sig"] else ""
    latex_lines.append(
        f"{row['factor']} & {row['regime']} & {row['n']} & "
        f"{row['ann_ret']:.2f} & {row['vol']:.2f} & "
        f"{row['sharpe']:.3f} & {row['t']:.2f}{sig} \\\\"
    )
latex_lines += [
    r"\hline",
    r"\end{tabular}",
    r"\begin{tablenotes}\footnotesize",
    r"\item Factors: Agarwalla, Jacob \& Varma (2013) IIM Ahmedabad, survivorship-bias adjusted.",
    r"\item MOM = WML momentum (IIMA library has 4 factors: MKT, SMB, HML, WML; RMW/CMA not available).",
    r"\item $^\dagger p<0.10$, $^* p<0.05$, $^{**} p<0.01$, $^{***} p<0.001$",
    r"\end{tablenotes}",
    r"\end{table}",
]
with open(OUT_DIR / "table_a1_iima_factors.tex", "w") as f:
    f.write("\n".join(latex_lines))

print(f"\n→ Saved: outputs/table_a1_iima_factors.csv")
print(f"→ Saved: outputs/table_a1_iima_factors.tex")

# ── Cross-validation: own factors vs IIMA ────────────────────────────────────
print("\n" + "=" * 70)
print("CROSS-VALIDATION: Own-constructed vs IIMA factors (correlation check)")
print("=" * 70)

own_fr = pd.read_csv(DATA_DIR / "factor_returns.csv",
                     index_col=0, parse_dates=True)
overlap = own_fr.index.intersection(iima.index)
print(f"Overlap: {len(overlap)} days")

for own_col, iima_col in [("MKT","MKT"), ("SMB","SMB"), ("HML","HML")]:
    if own_col in own_fr.columns and iima_col in iima.columns:
        r = own_fr.loc[overlap, own_col].corr(iima.loc[overlap, iima_col])
        print(f"  {own_col:>5} vs IIMA {iima_col}: r = {r:.4f}  "
              f"{'HIGH ✓' if abs(r) > 0.70 else 'LOW — investigate'}")

print("\nDone. Run Table 2 next to compare regime patterns across both factor sets.")
