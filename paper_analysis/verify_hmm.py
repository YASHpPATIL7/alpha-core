"""
HMM Fix Verification
====================
Run this BEFORE re-running any paper tables.
Checks that the fixed HMM produces economically sensible regime labels.

Pass criteria:
  ✓ Bear has negative or near-zero mean return
  ✓ Bull has highest mean return
  ✓ Bear has highest VIX / lowest VIX is in Bull or Sideways
  ✓ Regime distribution roughly: Bull ~30-50%, Sideways ~30-50%, Bear ~15-30%
  ✓ March 2020 crash week is Bear
  ✓ 2021 (Jan-Dec) is majority Bull or Sideways (NOT Bear)
  ✓ Regime persistence: transition matrix diagonal > 0.85 (regimes stick)

If any of these fail → HMM still wrong, do not re-run paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

PASS = "✓"
FAIL = "✗"

def check(condition, label, detail=""):
    mark = PASS if condition else FAIL
    print(f"  {mark}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition

print("=" * 60)
print("HMM FIX VERIFICATION")
print("=" * 60)

# ── Load regime labels ────────────────────────────────────────────────────────
rl = pd.read_csv(DATA_DIR / "regime_labels.csv", index_col=0, parse_dates=True)
print(f"\nLoaded: {len(rl)} days | {rl.index[0].date()} → {rl.index[-1].date()}")

# ── 1. Regime distribution ────────────────────────────────────────────────────
print("\n[1] REGIME DISTRIBUTION")
counts = rl["regime_name"].value_counts()
total  = len(rl)
for r in ["Bull", "Sideways", "Bear"]:
    n   = counts.get(r, 0)
    pct = n / total * 100
    print(f"  {r:<10} {n:>5} days  ({pct:.1f}%)")

bull_pct = counts.get("Bull", 0) / total * 100
bear_pct = counts.get("Bear", 0) / total * 100
check(25 <= bull_pct <= 60,  "Bull ~25–60% of days", f"{bull_pct:.1f}%")
check(10 <= bear_pct <= 35,  "Bear ~10–35% of days", f"{bear_pct:.1f}%")

# ── 2. Year-by-year breakdown ─────────────────────────────────────────────────
print("\n[2] YEAR-BY-YEAR REGIME BREAKDOWN")
rl["year"] = rl.index.year
yearly = rl.groupby(["year", "regime_name"]).size().unstack(fill_value=0)
for col in ["Bull", "Sideways", "Bear"]:
    if col not in yearly.columns:
        yearly[col] = 0
print(yearly[["Bull", "Sideways", "Bear"]].to_string())

yr2021_bear_pct = yearly.loc[2021, "Bear"] / yearly.loc[2021].sum() * 100 if 2021 in yearly.index else 100
check(yr2021_bear_pct < 40, "2021 is NOT majority Bear", f"Bear={yr2021_bear_pct:.0f}% of 2021")

# ── 3. Known-date spot checks ─────────────────────────────────────────────────
print("\n[3] KNOWN-DATE SPOT CHECKS")
spot_checks = {
    "2020-03-23": ("Bear", "Nifty -13.9%, worst day of COVID crash"),
    "2020-03-20": ("Bear", "COVID crash week"),
    "2020-03-24": ("Bear", "COVID crash week"),
    "2021-07-01": ("Bull", "Mid-2021 bull run"),
    "2021-11-01": ("Bull", "Late-2021 bull run"),
    "2024-06-04": ("Bear", "Election result shock day"),
}
for date_str, (expected, desc) in spot_checks.items():
    try:
        actual = rl.loc[date_str, "regime_name"]
        ok = (actual == expected)
        check(ok, f"{date_str} ({desc})", f"expected={expected}, got={actual}")
    except KeyError:
        print(f"  —  {date_str} not in dataset (holiday/weekend)")

# ── 4. Regime-conditional return statistics ───────────────────────────────────
print("\n[4] REGIME-CONDITIONAL ECONOMICS")
fr_path = DATA_DIR / "factor_returns.csv"
if fr_path.exists():
    fr = pd.read_csv(fr_path, index_col=0, parse_dates=True)
    common = rl.index.intersection(fr.index)
    rl_a = rl.loc[common]
    fr_a = fr.loc[common]

    for regime in ["Bull", "Sideways", "Bear"]:
        mask = rl_a["regime_name"] == regime
        mkt_days = fr_a.loc[mask, "MKT"]
        ann_ret = mkt_days.mean() * 252 * 100
        ann_vol = mkt_days.std() * np.sqrt(252) * 100
        n = mask.sum()
        print(f"  {regime:<10} n={n:>4}  ann_ret={ann_ret:+.1f}%  ann_vol={ann_vol:.1f}%")

    # Key checks
    bear_ret  = fr_a.loc[rl_a["regime_name"]=="Bear",  "MKT"].mean() * 252 * 100
    bull_ret  = fr_a.loc[rl_a["regime_name"]=="Bull",  "MKT"].mean() * 252 * 100
    check(bear_ret < bull_ret,   "Bull ann_return > Bear ann_return",
          f"Bull={bull_ret:.1f}%  Bear={bear_ret:.1f}%")
    check(bear_ret < 10,         "Bear ann_return < +10% (not a bull regime)",
          f"Bear={bear_ret:.1f}%")
else:
    print("  factor_returns.csv not found — skipping return checks")

# ── 5. VIX by regime (if available) ──────────────────────────────────────────
print("\n[5] VIX BY REGIME")
vix_path = DATA_DIR / "india_vix_history.csv"
if vix_path.exists():
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    vix.columns = vix.columns.str.strip().str.lower()
    close_col = next((c for c in vix.columns if "close" in c), vix.columns[0])
    vix = vix[[close_col]].rename(columns={close_col: "vix"})
    common_v = rl.index.intersection(vix.index)
    if len(common_v) > 100:
        rl_v  = rl.loc[common_v]
        vix_v = vix.loc[common_v]
        for regime in ["Bull", "Sideways", "Bear"]:
            mask = rl_v["regime_name"] == regime
            mean_vix = vix_v.loc[mask, "vix"].mean()
            print(f"  {regime:<10} mean VIX = {mean_vix:.1f}")
        bear_vix = vix_v.loc[rl_v["regime_name"]=="Bear",  "vix"].mean()
        bull_vix = vix_v.loc[rl_v["regime_name"]=="Bull",  "vix"].mean()
        check(bear_vix > bull_vix, "Bear VIX > Bull VIX", f"Bear={bear_vix:.1f}  Bull={bull_vix:.1f}")
    else:
        print("  Insufficient VIX overlap with regime labels")
else:
    print("  india_vix_history.csv not found — skipping VIX checks")

# ── 6. Regime persistence (transition matrix) ─────────────────────────────────
print("\n[6] REGIME PERSISTENCE")
regimes_seq = rl["regime_name"]
for regime in ["Bull", "Sideways", "Bear"]:
    in_regime   = (regimes_seq == regime)
    stays       = (in_regime & (regimes_seq.shift(-1) == regime)).sum()
    total_in    = in_regime.sum() - 1
    persistence = stays / total_in if total_in > 0 else 0
    check(persistence > 0.85, f"{regime} persistence > 85%", f"{persistence:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
print("If all checks above show ✓: HMM is fixed → proceed to re-run paper.")
print("If any ✗ remain: paste this output and we'll diagnose further.")
print("\nNext step (if passing):")
print("  python paper_analysis/table2_regime_factor_matrix.py")
print("  python paper_analysis/table3_fii_factor_regimes.py")
print("  python paper_analysis/table4_strategy_comparison.py")
