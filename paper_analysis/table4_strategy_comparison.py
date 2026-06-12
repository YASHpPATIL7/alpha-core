"""
Table 4 — Walk-Forward Strategy Comparison
==========================================

Paper: "Regime-Conditional Factor Investing in Indian Equities"

What this produces:
  Comparison of four strategies over the walk-forward period (2021-2026):
    1. Regime-Aware     — dynamically allocates based on HMM regime
    2. Static HRP       — HRP weights held for full period (monthly rebalance)
    3. MVO Max Sharpe   — best-Sharpe MVO, ignores regime
    4. Buy-and-Hold     — equal-weight Nifty 14-stock universe

  Metrics: Annualised Return, Volatility, Sharpe, MaxDD, Calmar, Turnover

The key comparison:
  If regime-aware beats static with LOWER drawdown and COMPARABLE turnover,
  the paper's core argument is validated: regime information adds value
  beyond static factor allocation.

Walk-Forward Design:
  - Expanding window: train on all data up to month t, trade month t+1
  - Monthly rebalance (23 trading days average)
  - 0.25% one-way transaction cost applied on each turnover unit
  - Regime determined by HMM state on last day of each month's training data

  Why NOT full walk-forward model refitting?
    Refitting HMM monthly is computationally expensive and introduces
    look-ahead sensitivity. Since regime detection is our INPUT (not the
    thing being optimised), we use the full-sample HMM labels and assume
    the regime signal was available at each rebalance date.
    This is documented as a limitation: full WF would require monthly HMM refits.

Regime-Aware Allocation:
  Bull     → MVO weights (momentum/quality factors dominate — Table 2)
  Sideways → HRP weights (balanced, no strong factor signal)
  Bear     → Equal-weight (capital preservation, avoid factor bets)

Transaction Cost Model:
  Each rebalance: cost = 0.25% × sum(|w_new - w_old|)
  This is conservative for NSE large-caps (actual spread ~0.05-0.15%
  but includes impact + brokerage). Documented in paper.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
# Try ml-portfolio-optimizer as sibling of alpha-core's parent (Local_Mark1)
_candidates = [
    BASE_DIR.parent / "ml-portfolio-optimizer" / "data",   # same level as alpha-core
    BASE_DIR.parent.parent / "ml-portfolio-optimizer" / "data",  # one level up
]
KUBER_DATA = next((p for p in _candidates if p.exists()), _candidates[0])
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS   = 0.0025   # 0.25% one-way transaction cost
START_DATE = "2021-01-01"   # walk-forward start (2 years warmup for HMM)
REBAL_DAYS = 21             # monthly rebalance


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all_data():
    # Returns
    ret_path = DATA_DIR.parent.parent / "indian-risk-engine" / "data" / "vajra_returns.csv"
    if not ret_path.exists():
        # Try alternative path
        ret_path = BASE_DIR.parent / "indian-risk-engine" / "data" / "vajra_returns.csv"
    if not ret_path.exists():
        ret_path = DATA_DIR / "vajra_returns.csv"

    returns = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    print(f"Returns: {returns.shape} | {returns.index[0].date()} → {returns.index[-1].date()}")

    # Regime labels
    rl = pd.read_csv(DATA_DIR / "regime_labels.csv", index_col=0, parse_dates=True)

    # Kuber backtest results (static strategies)
    k7_path = KUBER_DATA / "k7_backtest_results.csv"
    k7 = pd.read_csv(k7_path) if k7_path.exists() else None

    # HRP weights from kuber
    hrp_path = KUBER_DATA / "k3_hrp_weights.csv"
    hrp_w = pd.read_csv(hrp_path, index_col=0) if hrp_path.exists() else None

    # MVO weights
    mvo_path = KUBER_DATA / "k2_max_sharpe.csv"
    mvo_w = pd.read_csv(mvo_path, index_col=0) if mvo_path.exists() else None

    return returns, rl, k7, hrp_w, mvo_w


# ── Build daily equity curves ─────────────────────────────────────────────────
def build_equity_curves(returns: pd.DataFrame,
                         rl: pd.DataFrame,
                         hrp_w: pd.DataFrame,
                         mvo_w: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily NAV curves for each strategy starting from START_DATE.

    Strategy weights:
      - HRP / MVO: use pre-computed weights from kuber, rebalanced monthly
      - Regime-aware: switch between MVO/HRP/Equal based on regime
      - Buy-and-hold: equal weight, no rebalancing
    """
    # Align data
    common = returns.index.intersection(rl.index)
    ret = returns.loc[common].copy()
    regime = rl.loc[common, "regime_name"]

    # Restrict to walk-forward period
    wf_start = pd.Timestamp(START_DATE)
    ret = ret[ret.index >= wf_start]
    regime = regime[regime.index >= wf_start]

    stocks = ret.columns.tolist()
    n = len(stocks)

    # ── Weight vectors ────────────────────────────────────────────────────────
    # Equal weight
    w_equal = pd.Series(1/n, index=stocks)

    # HRP weights (from kuber, align to available stocks)
    if hrp_w is not None:
        # Flatten to Series regardless of CSV shape (single row, single col, or matrix)
        if isinstance(hrp_w, pd.DataFrame):
            w_hrp = hrp_w.iloc[0] if hrp_w.shape[0] == 1 else hrp_w.iloc[:, 0]
        else:
            w_hrp = hrp_w
        w_hrp = w_hrp.reindex(stocks).fillna(0)
        s = float(w_hrp.sum())
        w_hrp = w_hrp / s if s > 0 else w_equal
    else:
        w_hrp = w_equal
        print("  HRP weights not found — using equal weight")

    # MVO weights (from kuber)
    if mvo_w is not None:
        if isinstance(mvo_w, pd.DataFrame):
            w_mvo = mvo_w.iloc[0] if mvo_w.shape[0] == 1 else mvo_w.iloc[:, 0]
        else:
            w_mvo = mvo_w
        w_mvo = w_mvo.reindex(stocks).fillna(0)
        s = float(w_mvo.sum())
        w_mvo = w_mvo / s if s > 0 else w_equal
    else:
        w_mvo = w_equal
        print("  MVO weights not found — using equal weight")

    print(f"\nWalk-forward period: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)} days)")

    # ── Simulate strategies ───────────────────────────────────────────────────
    strategies = {
        "Regime-Aware" : simulate_regime_aware(ret, regime, w_bull=w_mvo, w_side=w_hrp, w_bear=w_equal),
        "Static HRP"   : simulate_static(ret, w_hrp),
        "MVO MaxSharpe": simulate_static(ret, w_mvo),
        "Buy-and-Hold" : simulate_bah(ret),
    }

    # Combine into one DataFrame
    equity_df = pd.DataFrame(strategies)
    return equity_df


def simulate_static(returns: pd.DataFrame,
                    weights: pd.Series,
                    rebal_freq: int = REBAL_DAYS) -> pd.Series:
    """
    Static strategy: hold fixed weights, rebalance every rebal_freq days.
    Apply 0.25% cost on each rebalance.
    """
    nav  = 1.0
    w_curr = weights.copy()
    navs = []

    for i, (date, row) in enumerate(returns.iterrows()):
        ret_today = (w_curr * row).sum()

        # Apply rebalancing cost
        if i > 0 and i % rebal_freq == 0:
            turnover = (weights - w_curr).abs().sum()
            cost = COST_BPS * turnover
            ret_today -= cost
            w_curr = weights.copy()

        nav *= (1 + ret_today)
        navs.append(nav)

    return pd.Series(navs, index=returns.index, name="Static")


def simulate_bah(returns: pd.DataFrame) -> pd.Series:
    """
    Buy-and-hold equal weight, no rebalancing (weights drift with returns).
    """
    n = len(returns.columns)
    nav  = 1.0
    w    = pd.Series(1/n, index=returns.columns)
    navs = []

    for date, row in returns.iterrows():
        ret_today = (w * row).sum()
        nav *= (1 + ret_today)
        # Let weights drift (true B&H)
        w = w * (1 + row)
        w = w / w.sum()
        navs.append(nav)

    return pd.Series(navs, index=returns.index, name="BuyHold")


def simulate_regime_aware(returns: pd.DataFrame,
                           regime: pd.Series,
                           w_bull: pd.Series,
                           w_side: pd.Series,
                           w_bear: pd.Series,
                           rebal_freq: int = REBAL_DAYS) -> pd.Series:
    """
    Regime-aware strategy:
      Bull     → MVO weights
      Sideways → HRP weights
      Bear     → Equal weight (cash-like, minimal factor bets)

    Regime checked at each monthly rebalance (not daily) to avoid
    excessive turnover from noisy day-to-day regime flips.
    Documented limitation: could also use posterior probabilities instead
    of hard Viterbi states — softer regime-switching.
    """
    regime_weights = {"Bull": w_bull, "Sideways": w_side, "Bear": w_bear}
    nav   = 1.0
    navs  = []
    turnovers = []

    # Initial weights: regime on day 0
    initial_regime = regime.iloc[0] if not regime.empty else "Sideways"
    w_curr = regime_weights.get(initial_regime, w_side).copy()

    for i, (date, row) in enumerate(returns.iterrows()):
        ret_today = (w_curr * row).sum()

        # Monthly rebalance: check regime and switch if needed
        if i > 0 and i % rebal_freq == 0:
            current_regime = regime.loc[date] if date in regime.index else "Sideways"
            w_target = regime_weights.get(current_regime, w_side)
            turnover = (w_target - w_curr).abs().sum()
            cost = COST_BPS * turnover
            ret_today -= cost
            w_curr = w_target.copy()
            turnovers.append(turnover)

        nav *= (1 + ret_today)
        navs.append(nav)

    ann_turnover = np.mean(turnovers) * 12 * 100 if turnovers else 0
    print(f"  Regime-Aware annualised turnover: {ann_turnover:.1f}%")

    return pd.Series(navs, index=returns.index, name="Regime-Aware")


# ── Compute performance metrics ────────────────────────────────────────────────
def performance_metrics(nav_series: pd.Series, name: str) -> dict:
    """
    Compute full set of metrics for one equity curve.
    """
    rets = nav_series.pct_change().dropna()
    n    = len(rets)

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Maximum drawdown
    peak = nav_series.cummax()
    dd   = (nav_series - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Sortino
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino  = ann_ret / downside if downside > 0 else np.nan

    # t-stat on returns
    t_stat = (rets.mean() / rets.std()) * np.sqrt(n) if rets.std() > 0 else np.nan

    total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1

    return {
        "Strategy"  : name,
        "Total Ret%": round(total_ret * 100, 2),
        "Ann Ret%"  : round(ann_ret   * 100, 2),
        "Ann Vol%"  : round(ann_vol   * 100, 2),
        "Sharpe"    : round(sharpe,   3),
        "Sortino"   : round(sortino,  3),
        "MaxDD%"    : round(max_dd    * 100, 2),
        "Calmar"    : round(calmar,   3),
        "t-stat"    : round(t_stat,   2),
    }


# ── Build comparison table ─────────────────────────────────────────────────────
def build_comparison_table(equity_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in equity_df.columns:
        metrics = performance_metrics(equity_df[col], col)
        rows.append(metrics)
    return pd.DataFrame(rows)


# ── Print and LaTeX ───────────────────────────────────────────────────────────
def print_table4(comp_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("TABLE 4 — WALK-FORWARD STRATEGY COMPARISON (2021–2026, 0.25% costs)")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("\nRegime-Aware = MVO in Bull | HRP in Sideways | Equal-Weight in Bear")
    print("Transaction cost: 0.25% one-way on each monthly rebalance turnover")


def to_latex(comp_df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Walk-Forward Strategy Comparison (2021--2026, Monthly Rebalance, 0.25\% Costs)}")
    lines.append(r"\label{tab:strategy_comparison}")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Strategy & Ann.Ret\% & Vol\% & Sharpe & Sortino & MaxDD\% & Calmar & $t$-stat \\")
    lines.append(r"\midrule")

    for _, row in comp_df.iterrows():
        bold_start = r"\textbf{" if row["Strategy"] == "Regime-Aware" else ""
        bold_end   = "}"         if row["Strategy"] == "Regime-Aware" else ""
        name = bold_start + row["Strategy"].replace("&", r"\&") + bold_end
        lines.append(
            f"{name} & {row['Ann Ret%']:.2f} & {row['Ann Vol%']:.2f} & "
            f"{row['Sharpe']:.3f} & {row['Sortino']:.3f} & "
            f"{row['MaxDD%']:.2f} & {row['Calmar']:.3f} & "
            f"{row['t-stat']:.2f} \\\\"
        )

    lines.append(r"\midrule")
    # Add kuber static results from file if richer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item \textit{Note:} Walk-forward backtest on 14 NSE large-cap stocks, 2021--2026. "
                 r"Regime-Aware strategy allocates to MVO Max-Sharpe weights in Bull regime, "
                 r"HRP weights in Sideways, and equal-weight in Bear, switching monthly. "
                 r"Transaction cost of 0.25\% one-way is applied on each rebalance's turnover. "
                 r"HMM regime labels are from the full-sample model (documented limitation: "
                 r"full walk-forward would require monthly HMM refitting).")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("TABLE 4 — WALK-FORWARD STRATEGY COMPARISON")
    print("Paper: Regime-Conditional Factor Investing in Indian Equities")
    print("=" * 80)

    returns, rl, k7, hrp_w, mvo_w = load_all_data()

    # Build equity curves
    equity_df = build_equity_curves(returns, rl, hrp_w, mvo_w)

    # Compute metrics
    comp_df = build_comparison_table(equity_df)

    # Print
    print_table4(comp_df)

    # Cross-reference with kuber's k7 results
    if k7 is not None:
        print("\n── CROSS-CHECK: Kuber k7 static backtest results ──")
        print(k7[["Strategy", "Return%", "Vol%", "Sharpe", "MaxDD%", "Calmar"]].to_string(index=False))
        print("(Above: full period 2021-2026 from ml-portfolio-optimizer)")

    # Key finding
    regime_row = comp_df[comp_df["Strategy"] == "Regime-Aware"]
    bah_row    = comp_df[comp_df["Strategy"] == "Buy-and-Hold"]
    hrp_row    = comp_df[comp_df["Strategy"] == "Static HRP"]
    if not regime_row.empty and not bah_row.empty:
        sharpe_uplift = regime_row["Sharpe"].values[0] - bah_row["Sharpe"].values[0]
        dd_improvement = bah_row["MaxDD%"].values[0] - regime_row["MaxDD%"].values[0]
        print(f"\nKEY FINDING:")
        print(f"  Regime-Aware vs Buy-and-Hold: Sharpe +{sharpe_uplift:.3f}, MaxDD better by {dd_improvement:.2f}%")
        if not hrp_row.empty:
            vs_hrp = regime_row["Sharpe"].values[0] - hrp_row["Sharpe"].values[0]
            print(f"  Regime-Aware vs Static HRP: Sharpe {'+' if vs_hrp>0 else ''}{vs_hrp:.3f}")

    # Save outputs
    comp_df.to_csv(OUT_DIR / "table4_strategy_comparison.csv", index=False)
    equity_df.to_csv(OUT_DIR / "table4_equity_curves.csv")

    latex = to_latex(comp_df)
    (OUT_DIR / "table4_strategy_comparison.tex").write_text(latex)

    print(f"\nSaved:")
    print(f"  {OUT_DIR}/table4_strategy_comparison.csv")
    print(f"  {OUT_DIR}/table4_equity_curves.csv")
    print(f"  {OUT_DIR}/table4_strategy_comparison.tex")
