"""
Kelly Position Sizing — M5
===========================

Computes optimal position sizes for:
  1. Pairs trades (from M3 cointegration signals)
  2. Factor alpha bets (from M1 Fama-French residuals)

Kelly Criterion chosen: CONTINUOUS HALF-KELLY gated by HMM regime (M4)

Why these design choices:
--------------------------
Full Kelly:
  f* = μ / σ²    (for continuous Gaussian returns)
  Maximises expected log(wealth). Theoretically optimal.
  Problem: requires perfect knowledge of μ and σ. In practice, μ is
  estimated from noisy historical data. Overestimating μ → full Kelly
  bets more than 100% of capital and blows up.

Half-Kelly:
  f = 0.5 × (μ / σ²)
  Industry standard for live trading. If your μ estimate is off by 2×,
  you're still at full Kelly instead of 2× Kelly (which ruins you).
  Reduces drawdown by ~50% vs full Kelly, gives up ~25% of long-run growth.
  At quant interview: "I use half-Kelly to account for estimation error in μ."

Regime Gate (M4 HMM):
  Bull     → 100% of half-Kelly (regime supports position)
  Sideways → 50%  of half-Kelly (reduced — uncertainty high)
  Bear     → 0%   (flat — Kelly says 'don't fight the regime')

  Why? Kelly assumes the edge (μ) is stable. In a Bear regime,
  every factor signal's μ shrinks — momentum reverses, value traps.
  Multiplying by the regime gate is equivalent to downward-revising
  the μ estimate based on macro context.

Position cap: 5% per stock (hard limit regardless of Kelly output)
  Protects against: over-concentration, liquidity limits, SEBI lot sizes.
  Even if Kelly says 30%, we cap at 5% — the remaining 25% sits in cash.

Position floor: 0% (no shorting the portfolio itself)
  We can go flat but never negative on individual positions.
  Pairs trades are self-hedging (long + short = market neutral),
  so the "net" portfolio exposure is naturally low.
"""

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Constants ──────────────────────────────────────────────────────────────────
HALF_KELLY        = 0.5     # safety fraction — industry standard
MAX_POSITION_PCT  = 0.05    # 5% hard cap per stock leg
MIN_POSITION_PCT  = 0.00    # 0% floor (no leverage on individual legs)
LOOKBACK_DAYS     = 252     # 1 trading year for vol estimation

# Regime multipliers — applied ON TOP of half-Kelly
REGIME_MULTIPLIER = {
    "Bull":     1.0,   # full half-Kelly
    "Sideways": 0.5,   # 25% of full Kelly (half-Kelly × 0.5)
    "Bear":     0.0,   # flat
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load inputs
# ═══════════════════════════════════════════════════════════════
def load_inputs():
    """
    Load all three upstream module outputs:
      M3 → cointegration_signals.csv  (pairs z-scores, α weights, signals)
      M4 → regime_labels.csv          (daily regime: Bull/Bear/Sideways)
      M1 → factor_scores.csv          (per-stock alpha, betas, t-stats)
    """
    signals = pd.read_csv(DATA_DIR / "cointegration_signals.csv",
                          index_col=0, parse_dates=True)
    regime  = pd.read_csv(DATA_DIR / "regime_labels.csv",
                          index_col=0, parse_dates=True)
    scores  = pd.read_csv(DATA_DIR / "factor_scores.csv",
                          index_col=0)

    logger.info("Loaded signals: %s", signals.shape)
    logger.info("Loaded regime:  %s", regime.shape)
    logger.info("Loaded scores:  %s", scores.shape)
    return signals, regime, scores


# ═══════════════════════════════════════════════════════════════
# STEP 2: Get current regime
# ═══════════════════════════════════════════════════════════════
def get_regime_multiplier(regime_df: pd.DataFrame) -> tuple:
    """
    Returns (regime_name, multiplier) from the latest observation.

    Why latest row?
      regime_labels.csv is produced by M4's forward-algorithm filtered decode —
      every row is the argmax of P(state | obs_1..obs_t), conditioning ONLY on
      data up to and including that date (no look-ahead). The last row is today's
      (or most recent trading day's) live regime readout.

      Note: historical rows are NOT Viterbi-smoothed; using Viterbi would mean
      the label on any past date conditions on future data, contaminating XGBoost
      features and Kelly backtest history. The live single-day readout uses Viterbi
      since for the terminal observation both methods are equivalent.
    """
    latest = regime_df.iloc[-1]
    name   = latest["regime_name"]
    mult   = REGIME_MULTIPLIER.get(name, 0.5)  # default 0.5 if unknown

    logger.info("\n── Current Regime ──────────────────────────────────────────────")
    logger.info("  Regime:     %s", name)
    logger.info("  Multiplier: %.1f × half-Kelly = %.3f× full Kelly",
                mult, HALF_KELLY * mult)
    logger.info("  Effective Kelly fraction: %.1f%%", HALF_KELLY * mult * 100)
    return name, mult


# ═══════════════════════════════════════════════════════════════
# STEP 3: Pairs Kelly sizing (continuous Kelly for mean-reverting spread)
# ═══════════════════════════════════════════════════════════════
def compute_pairs_kelly(signals: pd.DataFrame,
                        regime_mult: float) -> pd.DataFrame:
    """
    Computes Kelly-optimal position sizes for each cointegrated pair.

    For a pairs trade (long-short spread), the Kelly formula is:

        f* = μ_spread / σ²_spread

    Where:
        μ_spread = expected daily P&L from the spread reversion
                 = rolling mean of spread daily returns (LOOKBACK_DAYS)
        σ²_spread = variance of daily spread returns
                  = rolling variance of spread returns

    This is the CONTINUOUS Kelly for Gaussian returns — derived from
    maximising E[log(W)] when returns ~ N(μ, σ²).

    For a pairs trade specifically:
        The "return" of the spread on day t =
            (spread_t - spread_{t-1}) / |spread_{t-1}|
        If you're LONG the spread (z < -2): profit when spread rises
        If you're SHORT the spread (z > +2): profit when spread falls

    Final position size per leg:
        raw_f = HALF_KELLY × f* × regime_mult
        f_leg = clip(raw_f × alpha_weight, MIN_POSITION_PCT, MAX_POSITION_PCT)

    The α-weight from M3 reflects how much of the spread correction
    is borne by each leg — the faster error-corrector gets more sizing.
    """
    results = []
    pair_ids = signals.groupby(["stock_a", "stock_b"])

    logger.info("\n── Pairs Kelly Sizing ───────────────────────────────────────────")
    logger.info("  %-28s  %-8s  %-8s  %-10s  %-10s  %-10s  %s",
                "Pair", "f* raw", "Signal", "Size A%", "Size B%",
                "Regime×", "Rationale")
    logger.info("  " + "─" * 88)

    for (stock_a, stock_b), grp in pair_ids:
        grp = grp.sort_index()

        # Daily spread returns (dollar P&L on normalised prices)
        # Using pct_change() on a zero-crossing spread is mathematically invalid and causes infinite returns.
        # We calculate P&L divided by gross capital. Since prices start at 100:
        latest = grp.iloc[-1]
        beta_b = latest["beta_b"]
        gross_cap = 100.0 * (1.0 + abs(beta_b))
        spread_ret = grp["spread"].diff().dropna() / gross_cap

        if len(spread_ret) < LOOKBACK_DAYS:
            logger.warning("  %s/%s — insufficient history (%d days), skipping",
                           stock_a, stock_b, len(spread_ret))
            continue

        # Rolling statistics over past 252 days
        mu_spread    = spread_ret.iloc[-LOOKBACK_DAYS:].mean()
        sigma2_spread = spread_ret.iloc[-LOOKBACK_DAYS:].var()

        if sigma2_spread < 1e-10:
            logger.warning("  %s/%s — zero variance, skipping", stock_a, stock_b)
            continue

        # Raw full Kelly fraction
        full_kelly = mu_spread / sigma2_spread

        # Apply half-Kelly + regime gate
        # Bug fix B1: abs(full_kelly) would treat negative-edge pairs (μ≤0)
        # identically to positive-edge pairs. Kelly says "no bet" when μ≤0;
        # we floor at 0 so magnitude correctly collapses to zero.
        effective_f = HALF_KELLY * max(full_kelly, 0.0) * regime_mult

        # Get latest signal and α weights
        latest = grp.iloc[-1]
        signal    = latest["signal"]
        alpha_a   = abs(latest.get("alpha_a", 0.5))
        alpha_b   = abs(latest.get("alpha_b", 0.5))
        size_a_pct = latest.get("size_a_pct", 0.5)
        size_b_pct = latest.get("size_b_pct", 0.5)

        # Only size if there's an active signal — flat = zero position
        if signal == "FLAT":
            pos_a = 0.0
            pos_b = 0.0
            rationale = "No signal (z-score inside ±2.0 band)"
        else:
            # Allocate across legs proportionally to α weights (M3 output)
            # Leg A gets more if it corrects the spread faster (higher |alpha_a|)
            total_alpha = alpha_a + alpha_b + 1e-9
            raw_a = effective_f * (alpha_a / total_alpha)
            raw_b = effective_f * (alpha_b / total_alpha)

            pos_a = float(np.clip(raw_a, MIN_POSITION_PCT, MAX_POSITION_PCT))
            pos_b = float(np.clip(raw_b, MIN_POSITION_PCT, MAX_POSITION_PCT))
            rationale = f"z={latest.get('z_score', 0):.2f} | signal={signal}"

        results.append({
            "pair":        f"{stock_a}/{stock_b}",
            "stock_a":     stock_a,
            "stock_b":     stock_b,
            "signal":      signal,
            "full_kelly":  round(full_kelly, 4),
            "effective_f": round(effective_f, 4),
            "mu_spread":   round(mu_spread, 6),
            "sigma_spread": round(np.sqrt(sigma2_spread), 6),
            "sharpe_spread": round(mu_spread / (np.sqrt(sigma2_spread) + 1e-9), 4),
            "pos_a_pct":   round(pos_a * 100, 3),
            "pos_b_pct":   round(pos_b * 100, 3),
            "regime_mult": regime_mult,
            "rationale":   rationale,
        })

        logger.info("  %-28s  %-8.4f  %-8s  %-10.3f  %-10.3f  %-10.1f  %s",
                    f"{stock_a}/{stock_b}", full_kelly, signal,
                    pos_a * 100, pos_b * 100, regime_mult, rationale)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# STEP 4: Factor alpha Kelly sizing (single-stock bets)
# ═══════════════════════════════════════════════════════════════
def compute_factor_kelly(scores: pd.DataFrame,
                         regime_df: pd.DataFrame,
                         regime_name: str,
                         regime_mult: float) -> pd.DataFrame:
    """
    Computes Kelly-optimal position sizes for individual stocks
    based on their Fama-French alpha (M1 output).

    For single-stock bets:
        f* = α / σ²_α

    Where:
        α    = daily alpha from Fama-French regression (factor_scores.csv)
        σ²_α = variance of the residual returns (idiosyncratic risk)

    Why use α and not total return?
        Kelly on total returns would include market beta exposure.
        We've already accounted for market risk by hedging with MKT.
        The α is the TRUE edge — what the stock earns ABOVE what its
        beta exposure predicts. This is the only part Kelly should bet on.

    Regime adjustment for factor bets:
        Bull:     Momentum (high-β, high-RMW) stocks favoured
        Sideways: Quality factors (low-vol, high-profitability) — be selective
        Bear:     All factor bets zeroed (even positive α stocks tend to drop)

    Additional filter: only bet on stocks where alpha t-stat > 1.5
        (marginal statistical significance). Bets on t<1 alphas are noise.
    """
    results = []

    # Estimate residual variance from the residual returns file
    try:
        residuals = pd.read_csv(DATA_DIR / "factor_residuals.csv",
                                index_col=0, parse_dates=True)
        resid_vol = residuals.std()   # per-stock residual daily std
    except FileNotFoundError:
        logger.warning("factor_residuals.csv not found — using alpha std proxy")
        resid_vol = None

    logger.info("\n── Factor Alpha Kelly Sizing ────────────────────────────────────")
    logger.info("  %-12s  %-10s  %-8s  %-8s  %-8s  %-8s  %s",
                "Stock", "Alpha/day", "t-stat", "f* raw", "f final%", "Action", "Note")
    logger.info("  " + "─" * 80)

    for ticker, row in scores.iterrows():
        alpha    = row["alpha"]          # daily alpha from M1
        alpha_t  = row["alpha_t"]        # t-statistic
        alpha_p  = row["alpha_pvalue"]

        # Residual vol — use per-stock residuals if available, else proxy
        if resid_vol is not None and ticker in resid_vol.index:
            sigma_resid = resid_vol[ticker]
        else:
            # Fallback: approximate from alpha and t-stat
            # t = alpha / se_alpha → se_alpha = alpha / t
            se_alpha = abs(alpha / (alpha_t + 1e-9))
            sigma_resid = se_alpha * np.sqrt(row.get("n_obs", 1476))
            sigma_resid = max(sigma_resid, 0.005)   # floor at 0.5% daily vol

        sigma2_resid = sigma_resid ** 2

        # Full Kelly
        full_kelly = alpha / (sigma2_resid + 1e-12)

        # Apply half-Kelly + regime gate
        # Bug fix B1: abs(full_kelly) inflates negative-alpha bets.
        # Floor at 0: Kelly says "no bet" when alpha ≤ 0; effective_f collapses.
        effective_f = HALF_KELLY * max(full_kelly, 0.0) * regime_mult

        # Significance filter: only take positions with t-stat > 1.5
        T_THRESH = 1.5
        if abs(alpha_t) < T_THRESH:
            pos = 0.0
            action = "SKIP"
            note = f"t={alpha_t:.2f} < {T_THRESH} (insufficient edge)"
        elif alpha <= 0:
            # Negative alpha: Kelly already zeroed effective_f, skip cleanly.
            pos = 0.0
            action = "SKIP"
            note = f"Non-positive alpha ({alpha:.5f}/day) — no edge"
        elif regime_mult == 0.0:
            pos = 0.0
            action = "FLAT"
            note = "Bear regime — all factor bets off"
        else:
            pos = float(np.clip(effective_f, MIN_POSITION_PCT, MAX_POSITION_PCT))
            action = "LONG"   # SHORT removed: was unreachable (negative alpha skipped above)
            note = f"t={alpha_t:.2f} | α={alpha:.5f}/day"

        results.append({
            "ticker":       ticker,
            "alpha_daily":  round(alpha, 6),
            "alpha_annual": round(alpha * 252 * 100, 2),
            "alpha_tstat":  round(alpha_t, 3),
            "full_kelly":   round(full_kelly, 4),
            "effective_f":  round(effective_f, 4),
            "pos_pct":      round(pos * 100, 3),
            "action":       action,
            "regime":       regime_name,
            "note":         note,
        })

        logger.info("  %-12s  %-10.5f  %-8.2f  %-8.3f  %-8.3f  %-8s  %s",
                    ticker, alpha, alpha_t, full_kelly, pos * 100, action, note)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# STEP 5: Portfolio summary
# ═══════════════════════════════════════════════════════════════
def portfolio_summary(pairs_df: pd.DataFrame,
                      factor_df: pd.DataFrame,
                      regime_name: str) -> None:
    """
    Prints a consolidated portfolio view:
      - Total gross exposure (sum of all position sizes)
      - Net exposure (longs - shorts)
      - Number of active positions
      - Cash remaining (1 - gross_exposure)

    Gross < 100% = we're not fully deployed (expected in Bear/Sideways)
    Net   ≈ 0%   = market neutral (good for pairs strategies)
    """
    logger.info("\n── Portfolio Summary ────────────────────────────────────────────")
    logger.info("  Regime: %s", regime_name)
    logger.info("")

    # Pairs positions
    active_pairs = pairs_df[pairs_df["signal"] != "FLAT"]
    pairs_gross = (pairs_df["pos_a_pct"] + pairs_df["pos_b_pct"]).sum()

    logger.info("  PAIRS TRADES")
    logger.info("  %-24s  %d / %d active", "Active pairs:", len(active_pairs), len(pairs_df))
    logger.info("  %-24s  %.2f%%", "Gross exposure:", pairs_gross)

    for _, row in active_pairs.iterrows():
        logger.info("    → %s: LONG %s %.2f%% | SHORT %s %.2f%%  [%s]",
                    row["pair"], row["stock_a"], row["pos_a_pct"],
                    row["stock_b"], row["pos_b_pct"], row["signal"])

    # Factor positions
    active_factor = factor_df[factor_df["action"].isin(["LONG", "SHORT"])]
    factor_longs  = factor_df[factor_df["action"] == "LONG"]["pos_pct"].sum()
    factor_shorts = factor_df[factor_df["action"] == "SHORT"]["pos_pct"].sum()
    factor_gross  = factor_longs + factor_shorts
    factor_net    = factor_longs - factor_shorts

    logger.info("")
    logger.info("  FACTOR ALPHA BETS")
    logger.info("  %-24s  %d / %d active", "Active stocks:", len(active_factor), len(factor_df))
    logger.info("  %-24s  %.2f%%", "Gross exposure:", factor_gross)
    logger.info("  %-24s  +%.2f%%  -%.2f%%  net=%.2f%%",
                "Long / Short:", factor_longs, factor_shorts, factor_net)

    for _, row in active_factor.iterrows():
        logger.info("    → %s %s %.2f%%  (α=%.4f/day, t=%.2f)",
                    row["action"], row["ticker"], row["pos_pct"],
                    row["alpha_daily"], row["alpha_tstat"])

    # Totals
    total_gross = pairs_gross + factor_gross
    cash_pct    = max(0, 100 - total_gross)

    logger.info("")
    logger.info("  ═" * 35)
    logger.info("  %-24s  %.2f%%", "TOTAL GROSS EXPOSURE:", total_gross)
    logger.info("  %-24s  %.2f%%", "CASH / UNDEPLOYED:", cash_pct)
    logger.info("  %-24s  %.2f%%", "NET MARKET EXPOSURE:", factor_net)

    if total_gross > 100:
        logger.warning("  ⚠️  Total gross > 100%% — check position caps!")
    elif total_gross < 20:
        logger.info("  ℹ️  Low deployment (%.1f%%) — regime is conservative (%s)",
                    total_gross, regime_name)


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_kelly_pipeline() -> tuple:
    """
    Full M5 Kelly Sizing pipeline:
      1. Load M3 signals, M4 regime, M1 factor scores
      2. Get current regime → multiplier
      3. Compute pairs Kelly sizes (continuous Kelly on spread returns)
      4. Compute factor alpha Kelly sizes (single-stock idiosyncratic bets)
      5. Portfolio summary
      6. Save kelly_positions_pairs.csv + kelly_positions_factor.csv
    """
    logger.info("=" * 70)
    logger.info("KELLY POSITION SIZING — M5")
    logger.info("=" * 70)

    # Load
    signals, regime_df, scores = load_inputs()

    # Regime gate
    regime_name, regime_mult = get_regime_multiplier(regime_df)

    # Pairs Kelly
    pairs_df = compute_pairs_kelly(signals, regime_mult)

    # Factor Kelly
    factor_df = compute_factor_kelly(scores, regime_df, regime_name, regime_mult)

    # Summary
    portfolio_summary(pairs_df, factor_df, regime_name)

    # Save
    pairs_out  = DATA_DIR / "kelly_positions_pairs.csv"
    factor_out = DATA_DIR / "kelly_positions_factor.csv"
    pairs_df.to_csv(pairs_out, index=False)
    factor_df.to_csv(factor_out, index=False)

    logger.info("\n── SAVED ────────────────────────────────────────────────────────")
    logger.info("  kelly_positions_pairs.csv  — %d rows", len(pairs_df))
    logger.info("  kelly_positions_factor.csv — %d rows", len(factor_df))
    logger.info("  → Downstream: M6 (FinBERT gate), M10 (Alpaca execution)")
    logger.info("=" * 70)

    return pairs_df, factor_df


# ═══════════════════════════════════════════════════════════════
def get_current_positions() -> dict:
    """
    Lightweight interface for M6 / M10 — loads saved CSVs and
    returns today's position targets as a dict.

    Usage in M10 (Alpaca):
        from alpha_core.kelly_sizing import get_current_positions
        positions = get_current_positions()
        # {'BAJFINANCE': {'action': 'LONG', 'pct': 4.2, 'source': 'pairs'},
        #  'HDFCBANK':   {'action': 'SHORT', 'pct': 0.5, 'source': 'pairs'},
        #  'SUNPHARMA':  {'action': 'LONG', 'pct': 2.1, 'source': 'factor'}}
    """
    positions = {}

    pairs_path  = DATA_DIR / "kelly_positions_pairs.csv"
    factor_path = DATA_DIR / "kelly_positions_factor.csv"

    if pairs_path.exists():
        pdf = pd.read_csv(pairs_path)
        for _, row in pdf[pdf["signal"] != "FLAT"].iterrows():
            if row["pos_a_pct"] > 0:
                positions[row["stock_a"]] = {
                    "action": "LONG" if "LONG_A" in row["signal"] else "SHORT",
                    "pct": row["pos_a_pct"], "source": "pairs"
                }
            if row["pos_b_pct"] > 0:
                positions[row["stock_b"]] = {
                    "action": "SHORT" if "LONG_A" in row["signal"] else "LONG",
                    "pct": row["pos_b_pct"], "source": "pairs"
                }

    if factor_path.exists():
        fdf = pd.read_csv(factor_path)
        for _, row in fdf[fdf["action"].isin(["LONG", "SHORT"])].iterrows():
            positions[row["ticker"]] = {
                "action": row["action"],
                "pct": row["pos_pct"], "source": "factor"
            }

    return positions


if __name__ == "__main__":
    pairs_df, factor_df = run_kelly_pipeline()
    print("\n── Pairs Positions ──")
    print(pairs_df[["pair", "signal", "full_kelly", "effective_f",
                     "pos_a_pct", "pos_b_pct"]].to_string(index=False))
    print("\n── Factor Positions ──")
    print(factor_df[["ticker", "alpha_annual", "alpha_tstat",
                      "full_kelly", "pos_pct", "action"]].to_string(index=False))
