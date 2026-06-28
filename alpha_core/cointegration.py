"""
Cointegration Pairs Scanner — NSE Equities (Johansen Method)
=============================================================

Finds pairs (and multi-stock groups) of NSE stocks where the price spread
is mean-reverting using the Johansen Trace + Max-Eigenvalue test.

Used for pairs trading: β gives exact hedge weights, α tells you which
stock does the error-correcting (execution sizing anchor).

Method: Johansen Test (VAR + MLE)
  Instead of regressing A on B (Engle-Granger's asymmetric flaw), Johansen
  treats all stocks as endogenous in a VAR system and finds all independent
  cointegrating vectors via eigendecomposition.

  ΔX_t = Π·X_{t-1} + Γ₁·ΔX_{t-1} + ... + Γₚ·ΔX_{t-p} + ε_t

  The rank of Π determines how many independent relationships exist:
    Rank(Π) = 0     → No cointegration
    Rank(Π) = r     → r independent cointegrating vectors
    Rank(Π) = n     → All series already stationary

  Π = α·β'
    β (Leash Matrix):   portfolio weights for each cointegrating vector
    α (Speed Matrix):   how fast each stock corrects deviations from the spread

Why Johansen beats Engle-Granger:
  1. Symmetric — no "boss" stock. Reordering tickers = same result.
  2. Finds ALL r independent vectors, not just one.
  3. α matrix reveals WHICH stock does the correcting → execution insight.
  4. Two test statistics: Trace (conservative) + Max-Eigenvalue (precise).

Failure Modes:
  1. Cointegration breaks during crises (COVID 2020) — correlations spike
     to 1.0, all spreads blow out simultaneously. This is when the pairs
     trade loses the most. Mitigate: halt pairs trades when DCC mean_corr > 0.45.
  2. With 14 stocks: 91 possible pairs. Expect 3-8 false positives at p<0.05
     by random chance (5% of 91 = ~4.5). Always verify economic logic.
  3. Half-life < 5 days = noise. Half-life > 60 days = too slow to trade.
     Sweet spot: 10-30 day half-life.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

BASE_DIR = Path(__file__).parent.parent
RISK_ENGINE_DATA = BASE_DIR.parent / "indian-risk-engine" / "data"
DATA_DIR = BASE_DIR / "data"

# Johansen significance level — columns in crit value table
# 0 = 90%, 1 = 95%, 2 = 99%
JOHANSEN_SIG = 1       # 95% confidence

# Half-life bounds — practical tradeable range (days)
MIN_HALFLIFE = 5
MAX_HALFLIFE = 60

# Johansen VAR lag order (AIC-selected; 1 is standard for daily equity prices)
VAR_LAGS = 1

# ── Economic-mechanism screen ───────────────────────────────────────────────
# A statistical spread is only worth trading if there is a REASON for the two
# names to share a common stochastic trend. We require same-sector membership
# and attach the mechanism. Cross-sector "cointegration" on raw prices is almost
# always shared market beta or noise — which is why we also factor-neutralise.
SECTOR_MAP = {
    "RELIANCE": "Energy",     "ONGC": "Energy",
    "HDFCBANK": "Financials", "ICICIBANK": "Financials",
    "AXISBANK": "Financials", "BAJFINANCE": "Financials",
    "INFY": "IT",             "TCS": "IT",          "WIPRO": "IT",
    "DRREDDY": "Pharma",      "SUNPHARMA": "Pharma",
    "HINDUNILVR": "FMCG",     "ITC": "FMCG",
    "MARUTI": "Auto",
}
SECTOR_MECHANISM = {
    "Energy":     "shared crude-oil & energy-policy exposure",
    "Financials": "shared rate cycle, credit growth & deposit dynamics",
    "IT":         "shared USD/INR & global tech-spend exposure",
    "Pharma":     "shared US-generics pricing & USD exposure",
    "FMCG":       "shared rural demand & input-cost cycle",
    "Auto":       "shared demand & commodity-input cycle",
}

def pair_mechanism(a: str, b: str):
    """Return (same_sector: bool, sector_a, sector_b, mechanism_str)."""
    sa = SECTOR_MAP.get(a, "Other")
    sb = SECTOR_MAP.get(b, "Other")
    same = (sa == sb) and sa != "Other"
    mech = SECTOR_MECHANISM.get(sa, "—") if same else "cross-sector — no economic anchor"
    return same, sa, sb, mech

# Gates that separate a real pair from a market-beta artefact:
REQUIRE_FACTOR_NEUTRAL = True   # spread must survive FF5 residualisation
REQUIRE_SAME_SECTOR    = True   # spread must have an economic mechanism
MIN_OBS                = 60     # minimum overlapping observations per pair


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load price levels (NOT returns — cointegration works on prices)
# ═══════════════════════════════════════════════════════════════
def load_prices() -> pd.DataFrame:
    """
    Load reconstructed price levels from log returns.

    Why prices and not returns?
    Cointegration is about the LEVEL of prices moving together over time.
    Returns are already differenced — you lose the long-run relationship.
    Think of it like: you want to know if two rivers flow at the same level,
    not just if their daily rainfall is correlated.
    """
    returns_path = RISK_ENGINE_DATA / "vajra_returns.csv"

    if returns_path.exists():
        logger.info("Loading returns from Risk Engine: %s", returns_path)
        rets = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        # Reconstruct prices: start at 100, compound the log returns
        # log return → price: P_t = P_0 * exp(sum of log returns up to t)
        prices = np.exp(rets.cumsum()) * 100
        logger.info("Reconstructed prices: %d days × %d stocks", *prices.shape)
        return prices
    else:
        logger.error("Returns file not found at %s", returns_path)
        raise FileNotFoundError(f"Run the Risk Engine first: {returns_path}")


def load_residual_prices() -> pd.DataFrame:
    """
    Reconstruct *idiosyncratic* price levels from Fama-French-5 residuals.

    Why this is the whole point of the fix:
      Two stocks can look cointegrated on raw prices simply because they both
      load on the market (and SMB/HML/RMW/CMA). That is shared beta, not a
      genuine pair relationship — exactly the "fitting to noise" failure mode.
      Upstream (fama_french.py) we regress each stock's returns on the FF5
      factors and store the residual (idiosyncratic) returns in
      factor_residuals.csv. Here we rebuild prices from that RESIDUAL stream.
      Cointegration that survives on residuals reflects a real stock-specific
      long-run link, not common-factor co-movement.

    Returns a price-level DataFrame (exp of cumulative residual returns × 100),
    or None if residuals are unavailable (factor-neutral test then disabled).
    """
    resid_path = DATA_DIR / "factor_residuals.csv"
    if not resid_path.exists():
        logger.warning("factor_residuals.csv not found — factor-neutral test "
                       "DISABLED. Run fama_french.py first to enable it.")
        return None
    resid = pd.read_csv(resid_path, index_col=0, parse_dates=True).dropna(how="all")
    prices = np.exp(resid.fillna(0.0).cumsum()) * 100.0
    logger.info("Reconstructed FACTOR-NEUTRAL prices: %d days × %d stocks", *prices.shape)
    return prices


def _eval_pair(prices: pd.DataFrame, a: str, b: str) -> dict:
    """
    Run the full Johansen + spread + half-life evaluation for one pair on a
    given price frame (works for raw OR residual prices). Returns a metrics
    dict, or None if the pair can't be evaluated.
    """
    if prices is None or a not in prices.columns or b not in prices.columns:
        return None
    pair = prices[[a, b]].dropna()
    if len(pair) < MIN_OBS:
        return None
    j = johansen_test(pair)
    out = {
        "cointegrated": bool(j["cointegrated"]),
        "r": int(j["r"]),
        "trace_stat": float(j["trace_stats"][0]),
        "trace_crit": float(j["crit_trace"][0]),
        "maxeig_stat": float(j["max_eigen_stats"][0]),
        "maxeig_crit": float(j["crit_max"][0]),
        "beta_b": np.nan, "alpha_a": np.nan, "alpha_b": np.nan,
        "halflife": np.nan, "spread_mean": np.nan, "spread_std": np.nan,
    }
    if j["cointegrated"] and j["beta_vectors"] is not None:
        spread = compute_johansen_spread(pair, j["beta_vectors"])
        b0 = j["beta_vectors"][:, 0]
        b0 = b0 / b0[0]
        out["beta_b"]      = float(b0[1])
        out["halflife"]    = compute_halflife(spread)
        out["spread_mean"] = float(spread.mean())
        out["spread_std"]  = float(spread.std())
        if j["alpha_vectors"] is not None:
            out["alpha_a"] = float(j["alpha_vectors"][0, 0])
            out["alpha_b"] = float(j["alpha_vectors"][1, 0])
    return out


# ═══════════════════════════════════════════════════════════════
# STEP 2: Johansen test for a pair (or group)
# ═══════════════════════════════════════════════════════════════
def johansen_test(price_matrix: pd.DataFrame) -> dict:
    """
    Run the Johansen cointegration test on a matrix of price series.

    Returns:
        cointegrated  : bool — at least one cointegrating vector found
        r             : int  — number of cointegrating vectors (rank of Π)
        beta_vectors  : np.ndarray — β matrix (n × r) of portfolio weights
        alpha_vectors : np.ndarray — α matrix (n × r) of adjustment speeds
        trace_stats   : list of trace statistics
        max_eigen_stats: list of max-eigenvalue statistics
        crit_trace    : critical values for trace test at JOHANSEN_SIG level
        crit_max      : critical values for max-eigen test at JOHANSEN_SIG level

    How to read the output:
        beta[:, 0]    — weights for the FIRST (strongest) cointegrating spread
        alpha[:, 0]   — adjustment speeds for the first spread
        If alpha[0, 0] = -0.25 and alpha[1, 0] = 0.00:
            Stock 0 corrects 25% of the gap per day. Stock 1 is the anchor.
    """
    data = price_matrix.dropna().values  # (T, n)

    # coint_johansen(endog, det_order, k_ar_diff)
    # det_order = 0: constant in cointegration space (most common for prices)
    # k_ar_diff = VAR_LAGS: number of lagged difference terms
    result = coint_johansen(data, det_order=0, k_ar_diff=VAR_LAGS)

    n = data.shape[1]

    # Count cointegrating vectors where BOTH trace AND max-eigen reject H0
    # result.lr1 = trace statistics (length n)
    # result.lr2 = max-eigenvalue statistics (length n)
    # result.cvt = trace critical values (n × 3) — columns: 90%, 95%, 99%
    # result.cvm = max-eigen critical values (n × 3)
    r = 0
    for i in range(n):
        trace_reject   = result.lr1[i] > result.cvt[i, JOHANSEN_SIG]
        maxeig_reject  = result.lr2[i] > result.cvm[i, JOHANSEN_SIG]
        if trace_reject and maxeig_reject:
            r += 1
        else:
            break  # Johansen sequential: stop at first failure

    cointegrated = r > 0

    # β matrix: each column is a cointegrating vector (portfolio weights)
    # result.evec shape: (n, n) — columns are eigenvectors sorted by eigenvalue desc
    beta_vectors  = result.evec[:, :r] if r > 0 else None

    # α matrix from VECM: adjustment speed coefficients
    # Compute from the full VECM — statsmodels gives us this via the VECM fit
    alpha_vectors = None
    if r > 0:
        try:
            from statsmodels.tsa.vector_ar.vecm import VECM
            vecm = VECM(data, k_ar_diff=VAR_LAGS, coint_rank=r, deterministic="ci")
            vecm_fit = vecm.fit()
            alpha_vectors = vecm_fit.alpha   # (n, r)
        except Exception as e:
            logger.warning("  Could not fit VECM for α extraction: %s", e)

    return {
        "cointegrated"     : cointegrated,
        "r"                : r,
        "beta_vectors"     : beta_vectors,
        "alpha_vectors"    : alpha_vectors,
        "trace_stats"      : result.lr1.tolist(),
        "max_eigen_stats"  : result.lr2.tolist(),
        "crit_trace"       : result.cvt[:, JOHANSEN_SIG].tolist(),
        "crit_max"         : result.cvm[:, JOHANSEN_SIG].tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 3: Build the spread from β weights
# ═══════════════════════════════════════════════════════════════
def compute_johansen_spread(price_matrix: pd.DataFrame,
                             beta: np.ndarray) -> pd.Series:
    """
    Construct the cointegrating spread using Johansen β weights.

    spread_t = β' · X_t   (β normalised so first element = 1.0)

    This is NOT regression-based — no asymmetry. β comes from eigendecomposition
    of the VAR system, treating all stocks as symmetric.

    For a pair (HDFC, ICICI) with β = [1.0, -0.82]:
      spread_t = price_HDFC_t - 0.82 × price_ICICI_t
    When spread > mean: HDFC is overpriced relative to ICICI → short HDFC, long ICICI
    """
    data = price_matrix.dropna().values  # (T, n)
    # Use the first (strongest) cointegrating vector
    beta_0 = beta[:, 0]
    # Normalise: first component = 1.0 for interpretability
    beta_0 = beta_0 / beta_0[0]
    spread = data @ beta_0           # (T,)
    return pd.Series(spread, index=price_matrix.dropna().index, name="spread")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Half-life — how fast does the spread snap back?
# ═══════════════════════════════════════════════════════════════
def compute_halflife(spread: pd.Series) -> float:
    """
    Compute half-life of mean reversion using the Ornstein-Uhlenbeck model.

    Half-life = how many days until the spread closes HALF the gap to its mean.

    Example:
      Spread is 10 points above average. Half-life = 15 days.
      In 15 days, spread will be ~5 points above average (50% closed).
      In 30 days, ~2.5 points above average. And so on.

    How we compute it:
      Regress (spread_today - spread_yesterday) on spread_yesterday.
      The coefficient θ tells you the mean-reversion speed.
      Half-life = -log(2) / log(1 + θ)

    Why this matters practically:
      Half-life < 5 days: too fast, transaction costs eat the profit
      Half-life > 60 days: too slow, capital tied up too long
      Sweet spot: 10-30 days
    """
    spread_lag  = spread.shift(1)
    spread_diff = spread - spread_lag
    df = pd.concat([spread_diff, spread_lag], axis=1).dropna()
    df.columns = ["diff", "lag"]

    X     = sm.add_constant(df["lag"])
    model = sm.OLS(df["diff"], X).fit()
    theta = model.params["lag"]   # mean-reversion speed (should be negative)

    if theta >= 0:
        return np.inf   # Not mean-reverting

    halflife = -np.log(2) / np.log(1 + theta)
    return round(halflife, 1)


# ═══════════════════════════════════════════════════════════════
# STEP 5: Test ALL pairs
# ═══════════════════════════════════════════════════════════════
def scan_all_pairs(prices: pd.DataFrame,
                   resid_prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Run Johansen on every pair, on BOTH raw and factor-neutral (FF5-residual)
    prices, and attach the economic-mechanism (sector) screen.

    14 stocks → 91 pairs. For each we record:
      • raw_*    : Johansen on raw reconstructed prices (the old, naive test)
      • resid_*  : Johansen on FF5-residual prices (the meaningful test)
      • sector / same_sector / mechanism (economic-anchor screen)

    The reported headline columns (trace_stat, beta_b, halflife_days, …) come
    from the RESIDUAL test when factor-neutralisation is on, because that is the
    spread you would actually trade. Raw values are kept for the before/after
    transparency story.
    """
    tickers  = prices.columns.tolist()
    all_pairs = list(combinations(tickers, 2))
    fn = resid_prices is not None and REQUIRE_FACTOR_NEUTRAL
    logger.info("Testing %d pairs (%d stocks) via Johansen | factor-neutral=%s",
                len(all_pairs), len(tickers), fn)

    results = []
    for a, b in all_pairs:
        try:
            same, sec_a, sec_b, mech = pair_mechanism(a, b)
            raw = _eval_pair(prices, a, b)
            res = _eval_pair(resid_prices, a, b) if resid_prices is not None else None

            # headline = residual test when available, else raw
            head = res if (fn and res is not None) else raw
            if head is None:
                continue

            results.append({
                "stock_a": a, "stock_b": b,
                "sector_a": sec_a, "sector_b": sec_b,
                "same_sector": bool(same), "mechanism": mech,
                # ── headline (the spread we'd actually trade) ──
                "r"             : head["r"],
                "trace_stat"    : round(head["trace_stat"], 4),
                "trace_crit_95" : round(head["trace_crit"], 4),
                "maxeig_stat"   : round(head["maxeig_stat"], 4),
                "maxeig_crit_95": round(head["maxeig_crit"], 4),
                "beta_a"        : 1.0 if not np.isnan(head["beta_b"]) else np.nan,
                "beta_b"        : round(head["beta_b"], 4) if not np.isnan(head["beta_b"]) else np.nan,
                "alpha_a"       : round(head["alpha_a"], 4) if not np.isnan(head["alpha_a"]) else np.nan,
                "alpha_b"       : round(head["alpha_b"], 4) if not np.isnan(head["alpha_b"]) else np.nan,
                "halflife_days" : head["halflife"],
                "spread_mean"   : round(head["spread_mean"], 4) if not np.isnan(head["spread_mean"]) else np.nan,
                "spread_std"    : round(head["spread_std"], 4) if not np.isnan(head["spread_std"]) else np.nan,
                # ── before/after transparency ──
                "raw_cointegrated"   : bool(raw["cointegrated"]) if raw else False,
                "raw_trace_stat"     : round(raw["trace_stat"], 4) if raw else np.nan,
                "resid_cointegrated" : bool(res["cointegrated"]) if res else False,
                "resid_trace_stat"   : round(res["trace_stat"], 4) if res else np.nan,
                # ── final verdict: survives factor-neutral test AND has a mechanism ──
                "cointegrated"  : bool(
                    (res["cointegrated"] if (fn and res is not None)
                     else (raw["cointegrated"] if raw else False))
                    and (same or not REQUIRE_SAME_SECTOR)
                ),
            })
        except Exception as e:
            logger.warning("  %s/%s failed: %s", a, b, e)

    df = pd.DataFrame(results).sort_values("trace_stat", ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_cointegration_scanner() -> pd.DataFrame:
    """
    Run the complete Johansen cointegration scanner.
    """
    logger.info("=" * 70)
    logger.info("COINTEGRATION PAIRS SCANNER — NSE EQUITIES (JOHANSEN)")
    logger.info("=" * 70)

    # Step 1: Load raw price levels + factor-neutral (FF5-residual) price levels
    prices       = load_prices()
    resid_prices = load_residual_prices()

    # Step 2: Scan all 91 pairs on both, with the sector screen
    results = scan_all_pairs(prices, resid_prices)

    # ── ELIMINATION FUNNEL ─────────────────────────────────────────────────
    total          = len(results)
    raw_pass       = results[results["raw_cointegrated"]]
    resid_pass     = results[results["resid_cointegrated"]]
    # spurious = looked cointegrated on raw prices but DIED after factor-neutralising
    spurious       = results[results["raw_cointegrated"] & ~results["resid_cointegrated"]]
    # survived factor-neutral test
    survived       = resid_pass if REQUIRE_FACTOR_NEUTRAL else raw_pass
    with_mechanism = survived[survived["same_sector"]] if REQUIRE_SAME_SECTOR else survived
    halflife_ok    = with_mechanism[
        (with_mechanism["halflife_days"] >= MIN_HALFLIFE) &
        (with_mechanism["halflife_days"] <= MAX_HALFLIFE)
    ]
    cointegrated   = halflife_ok.copy()
    cointegrated["cointegrated"] = True          # final verdict for these rows
    tradeable      = len(cointegrated)

    logger.info("\n── ELIMINATION FUNNEL (why so few pairs survive) ───────────────────")
    logger.info("  ① All pairs tested (14C2) ............................ %2d", total)
    logger.info("  ② Cointegrated on RAW prices ........................ %2d", len(raw_pass))
    logger.info("  ③ Factor-neutral test (FF5 residuals):")
    logger.info("       • SPURIOUS — raw-only, died on residuals ....... %2d  ← shared market beta, not a pair",
                len(spurious))
    logger.info("       • SURVIVED — still cointegrated on residuals ... %2d", len(resid_pass))
    logger.info("  ④ Economic mechanism (same sector) .................. %2d", len(with_mechanism))
    logger.info("  ⑤ Half-life in %d–%dd tradeable band ............... %2d  ✅ TRADEABLE",
                MIN_HALFLIFE, MAX_HALFLIFE, tradeable)

    if len(spurious) > 0:
        logger.info("\n── SPURIOUS PAIRS (raw-price artefacts removed by factor-neutralising) ──")
        logger.info("  These looked cointegrated on raw prices but had NO idiosyncratic")
        logger.info("  link once market/size/value/profitability/investment factors were")
        logger.info("  stripped out. This is precisely the 'fitting to noise' trap.")
        logger.info("  %-13s  %-13s  %-22s  %8s  %8s", "Stock A", "Stock B",
                    "Sectors", "RawTrace", "ResTrace")
        logger.info("  " + "─" * 70)
        for _, r in spurious.sort_values("raw_trace_stat", ascending=False).iterrows():
            logger.info("  %-13s  %-13s  %-22s  %8.2f  %8.2f",
                        r["stock_a"], r["stock_b"], f"{r['sector_a']}/{r['sector_b']}",
                        r["raw_trace_stat"],
                        r["resid_trace_stat"] if not np.isnan(r["resid_trace_stat"]) else 0.0)

    if tradeable > 0:
        logger.info("\n── TRADEABLE PAIRS — survived factor-neutral test + mechanism ──────")
        for _, row in cointegrated.iterrows():
            logger.info("  ┌─ %s / %s   [%s · %s]", row["stock_a"], row["stock_b"],
                        row["sector_a"], row["mechanism"])
            logger.info("  │  Residual Johansen: Trace=%.2f > crit=%.2f  (raw was %.2f)",
                        row["trace_stat"], row["trace_crit_95"], row["raw_trace_stat"])
            logger.info("  │  β = [1.0000, %.4f]  ·  half-life %.1fd  ·  spread σ=%.4f",
                        row["beta_b"], row["halflife_days"], row["spread_std"])
            logger.info("  └─────────────────────────────────────────────────────────────")
    else:
        logger.info("\n  No pair survives factor-neutralisation + mechanism + half-life.")
        logger.info("  That is an honest result: among 14 diversified large-caps there may")
        logger.info("  simply be no idiosyncratic cointegration right now. Better to show")
        logger.info("  zero than to trade market-beta artefacts.")

    # ── MONITORED WATCHLIST ─────────────────────────────────────────────────
    # Same-sector pairs ranked by factor-neutral (residual) cointegration
    # strength. These are economically coherent candidates that did NOT clear
    # the strict Johansen bar (or the half-life band) — shown as "monitored,
    # not traded" so the methodology is visible even when nothing is tradeable.
    same_sector = results[results["same_sector"]].copy()
    same_sector = same_sector.sort_values("resid_trace_stat", ascending=False)
    # exclude any that are already tradeable
    trade_keys = set(zip(cointegrated["stock_a"], cointegrated["stock_b"]))
    same_sector = same_sector[~same_sector.apply(
        lambda r: (r["stock_a"], r["stock_b"]) in trade_keys, axis=1)]
    watchlist = same_sector.head(5).copy()
    watchlist["gap_to_crit"] = (watchlist["trace_crit_95"]
                                - watchlist["resid_trace_stat"]).round(2)

    if len(watchlist):
        logger.info("\n── MONITORED WATCHLIST (same-sector, not yet tradeable) ────────────")
        logger.info("  %-12s %-12s %-11s %9s %8s  %s",
                    "Stock A", "Stock B", "Sector", "ResTrace", "Crit", "Mechanism")
        logger.info("  " + "─" * 78)
        for _, r in watchlist.iterrows():
            logger.info("  %-12s %-12s %-11s %9.2f %8.2f  %s",
                        r["stock_a"], r["stock_b"], r["sector_a"],
                        r["resid_trace_stat"], r["trace_crit_95"], r["mechanism"])

    # ── Save outputs ───────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    results.to_csv(DATA_DIR / "cointegration_all_pairs.csv", index=False)
    cointegrated.to_csv(DATA_DIR / "cointegration_tradeable.csv", index=False)
    watchlist.to_csv(DATA_DIR / "cointegration_watchlist.csv", index=False)

    logger.info("\n  SAVED:")
    logger.info("  cointegration_all_pairs.csv  — %d pairs (raw + residual + sector)", len(results))
    logger.info("  cointegration_tradeable.csv  — %d tradeable pairs", tradeable)
    logger.info("  cointegration_watchlist.csv  — %d monitored same-sector pairs", len(watchlist))
    logger.info("=" * 70)

    return cointegrated



# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATOR — pairs detected → trades signalled
# ═══════════════════════════════════════════════════════════════
def generate_signals(prices: pd.DataFrame,
                     tradeable_pairs: pd.DataFrame,
                     entry_z: float = 2.0,
                     exit_z: float  = 0.5,
                     lookback: int  = 252) -> pd.DataFrame:
    """
    Convert Johansen cointegration results into actionable trade signals.

    Logic (Ornstein-Uhlenbeck z-score strategy):
      1. Compute rolling z-score of the Johansen spread over `lookback` days
         z_t = (spread_t − μ_rolling) / σ_rolling
      2. Entry rules:
           z > +entry_z  → spread is WIDE and HIGH  → SHORT stock_a, LONG stock_b
           z < −entry_z  → spread is WIDE and LOW   → LONG stock_a, SHORT stock_b
      3. Exit rules:
           |z| < exit_z  → spread has mean-reverted → EXIT both legs
      4. Position sizing via α:
           The stock with larger |α| does more correcting → size that leg larger.
           Size ratio = |α_a| / (|α_a| + |α_b|)  for stock_a
           This anchors capital where the reversion force is concentrated.

    Parameters:
      entry_z  : z-score threshold to enter a trade (default 2.0 = 2 std devs)
      exit_z   : z-score threshold to exit a trade (default 0.5)
      lookback : rolling window for z-score normalisation (default 252 = 1 year)

    Returns:
      DataFrame with columns:
        date, stock_a, stock_b, spread, z_score,
        signal        (LONG_A, SHORT_A, EXIT, FLAT)
        pos_a         (+1 long stock_a, -1 short stock_a, 0 flat)
        pos_b         (+1 long stock_b, -1 short stock_b, 0 flat)
        size_a_pct    (% of capital to allocate to stock_a leg)
        size_b_pct    (% of capital to allocate to stock_b leg)
        beta_b        (Johansen hedge ratio — for reference)
    """
    if tradeable_pairs.empty:
        logger.info("No tradeable pairs — no signals to generate.")
        return pd.DataFrame()

    all_signals = []

    for _, row in tradeable_pairs.iterrows():
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]
        beta_b  = row["beta_b"]       # Johansen hedge ratio (normalised)
        alpha_a = row["alpha_a"]      # adjustment speed stock_a
        alpha_b = row["alpha_b"]      # adjustment speed stock_b

        # ── Build spread using Johansen β weights ──────────────────────
        # spread_t = price_a_t − beta_b × price_b_t
        # β was normalised so beta_a = 1.0 always
        if stock_a not in prices.columns or stock_b not in prices.columns:
            logger.warning("  %s or %s not in price data — skipping", stock_a, stock_b)
            continue

        pair_prices = prices[[stock_a, stock_b]].dropna()
        spread = pair_prices[stock_a] - beta_b * pair_prices[stock_b]

        # ── Rolling z-score ────────────────────────────────────────────
        rolling_mean = spread.rolling(lookback, min_periods=lookback // 2).mean()
        rolling_std  = spread.rolling(lookback, min_periods=lookback // 2).std()
        z_score = (spread - rolling_mean) / rolling_std

        # ── Position sizing from α ─────────────────────────────────────
        # The stock with larger |α| corrects faster → size that leg heavier
        abs_a = abs(alpha_a) if not np.isnan(alpha_a) else 0.5
        abs_b = abs(alpha_b) if not np.isnan(alpha_b) else 0.5
        total = abs_a + abs_b if (abs_a + abs_b) > 0 else 1.0
        size_a = round(abs_a / total, 4)   # fraction of capital to stock_a leg
        size_b = round(abs_b / total, 4)   # fraction of capital to stock_b leg

        # ── Signal generation (vectorised) ────────────────────────────
        signals = pd.DataFrame({
            "date"      : pair_prices.index,
            "stock_a"   : stock_a,
            "stock_b"   : stock_b,
            "spread"    : spread.values,
            "z_score"   : z_score.values,
            "beta_b"    : beta_b,
            "alpha_a"   : alpha_a,
            "alpha_b"   : alpha_b,
            "size_a_pct": size_a,
            "size_b_pct": size_b,
        })

        # Assign signal column
        # LONG_A  = long stock_a, short stock_b (spread is too low → will rise)
        # SHORT_A = short stock_a, long stock_b (spread is too high → will fall)
        # EXIT    = z-score near zero → close positions
        # FLAT    = no active position
        conditions = [
            z_score < -entry_z,                        # spread too low
            z_score >  entry_z,                        # spread too high
            z_score.abs() < exit_z,                    # mean-reverted
        ]
        choices = ["LONG_A", "SHORT_A", "EXIT"]
        signals["signal"] = np.select(conditions, choices, default="FLAT")

        # Map signal → position integers
        pos_map = {"LONG_A": 1, "SHORT_A": -1, "EXIT": 0, "FLAT": 0}
        signals["pos_a"] = signals["signal"].map(pos_map)
        signals["pos_b"] = -signals["pos_a"]   # always opposite leg

        all_signals.append(signals)

        # ── Log latest signal ──────────────────────────────────────────
        latest = signals.iloc[-1]
        logger.info(
            "  %s / %s  |  z=%.2f  |  Signal: %-8s  |  β=%.4f  |  "
            "Size A=%.0f%%  B=%.0f%%",
            stock_a, stock_b,
            latest["z_score"] if not np.isnan(latest["z_score"]) else 0,
            latest["signal"],
            beta_b,
            size_a * 100, size_b * 100
        )

    if not all_signals:
        return pd.DataFrame()

    combined = pd.concat(all_signals, ignore_index=True)
    return combined


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Step 1: Run Johansen scanner ──────────────────────────
    tradeable_pairs = run_cointegration_scanner()

    # ── Step 2: Generate trading signals ──────────────────────
    if not tradeable_pairs.empty:
        logger.info("\n" + "=" * 70)
        logger.info("PAIRS TRADING SIGNAL GENERATOR")
        logger.info("=" * 70)

        # Generate signals on the SAME price space the pairs were selected in.
        # When factor-neutralisation is on, β/α/spread come from residual prices,
        # so the z-score must be computed on residual prices too — otherwise the
        # spread and its statistics are inconsistent.
        resid_prices = load_residual_prices()
        signal_prices = resid_prices if (REQUIRE_FACTOR_NEUTRAL and resid_prices is not None) \
                        else load_prices()
        signals_df = generate_signals(
            prices          = signal_prices,
            tradeable_pairs = tradeable_pairs,
            entry_z         = 2.0,    # enter at 2 standard deviations
            exit_z          = 0.5,    # exit when z-score < 0.5
            lookback        = 252,    # 1 year rolling window for normalisation
        )

        if not signals_df.empty:
            # ── Show latest signals for each pair ─────────────
            logger.info("\n── TODAY'S SIGNALS ─────────────────────────────────────────────")
            logger.info("  %-12s  %-12s  %8s  %8s  %-10s  %6s  %6s",
                        "Stock A", "Stock B", "Z-Score", "Spread", "Signal", "Pos A", "Pos B")
            logger.info("  " + "─" * 70)
            latest = signals_df.groupby(["stock_a", "stock_b"]).last().reset_index()
            for _, r in latest.iterrows():
                z = r["z_score"] if not np.isnan(r["z_score"]) else 0.0
                logger.info("  %-12s  %-12s  %8.3f  %8.4f  %-10s  %6d  %6d",
                            r["stock_a"], r["stock_b"],
                            z, r["spread"],
                            r["signal"], int(r["pos_a"]), int(r["pos_b"]))

            # ── Active trade summary ───────────────────────────
            active = latest[latest["signal"].isin(["LONG_A", "SHORT_A"])]
            logger.info("\n  Active pairs trades: %d", len(active))
            for _, r in active.iterrows():
                action_a = "LONG " if r["pos_a"] == 1 else "SHORT"
                action_b = "LONG " if r["pos_b"] == 1 else "SHORT"
                logger.info("  → %s %s (%.0f%% capital)  +  %s %s (%.0f%% capital)",
                            action_a, r["stock_a"], r["size_a_pct"] * 100,
                            action_b, r["stock_b"], r["size_b_pct"] * 100)

            # ── Save ──────────────────────────────────────────
            os.makedirs(DATA_DIR, exist_ok=True)
            out_path = DATA_DIR / "cointegration_signals.csv"
            signals_df.to_csv(out_path, index=False)
            logger.info("\n  Saved → data/cointegration_signals.csv  (%d rows)", len(signals_df))
        else:
            logger.info("No signals generated — check price data alignment.")
    else:
        logger.info("No tradeable pairs found — skipping signal generation.")

