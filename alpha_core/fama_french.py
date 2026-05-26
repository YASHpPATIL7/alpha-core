"""
Fama-French 5-Factor Engine for NSE Equities
=============================================

Constructs 5 risk factors from NSE large-cap returns and decomposes
each stock's return into factor exposures (betas) + unexplained alpha.

The 5 Factors:
  MKT : Market Risk Premium     = R_nifty - R_f
  SMB : Small Minus Big          = avg(Small) - avg(Big)
  HML : High Minus Low           = avg(Value) - avg(Growth)
  RMW : Robust Minus Weak        = avg(High Profit) - avg(Low Profit)
  CMA : Conservative Minus Agg.  = avg(Low Invest) - avg(High Invest)

Model:  R_i - R_f = α_i + β₁·MKT + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε_i

Failure Modes:
  1. SMB is often NEGATIVE in India — large caps dominate. The "small-cap
     premium" from US literature doesn't consistently hold on NSE.
  2. Factor construction requires fundamental data (P/B, ROE, CapEx).
     We use market-cap proxies from price data when fundamentals unavailable.
  3. With only 14 stocks, factor portfolios have very few members per bin.
     Production would use full Nifty 200 (216 stocks in our DB).

References:
  Fama, French (2015): "A five-factor model" — Journal of Financial Economics
  NSE India data via yfinance
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent.parent
RISK_ENGINE_DATA = BASE_DIR.parent / "indian-risk-engine" / "data"
DATA_DIR = BASE_DIR / "data"

# Risk-free rate: RBI repo rate ÷ 252 (daily)
RBI_REPO_RATE = 0.065           # 6.5% annual (as of 2025-26)
DAILY_RF = RBI_REPO_RATE / 252  # ~0.0258% per day

# Tickers in our universe (from Risk Engine)
TICKERS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "AXISBANK", "BAJFINANCE", "DRREDDY", "HINDUNILVR", "ITC",
    "MARUTI", "ONGC", "SUNPHARMA", "WIPRO",
]


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_returns() -> pd.DataFrame:
    """
    Load daily log returns from the Risk Engine's vajra_returns.csv.
    Falls back to yfinance if Risk Engine data not available.
    """

    csv_path = RISK_ENGINE_DATA / "vajra_returns.csv"

    if csv_path.exists():
        logger.info("Loading returns from Risk Engine: %s", csv_path)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        logger.info("Loaded %d days × %d stocks", *df.shape)
        return df

    # Fallback: download from yfinance
    logger.warning("Risk Engine data not found. Downloading from yfinance...")
    nse_tickers = [f"{t}.NS" for t in TICKERS]
    prices = yf.download(nse_tickers, start="2019-01-01", auto_adjust=True)["Close"]
    prices.columns = TICKERS
    returns = np.log(prices / prices.shift(1)).dropna()
    logger.info("Downloaded %d days × %d stocks from yfinance", *returns.shape)
    return returns


def load_nifty_returns(start: str = "2019-01-01") -> pd.Series:
    """
    Load Nifty 50 daily log returns as the TRUE market benchmark.

    Bug fixed: Previously MKT was computed as equal-weight average of
    our 14 stocks — circular/self-referential. The proper market factor
    must be an INDEPENDENT broad market index.

    We use ^NSEI (Nifty 50 index) from yfinance.
    """
    logger.info("Fetching Nifty 50 (^NSEI) as market benchmark...")
    try:
        nifty = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)["Close"]
        nifty = nifty.squeeze()  # DataFrame → Series
        nifty_ret = np.log(nifty / nifty.shift(1)).dropna()
        nifty_ret.name = "NIFTY50"
        logger.info("  Nifty 50: %d days loaded (%.1f%% annualised)",
                    len(nifty_ret), nifty_ret.mean() * 252 * 100)
        return nifty_ret
    except Exception as e:
        logger.warning("  Nifty 50 download failed: %s — falling back to equal-weight", e)
        return None


def load_market_caps() -> pd.DataFrame:
    """
    Load market capitalisation data for factor sorting.

    Known limitation (Look-Ahead Bias): We use CURRENT market cap
    from yfinance for all historical sorts. In production, you need
    Point-in-Time (PIT) market cap from NSE bhavcopy or CMIE Prowess
    so that on 2021-03-31 you only use data available on that date.
    This is documented as a known limitation — it inflates historical
    factor exposures for growth stocks.
    """
    logger.info("Fetching market caps from yfinance...")
    caps = {}
    for ticker in TICKERS:
        try:
            info = yf.Ticker(f"{ticker}.NS").info
            cap = info.get("marketCap", None)
            if cap:
                caps[ticker] = cap
                logger.info("  %s: ₹%.0f Cr", ticker, cap / 1e7)
            else:
                logger.warning("  %s: market cap not available", ticker)
        except Exception as e:
            logger.warning("  %s: failed — %s", ticker, e)

    # Fallback for missing tickers: use median cap
    if caps:
        median_cap = np.median(list(caps.values()))
        for t in TICKERS:
            if t not in caps:
                caps[t] = median_cap
                logger.info("  %s: using median cap (fallback)", t)

    return pd.Series(caps, name="market_cap")


def load_fundamentals() -> pd.DataFrame:
    """
    Load fundamental data for HML (book-to-market), RMW (profitability),
    and CMA (investment rate) factor construction.

    Pulls: P/B ratio, ROE, and CapEx growth from yfinance.
    Missing values are filled with sector medians.
    """
    logger.info("Fetching fundamentals from yfinance...")
    fundamentals = []

    for ticker in TICKERS:
        try:
            info = yf.Ticker(f"{ticker}.NS").info
            row = {
                "ticker": ticker,
                "pb_ratio": info.get("priceToBook", None),
                "roe": info.get("returnOnEquity", None),
                "revenue_growth": info.get("revenueGrowth", None),
            }
            fundamentals.append(row)
            logger.info("  %s: P/B=%.2f  ROE=%.2f%%  RevGrowth=%.1f%%",
                        ticker,
                        row["pb_ratio"] or 0,
                        (row["roe"] or 0) * 100,
                        (row["revenue_growth"] or 0) * 100)
        except Exception as e:
            logger.warning("  %s: fundamentals failed — %s", ticker, e)
            fundamentals.append({
                "ticker": ticker,
                "pb_ratio": None, "roe": None, "revenue_growth": None,
            })

    df = pd.DataFrame(fundamentals).set_index("ticker")

    # Fill missing with median (documented simplification)
    for col in df.columns:
        median = df[col].median()
        missing = df[col].isna().sum()
        if missing > 0:
            logger.warning("  Filling %d missing %s values with median=%.4f",
                          missing, col, median)
            df[col] = df[col].fillna(median)

    return df


# ═══════════════════════════════════════════════════════════════
# FACTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════
def construct_factors(returns: pd.DataFrame,
                      market_caps: pd.Series,
                      fundamentals: pd.DataFrame,
                      nifty_returns: pd.Series = None) -> pd.DataFrame:
    """
    Construct daily Fama-French 5 factors from NSE data.

    Method:
      1. MKT = Nifty 50 log return - R_f  (FIXED: was self-referential equal-weight avg)
      2. SMB = sort by market cap → bottom half avg - top half avg
      3. HML = sort by 1/P/B (book-to-market) → top half - bottom half
      4. RMW = sort by ROE → top half - bottom half
      5. CMA = sort by revenue growth → bottom half - top half
         (conservative = LOW growth firms outperform)

    Known Limitations (Look-Ahead Bias):
      - SMB sort uses CURRENT market cap (not PIT historical)
      - HML sort uses CURRENT P/B ratio (not PIT quarterly book value)
      - RMW sort uses CURRENT ROE (not PIT quarterly income statement)
      - CMA sort uses CURRENT revenue growth (not PIT)
      These are documented limitations. Full fix requires CMIE Prowess /
      AceEquity PIT data. Use factor regressions for exposure analysis
      only — do NOT run historical backtests claiming HML/RMW alpha.

    Returns DataFrame with columns: [MKT, SMB, HML, RMW, CMA]
    """
    logger.info("Constructing 5 Fama-French factors...")

    stocks = [t for t in TICKERS if t in returns.columns]
    ret = returns[stocks]

    # ── MKT: Market Risk Premium ───────────────────────────────
    # FIX: Use Nifty 50 (^NSEI) as true market, not equal-weight of 14 stocks
    if nifty_returns is not None:
        mkt = nifty_returns.reindex(ret.index).dropna() - DAILY_RF
        logger.info("  MKT (Nifty 50): mean=%.5f, std=%.5f", mkt.mean(), mkt.std())
    else:
        # Fallback (documented: self-referential, use with caution)
        mkt = ret.mean(axis=1) - DAILY_RF
        logger.warning("  MKT: using equal-weight of 14 stocks (self-referential fallback)")

    # ── Sort stocks into groups ────────────────────────────────
    # Size sort (SMB)
    # KNOWN LIMITATION — "Size Illusion":
    # Our 14 tickers are ALL Nifty 50 mega-caps (₹1L Cr to ₹18L Cr).
    # The median split puts MARUTI/AXISBANK (₹4L Cr) as "Small" vs
    # RELIANCE/HDFCBANK (₹12-18L Cr) as "Big" — both are large-caps in reality.
    # True SMB requires Nifty Smallcap 250 (₹500-5000 Cr) vs Nifty 50.
    # Fix: use 216-stock PostgreSQL DB for intra-universe size variation.
    # Current output measures intra-large-cap size, not true size premium.
    cap_median = market_caps[stocks].median()
    small = [t for t in stocks if market_caps[t] <= cap_median]
    big = [t for t in stocks if market_caps[t] > cap_median]
    logger.info("  Size split: %d small, %d big (median cap: ₹%.0f Cr)",
                len(small), len(big), cap_median / 1e7)

    # Value sort (HML) — high book-to-market = value stocks
    # P/B inverse: low P/B = high book-to-market = value
    btm = (1 / fundamentals.loc[stocks, "pb_ratio"]).sort_values(ascending=False)
    n_half = len(stocks) // 2
    value = btm.index[:n_half].tolist()
    growth = btm.index[n_half:].tolist()
    logger.info("  Value split: %d value (low P/B), %d growth (high P/B)", len(value), len(growth))

    # Profitability sort (RMW)
    roe_sorted = fundamentals.loc[stocks, "roe"].sort_values(ascending=False)
    robust = roe_sorted.index[:n_half].tolist()
    weak = roe_sorted.index[n_half:].tolist()
    logger.info("  Profit split: %d robust (high ROE), %d weak", len(robust), len(weak))

    # Investment sort (CMA)
    inv_sorted = fundamentals.loc[stocks, "revenue_growth"].sort_values(ascending=True)
    conservative = inv_sorted.index[:n_half].tolist()
    aggressive = inv_sorted.index[n_half:].tolist()
    logger.info("  Investment split: %d conservative, %d aggressive", len(conservative), len(aggressive))

    # ── Compute factor returns ─────────────────────────────────
    smb = ret[small].mean(axis=1) - ret[big].mean(axis=1)
    hml = ret[value].mean(axis=1) - ret[growth].mean(axis=1)
    rmw = ret[robust].mean(axis=1) - ret[weak].mean(axis=1)
    cma = ret[conservative].mean(axis=1) - ret[aggressive].mean(axis=1)

    factors = pd.DataFrame({
        "MKT": mkt, "SMB": smb, "HML": hml, "RMW": rmw, "CMA": cma,
    }, index=ret.index)

    # ── Log factor statistics ──────────────────────────────────
    logger.info("\n  Factor Statistics (annualised):")
    logger.info("  %-6s  %10s  %10s  %10s", "Factor", "Mean(%)", "Std(%)", "Sharpe")
    for col in factors.columns:
        ann_mean = factors[col].mean() * 252 * 100
        ann_std = factors[col].std() * np.sqrt(252) * 100
        sharpe = (factors[col].mean() * 252) / (factors[col].std() * np.sqrt(252)) if factors[col].std() > 0 else 0
        logger.info("  %-6s  %10.2f  %10.2f  %10.3f", col, ann_mean, ann_std, sharpe)

    return factors


# ═══════════════════════════════════════════════════════════════
# FACTOR REGRESSION (per stock)
# ═══════════════════════════════════════════════════════════════
def regress_stock(stock_returns: pd.Series,
                  factors: pd.DataFrame,
                  ticker: str) -> dict:
    """
    Run OLS: R_i - R_f = α + β₁·MKT + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε

    Returns dict with:
      - alpha (Jensen's Alpha): unexplained return
      - betas: exposure to each factor
      - r_squared: how much the 5 factors explain
      - residuals: ε (used by XGBoost in Module 7)
      - t_stats: significance of each coefficient
    """
    # Align dates
    df = pd.concat([stock_returns - DAILY_RF, factors], axis=1, join="inner").dropna()
    y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1:])

    model = sm.OLS(y, X).fit()

    result = {
        "ticker": ticker,
        "alpha": model.params["const"],
        "alpha_t": model.tvalues["const"],
        "alpha_pvalue": model.pvalues["const"],
        "beta_MKT": model.params["MKT"],
        "beta_SMB": model.params["SMB"],
        "beta_HML": model.params["HML"],
        "beta_RMW": model.params["RMW"],
        "beta_CMA": model.params["CMA"],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
    }

    logger.info(
        "  %-12s  α=%+.5f (t=%.2f)  R²=%.3f  βMKT=%.3f  βSMB=%.3f  βHML=%.3f",
        ticker, result["alpha"], result["alpha_t"], result["r_squared"],
        result["beta_MKT"], result["beta_SMB"], result["beta_HML"],
    )

    return result, model.resid


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_factor_engine() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full Fama-French pipeline:
      1. Load returns from Risk Engine
      2. Fetch market caps + fundamentals
      3. Construct 5 factors
      4. Regress each stock → betas + alpha + residuals

    Returns:
      - factor_scores: DataFrame with all betas/alpha per stock
      - factors: daily factor returns (MKT, SMB, HML, RMW, CMA)
      - residuals: DataFrame of regression residuals per stock
    """
    logger.info("=" * 65)
    logger.info("FAMA-FRENCH 5-FACTOR ENGINE — NSE EQUITIES")
    logger.info("=" * 65)

    # Step 1: Load data
    returns = load_returns()
    nifty_returns = load_nifty_returns(start=str(returns.index[0].date()))
    market_caps = load_market_caps()
    fundamentals = load_fundamentals()

    # Step 2: Construct factors (MKT now uses Nifty 50, not self-referential)
    factors = construct_factors(returns, market_caps, fundamentals, nifty_returns)

    # Step 3: Regress each stock
    logger.info("\n" + "─" * 65)
    logger.info("FACTOR REGRESSIONS")
    logger.info("%-14s  %-22s  %-8s  %-8s  %-8s  %-8s",
                "Ticker", "Alpha (t-stat)", "R²", "βMKT", "βSMB", "βHML")
    logger.info("─" * 65)

    all_results = []
    all_residuals = {}

    for ticker in TICKERS:
        if ticker not in returns.columns:
            logger.warning("  %s not in returns data, skipping", ticker)
            continue

        result, resid = regress_stock(returns[ticker], factors, ticker)
        all_results.append(result)
        all_residuals[ticker] = resid

    factor_scores = pd.DataFrame(all_results)
    residuals = pd.DataFrame(all_residuals)

    # Step 4: Save outputs
    os.makedirs(DATA_DIR, exist_ok=True)

    factor_scores.to_csv(DATA_DIR / "factor_scores.csv", index=False)
    factors.to_csv(DATA_DIR / "factor_returns.csv")
    residuals.to_csv(DATA_DIR / "factor_residuals.csv")

    logger.info("\n" + "=" * 65)
    logger.info("SAVED:")
    logger.info("  factor_scores.csv    — %d stocks × %d columns", *factor_scores.shape)
    logger.info("  factor_returns.csv   — %d days × %d factors", *factors.shape)
    logger.info("  factor_residuals.csv — %d days × %d stocks", *residuals.shape)
    logger.info("=" * 65)

    # Step 5: Key findings
    logger.info("\n── KEY FINDINGS ──")
    smb_mean = factors["SMB"].mean() * 252 * 100
    logger.info("  SMB annual mean: %.2f%% %s",
                smb_mean,
                "⚠️ NEGATIVE — large-cap premium in India" if smb_mean < 0
                else "✓ positive small-cap premium")

    top_alpha = factor_scores.nlargest(3, "alpha")[["ticker", "alpha", "alpha_t", "r_squared"]]
    logger.info("  Top 3 by Jensen's Alpha:")
    for _, row in top_alpha.iterrows():
        sig = "***" if abs(row["alpha_t"]) > 2.58 else "**" if abs(row["alpha_t"]) > 1.96 else ""
        logger.info("    %s: α=%+.5f (t=%.2f)%s  R²=%.3f",
                    row["ticker"], row["alpha"], row["alpha_t"], sig, row["r_squared"])

    avg_r2 = factor_scores["r_squared"].mean()
    logger.info("  Average R²: %.3f — factors explain %.1f%% of returns", avg_r2, avg_r2 * 100)

    return factor_scores, factors, residuals


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    scores, factors, residuals = run_factor_engine()
