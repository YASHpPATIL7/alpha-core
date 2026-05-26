"""
Alpaca Execution Gate — M10 (Paper Trading Stub)
==================================================

What this module does:
  Reads the final gated Kelly positions (after Regime → Kelly → FinBERT → XGBoost)
  and submits paper trades to Alpaca's API.

The honest architectural limitation (document this in interviews):
  This project's signals are computed on NSE (Indian) stocks.
  Alpaca's paper trading API only supports US equities.
  The connection between them requires a PROXY MAPPING:

    NSE Stock     → US ETF Proxy         Rationale
    ──────────────────────────────────────────────────────────
    HDFCBANK      → XLF  (Financials)    Indian bank ≈ US financial sector
    ICICIBANK     → XLF  (Financials)    Same sector proxy
    AXISBANK      → XLF  (Financials)    Same sector proxy
    BAJFINANCE    → XLF  (Financials)    NBFC ≈ US consumer finance
    INFY          → QQQ  (Tech/Nasdaq)   Indian IT ≈ US tech
    TCS           → QQQ  (Tech/Nasdaq)   Same
    WIPRO         → QQQ  (Tech/Nasdaq)   Same
    RELIANCE      → XLE  (Energy) + XLK  (Tech)  Conglomerate — split proxy
    ONGC          → XLE  (Energy)        PSU oil ≈ US energy sector
    SUNPHARMA     → XLV  (Healthcare)    Indian pharma ≈ US healthcare
    DRREDDY       → XLV  (Healthcare)    Same
    MARUTI        → XLY  (Consumer Disc) Automotive ≈ US consumer discretionary
    HINDUNILVR    → XLP  (Consumer Stap) FMCG ≈ US consumer staples
    ITC           → XLP  (Consumer Stap) FMCG + tobacco

  This is NOT signal arbitrage — it's architecture demonstration.
  In a real NSE deployment you'd use Zerodha Kite API (Indian broker) or
  Interactive Brokers with NSE access. Alpaca demonstrates the execution
  pattern: read signals → risk check → place order → log.

Why Alpaca for a demo:
  1. Free paper trading account — no real money at risk
  2. Clean REST API: alpaca-trade-api Python client
  3. Portfolio-level P&L tracking available via /v2/account endpoint
  4. Orders persist in paper account — you can show live order history
"""

import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Load .env — prefer alpha-core/.env (dedicated paper account for this strategy)
# Falls back to live-trading-alpha/.env if not found
_ALPHA_ENV  = BASE_DIR / ".env"
_LEGACY_ENV = BASE_DIR.parent / "live-trading-alpha" / ".env"

if _ALPHA_ENV.exists():
    load_dotenv(_ALPHA_ENV)
    logger.info("Loaded credentials from alpha-core/.env (dedicated account)")
elif _LEGACY_ENV.exists():
    load_dotenv(_LEGACY_ENV)
    logger.info("Loaded credentials from live-trading-alpha/.env (shared account)")
else:
    logger.warning("No .env found — will run in DRY_RUN mode")

# ── NSE → US ETF proxy mapping ───────────────────────────────────────────────
NSE_TO_ETF = {
    "HDFCBANK":   "XLF",   # Financials ETF
    "ICICIBANK":  "XLF",
    "AXISBANK":   "XLF",
    "BAJFINANCE": "XLF",
    "INFY":       "QQQ",   # Nasdaq-100 (IT proxy)
    "TCS":        "QQQ",
    "WIPRO":      "QQQ",
    "RELIANCE":   "XLE",   # Energy ETF (closest single proxy)
    "ONGC":       "XLE",
    "SUNPHARMA":  "XLV",   # Healthcare ETF
    "DRREDDY":    "XLV",
    "MARUTI":     "XLY",   # Consumer Discretionary ETF
    "HINDUNILVR": "XLP",   # Consumer Staples ETF
    "ITC":        "XLP",
}

# Position size for paper trading (fixed $1000 per active signal)
# In production: use gated_pct × portfolio_value
PAPER_SIZE_USD = 1000


def load_alpaca_client():
    """
    Load Alpaca client from environment variables.
    Keys are auto-loaded from live-trading-alpha/.env:
      ALPACA_API_KEY    → paper trading key
      ALPACA_SECRET_KEY → paper trading secret
    Runs DRY_RUN if keys missing.
    """
    # Support both naming conventions
    key    = os.getenv("ALPACA_KEY") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    base   = os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")

    if not key or not secret:
        logger.warning("Alpaca keys not found — DRY_RUN mode")
        return None

    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(key, secret, base, api_version="v2")
        account = api.get_account()
        logger.info("Alpaca ACTIVE | Portfolio: $%s | Cash: $%s | Equity: $%s",
                    account.portfolio_value, account.cash, account.equity)
        return api
    except ImportError:
        logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
        return None
    except Exception as e:
        logger.error("Alpaca connection failed: %s", e)
        return None


def fetch_account_summary(api) -> dict:
    """
    Pull live account + position state from Alpaca.
    This is the LIVE PnL — real paper-traded performance since strategy inception.

    Shows:
      - Current portfolio value vs starting $100K
      - All open positions with unrealized PnL
      - Recent order history (last 10)
    """
    if api is None:
        return {}

    acct = api.get_account()
    positions = api.list_positions()
    orders = api.list_orders(status="all", limit=10)

    total_pnl = float(acct.equity) - 100_000   # paper accounts start at $100K
    pnl_pct   = total_pnl / 100_000 * 100

    logger.info("\n── Live Paper Account (Alpaca) ──────────────────────────────")
    logger.info("  Portfolio value: $%s", acct.portfolio_value)
    logger.info("  Total PnL vs $100K start: $%.2f  (%.2f%%)", total_pnl, pnl_pct)
    logger.info("  Cash available: $%s", acct.cash)
    logger.info("  Open positions: %d", len(positions))
    for p in positions:
        logger.info("    %-8s qty=%-6s mkt=$%-10.2f unrealized_pl=$%.2f",
                    p.symbol, p.qty, float(p.market_value), float(p.unrealized_pl))

    return {
        "portfolio_value":  float(acct.portfolio_value),
        "cash":            float(acct.cash),
        "equity":          float(acct.equity),
        "total_pnl_usd":   total_pnl,
        "total_pnl_pct":   pnl_pct,
        "n_positions":     len(positions),
        "positions":       [{"symbol": p.symbol, "qty": p.qty,
                             "mkt_val": float(p.market_value),
                             "unrealized_pl": float(p.unrealized_pl)}
                            for p in positions],
    }


def build_order_book() -> pd.DataFrame:
    """
    Reads the three-layer gated positions and maps NSE tickers to US ETF proxies.

    Signal hierarchy (from strongest to weakest confidence):
      1. FinBERT gate: multiplier < 0.3 → SKIP entirely
      2. Kelly position: 0% → SKIP
      3. XGBoost signal: SHORT_BIAS on IC > 0.05 stocks → REDUCE position by 50%
      4. Remaining: SUBMIT as LONG at gated_pct × $PAPER_SIZE_USD

    Position direction:
      LONG  = buy the ETF proxy (signal positive or neutral)
      SHORT = sell the ETF proxy (XGB SHORT_BIAS + IC > 0.05)

    Note: ETF-level aggregation —
      If HDFCBANK=LONG(2%) and ICICIBANK=SKIP, both map to XLF.
      Final XLF position = HDFCBANK's signal only.
      If both had signals, we'd average — but current output has only one active.
    """
    # Load FinBERT-gated positions
    try:
        factor_gated = pd.read_csv(DATA_DIR / "kelly_positions_factor_gated.csv")
    except FileNotFoundError:
        logger.error("kelly_positions_factor_gated.csv not found. Run finbert_sentiment.py first.")
        return pd.DataFrame()

    # Load XGBoost signals for IC filter
    try:
        xgb_preds = pd.read_csv(DATA_DIR / "xgb_predictions.csv")
        xgb_map = dict(zip(xgb_preds["ticker"], xgb_preds["signal"]))
        xgb_ic  = dict(zip(xgb_preds["ticker"], xgb_preds["ic_test"]))
    except FileNotFoundError:
        logger.warning("xgb_predictions.csv not found. Running without XGB filter.")
        xgb_map = {}
        xgb_ic  = {}

    orders = []
    for _, row in factor_gated.iterrows():
        ticker      = row["ticker"]
        gated_pct   = float(row.get("gated_pos_pct", row.get("gated_pct", row.get("kelly_pct", 0))))

        sentiment   = row.get("sentiment", "neutral")
        multiplier  = float(row.get("multiplier", 0.7))
        etf         = NSE_TO_ETF.get(ticker, "SPY")   # fallback to SPY

        # Gate 1: FinBERT multiplier < 0.3 = too uncertain → skip
        if multiplier < 0.3:
            action = "SKIP_SENTIMENT"
            qty = 0
        # Gate 2: Zero Kelly position → skip
        elif gated_pct <= 0:
            action = "SKIP_FLAT"
            qty = 0
        else:
            # Gate 3: XGB SHORT_BIAS on confident stocks → halve position
            xgb_signal = xgb_map.get(ticker, "NEUTRAL")
            ic_val     = xgb_ic.get(ticker, 0)
            if xgb_signal == "SHORT_BIAS" and ic_val > 0.05:
                action = "SHORT_XGB"
                qty = -int(PAPER_SIZE_USD * gated_pct / 100 / 100)  # fractional → shares
                qty = max(qty, -10)  # max 10 shares short per position
            else:
                action = "LONG"
                qty = int(PAPER_SIZE_USD * gated_pct / 100 / 100)
                qty = max(qty, 1)  # min 1 share

        orders.append({
            "nse_ticker":     ticker,
            "etf_proxy":      etf,
            "gated_pct":      gated_pct,
            "sentiment":      sentiment,
            "multiplier":     multiplier,
            "xgb_signal":     xgb_map.get(ticker, "N/A"),
            "xgb_ic":         round(xgb_ic.get(ticker, 0), 4),
            "action":         action,
            "qty":            qty,
            "side":           "buy" if qty > 0 else ("sell" if qty < 0 else "none"),
            "notional_usd":   abs(qty) * 100,   # approximate ($100/share for ETFs)
        })

    return pd.DataFrame(orders)


def submit_orders(order_book: pd.DataFrame, api) -> list:
    """
    Submit orders to Alpaca paper API.
    Only submits where action is LONG or SHORT_XGB and qty != 0.

    Why market orders (not limit)?
      Paper trading at end of day — market orders fill immediately.
      Limit orders add complexity without benefit in paper mode.
      In production: use TWAP/VWAP for orders > 1% of ADV.
    """
    submitted = []
    skipped   = []

    for _, row in order_book.iterrows():
        if row["qty"] == 0:
            skipped.append(row["nse_ticker"])
            continue

        symbol = row["etf_proxy"]
        side   = row["side"]
        qty    = abs(int(row["qty"]))

        if api is None:
            # DRY_RUN: log what would be submitted
            logger.info("  [DRY_RUN] %s %d × %s (proxy for %s | %.2f%% Kelly | XGB: %s)",
                        side.upper(), qty, symbol, row["nse_ticker"],
                        row["gated_pct"], row["xgb_signal"])
            submitted.append({**row.to_dict(), "status": "DRY_RUN",
                              "order_id": f"DRY_{row['nse_ticker']}"})
        else:
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="day",
                    client_order_id=f"alphacore_{row['nse_ticker']}_{datetime.now().strftime('%Y%m%d')}"
                )
                logger.info("  [SUBMITTED] %s %d × %s → order_id=%s",
                            side.upper(), qty, symbol, order.id)
                submitted.append({**row.to_dict(), "status": "SUBMITTED",
                                  "order_id": order.id})
            except Exception as e:
                logger.error("  [FAILED] %s %s: %s", symbol, side, e)
                submitted.append({**row.to_dict(), "status": f"ERROR: {e}",
                                  "order_id": "N/A"})

    logger.info("  Submitted: %d | Skipped: %d", len(submitted), len(skipped))
    return submitted


def run_alpaca_pipeline():
    logger.info("=" * 65)
    logger.info("ALPACA EXECUTION GATE — M10")
    logger.info("NSE signal -> US ETF proxy -> Live paper account")
    logger.info("=" * 65)

    # Connect and pull live account state first
    api = load_alpaca_client()
    mode = "LIVE PAPER" if api else "DRY_RUN"
    logger.info("Mode: %s", mode)

    account_summary = fetch_account_summary(api)
    portfolio_value = account_summary.get("portfolio_value", 100_000)

    # Build order book
    logger.info("\nBuilding order book (sizing against $%.2f)...", portfolio_value)
    order_book = build_order_book()

    if order_book.empty:
        logger.error("Empty order book — check upstream CSVs.")
        return None, []

    # Re-size qty using real portfolio value
    # notional = portfolio_value * gated_pct%
    # qty      = floor(notional / etf_approx_price)
    ETF_APPROX_PRICE = {
        "XLF": 45, "QQQ": 480, "XLE": 90,
        "XLV": 140, "XLY": 195, "XLP": 80, "SPY": 580,
    }
    for i, row in order_book.iterrows():
        if row["qty"] == 0:
            continue
        etf_price = ETF_APPROX_PRICE.get(row["etf_proxy"], 100)
        notional  = portfolio_value * row["gated_pct"] / 100
        new_qty   = max(1, int(notional / etf_price))
        order_book.at[i, "qty"] = new_qty if row["side"] == "buy" else -new_qty
        order_book.at[i, "notional_usd"] = new_qty * etf_price

    logger.info("\n-- Order Book ------------------------------------------------------")
    logger.info("  %-12s %-6s %-8s %-8s %-5s %-12s %-10s",
                "NSE", "ETF", "Kelly%", "Mult", "Qty", "Action", "XGB")
    logger.info("  " + "-" * 72)
    for _, r in order_book.iterrows():
        logger.info("  %-12s %-6s %-8.2f %-8.3f %-5d %-12s %-10s",
                    r["nse_ticker"], r["etf_proxy"], r["gated_pct"],
                    r["multiplier"], r["qty"], r["action"], r["xgb_signal"])

    # Submit orders
    logger.info("\nSubmitting orders (%s)...", mode)
    submitted = submit_orders(order_book, api)

    # Save all outputs
    order_book.to_csv(DATA_DIR / "alpaca_order_book.csv", index=False)
    pd.DataFrame(submitted).to_csv(DATA_DIR / "alpaca_submitted.csv", index=False)
    if account_summary:
        pd.DataFrame([account_summary]).to_csv(DATA_DIR / "alpaca_account.csv", index=False)
        logger.info("  alpaca_account.csv     — live account snapshot saved")

    logger.info("\n-- SAVED -----------------------------------------------------------")
    logger.info("  alpaca_order_book.csv  — order book with rationale")
    logger.info("  alpaca_submitted.csv   — submission status per order")
    logger.info("=" * 65)

    return order_book, submitted


if __name__ == "__main__":
    result = run_alpaca_pipeline()
    if result[0] is not None:
        order_book, submitted = result
        print("\n-- Final Order Book --")
        active = order_book[order_book["qty"] != 0]
        if active.empty:
            print("No active orders today (all positions flat — z-scores inside threshold).")
        else:
            print(active[["nse_ticker", "etf_proxy", "action",
                           "qty", "gated_pct", "xgb_signal"]].to_string(index=False))

