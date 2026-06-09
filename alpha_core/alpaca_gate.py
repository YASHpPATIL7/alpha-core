"""
Alpaca Execution Gate — M10 (Paper Trading Stub)
==================================================

What this module does:
  Reads the final gated Kelly positions (after Regime → Kelly → FinBERT → XGBoost)
  and submits paper trades to Alpaca's API using delta-based rebalancing.

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
    RELIANCE      → XLE  (Energy)        Conglomerate — closest single proxy
    ONGC          → XLE  (Energy)        PSU oil ≈ US energy sector
    SUNPHARMA     → XLV  (Healthcare)    Indian pharma ≈ US healthcare
    DRREDDY       → XLV  (Healthcare)    Same
    MARUTI        → XLY  (Consumer Disc) Automotive ≈ US consumer discretionary
    HINDUNILVR    → XLP  (Consumer Stap) FMCG ≈ US consumer staples
    ITC           → XLP  (Consumer Stap) FMCG + tobacco

  This is NOT signal arbitrage — it's architecture demonstration.
  In a real NSE deployment you'd use Zerodha Kite API or Interactive Brokers
  with NSE access. Alpaca demonstrates the execution pattern:
  read signals → risk check → delta rebalance → log.

Position management (v2 — delta rebalancing):
  target_qty = Kelly% × portfolio_value / ETF_price
  current_qty = shares currently held
  delta = target_qty − current_qty
  Trade only the delta. Stop-loss overrides all signals.
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

# Load .env — prefer alpha-core/.env (dedicated paper account)
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
    "RELIANCE":   "XLE",   # Energy ETF
    "ONGC":       "XLE",
    "SUNPHARMA":  "XLV",   # Healthcare ETF
    "DRREDDY":    "XLV",
    "MARUTI":     "XLY",   # Consumer Discretionary ETF
    "HINDUNILVR": "XLP",   # Consumer Staples ETF
    "ITC":        "XLP",
}

# ── Risk parameters ──────────────────────────────────────────────────────────
STOP_LOSS_PCT       = 0.04   # Close if unrealized loss > 4%
MAX_ETF_WEIGHT      = 0.15   # No single ETF > 15% of portfolio
REBALANCE_THRESHOLD = 2      # Min share delta to trade (avoids noise churn)


def load_alpaca_client():
    """
    Load Alpaca REST client from environment variables.
    Returns None (DRY_RUN) if keys are missing.
    """
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
    Pull live account state from Alpaca.
    Returns portfolio value, per-position details with cost basis + weight,
    and cumulative PnL vs $100K starting capital.
    """
    if api is None:
        return {}

    acct      = api.get_account()
    positions = api.list_positions()

    equity    = float(acct.equity)
    total_pnl = equity - 100_000
    pnl_pct   = total_pnl / 100_000 * 100

    pos_details = []
    logger.info("\n── Live Paper Account (Alpaca) ──────────────────────────────────")
    logger.info("  Portfolio equity : $%.2f", equity)
    logger.info("  Total PnL vs $100K: $%+.2f  (%+.2f%%)", total_pnl, pnl_pct)
    logger.info("  Cash             : $%.2f  (%.1f%% of equity)",
                float(acct.cash), float(acct.cash) / equity * 100)
    logger.info("  Open positions   : %d", len(positions))

    for p in positions:
        pnl_pct_pos = float(p.unrealized_plpc) * 100
        weight      = float(p.market_value) / equity * 100
        logger.info(
            "    %-6s  qty=%-4s  entry=$%-8.2f  mkt=$%-10.2f  "
            "unreal=$%+.2f (%+.1f%%)  weight=%.1f%%",
            p.symbol, p.qty, float(p.avg_entry_price),
            float(p.market_value), float(p.unrealized_pl), pnl_pct_pos, weight
        )
        pos_details.append({
            "symbol":         p.symbol,
            "qty":            int(p.qty),
            "avg_entry":      float(p.avg_entry_price),
            "cost_basis":     float(p.avg_entry_price) * float(p.qty),
            "mkt_val":        float(p.market_value),
            "unrealized_pl":  float(p.unrealized_pl),
            "unrealized_pct": pnl_pct_pos,
            "weight_pct":     weight,
        })

    return {
        "portfolio_value": float(acct.portfolio_value),
        "cash":            float(acct.cash),
        "equity":          equity,
        "total_pnl_usd":   total_pnl,
        "total_pnl_pct":   pnl_pct,
        "n_positions":     len(positions),
        "positions":       pos_details,
    }


def fetch_current_positions(api) -> dict:
    """
    Returns {etf_symbol: qty_held} for all open positions.
    Used to compute delta = target - current before every order.
    """
    if api is None:
        return {}
    try:
        positions = api.list_positions()
        pos_map   = {p.symbol: int(p.qty) for p in positions}
        logger.info("  Current holdings: %s", pos_map if pos_map else "FLAT")
        return pos_map
    except Exception as e:
        logger.warning("  Could not fetch positions: %s — assuming flat", e)
        return {}


def build_order_book() -> pd.DataFrame:
    """
    Reads three-layer gated positions and maps NSE tickers to US ETF proxies.
    Returns target_pct per row — actual qty is computed in run_alpaca_pipeline()
    using live portfolio value and live ETF prices.

    Signal hierarchy:
      1. FinBERT gate: multiplier < 0.3 → SKIP
      2. Kelly position: 0% → SKIP
      3. XGBoost SHORT_BIAS + IC > 0.05 → SHORT (negative target_pct)
      4. Remaining → LONG at gated_pct%
    """
    try:
        factor_gated = pd.read_csv(DATA_DIR / "kelly_positions_factor_gated.csv")
    except FileNotFoundError:
        logger.error("kelly_positions_factor_gated.csv not found. Run finbert_sentiment.py first.")
        return pd.DataFrame()

    try:
        xgb_preds = pd.read_csv(DATA_DIR / "xgb_predictions.csv")
        xgb_map   = dict(zip(xgb_preds["ticker"], xgb_preds["signal"]))
        xgb_ic    = dict(zip(xgb_preds["ticker"], xgb_preds["ic_test"]))
    except FileNotFoundError:
        logger.warning("xgb_predictions.csv not found — running without XGB filter.")
        xgb_map = {}
        xgb_ic  = {}

    rows = []
    for _, row in factor_gated.iterrows():
        ticker     = row["ticker"]
        gated_pct  = float(row.get("gated_pos_pct",
                             row.get("gated_pct", row.get("kelly_pct", 0))))
        sentiment  = row.get("sentiment", "neutral")
        multiplier = float(row.get("multiplier", 0.7))
        etf        = NSE_TO_ETF.get(ticker, "SPY")
        xgb_signal = xgb_map.get(ticker, "NEUTRAL")
        ic_val     = xgb_ic.get(ticker, 0)

        if multiplier < 0.3:
            action     = "SKIP_SENTIMENT"
            target_pct = 0.0
        elif gated_pct <= 0:
            action     = "SKIP_FLAT"
            target_pct = 0.0
        elif xgb_signal == "SHORT_BIAS" and ic_val > 0.05:
            action     = "SHORT_XGB"
            target_pct = -gated_pct     # negative = short
        else:
            action     = "LONG"
            target_pct = gated_pct

        rows.append({
            "nse_ticker": ticker,
            "etf_proxy":  etf,
            "target_pct": round(target_pct, 4),
            "sentiment":  sentiment,
            "multiplier": multiplier,
            "xgb_signal": xgb_signal,
            "xgb_ic":     round(ic_val, 4),
            "action":     action,
        })

    return pd.DataFrame(rows)


def submit_orders(order_book: pd.DataFrame, api,
                  current_positions: dict,
                  pos_details: list,
                  portfolio_value: float,
                  etf_prices: dict) -> list:
    """
    Delta-based rebalancing — the institutional way.

    For each ETF proxy:
      target_qty  = floor(portfolio_value × |target_pct|% / etf_price)
      current_qty = shares currently held
      delta       = target_qty − current_qty

    Rules applied in priority order:
      1. STOP-LOSS:  unrealized_pct < -4% → close regardless of signal
      2. MAX CAP:    cap target at 15% of portfolio per ETF
      3. THRESHOLD:  |delta| < 2 shares → HOLD (avoid noise trades)
      4. delta > 0 → BUY delta shares (new position or add)
      5. delta < 0 → SELL |delta| shares (trim or close)

    Labels:
      BUY   = opening new long
      ADD   = adding to existing long
      TRIM  = reducing long (target still > 0)
      CLOSE = fully exiting (target = 0)
      HOLD  = within rebalance threshold
    """
    stoploss_map = {p["symbol"]: p["unrealized_pct"] for p in pos_details}
    submitted, skipped = [], []
    today = datetime.now().strftime("%Y%m%d")

    # ── 0. Aggregate target_pct by ETF Proxy ──────────────────────────────
    # BUGFIX: Previously, if multiple NSE stocks mapped to the same ETF
    # (e.g. HDFCBANK, ICICIBANK -> XLF), the script would submit separate
    # conflicting orders per NSE stock against the aggregate XLF holding.
    # This caused "client_order_id must be unique" and massive over/under-leveraging.
    # We must sum the targets for each ETF before trading.
    
    etf_targets = {}
    etf_sources = {}
    
    for _, row in order_book.iterrows():
        symbol = row["etf_proxy"]
        nse    = row["nse_ticker"]
        pct    = row["target_pct"]
        
        if symbol not in etf_targets:
            etf_targets[symbol] = 0.0
            etf_sources[symbol] = []
            
        etf_targets[symbol] += pct
        if pct != 0:
            etf_sources[symbol].append(f"{nse}({pct:.2f}%)")
        else:
            etf_sources[symbol].append(f"{nse}(0%)")

    # ── Process each unique ETF ──────────────────────────────────────────
    for symbol, aggregated_pct in etf_targets.items():
        sources_str = ", ".join(etf_sources[symbol])
        current_qty = current_positions.get(symbol, 0)
        etf_price   = etf_prices.get(symbol, 100)

        # Base record for logging
        rec = {
            "etf_proxy": symbol,
            "nse_sources": sources_str,
            "aggregated_pct": round(aggregated_pct, 4)
        }

        # ── 1. STOP-LOSS ─────────────────────────────────────────────────────
        unreal_pct = stoploss_map.get(symbol, 0.0)
        if current_qty > 0 and unreal_pct < -(STOP_LOSS_PCT * 100):
            logger.info("  [STOP-LOSS] %s down %.1f%% (threshold %.0f%%) — closing %d shares",
                        symbol, unreal_pct, STOP_LOSS_PCT * 100, current_qty)
            if api:
                try:
                    api.close_position(symbol)
                    submitted.append({**rec, "status": "STOP_LOSS",
                                      "order_id": "close_position",
                                      "delta_qty": -current_qty, "final_qty": 0})
                except Exception as e:
                    logger.error("  [STOP-LOSS FAILED] %s: %s", symbol, e)
            else:
                logger.info("  [DRY_RUN] STOP-LOSS close %s", symbol)
                submitted.append({**rec, "status": "DRY_STOP_LOSS",
                                  "order_id": f"DRY_SL_{symbol}",
                                  "delta_qty": -current_qty, "final_qty": 0})
            continue

        # ── 2. Compute target_qty with position cap ───────────────────────────
        if aggregated_pct == 0:
            target_qty = 0
        else:
            raw_notional    = portfolio_value * abs(aggregated_pct) / 100
            capped_notional = min(raw_notional, portfolio_value * MAX_ETF_WEIGHT)
            target_qty      = int(capped_notional / etf_price)
            if aggregated_pct < 0:
                target_qty = -target_qty    # short
            if capped_notional < raw_notional:
                logger.info("  [CAP] %s capped at %.0f%% max (Kelly was %.2f%%)",
                            symbol, MAX_ETF_WEIGHT * 100, abs(aggregated_pct))

        # ── 3. Delta ─────────────────────────────────────────────────────────
        delta = target_qty - current_qty

        # ── 4. Rebalance threshold ────────────────────────────────────────────
        if abs(delta) < REBALANCE_THRESHOLD:
            status = "HOLD" if current_qty != 0 else "FLAT_HOLD"
            logger.info("  [%-9s] %-6s  held=%-4d  target=%-4d  delta=%+d  (below threshold)",
                        status, symbol, current_qty, target_qty, delta)
            submitted.append({**rec, "status": status, "order_id": "N/A",
                              "delta_qty": delta, "final_qty": current_qty})
            if current_qty == 0 and target_qty == 0:
                skipped.append(symbol)
            continue

        # ── 5. Execute delta trade ────────────────────────────────────────────
        side  = "buy" if delta > 0 else "sell"
        qty   = abs(delta)
        label = ("ADD"   if delta > 0 and current_qty > 0 else
                 "BUY"   if delta > 0 else
                 "TRIM"  if delta < 0 and target_qty > 0 else "CLOSE")

        logger.info("  [%-5s]  %-6s  held=%-4d  target=%-4d  delta=%+d  → %s %d shares",
                    label, symbol, current_qty, target_qty, delta, side.upper(), qty)

        if api is None:
            logger.info("  [DRY_RUN] %s %d × %s  (target %.2f%% | δ %+d)",
                        side.upper(), qty, symbol, aggregated_pct, delta)
            submitted.append({**rec, "status": f"DRY_{label}",
                              "order_id": f"DRY_{symbol}_{today}",
                              "delta_qty": delta, "final_qty": target_qty})
        else:
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="day",
                    client_order_id=f"alphacore_{symbol}_{today}"
                )
                logger.info("  [SUBMITTED] %s %d × %s → order_id=%s",
                            side.upper(), qty, symbol, order.id)
                submitted.append({**rec, "status": f"SUBMITTED_{label}",
                                  "order_id": order.id,
                                  "delta_qty": delta, "final_qty": target_qty})
            except Exception as e:
                logger.error("  [FAILED] %s %s: %s", symbol, side, e)
                submitted.append({**rec, "status": f"ERROR: {e}",
                                  "order_id": "N/A",
                                  "delta_qty": delta, "final_qty": current_qty})

    active  = [s for s in submitted if "HOLD" not in s["status"] and "FLAT" not in s["status"]]
    holding = [s for s in submitted if s["status"] == "HOLD"]
    logger.info("  Traded: %d ETFs | Holding: %d ETFs | Skipped: %d ETFs",
                len(active), len(holding), len(skipped))
    return submitted


def run_alpaca_pipeline():
    logger.info("=" * 65)
    logger.info("ALPACA EXECUTION GATE — M10")
    logger.info("NSE signal -> US ETF proxy -> delta rebalancing -> live paper")
    logger.info("=" * 65)

    api  = load_alpaca_client()
    mode = "LIVE PAPER" if api else "DRY_RUN"
    logger.info("Mode: %s", mode)

    account_summary   = fetch_account_summary(api)
    portfolio_value   = account_summary.get("portfolio_value", 100_000)
    pos_details       = account_summary.get("positions", [])
    current_positions = fetch_current_positions(api)

    order_book = build_order_book()
    if order_book.empty:
        logger.error("Empty order book — check upstream CSVs.")
        return None, []

    # Live ETF prices via yfinance download
    ETF_FALLBACK = {"XLF": 45, "QQQ": 480, "XLE": 90,
                    "XLV": 150, "XLY": 195, "XLP": 80, "SPY": 580}
    symbols = order_book["etf_proxy"].unique().tolist()
    try:
        import yfinance as yf
        raw = yf.download(symbols, period="2d", auto_adjust=True,
                          progress=False, group_by="ticker")
        etf_prices = {}
        for sym in symbols:
            try:
                series = raw[sym]["Close"] if len(symbols) > 1 else raw["Close"]
                etf_prices[sym] = float(series.dropna().iloc[-1])
            except Exception:
                etf_prices[sym] = ETF_FALLBACK.get(sym, 100)
        logger.info("  Live ETF prices: %s",
                    {k: f"${v:.2f}" for k, v in etf_prices.items()})
    except Exception as exc:
        logger.warning("  yfinance failed (%s) — using fallback prices", exc)
        etf_prices = {s: ETF_FALLBACK.get(s, 100) for s in symbols}

    # Log order book: held / target / delta preview
    logger.info("\n-- Order Book (current holdings: %s) --------------------------------",
                current_positions if current_positions else "FLAT")
    logger.info("  %-12s %-6s %-8s %-8s %-6s %-6s %-+6s %-12s",
                "NSE", "ETF", "Target%", "Mult", "Held", "Target", "Delta", "Action")
    logger.info("  " + "-" * 74)
    for _, r in order_book.iterrows():
        held = current_positions.get(r["etf_proxy"], 0)
        ep   = etf_prices.get(r["etf_proxy"], 100)
        tgt  = int(portfolio_value * abs(r["target_pct"]) / 100 / ep) if r["target_pct"] else 0
        logger.info("  %-12s %-6s %-8.2f %-8.3f %-6d %-6d %-+6d %-12s",
                    r["nse_ticker"], r["etf_proxy"], r["target_pct"],
                    r["multiplier"], held, tgt, tgt - held, r["action"])

    # Submit — delta-based rebalancing
    logger.info("\nSubmitting orders (%s)...", mode)
    submitted = submit_orders(
        order_book, api,
        current_positions=current_positions,
        pos_details=pos_details,
        portfolio_value=portfolio_value,
        etf_prices=etf_prices,
    )

    # Persist
    order_book.to_csv(DATA_DIR / "alpaca_order_book.csv", index=False)
    pd.DataFrame(submitted).to_csv(DATA_DIR / "alpaca_submitted.csv", index=False)
    pd.DataFrame([account_summary]).to_csv(DATA_DIR / "alpaca_account.csv", index=False)

    logger.info("\n-- SAVED -----------------------------------------------------------")
    logger.info("  alpaca_order_book.csv  — order book")
    logger.info("  alpaca_submitted.csv   — submission log")
    logger.info("  alpaca_account.csv     — live account snapshot")
    logger.info("=" * 65)

    return order_book, submitted


if __name__ == "__main__":
    result = run_alpaca_pipeline()
    if result[0] is not None:
        _, submitted = result
        print("\n-- Final Submission Summary --")
        df = pd.DataFrame(submitted)
        active = df[~df["status"].isin(["HOLD", "FLAT_HOLD"])]
        if active.empty:
            print("No trades today — all positions at target weight.")
        else:
            cols = [c for c in ["etf_proxy", "aggregated_pct", "delta_qty", 
                                "final_qty", "status"] if c in active.columns]
            print(active[cols].to_string(index=False))
