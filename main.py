"""
main.py — Alpha-Core Unified Pipeline
=======================================

Single entry point for the entire Factor Alpha Engine.
Run: python main.py

Execution order (data flows top to bottom):
  M1: Fama-French 5-Factor Decomposition
  M3: Johansen Cointegration + Signal Generation
  M4: HMM Regime Detection
  M5: Kelly Position Sizing (regime-gated, alpha-weighted)
  M6: FinBERT Sentiment Gating (live Yahoo Finance news)
  M7: XGBoost Residual Prediction (with black swan features)
  M8: SHAP Explainability
  M9: Drawdown Analytics
 M10: Alpaca Execution Gate (DRY_RUN or live paper)

Each module saves CSV outputs to data/ which the next module reads.
No module has an import dependency on another — only file dependencies.
This is intentional: modules can be re-run independently.
"""

import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAIN] %(levelname)s — %(message)s",
)
logger = logging.getLogger("main")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def section(title: str):
    logger.info("")
    logger.info("█" * 65)
    logger.info("  %s", title)
    logger.info("█" * 65)


def run_pipeline(skip_finbert: bool = False):
    t_start = time.time()
    results = {}

    # ── M1 ───────────────────────────────────────────────────────
    section("M1 — Fama-French 5-Factor Decomposition")
    try:
        from alpha_core.fama_french import run_factor_engine
        scores, residuals, factor_returns = run_factor_engine()
        results["M1"] = "OK"
        logger.info("M1 complete. Residuals: %s | last date: %s",
                    residuals.shape, residuals.index[-1].date())
    except Exception as e:
        logger.error("M1 FAILED: %s", e)
        results["M1"] = f"FAIL: {e}"

    # ── M3 ───────────────────────────────────────────────────────
    section("M3 — Johansen Cointegration + Signal Generation")
    try:
        from alpha_core.cointegration import run_cointegration_scanner
        coint_df = run_cointegration_scanner()
        results["M3"] = "OK"
        logger.info("M3 complete. %d pairs scanned.", len(coint_df))
    except Exception as e:
        logger.error("M3 FAILED: %s", e)
        results["M3"] = f"FAIL: {e}"

    # ── M4 ───────────────────────────────────────────────────────
    section("M4 — HMM Regime Detection")
    try:
        from alpha_core.hmm_regime import run_hmm_pipeline, detect_current_regime
        regime_df = run_hmm_pipeline()
        current_regime = detect_current_regime()
        results["M4"] = "OK"
        logger.info("M4 complete. Current regime: %s", current_regime)
    except Exception as e:
        logger.error("M4 FAILED: %s", e)
        results["M4"] = f"FAIL: {e}"
        current_regime = "Sideways"

    # ── M5 ───────────────────────────────────────────────────────
    section("M5 — Kelly Position Sizing")
    try:
        from alpha_core.kelly_sizing import run_kelly_pipeline
        kelly_pairs, kelly_factor = run_kelly_pipeline()
        results["M5"] = "OK"
        logger.info("M5 complete.")
    except Exception as e:
        logger.error("M5 FAILED: %s", e)
        results["M5"] = f"FAIL: {e}"

    # ── M6 ───────────────────────────────────────────────────────
    section("M6 — FinBERT Sentiment Gating")
    if skip_finbert:
        # Bug 21 root-cause fix (2026-06-12): skipping FinBERT must NOT skip
        # the gate. apply_gate_only() re-applies the last saved sentiment to
        # TODAY's Kelly output (neutralised if stale), so the gated file M10
        # reads is always fresh.
        logger.warning("M6 model SKIPPED (--skip-finbert). Applying gate from "
                       "last saved sentiment to today's Kelly output.")
        try:
            from alpha_core.finbert_sentiment import apply_gate_only
            apply_gate_only()
            results["M6"] = "GATE-ONLY (sentiment reused)"
        except Exception as e:
            logger.error("M6 gate-only FAILED: %s", e)
            results["M6"] = f"FAIL: {e}"
    else:
        try:
            from alpha_core.finbert_sentiment import run_finbert_pipeline
            sentiment_df = run_finbert_pipeline()
            results["M6"] = "OK"
            logger.info("M6 complete. %d tickers scored.", len(sentiment_df))
        except Exception as e:
            logger.error("M6 FAILED: %s", e)
            results["M6"] = f"FAIL: {e}"

    # ── M7 ───────────────────────────────────────────────────────
    section("M7 — XGBoost Residual Predictor (+ Black Swan Features)")
    try:
        from alpha_core.xgb_predictor import run_xgb_pipeline
        xgb_df, xgb_results = run_xgb_pipeline()
        results["M7"] = "OK"
        n_good = sum(1 for r in xgb_results if r["ic_test"] > 0.05)
        logger.info("M7 complete. IC>0.05 stocks: %d/14", n_good)
    except Exception as e:
        logger.error("M7 FAILED: %s", e)
        results["M7"] = f"FAIL: {e}"

    # ── M8 ───────────────────────────────────────────────────────
    section("M8 — SHAP Explainability")
    try:
        from alpha_core.shap_explainer import run_shap_pipeline
        shap_results, shap_df = run_shap_pipeline()
        results["M8"] = "OK"
        logger.info("M8 complete. %d SHAP rows.", len(shap_df))
    except Exception as e:
        logger.error("M8 FAILED: %s", e)
        results["M8"] = f"FAIL: {e}"

    # ── M9 ───────────────────────────────────────────────────────
    section("M9 — Drawdown Analytics")
    try:
        from alpha_core.analytics import (build_sentiment_return_data,
                                          plot_sentiment_scatter,
                                          plot_drawdown_analytics)
        import pandas as pd
        scatter_df = build_sentiment_return_data()
        plot_sentiment_scatter(scatter_df)
        vajra_path = BASE_DIR.parent / "indian-risk-engine" / "data" / "vajra_returns.csv"
        if vajra_path.exists():
            price_rets = pd.read_csv(vajra_path, index_col=0, parse_dates=True)
            for t in ["SUNPHARMA", "ICICIBANK"]:
                if t in price_rets.columns:
                    plot_drawdown_analytics(price_rets[t].dropna(), t)
        results["M9"] = "OK"
        logger.info("M9 complete.")
    except Exception as e:
        logger.error("M9 FAILED: %s", e)
        results["M9"] = f"FAIL: {e}"

    # ── M10 ──────────────────────────────────────────────────────
    section("M10 — Alpaca Execution Gate")
    try:
        from alpha_core.alpaca_gate import run_alpaca_pipeline
        order_book, submitted = run_alpaca_pipeline()
        results["M10"] = "OK"
        active = sum(1 for s in submitted if s.get("qty", 0) != 0)
        logger.info("M10 complete. %d active orders.", active)
    except Exception as e:
        logger.error("M10 FAILED: %s", e)
        results["M10"] = f"FAIL: {e}"

    # ── SUMMARY ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    section(f"PIPELINE COMPLETE — {elapsed:.1f}s")

    for module, status in results.items():
        icon = "OK" if status == "OK" else ("--" if "SKIP" in status else "!!")
        logger.info("    [%s] %s: %s", icon, module, status)

    # Final portfolio print
    try:
        import pandas as pd
        print("\n" + "═" * 65)
        print("  ALPHA-CORE — FINAL PORTFOLIO")
        print("═" * 65)

        reg = pd.read_csv(BASE_DIR / "data" / "regime_labels.csv",
                          index_col=0, parse_dates=True)
        print(f"\n  Regime:   {reg['regime_name'].iloc[-1]}  ({reg.index[-1].date()})")

        try:
            fg = pd.read_csv(BASE_DIR / "data" / "kelly_positions_factor_gated.csv")
            active = fg[fg["gated_pct"] > 0]
            print("\n  Factor Positions:")
            if active.empty:
                print("    All FLAT")
            for _, r in active.iterrows():
                print(f"    {r['ticker']:12} LONG {r['gated_pct']:.2f}%  "
                      f"(sentiment: {r.get('sentiment','n/a')})")
        except Exception:
            pass

        try:
            xp = pd.read_csv(BASE_DIR / "data" / "xgb_predictions.csv")
            hi = xp[(xp["signal"] != "NEUTRAL") & (xp["ic_test"] > 0.05)]
            print("\n  XGBoost High-IC Signals:")
            if hi.empty:
                print("    None above IC threshold")
            for _, r in hi.iterrows():
                print(f"    {r['ticker']:12} {r['signal']:12} "
                      f"pred={r['predicted_resid_pct_next_day']:+.4f}%  IC={r['ic_test']:.4f}")
        except Exception:
            pass

        print("\n" + "═" * 65)
    except Exception as e:
        logger.warning("Portfolio summary failed: %s", e)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpha-Core Factor Engine")
    parser.add_argument("--skip-finbert", action="store_true",
                        help="Skip FinBERT live fetch (use last saved CSV)")
    args = parser.parse_args()
    run_pipeline(skip_finbert=args.skip_finbert)
