"""
SHAP Explainability — M8
==========================

SHAP (SHapley Additive exPlanations) answers the question:
  "Why did the XGBoost model predict this specific value for this stock today?"

Why SHAP matters for this project:
  M7 XGBoost is a black box. It says "ICICIBANK: SHORT_BIAS, predicted -0.087%/day."
  A PM or risk manager will ask: "WHY? What drove that prediction?"
  SHAP gives the exact answer: "Regime=Bear contributed -0.042%. resid_lag1=-1.2%
  contributed -0.031%. factor_SMB contributed +0.008%. Net = -0.087%."

  This is legally important too: SEBI's AI/ML guidelines require model explainability.
  "My model says short" is not enough. SHAP makes every prediction auditable.

How SHAP works (Shapley values from game theory):
  Imagine the prediction as a game where features are players.
  Each feature's Shapley value = its average marginal contribution across ALL
  possible orderings of features entering the model.

  For prediction = -0.087%:
    Base value (average prediction on training set) = 0.000%
    + regime contribution:     -0.042%
    + resid_lag1 contribution: -0.031%
    + factor_SMB contribution: +0.008%
    + resid_vol20d:            -0.022%
    ...
    = -0.087%  (sum of all SHAP values = prediction - base_value)

Why TreeExplainer (not KernelExplainer)?
  KernelExplainer is model-agnostic — works on anything, but is slow (O(2^n) features).
  TreeExplainer is specifically designed for tree models (XGBoost, LightGBM, RF).
  It exploits the tree structure for exact SHAP computation in O(TLD²) time.
  For 14 models × 250 test samples × 17 features: TreeExplainer takes ~0.5 seconds.
  KernelExplainer would take ~30 minutes.

Three outputs:
  1. GLOBAL: Beeswarm plot (what features matter most, how direction varies)
  2. LOCAL:  Waterfall plot (why THIS specific prediction for ICICIBANK today)
  3. SUMMARY: Bar chart of mean |SHAP| per feature across all stocks
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
from pathlib import Path

# Re-use M7's pipeline to get model objects
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from alpha_core.xgb_predictor import (
    load_data, build_features, train_one_stock,
    TRAIN_END, TEST_START
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

BASE_DIR = Path(__file__).parent.parent
FIG_DIR  = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"
FIG_DIR.mkdir(exist_ok=True)

# Stocks to focus SHAP plots on (IC > 0.05 from M7)
FOCUS_TICKERS = ["ICICIBANK", "HINDUNILVR", "ONGC"]


# ═══════════════════════════════════════════════════════════════
# STEP 1: Re-run M7 to get model objects (models weren't saved to disk)
# ═══════════════════════════════════════════════════════════════
def get_models():
    """
    Re-trains all 14 XGBoost models and returns model objects + feature data.
    Training takes ~4 seconds total (already fast from M7).

    Why not save/load models?
      For a project of this size, re-training on demand is cleaner than
      managing 14 pkl/joblib files. In production you'd use MLflow or
      model registry. For the demo: re-train is fine.
    """
    logger.info("Loading data and re-training XGBoost models for SHAP...")
    residuals, factors, regime = load_data()

    models = {}
    feature_data = {}

    for ticker in residuals.columns:
        df = build_features(ticker, residuals, factors, regime)
        result = train_one_stock(ticker, df)
        if result:
            feature_cols = result["feature_cols"]
            test_df = df[df.index >= TEST_START]
            models[ticker] = {
                "model":        result["model"],
                "feature_cols": feature_cols,
                "X_test":       test_df[feature_cols].values,
                "y_test":       test_df["target"].values,
                "test_index":   test_df.index,
                "X_latest":     df[feature_cols].iloc[-1:].values,
                "ic_test":      result["ic_test"],
            }

    logger.info("Models ready: %d stocks", len(models))
    return models


# ═══════════════════════════════════════════════════════════════
# STEP 2: Compute SHAP values for all stocks
# ═══════════════════════════════════════════════════════════════
def compute_shap_values(models: dict) -> dict:
    """
    Compute SHAP values for test set of each stock using TreeExplainer.

    shap_values[ticker] = array of shape (n_test, n_features)
      Each row = one test day
      Each column = SHAP value for that feature on that day
      Sum of row = prediction - base_value (intercept)

    Also compute SHAP for the LATEST observation (today's prediction).
    This is the "local explanation" — why did the model say what it said TODAY?
    """
    shap_results = {}

    for ticker, data in models.items():
        model = data["model"]
        X_test = data["X_test"]
        X_latest = data["X_latest"]

        explainer = shap.TreeExplainer(model)

        # Test set SHAP (for global importance plots)
        sv_test = explainer.shap_values(X_test)        # (n_test, n_features)
        base_val = explainer.expected_value             # scalar

        # Latest SHAP (for local waterfall)
        sv_latest = explainer.shap_values(X_latest)    # (1, n_features)

        shap_results[ticker] = {
            "shap_values":        sv_test,
            "shap_latest":        sv_latest[0],
            "base_value":         float(base_val),
            "feature_cols":       data["feature_cols"],
            "X_test":             X_test,
            "test_index":         data["test_index"],
            "ic_test":            data["ic_test"],
        }
        logger.info("  %s SHAP done. Base value: %.5f%%", ticker, base_val * 100)

    return shap_results


# ═══════════════════════════════════════════════════════════════
# STEP 3: Global beeswarm plot for focus tickers
# ═══════════════════════════════════════════════════════════════
def plot_beeswarm(shap_results: dict, ticker: str):
    """
    Beeswarm plot = the SHAP "signature" of a model.

    X axis: SHAP value (negative = pushes prediction down, positive = pushes up)
    Y axis: features ranked by mean |SHAP| (most important at top)
    Colour: feature value (red=high, blue=low) — shows direction of effect

    How to read it:
      If regime is at the top with red dots on the right:
        → High regime value (Bull=2) pushes prediction positive
        → Makes intuitive sense: Bull regime → positive idiosyncratic returns expected

      If resid_lag1 has blue dots on the right and red on the left:
        → Low lag1 residual (stock fell yesterday) → predicts positive tomorrow
        → Mean-reversion signal
    """
    sr = shap_results[ticker]
    sv = sr["shap_values"]       # (n_test, n_features)
    X  = sr["X_test"]
    feature_cols = sr["feature_cols"]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:12]   # top 12 features

    shap.summary_plot(
        sv[:, order],
        X[:, order],
        feature_names=[feature_cols[i] for i in order],
        plot_type="dot",
        show=False,
        color_bar=True,
    )

    plt.title(f"SHAP Beeswarm — {ticker} (IC={sr['ic_test']:.4f})\n"
              f"Each dot = one test day. X = SHAP value. Colour = feature value.",
              color="white", fontsize=11)
    plt.gca().tick_params(colors="#aaa")
    plt.gca().set_facecolor("#1a1d27")
    fig.patch.set_facecolor("#0f1117")

    out = FIG_DIR / f"m8_shap_beeswarm_{ticker.lower()}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    logger.info("  Saved: %s", out)


# ═══════════════════════════════════════════════════════════════
# STEP 4: Local waterfall — explain today's prediction
# ═══════════════════════════════════════════════════════════════
def plot_waterfall(shap_results: dict, models: dict, ticker: str):
    """
    Waterfall plot = explains ONE specific prediction (today's).

    Starts from the base value (average model output on training set).
    Each feature adds or subtracts its SHAP contribution.
    Final bar lands on the actual prediction.

    This is the "SEBI audit" plot — if asked "why did you take this position?",
    you pull up this chart and say:
      "Base expectation was +0.000%. Regime=Sideways pushed it down by -0.042%.
       resid_lag1 of -0.8% (fell yesterday) pushed it down further by -0.031%.
       factor_HML contributed +0.012%. Net prediction: -0.087% (SHORT_BIAS)."
    """
    sr = shap_results[ticker]
    sv_latest = sr["shap_latest"]       # (n_features,)
    base_val  = sr["base_value"]
    feature_cols = sr["feature_cols"]
    X_latest = models[ticker]["X_latest"][0]

    # Sort by absolute SHAP descending
    abs_shap = np.abs(sv_latest)
    order = np.argsort(abs_shap)[::-1][:10]   # top 10

    features_sorted = [feature_cols[i] for i in order]
    shap_sorted     = sv_latest[order]
    vals_sorted     = X_latest[order]
    prediction      = base_val + sv_latest.sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Running total
    running = base_val
    bar_lefts = []
    bar_widths = []
    bar_colors = []

    for sv in shap_sorted:
        bar_lefts.append(min(running, running + sv))
        bar_widths.append(abs(sv))
        bar_colors.append("#22c55e" if sv > 0 else "#ef4444")
        running += sv

    y_pos = np.arange(len(features_sorted))
    bars = ax.barh(y_pos, bar_widths, left=bar_lefts,
                   color=bar_colors, height=0.6, alpha=0.85)

    # Feature labels with values
    labels = [f"{f}  [{v:.4f}]" for f, v in zip(features_sorted, vals_sorted)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=9)

    # Base value line
    ax.axvline(base_val, color="#94a3b8", linewidth=1, linestyle="--",
               label=f"Base: {base_val*100:.4f}%")

    # Prediction line
    ax.axvline(prediction, color="#f59e0b", linewidth=1.5, linestyle="-",
               label=f"Prediction: {prediction*100:.4f}%")

    ax.set_xlabel("SHAP value (contribution to prediction)", color="#aaa", fontsize=9)
    ax.set_title(
        f"SHAP Waterfall — {ticker} (Today's Prediction)\n"
        f"Base={base_val*100:.4f}%  →  Predicted={prediction*100:.4f}%  "
        f"({'SHORT_BIAS' if prediction < -0.0002 else 'LONG_BIAS' if prediction > 0.0002 else 'NEUTRAL'})",
        color="white", fontsize=11
    )
    ax.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white", edgecolor="#333")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    out = FIG_DIR / f"m8_shap_waterfall_{ticker.lower()}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    logger.info("  Saved: %s", out)


# ═══════════════════════════════════════════════════════════════
# STEP 5: Cross-stock mean |SHAP| bar chart
# ═══════════════════════════════════════════════════════════════
def plot_cross_stock_importance(shap_results: dict):
    """
    Aggregates mean |SHAP| across all 14 stocks to show which features
    matter most globally — across the entire universe.

    This is the "portfolio-level" explainability:
    "Across all your positions, what drove predictions most?"
    Answer: regime from M4, then short-term momentum, then volatility.
    """
    all_mean_shap = {}

    for ticker, sr in shap_results.items():
        sv = sr["shap_values"]
        feature_cols = sr["feature_cols"]
        mean_abs = np.abs(sv).mean(axis=0)
        for feat, val in zip(feature_cols, mean_abs):
            all_mean_shap[feat] = all_mean_shap.get(feat, [])
            all_mean_shap[feat].append(val)

    # Average across stocks
    avg_shap = {f: np.mean(v) for f, v in all_mean_shap.items()}
    sorted_feats = sorted(avg_shap.items(), key=lambda x: -x[1])[:12]

    feats, vals = zip(*sorted_feats)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    colors = []
    for f in feats:
        if "regime" in f:       colors.append("#a78bfa")   # purple — M4
        elif "resid_lag" in f:  colors.append("#60a5fa")   # blue — momentum
        elif "resid_vol" in f or "resid_sq" in f: colors.append("#f97316")  # orange — vol
        elif "factor_" in f:    colors.append("#22c55e")   # green — FF factors
        else:                   colors.append("#94a3b8")   # grey — other

    bars = ax.barh(range(len(feats)), vals, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, color="white", fontsize=9)
    ax.set_xlabel("Mean |SHAP| across all 14 stocks", color="#aaa", fontsize=9)
    ax.set_title("Global Feature Importance — Cross-Stock SHAP (M8)\n"
                 "Purple=Regime(M4)  Blue=Momentum  Orange=Volatility  Green=FF Factors",
                 color="white", fontsize=11)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    out = FIG_DIR / "m8_shap_global_importance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    logger.info("  Saved: %s", out)
    return sorted_feats


# ═══════════════════════════════════════════════════════════════
# STEP 6: Save SHAP summary CSV
# ═══════════════════════════════════════════════════════════════
def save_shap_summary(shap_results: dict):
    rows = []
    for ticker, sr in shap_results.items():
        sv = sr["shap_values"]
        feature_cols = sr["feature_cols"]
        mean_abs = np.abs(sv).mean(axis=0)
        for feat, val in zip(feature_cols, mean_abs):
            rows.append({
                "ticker":          ticker,
                "feature":         feat,
                "mean_abs_shap":   round(float(val) * 100, 6),   # in %
                "shap_today":      round(float(sr["shap_latest"][feature_cols.index(feat)]) * 100, 6),
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "shap_summary.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_shap_pipeline():
    logger.info("=" * 70)
    logger.info("SHAP EXPLAINABILITY — M8")
    logger.info("=" * 70)

    # Get models (re-trains M7 — ~4 seconds)
    models = get_models()

    # Compute SHAP values
    logger.info("\nComputing SHAP values (TreeExplainer)...")
    shap_results = compute_shap_values(models)

    # Global cross-stock importance
    logger.info("\n[Global] Cross-stock feature importance...")
    top_feats = plot_cross_stock_importance(shap_results)
    logger.info("\nTop features across universe:")
    for feat, val in top_feats[:5]:
        logger.info("  %-22s  mean|SHAP|=%.5f%%", feat, val * 100)

    # Per-stock beeswarm + waterfall for focus tickers
    for ticker in FOCUS_TICKERS:
        if ticker not in shap_results:
            continue
        ic = shap_results[ticker]["ic_test"]
        logger.info("\n[%s] IC=%.4f — plotting beeswarm + waterfall", ticker, ic)
        plot_beeswarm(shap_results, ticker)
        plot_waterfall(shap_results, models, ticker)

    # Save summary CSV
    shap_df = save_shap_summary(shap_results)
    logger.info("\n── SAVED ────────────────────────────────────────────────────────")
    logger.info("  shap_summary.csv              — %d rows", len(shap_df))
    logger.info("  m8_shap_global_importance.png")
    logger.info("  m8_shap_beeswarm_*.png        — per-stock (3 focus tickers)")
    logger.info("  m8_shap_waterfall_*.png       — today's prediction explained")
    logger.info("=" * 70)

    return shap_results, shap_df


if __name__ == "__main__":
    shap_results, shap_df = run_shap_pipeline()

    print("\n── Top SHAP Features (ICICIBANK — highest IC) ──")
    ici = shap_df[shap_df["ticker"] == "ICICIBANK"].sort_values(
        "mean_abs_shap", ascending=False).head(8)
    print(ici[["feature", "mean_abs_shap", "shap_today"]].to_string(index=False))
