"""
XGBoost Residual Predictor — M7
=================================

What this module does:
  Fama-French (M1) decomposes each stock's return into factor exposure + residual.
  The residual is the idiosyncratic return — what the LINEAR model can't explain.
  XGBoost finds NON-LINEAR patterns in those residuals to predict the next day's
  idiosyncratic return for each stock.

Why this matters (the core idea):
  Fama-French says: Return_t = α + β_MKT×MKT_t + β_SMB×SMB_t + ... + ε_t
  The ε_t (residual) is what's left. It's not random noise — it contains:
    - Short-term momentum in idiosyncratic returns (winners keep winning for 1-5 days)
    - Mean-reversion after large idiosyncratic moves (overreaction correction)
    - Regime-conditional patterns (residuals behave differently in Bull vs Bear)
    - Cross-factor nonlinearities (high SMB PLUS high RMW together = different effect)

  A LINEAR model (OLS) cannot capture any of these because they're all nonlinear.
  XGBoost CAN — it builds an ensemble of decision trees that carve the feature
  space into regions, finding "if lagged_resid_1d < -2% AND regime=Bear THEN
  tomorrow's residual tends to revert upward by 0.8%."

Why XGBoost specifically (not LSTM, not Random Forest)?
  - LSTM (RNN): needs longer sequences, more data, harder to interpret. For 1-step
    prediction with tabular features, XGBoost consistently outperforms LSTM.
  - Random Forest: no boosting → slower convergence, worse on tabular data.
  - XGBoost: gradient boosting on decision trees. Industry standard for tabular ML.
    Wins most Kaggle financial prediction competitions. Built-in feature importance.
    Fast. Handles missing values natively. Interpretable with SHAP (M8).

Design: one model per stock (not one cross-sectional model):
  Rationale: SUNPHARMA's idiosyncratic returns are driven by pharma-specific events
  (FDA approvals, drug launches). HDFCBANK's are driven by RBI policy and NPA cycles.
  A single cross-sectional model would average these out. 14 separate models each
  learn the stock's specific idiosyncratic patterns.

Target variable: next-day residual (regression, not classification):
  Predict the actual magnitude, not just direction. Kelly sizing can then use
  the predicted μ directly: f* = μ_predicted / σ²_residual.
  Better than classification because it feeds continuously into Kelly.

Features (engineered from residuals + factors + regime):
  MOMENTUM features: lagged residuals (1d, 2d, 3d, 5d, 10d)
    → Captures short-term idiosyncratic momentum/mean-reversion
  VOLATILITY features: rolling std of residuals (5d, 20d), squared residuals
    → Captures GARCH-like conditional volatility in idiosyncratic component
  POSITION features: rolling z-score, rolling percentile rank (20d)
    → Where is today's residual in the recent distribution?
  FACTOR CONTEXT: current day's MKT, SMB, HML, RMW, CMA returns
    → Context for "what kind of factor day is today?"
  REGIME: HMM regime label (0/1/2 = Bear/Sideways/Bull from M4)
    → Allows XGBoost to condition predictions on market state

Train/Val/Test split — TIME SERIES ONLY (NO SHUFFLE):
  Train: 2019-01-03 → VAL_START-1   (~85% of train window)
  Val:   VAL_START  → 2023-12-29    (~15% of train window — used for early stopping)
  Test:  2024-01-02 → 2024-12-30    (~250 days, fully out-of-sample — NEVER seen by model)
  Why no shuffle: shuffling would cause look-ahead bias — day 1000's features
  would train on day 1001's residuals. Time series MUST be split chronologically.

  Bug fix 2026-06-12 (A1): Previously early_stopping used eval_set=[(X_test, y_test)].
  This means the number of boosting rounds was selected by minimising loss on the test
  set — every reported test metric (IC, R², DirAcc) was optimistically biased because
  the test set participated in model selection. Fixed by carving a validation slice from
  the end of the training window. Expected: reported IC drops from biased to honest value.
  That drop is documented — it IS the audit story.
  n_estimators=300:   300 trees in the boosting ensemble
  max_depth=4:        Shallow trees. Deeper = more overfitting on noisy returns.
                      Quant rule of thumb: max_depth ≤ 4 for daily return prediction.
  learning_rate=0.05: Slow learner. Each tree contributes 5% of its weight.
                      Lower learning rate + more trees = better generalisation.
  subsample=0.8:      80% of training data per tree (prevents overfitting, adds diversity)
  colsample_bytree=0.8: 80% of features per tree (feature bagging)
  early_stopping_rounds=30: Stop if val loss doesn't improve for 30 rounds.
                             Prevents training too long.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import xgboost as xgb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_END  = "2023-12-29"   # last training day (inclusive)
VAL_START  = "2023-07-01"   # start of val slice (last ~15% of train window)
TEST_START = "2024-01-02"   # first test day — NEVER used during training

# ── Black Swan Events — Indian Market Context ──────────────────────────────
# These are structural market dislocations where normal idiosyncratic patterns
# break down. XGBoost needs to KNOW these happened — otherwise it tries to
# pattern-match extreme residual moves to normal features, which fails.
#
# For each event we add:
#   1. is_black_swan_window:  binary 1 during the event window
#   2. days_since_black_swan: integer, counts days since nearest event ended
#      → allows XGBoost to learn "residuals mean-revert faster 30 days post-crash"
#
# Why this matters for the model:
#   During COVID crash (Feb-Mar 2020), EVERY stock's residual went to -3σ to -5σ.
#   If XGBoost sees resid_lag1 = -4% and doesn't know "this is a black swan",
#   it uses the normal mean-reversion pattern — which says "big negative yesterday
#   → expect positive today." But in a crash, -4% yesterday often means -3% tomorrow.
#   The black_swan flag tells the model: "normal patterns don't apply here."
#
# Events documented:
#   IL&FS (2018-09):   Non-banking financial crisis, credit crunch across NBFC sector
#   COVID crash (2020-02 to 2020-03): Nifty -38% in 5 weeks
#   COVID recovery (2020-04 to 2020-08): V-shaped, fastest recovery in Indian history
#   RBI rate shock (2022-05): Emergency 40bp hike outside MPC cycle
#   Russia-Ukraine (2022-02): Crude spiked, FII outflows from India ₹14,000Cr
#   Adani-Hindenburg (2023-01): Short-seller report, Adani group lost $100Bn market cap
#   SVB collapse (2023-03): US banking contagion, global risk-off, Indian IT sector hit
BLACK_SWAN_EVENTS = [
    {"name": "ILFS_Crisis",       "start": "2018-09-21", "end": "2018-12-31"},
    {"name": "COVID_Crash",       "start": "2020-02-20", "end": "2020-03-23"},
    {"name": "COVID_Recovery",    "start": "2020-03-24", "end": "2020-08-31"},
    {"name": "RBI_Shock",         "start": "2022-05-04", "end": "2022-06-30"},
    {"name": "Russia_Ukraine",    "start": "2022-02-24", "end": "2022-04-15"},
    {"name": "Adani_Hindenburg",  "start": "2023-01-24", "end": "2023-02-28"},
    {"name": "SVB_Collapse",      "start": "2023-03-10", "end": "2023-03-31"},
]

def add_black_swan_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add black swan binary flags and days-since-event to the feature matrix.

    is_in_black_swan:
      1 if this date falls within any black swan event window, else 0.
      Used by XGBoost to switch off normal mean-reversion logic during crashes.

    days_since_last_black_swan:
      Number of trading days since the most recent black swan window ended.
      Captures the "aftershock" decay — correlations remain elevated for weeks
      after a major event. At day 0 (during event): this is 0.
      At day 30 after: value = 30. By day 90 it's back to "normal."

    Why lagged by 1?
      Same reason as regime — we're predicting t+1, so we use the flag at t.
      But the event window itself is known in advance (calendar event), so
      no look-ahead bias — we know COVID crash started Feb 20, 2020.
    """
    index = df.index

    is_swan = pd.Series(0, index=index)
    for ev in BLACK_SWAN_EVENTS:
        start = pd.Timestamp(ev["start"])
        end   = pd.Timestamp(ev["end"])
        is_swan[(index >= start) & (index <= end)] = 1

    # days_since: 0 during event, counts up after each event ends
    days_since = pd.Series(999, index=index)  # 999 = "no prior event"
    for ev in BLACK_SWAN_EVENTS:
        end = pd.Timestamp(ev["end"])
        mask = index > end
        day_counts = (index[mask] - end).days
        days_since[mask] = np.minimum(days_since[mask], day_counts)

    df["is_black_swan"]           = is_swan.shift(1).fillna(0)
    df["days_since_black_swan"]   = days_since.shift(1).fillna(999).clip(upper=365)
    return df

XGB_PARAMS = {
    "n_estimators":       300,
    "max_depth":          4,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,      # min samples per leaf — regularisation
    "reg_alpha":          0.1,    # L1 regularisation (feature selection)
    "reg_lambda":         1.0,    # L2 regularisation (weight shrinkage)
    "objective":          "reg:squarederror",
    "random_state":       42,
    "tree_method":        "hist", # faster on CPU for medium datasets
    "verbosity":          0,
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load all inputs
# ═══════════════════════════════════════════════════════════════
def load_data():
    residuals = pd.read_csv(DATA_DIR / "factor_residuals.csv",
                            index_col=0, parse_dates=True)
    factors   = pd.read_csv(DATA_DIR / "factor_returns.csv",
                            index_col=0, parse_dates=True)
    regime    = pd.read_csv(DATA_DIR / "regime_labels.csv",
                            index_col=0, parse_dates=True)

    logger.info("Residuals: %s | Factors: %s | Regime: %s",
                residuals.shape, factors.shape, regime.shape)
    return residuals, factors, regime


# ═══════════════════════════════════════════════════════════════
# STEP 2: Feature engineering for one stock
# ═══════════════════════════════════════════════════════════════
def build_features(ticker: str,
                   residuals: pd.DataFrame,
                   factors: pd.DataFrame,
                   regime: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix X and target y for one stock.

    CRITICAL: all features use only PAST information (t-1, t-2, ...).
    The target is tomorrow's residual (t+1).
    No feature looks at t or future — this is the look-ahead bias prevention.

    Feature groups:
      1. Lagged residuals       — direct past signal
      2. Rolling statistics     — vol, z-score, rank
      3. Factor returns (same day) — context, not prediction target
         Why same day? When we predict residual_t+1, we know factors_t already
         (factors_t = today's market return etc., available before close)
      4. Regime label           — from M4 HMM (same-day label is available)
    """
    r = residuals[ticker].copy()

    df = pd.DataFrame(index=r.index)

    # ── Momentum: lagged residuals ─────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        df[f"resid_lag{lag}"] = r.shift(lag)

    # ── Volatility: rolling std and squared residuals ──────────
    for window in [5, 20]:
        df[f"resid_vol{window}d"] = r.shift(1).rolling(window).std()

    df["resid_sq_lag1"] = (r.shift(1)) ** 2   # proxy for GARCH conditional vol

    # ── Position in distribution ───────────────────────────────
    # z-score: how far is yesterday's residual from its 20d mean
    roll_mean = r.shift(1).rolling(20).mean()
    roll_std  = r.shift(1).rolling(20).std()
    df["resid_zscore_20d"] = (r.shift(1) - roll_mean) / (roll_std + 1e-9)

    # percentile rank: where is yesterday's residual in the trailing 20-day window
    df["resid_rank_20d"] = r.shift(1).rolling(20).rank(pct=True)

    # ── Cumulative short-term return of idiosyncratic component ─
    # "Has this stock been running up on its own recently?"
    df["resid_cum5d"]  = r.shift(1).rolling(5).sum()
    df["resid_cum10d"] = r.shift(1).rolling(10).sum()

    # ── Factor returns (same day — contextual, NOT look-ahead) ──
    # We're predicting residual_t+1. We know factors_t (today's factor return)
    # because factor indices are available at market close.
    for col in factors.columns:
        df[f"factor_{col}"] = factors[col]

    # ── Regime from M4 HMM ─────────────────────────────────────
    # regime_int: 0=Bear, 1=Sideways, 2=Bull
    # We use yesterday's regime (regime_t) as a feature for predicting residual_t+1
    # Why not today's? Today's regime label depends on today's returns → look-ahead.
    regime_aligned = regime["regime_int"].reindex(df.index, method="ffill")
    df["regime"] = regime_aligned.shift(1)

    # ── Black Swan event flags ─────────────────────────────────
    # 7 documented Indian market crises 2018-2023 — see BLACK_SWAN_EVENTS above
    df = add_black_swan_features(df)

    # ── Target: NEXT day's residual ───────────────────────────
    df["target"] = r.shift(-1)   # residual_t+1

    # Drop rows where features are NaN, but KEEP the last row where target is NaN
    feature_cols = [c for c in df.columns if c != "target"]
    df = df.dropna(subset=feature_cols)

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3: Train one XGBoost model per stock
# ═══════════════════════════════════════════════════════════════
def train_one_stock(ticker: str, df: pd.DataFrame) -> dict:
    """
    Train XGBoost regressor on the feature matrix for one stock.
    Returns model + test metrics + feature importances.

    Time split:
      Train: everything up to TRAIN_END
      Test:  everything from TEST_START onwards (out-of-sample)

    Why we pass eval_set to XGBoost:
      early_stopping_rounds monitors the validation loss during training.
      If the val MSE hasn't improved for 30 consecutive rounds, training stops.
      This prevents overfitting without needing a separate held-out set inside train.

    Why min_child_weight=5:
      Minimum samples required in a leaf. On daily financial data (~1260 train days),
      you don't want leaves with 1-2 samples — that's pure noise memorisation.
      min_child_weight=5 forces the model to find patterns that hold for at least
      5 trading days, not just 1-2 coincidental observations.
    """
    feature_cols = [c for c in df.columns if c != "target"]

    # Drop target NaNs ONLY for the train/test sets. (The last row has target=NaN).
    train = df[df.index <= TRAIN_END].dropna(subset=["target"])
    test  = df[df.index >= TEST_START].dropna(subset=["target"])

    # ── Train/Val split: val carved from END of train window ──────────
    # Fix A1: early stopping must use a validation fold, NOT the test set.
    # Carve the last ~15% of train data (July–Dec 2023) as the val fold.
    # Test set (2024) remains 100% unseen during model selection.
    train_pure = train[train.index < VAL_START]
    val        = train[train.index >= VAL_START]

    if len(train_pure) < 150 or len(val) < 30:
        logger.warning("  %s — insufficient train/val split (train=%d, val=%d), skipping",
                       ticker, len(train_pure), len(val))
        return None

    X_train_pure = train_pure[feature_cols].values
    y_train_pure = train_pure["target"].values
    X_val        = val[feature_cols].values
    y_val        = val["target"].values
    X_test       = test[feature_cols].values
    y_test       = test["target"].values

    model = xgb.XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        X_train_pure, y_train_pure,
        eval_set=[(X_val, y_val)],   # ← val fold only; test set stays untouched
        verbose=False,
    )

    # Predictions
    y_pred_train = model.predict(X_train_pure)
    y_pred_test  = model.predict(X_test)

    # Metrics
    r2_train  = r2_score(y_train_pure, y_pred_train)
    r2_test   = r2_score(y_test,  y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Directional accuracy on test set
    # (does the predicted sign match the actual sign?)
    dir_acc = np.mean(np.sign(y_pred_test) == np.sign(y_test))

    # IC (Information Coefficient) = Spearman RANK correlation of prediction vs actual
    # Grinold & Kahn (2000) explicitly define IC as rank correlation — fat-tail robust.
    # Pearson IC is distorted by a single 10-sigma prediction. Spearman is not.
    # Fix applied 2026-05-27: was np.corrcoef (Pearson), now spearmanr.
    ic = spearmanr(y_pred_test, y_test).correlation

    # Feature importance
    importance = dict(zip(feature_cols,
                          model.feature_importances_.round(4).tolist()))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

    # Latest prediction (today's → predicts tomorrow's residual)
    latest_features = df[feature_cols].iloc[-1:].values
    next_day_pred = float(model.predict(latest_features)[0])

    return {
        "ticker":          ticker,
        "model":           model,
        "feature_cols":    feature_cols,
        "r2_train":        round(r2_train, 4),
        "r2_test":         round(r2_test, 4),
        "rmse_test":       round(rmse_test * 100, 4),   # in % per day
        "dir_acc_test":    round(dir_acc, 4),
        "ic_test":         round(ic, 4),
        "top_features":    top_features,
        "next_day_pred":   round(next_day_pred * 100, 4),  # in % per day
        "n_train":         len(train_pure),
        "n_val":           len(val),
        "n_test":          len(test),
        "best_iteration":  model.best_iteration,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4: Log results
# ═══════════════════════════════════════════════════════════════
def log_results(results: list):
    logger.info("\n── XGBoost Results by Stock ─────────────────────────────────────")
    logger.info("  %-12s  %-8s  %-8s  %-8s  %-8s  %-10s  %s",
                "Ticker", "R²-test", "RMSE%", "DirAcc", "IC", "Pred+1d%", "Top Feature")
    logger.info("  " + "─" * 80)

    for r in sorted(results, key=lambda x: -x["ic_test"]):
        ic_flag = "📈" if r["ic_test"] > 0.05 else ("🔴" if r["ic_test"] < 0 else "–")
        logger.info("  %-12s  %-8.4f  %-8.4f  %-8.3f  %-8.4f  %-10.4f  %s %s",
                    r["ticker"], r["r2_test"], r["rmse_test"],
                    r["dir_acc_test"], r["ic_test"],
                    r["next_day_pred"],
                    r["top_features"][0][0], ic_flag)

    # Summary stats
    ics = [r["ic_test"] for r in results]
    dir_accs = [r["dir_acc_test"] for r in results]
    logger.info("\n  Mean IC:   %.4f  (>0.05 = meaningful, >0.10 = strong)", np.mean(ics))
    logger.info("  Mean DirAcc: %.3f  (>0.52 = edge over coin flip)", np.mean(dir_accs))
    logger.info("  Stocks with IC > 0.05: %d / %d",
                sum(1 for ic in ics if ic > 0.05), len(results))


# ═══════════════════════════════════════════════════════════════
# STEP 5: Save predictions + signal for downstream modules
# ═══════════════════════════════════════════════════════════════
def save_outputs(results: list, residuals: pd.DataFrame,
                 factors: pd.DataFrame, regime: pd.DataFrame):
    """
    Save two outputs:

    1. xgb_predictions.csv — next-day predicted residual per stock (today's signal)
       Format: one row per stock with prediction, IC, dir_acc
       Used by: M8 (SHAP), M10 (Alpaca — adjust Kelly μ with XGB signal)

    2. xgb_model_summary.json — training stats per stock
       Used by: reporting, interviews ("show me how the model performed")
    """
    # Prediction table (latest signal — feeds into Kelly)
    pred_rows = []
    for r in results:
        pred_rows.append({
            "ticker":         r["ticker"],
            "predicted_resid_pct_next_day": r["next_day_pred"],
            "ic_test":        r["ic_test"],
            "dir_acc_test":   r["dir_acc_test"],
            "r2_test":        r["r2_test"],
            "signal":         "LONG_BIAS" if r["next_day_pred"] > 0.02
                              else ("SHORT_BIAS" if r["next_day_pred"] < -0.02
                              else "NEUTRAL"),
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(DATA_DIR / "xgb_predictions.csv", index=False)

    # Model summary JSON
    summary = []
    for r in results:
        summary.append({
            "ticker":         r["ticker"],
            "r2_train":       r["r2_train"],
            "r2_test":        r["r2_test"],
            "rmse_test_pct":  r["rmse_test"],
            "dir_acc_test":   r["dir_acc_test"],
            "ic_test":        r["ic_test"],
            "n_train":        r["n_train"],
            "n_val":          r["n_val"],
            "n_test":         r["n_test"],
            "best_iteration": r["best_iteration"],
            "top_5_features": [{"feature": f, "importance": v}
                               for f, v in r["top_features"]],
        })

    with open(DATA_DIR / "xgb_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n── SAVED ────────────────────────────────────────────────────────")
    logger.info("  xgb_predictions.csv      — %d stocks, today's signal", len(pred_rows))
    logger.info("  xgb_model_summary.json   — full training stats")

    return pred_df


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_xgb_pipeline() -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("XGBOOST RESIDUAL PREDICTOR — M7")
    logger.info("Train: 2019-2023 | Test: 2024 (out-of-sample)")
    logger.info("=" * 70)

    residuals, factors, regime = load_data()
    tickers = residuals.columns.tolist()

    results = []
    for ticker in tickers:
        logger.info("Training %s ...", ticker)
        df = build_features(ticker, residuals, factors, regime)
        result = train_one_stock(ticker, df)
        if result:
            results.append(result)

    log_results(results)
    pred_df = save_outputs(results, residuals, factors, regime)

    logger.info("=" * 70)
    return pred_df, results


if __name__ == "__main__":
    pred_df, results = run_xgb_pipeline()
    print("\n── XGBoost Predictions (next trading day) ──")
    print(pred_df[["ticker", "predicted_resid_pct_next_day",
                   "signal", "ic_test", "dir_acc_test"]].to_string(index=False))
