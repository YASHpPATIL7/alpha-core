"""
Alpha Core — Factor Alpha Engine for NSE Equities
==================================================

Modules:
  fama_french     → 5-factor decomposition + Jensen's Alpha
  cointegration   → Engle-Granger pairs scanner
  hmm_regime      → Bull/Bear/Sideways regime detection (coming)
  kelly_sizing    → Kelly Criterion position sizing (coming)
  finbert         → FinBERT sentiment analysis (coming)
  xgboost_signal  → XGBoost residual predictor (coming)
  shap_explainer  → SHAP TreeExplainer (coming)

Usage:
  from alpha_core.fama_french import run_factor_engine
  from alpha_core.cointegration import run_cointegration_scanner

  scores, factors, residuals = run_factor_engine()
  pairs = run_cointegration_scanner()
"""

from alpha_core.fama_french import run_factor_engine
from alpha_core.cointegration import run_cointegration_scanner
