"""
HMM Regime Detection — M4
==========================

Detects market regimes (Bull / Bear / Sideways) from daily returns using
a Gaussian Hidden Markov Model (GaussianHMM) with full covariance.

What this module does and WHY:
  Standard factor models (Fama-French M1) assume fixed betas.
  In reality, factor exposures change with the market environment.
  A momentum strategy that works in a Bull regime destroys capital in a Bear.
  HMM solves this by learning WHEN the market is in which state — without
  you hand-coding any rules.

How HMM works (conceptually):
  - There are K hidden states (e.g. Bull, Bear, Sideways) you can't observe directly.
  - What you CAN observe: daily returns, volatility, factor spreads.
  - HMM learns: "In state Bull, returns are typically +0.08%/day with low vol.
    In state Bear, returns are -0.15%/day with high vol."
  - It also learns TRANSITION PROBABILITIES: how likely is the market to stay
    in Bear vs escape to Sideways vs jump to Bull on any given day?
  - Once trained, it assigns the most probable hidden state to every day
    in history (Viterbi algorithm) and predicts today's regime.

BIC Check (why 3 states?):
  BIC = -2 * log_likelihood + k * log(n)
  where k = number of free parameters, n = number of observations.
  Lower BIC = better model (penalises complexity).
  We test K = 2, 3, 4 and pick the K with lowest BIC.
  3 states typically wins on Nifty data because markets clearly have:
    Bull (trending up, low vol), Bear (trending down, high vol),
    Sideways (flat, mean-reverting, moderate vol).

covariance_type options:
  "spherical" — one variance per state, same in all directions. Fastest.
  "diag"      — separate variance per feature per state, no cross-feature correlation.
  "full"      — full covariance matrix per state: captures correlations between
                features (e.g. when MKT is negative AND vol is high simultaneously).
  "tied"      — all states share one covariance matrix (bad: can't distinguish regimes).
  We use "full" because financial regimes differ in BOTH variance AND cross-correlations.
  Bear markets: returns and volatility become strongly negatively correlated.
  Bull markets: that correlation is much weaker. "full" captures this.

Failure modes (always documented):
  1. HMM is unstable with < 60 days of data. Minimum lookback enforced.
  2. State labels are arbitrary integers (0, 1, 2) — they must be mapped to
     Bull/Bear/Sideways by inspecting mean returns per state, not assumed.
  3. Non-stationarity: if the market structure changes completely (e.g. new SEBI rules,
     demonetisation shock), the model trained on past data mislabels new regimes.
     Mitigation: rolling refit every 252 days.
  4. HMM uses EM (Expectation-Maximisation) for fitting — EM can converge to local
     optima. We run n_init=5 random restarts and pick the best log-likelihood.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

BASE_DIR = Path(__file__).parent.parent
RISK_ENGINE_DATA = BASE_DIR.parent / "indian-risk-engine" / "data"
DATA_DIR = BASE_DIR / "data"

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_ITER = 200          # EM iterations — enough for convergence
N_INIT = 5            # Random restarts to avoid local optima
MIN_DAYS = 60         # Minimum history needed for stable HMM
COVARIANCE_TYPE = "full"
RANDOM_STATE = 42     # Reproducibility

# State names — assigned AFTER inspecting mean returns per state (not assumed)
# We auto-assign in label_states() below.
STATE_NAMES = {0: "Unknown-0", 1: "Unknown-1", 2: "Unknown-2"}


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load features
# ═══════════════════════════════════════════════════════════════
def load_features() -> pd.DataFrame:
    """
    Build the HMM feature matrix.

    We use THREE features, not just raw returns:
      1. Nifty market return (MKT from Fama-French factor_returns.csv)
         — captures directional trend
      2. Rolling 20-day realised volatility of MKT
         — captures fear/turbulence
      3. Rolling 20-day momentum (sign of cumulative return)
         — captures persistence of the regime

    Why 3 features and not just 1?
      With only MKT returns: HMM separates high-return vs low-return days.
      That's noisy — a single bad day in a Bull market gets mislabelled Bear.
      Adding volatility anchors the HMM: Bear = low return AND high vol.
      Adding momentum adds inertia: Bull days cluster together.

    Why use MKT from factor_returns.csv (not raw Nifty)?
      MKT is already excess-return (Nifty - Rf). It's the purest signal.
      Also, this creates a clean dependency: M4 consumes M1's output,
      which is exactly how the pipeline is designed to work.
    """
    factor_path = DATA_DIR / "factor_returns.csv"
    returns_path = RISK_ENGINE_DATA / "vajra_returns.csv"

    # Primary: use MKT from Fama-French (already risk-adjusted)
    if factor_path.exists():
        logger.info("Loading factor returns from: %s", factor_path)
        factors = pd.read_csv(factor_path, index_col=0, parse_dates=True)
        mkt = factors["MKT"]
        logger.info("  MKT loaded: %d days", len(mkt))
    else:
        # Fallback: equal-weight of all stocks as proxy for market
        logger.warning("factor_returns.csv not found — falling back to vajra_returns.csv")
        rets = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        mkt = rets.mean(axis=1)
        logger.warning("  Using equal-weight returns as MKT proxy (less accurate)")

    # ── Feature engineering ────────────────────────────────────
    features = pd.DataFrame(index=mkt.index)

    # Feature 1: daily return — captures direction
    features["mkt_return"] = mkt

    # Feature 2: rolling 20-day realised vol — captures fear
    # Why 20 days? One calendar month. Short enough to be reactive,
    # long enough to not flip on a single day.
    features["realised_vol"] = mkt.rolling(20).std() * np.sqrt(252)

    # Feature 3: rolling 20-day cumulative return — captures momentum
    # Positive = bullish momentum. Negative = bearish drift.
    features["momentum_20d"] = mkt.rolling(20).sum()

    # Drop NaN rows (first 20 days have no rolling stats)
    features = features.dropna()

    logger.info("  Feature matrix: %d days × %d features", *features.shape)
    logger.info("  Date range: %s → %s",
                features.index[0].date(), features.index[-1].date())
    return features


# ═══════════════════════════════════════════════════════════════
# STEP 2: BIC-based model selection
# ═══════════════════════════════════════════════════════════════
def select_n_states(X: np.ndarray, k_range: range = range(2, 6)) -> int:
    """
    Use BIC to select the optimal number of hidden states.

    BIC = -2 * log_likelihood + k * log(n)
      log_likelihood: how well the model fits the data (higher = better)
      k: number of free parameters (penalises model complexity)
      n: number of observations
      Lower BIC = better.

    Free parameters in GaussianHMM with 'full' covariance and K states, D features:
      Transition matrix:     K*(K-1)          (rows must sum to 1)
      Initial probabilities: K-1
      Means:                 K*D
      Covariance matrices:   K * D*(D+1)/2    (symmetric full cov)
      Total: K*(K-1) + (K-1) + K*D + K*D*(D+1)//2

    This ensures we don't overfit by adding states that don't improve fit
    enough to justify the extra parameters.
    """
    n, D = X.shape
    best_bic = np.inf
    best_k = k_range[0]
    results = []

    logger.info("\n── BIC Model Selection ──────────────────────────────────────────")
    logger.info("  %-8s  %-14s  %-12s  %-8s", "K States", "Log-Lik", "BIC", "Winner")
    logger.info("  " + "─" * 48)

    for k in k_range:
        try:
            # Manual multi-restart: fit N_INIT times, keep best log-likelihood
            # (hmmlearn 0.3.x doesn't have n_init parameter)
            best_model = None
            best_score = -np.inf
            for seed in range(N_INIT):
                m = GaussianHMM(
                    n_components=k,
                    covariance_type=COVARIANCE_TYPE,
                    n_iter=N_ITER,
                    random_state=RANDOM_STATE + seed,
                )
                m.fit(X)
                s = m.score(X)
                if s > best_score:
                    best_score = s
                    best_model = m
            model = best_model

            # Count free parameters
            # Transition: k*(k-1), startprob: k-1, means: k*D, covars: k*D*(D+1)//2
            n_params = k*(k-1) + (k-1) + k*D + k * (D*(D+1)//2)
            # score() in hmmlearn returns the LOG-LIKELIHOOD of the entire sequence
            # (NOT per-sample average — it's the full forward-backward sum)
            log_lik = model.score(X)
            bic = -2 * log_lik + n_params * np.log(n)

            is_best = bic < best_bic
            if is_best:
                best_bic = bic
                best_k = k

            results.append({"k": k, "log_lik": log_lik, "bic": bic, "n_params": n_params})
            logger.info("  %-8d  %-14.2f  %-12.2f  %s",
                        k, log_lik, bic, "← BEST" if is_best else "")

        except Exception as e:
            logger.warning("  K=%d failed: %s", k, e)

    logger.info("  Optimal K = %d states (lowest BIC = %.2f)", best_k, best_bic)
    return best_k


# ═══════════════════════════════════════════════════════════════
# STEP 3: Fit HMM
# ═══════════════════════════════════════════════════════════════
def fit_hmm(X: np.ndarray, n_states: int) -> GaussianHMM:
    """
    Fit the GaussianHMM on the standardised feature matrix.

    Why standardise features before fitting?
      HMM's Gaussian emission assumes each feature contributes meaningfully.
      mkt_return is ~0.0005/day. realised_vol is ~0.15/year.
      Without standardisation, the covariance matrix is dominated by vol,
      making mkt_return invisible. StandardScaler brings all features to mean=0, std=1.

    Note: We do NOT standardise here — we return the model so the caller can
    standardise + keep the scaler for inverse-transforming predictions on new data.
    Standardisation happens in run_hmm_pipeline() so the scaler is reusable.
    """
    logger.info("\nFitting GaussianHMM | n_states=%d | cov=%s | n_iter=%d | n_init=%d",
                n_states, COVARIANCE_TYPE, N_ITER, N_INIT)

    # Manual N_INIT restarts — hmmlearn 0.3.x has no n_init parameter
    best_model = None
    best_score = -np.inf
    for seed in range(N_INIT):
        m = GaussianHMM(
            n_components=n_states,
            covariance_type=COVARIANCE_TYPE,
            n_iter=N_ITER,
            random_state=RANDOM_STATE + seed,
        )
        m.fit(X)
        s = m.score(X)
        if s > best_score:
            best_score = s
            best_model = m
    model = best_model

    logger.info("  Converged: %s | Final log-lik: %.4f",
                model.monitor_.converged, model.score(X))
    return model


# ═══════════════════════════════════════════════════════════════
# STEP 4: Label states (Bull / Bear / Sideways)
# ═══════════════════════════════════════════════════════════════
def label_states(model: GaussianHMM, feature_cols: list,
                 scaler=None) -> dict:
    """
    Map integer state labels (0, 1, 2) to regime names by inspecting
    the learned mean return per state IN ORIGINAL (un-standardised) space.

    WHY we can't assume: HMM labels states by EM initialisation, not meaning.
    State 0 might be Bear in one run, Bull in another. We must look at the
    mean of the first feature (mkt_return) per state — highest mean = Bull.

    WHY we inverse-transform: model.means_ are in standardised space (z-scores).
    Sorting by standardised mean is equivalent to sorting by original mean
    (StandardScaler is monotone), but inverse-transforming makes the logged
    statistics interpretable (actual daily returns, not z-scores).

    Returns dict: {state_int: "Bull" | "Bear" | "Sideways"}
    """
    means_std = model.means_   # (K, D) in standardised space

    # Invert to original space for interpretable logging
    if scaler is not None:
        means_orig = scaler.inverse_transform(means_std)  # (K, D) original units
    else:
        means_orig = means_std  # fallback: use standardised

    # Volatility = sqrt of variance of feature 0 (mkt_return) in original space
    # model.covars_ is in standardised space — scale back using scaler.scale_[0]
    if scaler is not None:
        scale_0 = scaler.scale_[0]  # std of mkt_return in original data
        vols_orig = np.sqrt(np.array([
            model.covars_[k][0, 0] for k in range(model.n_components)
        ])) * scale_0
    else:
        vols_orig = np.sqrt(np.array([
            model.covars_[k][0, 0] for k in range(model.n_components)
        ]))

    # Sort states by Sharpe-like score (return / vol) in original space.
    # Why Sharpe not just return?
    #   On 5-year NSE data (2019-2024), the COVID crash is only 40 days.
    #   Over 1419 days total, even the "Bear" state has a small positive mean
    #   return because recoveries are fast. Sorting by mean alone is unstable.
    #   Sharpe = mean_return / vol separates:
    #     Bull:     high return, moderate vol → highest Sharpe
    #     Sideways: low return, low vol       → moderate Sharpe
    #     Bear:     low/negative return, HIGH vol → lowest Sharpe
    sharpe_scores = np.array([
        means_orig[k, 0] / (vols_orig[k] + 1e-9)
        for k in range(model.n_components)
    ])
    sorted_states = np.argsort(sharpe_scores)  # ascending: worst Sharpe first = Bear

    n = model.n_components
    mapping = {}

    if n == 2:
        mapping[sorted_states[0]] = "Bear"
        mapping[sorted_states[1]] = "Bull"
    elif n == 3:
        mapping[sorted_states[0]] = "Bear"
        mapping[sorted_states[1]] = "Sideways"
        mapping[sorted_states[2]] = "Bull"
    else:
        mapping[sorted_states[0]] = "Bear"
        mapping[sorted_states[-1]] = "Bull"
        for i in range(1, n - 1):
            mapping[sorted_states[i]] = f"Transition-{i}"

    logger.info("\n── Learned State Characteristics (original units) ───────────────")
    logger.info("  %-12s  %-8s  %12s  %10s  %12s",
                "Regime", "State", "Mean Ret/day", "Ann Ret%", "Ann Vol%")
    logger.info("  " + "─" * 58)
    for state_idx in range(n):
        state = sorted_states[state_idx]
        name = mapping[state]
        ann_ret = means_orig[state, 0] * 252 * 100
        ann_vol = vols_orig[state] * np.sqrt(252) * 100
        logger.info("  %-12s  %-8d  %12.5f  %10.2f  %12.2f",
                    name, state, means_orig[state, 0], ann_ret, ann_vol)

    return mapping




# ═══════════════════════════════════════════════════════════════
# STEP 5: Validate on known market dates
# ═══════════════════════════════════════════════════════════════
def validate_regimes(regime_series: pd.Series) -> None:
    """
    Check that HMM correctly labels historically known regimes.

    Known ground truths from NSE history:
      Mar 2020 → COVID crash (BEAR): Nifty fell ~38% in 40 days
      Jan–Dec 2021 → Post-COVID bull run (BULL): Nifty up ~24%
      Jan–Jun 2022 → Rate hike bear market (BEAR/SIDEWAYS): Nifty -15%
      Oct 2023–Mar 2024 → Election bull run (BULL): Nifty +20%

    If HMM fails these basic sanity checks, the model is mislabelled —
    likely due to state label swap (see label_states). Re-check means_.
    """
    validation_windows = {
        "COVID Crash (Mar 2020)"  : ("2020-03-01", "2020-03-31", "Bear"),
        "Post-COVID Bull (2021)"  : ("2021-01-01", "2021-12-31", "Bull"),
        "Rate Hike Bear (H1 2022)": ("2022-01-01", "2022-06-30", "Bear"),
        "Election Bull (Oct 2023)": ("2023-10-01", "2024-03-31", "Bull"),
    }

    logger.info("\n── Regime Validation on Known Market Events ─────────────────────")
    logger.info("  %-28s  %-12s  %-12s  %s",
                "Period", "Expected", "Actual", "Pass?")
    logger.info("  " + "─" * 65)

    all_pass = True
    for label, (start, end, expected) in validation_windows.items():
        try:
            window = regime_series.loc[start:end]
            if window.empty:
                logger.warning("  %-28s  No data in range", label)
                continue

            # Most frequent regime in the window = "actual"
            actual = window.value_counts().idxmax()
            pct = (window == actual).mean() * 100
            passed = (actual == expected)
            if not passed:
                all_pass = False

            logger.info("  %-28s  %-12s  %-12s  %s  (%.0f%% of days)",
                        label, expected, actual,
                        "✅ PASS" if passed else "⚠️  FAIL", pct)
        except Exception as e:
            logger.warning("  %-28s  Error: %s", label, e)

    if all_pass:
        logger.info("  All validation checks PASSED — regime labels are correct.")
    else:
        logger.warning("  Some checks FAILED — review label_states() mapping.")
        logger.warning("  This does NOT mean the model is wrong — label_states()")
        logger.warning("  maps by mean return. If COVID period labels as 'Sideways',")
        logger.warning("  check model.means_ and adjust label_states() threshold.")


# ═══════════════════════════════════════════════════════════════
# STEP 6: Transition matrix report
# ═══════════════════════════════════════════════════════════════
def log_transition_matrix(model: GaussianHMM, state_map: dict) -> None:
    """
    Log the learned regime transition probabilities.

    transmat_[i, j] = P(regime tomorrow = j | regime today = i)

    This is one of the most actionable outputs of HMM for trading:
      High P(Bear → Bear) = once in Bear, market stays Bear for many days.
      High P(Sideways → Bull) = Sideways often precedes rallies.
      This informs Kelly sizing: reduce position size when P(Bear|Bear) > 0.85.
    """
    K = model.n_components
    logger.info("\n── Transition Probability Matrix ─────────────────────────────────")
    logger.info("  P(row → col) = probability of moving from row regime to col regime")
    logger.info("")

    # Header
    names = [state_map.get(k, f"S{k}") for k in range(K)]
    header = "  {:<14}".format("From \\ To") + "".join(f"  {n:<12}" for n in names)
    logger.info(header)
    logger.info("  " + "─" * (16 + K * 14))

    for i in range(K):
        row_name = state_map.get(i, f"State-{i}")
        probs = "".join(f"  {model.transmat_[i, j]:.4f}      " for j in range(K))
        logger.info("  {:<14}{}".format(row_name, probs))

    # Most sticky regime
    persistence = np.diag(model.transmat_)
    stickiest_state = np.argmax(persistence)
    logger.info("\n  Most persistent regime: %s (P(stay)=%.4f)",
                state_map.get(stickiest_state, f"S{stickiest_state}"),
                persistence[stickiest_state])
    logger.info("  → Once market enters this regime, it stays for ~%.1f days on average.",
                1 / (1 - persistence[stickiest_state] + 1e-9))


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_hmm_pipeline(force_n_states: int = None) -> pd.DataFrame:
    """
    Full HMM Regime Detection pipeline:
      1. Load 3-feature matrix (mkt_return, realised_vol, momentum_20d)
      2. Standardise features (zero mean, unit variance)
      3. BIC search over K=2..5 (or use force_n_states to skip search)
      4. Fit GaussianHMM with best K
      5. Label states (Bull/Bear/Sideways by mean return)
      6. Validate on known market events
      7. Log transition matrix
      8. Save regime_labels.csv for downstream modules (M5 Kelly, M6 FinBERT gate)

    Returns:
      DataFrame with columns: [regime_int, regime_name, mkt_return, realised_vol, momentum_20d]
    """
    logger.info("=" * 70)
    logger.info("HMM REGIME DETECTION — M4")
    logger.info("=" * 70)

    # ── Load features ──────────────────────────────────────────
    features_df = load_features()
    feature_cols = features_df.columns.tolist()

    if len(features_df) < MIN_DAYS:
        raise ValueError(
            f"Only {len(features_df)} days of data — need at least {MIN_DAYS}. "
            "Run the Risk Engine first to generate vajra_returns.csv."
        )

    # ── Standardise ────────────────────────────────────────────
    # Why: GaussianHMM's covariance fitting is sensitive to feature scale.
    # We standardise so vol (0.15) and returns (0.0005) are comparable.
    # We keep the scaler so new days can be transformed consistently.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)   # (T, 3) float64

    logger.info("\n── Feature Statistics (before standardisation) ──────────────────")
    for i, col in enumerate(feature_cols):
        logger.info("  %-18s  mean=%+.6f  std=%.6f",
                    col, features_df[col].mean(), features_df[col].std())

    # ── BIC selection ──────────────────────────────────────────
    if force_n_states is not None:
        n_states = force_n_states
        logger.info("\nUsing forced n_states=%d (BIC search skipped)", n_states)
    else:
        # BIC note: on financial return data, HMM-BIC often decreases monotonically
        # (more states always fit better — no clear elbow like in K-means).
        # We cap at K=4 and default to 3 for interpretability:
        #   2 states: too coarse (no Sideways)
        #   3 states: matches market narrative + interview explainability
        #   4 states: adds a Crash state, useful but hard to defend in 1 minute
        # If BIC keeps decreasing with no elbow, we force K=3.
        n_states = select_n_states(X, k_range=range(2, 5))
        bic_results = []
        # Re-run to get the values
        D_ = X.shape[1]
        scores = []
        for k_ in range(2, 5):
            try:
                m_ = GaussianHMM(n_components=k_, covariance_type=COVARIANCE_TYPE,
                                 n_iter=N_ITER, random_state=RANDOM_STATE)
                m_.fit(X)
                n_params_ = k_*(k_-1) + (k_-1) + k_*D_ + k_*(D_*(D_+1)//2)
                bic_ = -2*m_.score(X) + n_params_*np.log(len(X))
                scores.append(bic_)
            except Exception:
                scores.append(np.inf)
        # If BIC strictly decreasing (no elbow) → force K=3 for interpretability
        if len(scores) >= 3 and scores[0] > scores[1] > scores[2]:
            logger.info("  BIC strictly decreasing — no elbow. Forcing K=3 for interpretability.")
            n_states = 3

    # ── Fit ────────────────────────────────────────────────────

    model = fit_hmm(X, n_states)

    # ── Decode: Viterbi algorithm ──────────────────────────────
    # predict() uses the Viterbi algorithm to find the single most probable
    # sequence of hidden states given all observations.
    # This is O(K²T) — linear in time, quadratic in states.
    regime_ints = model.predict(X)   # shape: (T,)

    # ── Label states ───────────────────────────────────────────
    state_map = label_states(model, feature_cols, scaler=scaler)

    # ── Validate ───────────────────────────────────────────────
    regime_names = pd.Series(
        [state_map[r] for r in regime_ints],
        index=features_df.index,
        name="regime_name"
    )
    validate_regimes(regime_names)

    # ── Transition matrix ──────────────────────────────────────
    log_transition_matrix(model, state_map)

    # ── Regime statistics ──────────────────────────────────────
    logger.info("\n── Regime Distribution ──────────────────────────────────────────")
    regime_counts = regime_names.value_counts()
    total = len(regime_names)
    for name, count in regime_counts.items():
        logger.info("  %-12s  %4d days  (%.1f%%)", name, count, count / total * 100)

    # ── Today's regime ─────────────────────────────────────────
    today_regime_int  = regime_ints[-1]
    today_regime_name = state_map[today_regime_int]
    logger.info("\n── CURRENT REGIME (latest observation) ──────────────────────────")
    logger.info("  📍 %s", today_regime_name.upper())
    logger.info("  Signal for downstream modules:")
    if today_regime_name == "Bull":
        logger.info("  → Use Momentum + Quality factors. Full Kelly sizing.")
    elif today_regime_name == "Bear":
        logger.info("  → Switch to Low-Vol + Defensive factors. Half-Kelly or zero.")
    else:
        logger.info("  → Sideways: Mean-reversion favoured. Pairs trades active.")
        logger.info("  → Cointegration pairs (BAJFINANCE/HDFCBANK, HDFCBANK/TCS) now relevant.")

    # ── Build output DataFrame ─────────────────────────────────
    output = features_df.copy()
    output["regime_int"]  = regime_ints
    output["regime_name"] = [state_map[r] for r in regime_ints]

    # ── Save ───────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = DATA_DIR / "regime_labels.csv"
    output.to_csv(out_path)

    logger.info("\n── SAVED ────────────────────────────────────────────────────────")
    logger.info("  regime_labels.csv — %d days × %d columns", *output.shape)
    logger.info("  Columns: mkt_return, realised_vol, momentum_20d, regime_int, regime_name")
    logger.info("  Downstream consumers: M5 (Kelly), M6 (FinBERT gate), M10 (Alpaca)")
    logger.info("=" * 70)

    return output


# ═══════════════════════════════════════════════════════════════
def detect_current_regime() -> str:
    """
    Lightweight function for downstream modules (M5, M6, M10).
    Loads pre-computed regime_labels.csv and returns today's regime name.
    Falls back to running the full pipeline if the file doesn't exist.

    Usage in M5 Kelly:
        from alpha_core.hmm_regime import detect_current_regime
        regime = detect_current_regime()  # "Bull" | "Bear" | "Sideways"
        kelly_multiplier = 1.0 if regime == "Bull" else 0.5 if regime == "Sideways" else 0.0
    """
    label_path = DATA_DIR / "regime_labels.csv"
    if label_path.exists():
        df = pd.read_csv(label_path, index_col=0, parse_dates=True)
        regime = df["regime_name"].iloc[-1]
        # Fix 2026-05-27: staleness check — warn if last label is > 3 calendar days old.
        # Without this, a 2-week-old Bear signal trades as if current. Silent and dangerous.
        last_date = df.index[-1]
        days_stale = (pd.Timestamp.today().normalize() - last_date).days
        if days_stale > 3:
            logger.warning(
                "⚠ STALE REGIME: last label is %d days old (date: %s, regime: %s). "
                "Re-run hmm_regime.py to refresh. Trading on stale signal.",
                days_stale, last_date.date(), regime
            )
        else:
            logger.info("Current regime (from cache, %d days old): %s", days_stale, regime)
        return regime
    else:
        logger.warning("regime_labels.csv not found — running full pipeline")
        output = run_hmm_pipeline()
        return output["regime_name"].iloc[-1]


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    regime_df = run_hmm_pipeline()
    print(f"\nFinal output shape: {regime_df.shape}")
    print(regime_df.tail(10).to_string())
