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
from scipy.stats import multivariate_normal
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
VIX_PATH = DATA_DIR / "india_vix_history.csv"

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

    We use THREE features:
      1. Nifty market return (MKT from Fama-French factor_returns.csv)
         — captures directional trend
      2. India VIX (from india_vix_history.csv)
         — exogenous implied-volatility measure; captures fear/turbulence.
           VIX is forward-looking and does NOT lag after a crash, unlike
           rolling realised vol which stays elevated for 20 days post-crash.
           This prevents the COVID crash recovery period from being
           mislabelled as "Bear" for weeks after the bottom.
           FIX (2026-06-07): replaced self-computed realised_vol with VIX.
      3. Rolling 20-day Sharpe ratio (mean / std of returns)
         — directional momentum signal: positive in bull, negative in bear.
           FIX (2026-06-07): replaced raw 20-day sum (which conflated
           magnitude with direction, making the feature backwards) with
           the Sharpe ratio which is sign-correct and scale-free.

    Why use India VIX instead of self-computed realised vol?
      Self-computed: mkt.rolling(20).std() * sqrt(252) lags 20 days.
      After COVID crash (-40% in 40 days), the 20-day window keeps vol
      elevated all through April–May 2020 even as markets recover.
      This caused steady bull days to be labelled 'Bear' (high vol cluster).
      VIX is market-implied and forward-looking — it spiked to 83 on
      Mar 24 and fell to 30 by Apr 30 as recovery began. Much better signal.

    Why use rolling Sharpe instead of raw 20-day sum for momentum?
      Raw sum conflates magnitude with direction. The COVID crash period
      had the largest absolute 20-day sums (both negative and positive),
      making the 'Bull' cluster absorb both crash + recovery days.
      Rolling Sharpe = mean_20d / std_20d is dimensionless and directional:
      clearly positive in trending bull markets, negative in bear trends.
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

    # Feature 2: India VIX — exogenous fear/turbulence measure
    # FIX 2026-06-07: replaces self-computed realised_vol which lagged 20 days
    # and caused volatility clustering instead of directional regime detection.
    if VIX_PATH.exists():
        vix_df = pd.read_csv(VIX_PATH, index_col=0, parse_dates=True)
        vix_aligned = vix_df["vix_close"].reindex(mkt.index)
        # Forward-fill up to 5 days for weekends/holidays, then backward-fill start
        vix_aligned = vix_aligned.ffill(limit=5).bfill()
        features["india_vix"] = vix_aligned / 100.0  # normalise to decimal (9→0.09, 83→0.83)
        logger.info("  India VIX loaded: %d aligned days (range %.1f–%.1f)",
                    vix_aligned.notna().sum(), vix_aligned.min(), vix_aligned.max())
    else:
        # Fallback: rolling realised vol (old buggy method — labelled for audit trail)
        logger.warning("india_vix_history.csv not found — falling back to realised_vol "
                       "(NOTE: this is the buggy feature that caused volatility clustering; "
                       "fetch VIX data and re-run for correct regime detection)")
        features["india_vix"] = mkt.rolling(20).std() * np.sqrt(252)

    # Feature 3: rolling 20-day Sharpe ratio — directional momentum
    # FIX 2026-06-07: replaces raw 20-day sum which was magnitude-contaminated
    # and made the feature backwards (bull cluster had negative mean momentum).
    roll_mean = mkt.rolling(20).mean()
    roll_std  = mkt.rolling(20).std()
    features["momentum_sharpe"] = roll_mean / (roll_std + 1e-9)
    # Positive = persistently rising market (bull), negative = falling (bear), ~0 = sideways

    # Drop NaN rows (first 20 days have no rolling stats; VIX might have early NaN)
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

    FIX 2026-06-07: Replaced Sharpe-sort with mean-return sort.
    OLD METHOD (BUGGY):
      Sorted by Sharpe = mean_return / emission_vol.
      Problem: with VIX as feature 2 and rolling-Sharpe as feature 3, the
      emission covariance structure no longer corresponds to return vol alone.
      More critically: Sharpe-sort was the root cause of the Covid crash being
      labelled 'Bull' — the high-vol cluster got the highest Sharpe by accident
      because mean_return was similar across all states while vol varied 10x.
    NEW METHOD:
      Sort purely by mean of feature 0 (mkt_return) in original space.
      Lowest mean return = Bear. Highest mean return = Bull. Simple and correct.
      If two states have very similar mean returns, we use the VIX mean as a
      tiebreaker: higher VIX mean = more fearful = Bear.

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

    # FIX 2026-06-07: Sort using semantic features instead of raw returns.
    # High VIX = Crisis/Bear. High Momentum = Bull.
    # 1. Bull is the state with the highest mean momentum.
    # 2. Bear is the state with the highest mean VIX.
    # 3. Sideways is the remaining state.
    # This prevents crisis states (which often have massive dead-cat bounces
    # and thus positive mean returns) from being labelled 'Sideways' or 'Bull'.
    
    # Feature indices: 0=mkt_return, 1=india_vix, 2=momentum_sharpe
    vix_means = means_orig[:, 1]
    mom_means = means_orig[:, 2]

    n = model.n_components
    mapping = {}

    if n == 3:
        bull_state = np.argmax(mom_means)
        
        # Mask out the bull state to find the highest VIX among remaining
        remaining_states = [i for i in range(n) if i != bull_state]
        bear_state = remaining_states[np.argmax([vix_means[i] for i in remaining_states])]
        
        # Sideways is whatever is left
        side_state = [i for i in range(n) if i not in (bull_state, bear_state)][0]
        
        mapping[bull_state] = "Bull"
        mapping[bear_state] = "Bear"
        mapping[side_state] = "Sideways"
        
        # For logging order
        sorted_states = [bear_state, side_state, bull_state]
    else:
        # Fallback for 2 or 4+ states
        sorted_states = np.argsort(means_orig[:, 0])
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

    def filtered_state_probs(model, X):
        """True forward-pass filtered probabilities P(state_t | x_1..x_t).

        hmmlearn's predict_proba() runs forward-backward (smoothed posteriors —
        they condition on the FULL sample, future included). This function runs
        the forward recursion only, normalised at each step, so the probability
        at time t conditions exclusively on observations up to and including t.
        """
        T = X.shape[0]
        K = model.n_components

        # Per-frame emission likelihoods B[t, k] = p(x_t | state k)
        B = np.zeros((T, K))
        for k in range(K):
            cov = model.covars_[k]
            if cov.ndim == 1:                # covariance_type="diag"
                cov = np.diag(cov)
            B[:, k] = multivariate_normal.pdf(
                X, mean=model.means_[k], cov=cov, allow_singular=True)
        B = np.clip(B, 1e-300, None)         # underflow guard

        alpha = np.zeros((T, K))
        alpha[0] = model.startprob_ * B[0]
        alpha[0] /= alpha[0].sum()
        A = model.transmat_
        for t in range(1, T):
            alpha[t] = B[t] * (alpha[t - 1] @ A)
            s = alpha[t].sum()
            alpha[t] = alpha[t] / s if (np.isfinite(s) and s > 0) \
                       else np.full(K, 1.0 / K)
        return alpha

    # ── Decode: Forward-algorithm FILTERED probabilities (historical labels) ─────
    # Bug fix 1 (2026-06-12): hmmlearn's predict_proba() runs forward-backward
    # (smoothed posteriors), which means every historical label still uses
    # future data. We implement an explicit forward-only recursion using the
    # fitted model's parameters so P(state_t | x_1..x_t) conditions exclusively
    # on observations up to and including t.
    #
    # We take argmax of each row = the filtered estimate at each date.
    # This is clean: the regime label on day t uses only data up to and
    # including day t.
    #
    # Viterbi is still used for the final current-day LIVE readout only,
    # since for the last row both methods condition on the same data.
    filtered_probs = filtered_state_probs(model, X)   # shape: (T, K), P(state|past)
    regime_ints    = filtered_probs.argmax(axis=1).astype(int)  # shape: (T,)

    # Viterbi for live readout ONLY (last row) — identical to filtered at T
    viterbi_ints = model.predict(X)
    today_regime_int_viterbi = int(viterbi_ints[-1])

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

    # ── Today's regime: use Viterbi for live readout ───────────────────
    today_regime_int  = today_regime_int_viterbi
    today_regime_name = state_map[today_regime_int]
    today_probs = filtered_probs[-1]
    prob_str = " · ".join(
        f"{state_map[k]} {today_probs[k]*100:.0f}%" for k in range(model.n_components)
    )
    logger.info("\n── CURRENT REGIME (latest observation) ──────────────────────────")
    logger.info("  📍 %s  (confidence %.0f%%)", today_regime_name.upper(),
                today_probs.max() * 100)
    logger.info("  Posterior: %s", prob_str)
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

    # ── Per-day state PROBABILITIES (filtered, uses only past data) ─────
    # A single hard label ("BULL") is brittle and easy to call wrong. Saving the
    # filtered posterior P(state_t | x_1..x_t) lets the dashboard show an honest
    # "Bull 12% · Sideways 30% · Bear 58%" readout instead of one fragile word.
    # Columns are named by REGIME, not by arbitrary HMM state integer, so
    # consumers never have to re-derive the state→name mapping.
    for state_idx in range(model.n_components):
        col = "prob_" + state_map[state_idx].lower()   # prob_bull/prob_bear/prob_sideways
        output[col] = filtered_probs[:, state_idx]
    # Confidence of the label actually assigned each day (max posterior)
    output["regime_confidence"] = filtered_probs.max(axis=1)

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
