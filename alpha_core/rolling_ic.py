"""
Rolling IC Backtest — Walk-Forward Validation
==============================================

Answers: "Is the XGBoost signal stable across time, or just lucky in 2024?"

Walk-forward methodology:
  Window 1: Train 2019-2020 → Test 2021 → IC_2021
  Window 2: Train 2019-2021 → Test 2022 → IC_2022
  Window 3: Train 2019-2022 → Test 2023 → IC_2023
  Window 4: Train 2019-2023 → Test 2024 → IC_2024  (same as M7 final result)

For each window, compute IC per stock. Plot IC heatmap + mean line.
If IC is consistently positive across years → signal is real, not overfitting.
If IC only works in 2024 → the model got lucky.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from alpha_core.xgb_predictor import load_data, build_features, train_one_stock, XGB_PARAMS
import xgboost as xgb
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")

BASE_DIR = Path(__file__).parent.parent
FIG_DIR  = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"
FIG_DIR.mkdir(exist_ok=True)

# Walk-forward windows: (train_end, test_start, test_end, label)
WALK_FORWARD_WINDOWS = [
    ("2020-12-31", "2021-01-04", "2021-12-31", "2021"),
    ("2021-12-31", "2022-01-03", "2022-12-30", "2022"),
    ("2022-12-30", "2023-01-02", "2023-12-29", "2023"),
    ("2023-12-29", "2024-01-02", "2024-12-30", "2024"),
]

# Black swan events for reference lines on the IC plot
BLACK_SWANS = {
    "COVID\nCrash":     "2020",
    "RU/RBI\nShocks":   "2022",
    "Adani/\nSVB":      "2023",
}


def compute_ic_for_window(ticker: str, df: pd.DataFrame,
                           train_end: str, test_start: str, test_end: str) -> float:
    """Train on expanding window, return IC on held-out year."""
    feature_cols = [c for c in df.columns if c != "target"]
    train = df[df.index <= train_end]
    test  = df[(df.index >= test_start) & (df.index <= test_end)]

    if len(train) < 200 or len(test) < 30:
        return np.nan

    model = xgb.XGBRegressor(
        **{k: v for k, v in XGB_PARAMS.items() if k != "verbosity"},
        early_stopping_rounds=30,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        train[feature_cols].values, train["target"].values,
        eval_set=[(test[feature_cols].values, test["target"].values)],
        verbose=False,
    )

    y_pred = model.predict(test[feature_cols].values)
    y_true = test["target"].values

    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        return np.nan
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def run_rolling_ic():
    logger.info("=" * 60)
    logger.info("ROLLING IC BACKTEST — Walk-Forward Validation")
    logger.info("=" * 60)

    residuals, factors, regime = load_data()
    tickers = residuals.columns.tolist()

    # ic_table[year][ticker] = IC value
    ic_table = {label: {} for _, _, _, label in WALK_FORWARD_WINDOWS}

    for ticker in tickers:
        logger.info("  %s ...", ticker)
        df = build_features(ticker, residuals, factors, regime)
        for train_end, test_start, test_end, label in WALK_FORWARD_WINDOWS:
            ic = compute_ic_for_window(ticker, df, train_end, test_start, test_end)
            ic_table[label][ticker] = ic

    # Build DataFrame: rows=tickers, cols=years
    ic_df = pd.DataFrame(ic_table).T   # (years × tickers)
    ic_df.index.name = "year"

    logger.info("\n── Rolling IC Table ──────────────────────────────────────────")
    logger.info("  %s", ic_df.round(4).to_string())
    logger.info("\n  Mean IC per year:")
    for year, row in ic_df.iterrows():
        mean_ic = row.mean()
        flag = "GOOD" if mean_ic > 0.03 else ("WEAK" if mean_ic > 0 else "NEGATIVE")
        logger.info("    %s: %.4f  [%s]", year, mean_ic, flag)

    # Save
    ic_df.to_csv(DATA_DIR / "rolling_ic.csv")
    plot_rolling_ic(ic_df)
    return ic_df


def plot_rolling_ic(ic_df: pd.DataFrame):
    """
    Two-panel plot:
      Top: IC heatmap (years × stocks) — see which stocks are stable
      Bottom: Mean IC per year with std band + black swan reference lines
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                    gridspec_kw={"height_ratios": [2, 1.2]})
    fig.patch.set_facecolor("#0f1117")

    # ── Panel 1: Heatmap ──────────────────────────────────────
    ax1.set_facecolor("#1a1d27")
    data = ic_df.values.astype(float)   # (years × tickers)

    cmap = plt.cm.RdYlGn
    im = ax1.imshow(data, cmap=cmap, aspect="auto", vmin=-0.15, vmax=0.20)
    ax1.set_xticks(range(len(ic_df.columns)))
    ax1.set_xticklabels(ic_df.columns, rotation=35, ha="right",
                        color="white", fontsize=8)
    ax1.set_yticks(range(len(ic_df.index)))
    ax1.set_yticklabels(ic_df.index, color="white", fontsize=9)
    ax1.set_title("Rolling IC Heatmap — Walk-Forward Validation (M7+)\n"
                  "Green = positive IC (model works)  |  Red = negative IC",
                  color="white", fontsize=11)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7,
                         color="black" if abs(val) > 0.03 else "white")

    plt.colorbar(im, ax=ax1, label="IC", fraction=0.03, pad=0.01)

    # ── Panel 2: Mean IC per year ──────────────────────────────
    ax2.set_facecolor("#1a1d27")
    years = ic_df.index.tolist()
    mean_ics = ic_df.mean(axis=1).values
    std_ics  = ic_df.std(axis=1).values
    x = np.arange(len(years))

    ax2.fill_between(x, mean_ics - std_ics, mean_ics + std_ics,
                     alpha=0.25, color="#60a5fa", label="±1 std across stocks")
    ax2.plot(x, mean_ics, color="#60a5fa", linewidth=2, marker="o",
             markersize=7, label="Mean IC")
    ax2.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax2.axhline(0.05, color="#22c55e", linewidth=0.8, linestyle=":",
                alpha=0.6, label="IC=0.05 (meaningful)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(years, color="white", fontsize=10)
    ax2.set_ylabel("Mean IC across 14 stocks", color="#aaa", fontsize=9)
    ax2.set_title("Mean IC per Test Year  (higher = more predictive signal)",
                  color="white", fontsize=10)
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white", edgecolor="#333")
    ax2.tick_params(colors="#aaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333")

    out = FIG_DIR / "rolling_ic_backtest.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    logger.info("  Saved: %s", out)


if __name__ == "__main__":
    ic_df = run_rolling_ic()
    print("\n── Rolling IC Summary ──")
    print(ic_df.round(4).to_string())
    print(f"\nBest year:  {ic_df.mean(axis=1).idxmax()} (IC={ic_df.mean(axis=1).max():.4f})")
    print(f"Worst year: {ic_df.mean(axis=1).idxmin()} (IC={ic_df.mean(axis=1).min():.4f})")
