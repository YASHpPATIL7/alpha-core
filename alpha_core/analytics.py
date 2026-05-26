"""
M6 Scatter Plot + M9 Drawdown Analytics
=========================================

Two standalone analyses that complete M6 and M9:

  M6 finish: FinBERT sentiment score vs next-day stock return (scatter plot)
             Shows whether high sentiment scores predict positive next-day returns.

  M9: Drawdown Analytics on the simulated strategy equity curve.
      Max drawdown, Calmar ratio, underwater curve, recovery times.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no GUI — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# M6 SCATTER: Sentiment Score vs Next-Day Return
# ═══════════════════════════════════════════════════════════════
"""
Why this plot matters for interviews:
  The whole point of FinBERT gating is that sentiment PREDICTS returns.
  If P(positive earnings) is high, the stock should rise the next day
  as the market digests the news. This scatter tests that hypothesis
  on 10 real earnings announcements in our universe.

What we expect:
  Top-right quadrant: high sentiment score, positive next-day return (correct)
  Bottom-left quadrant: low sentiment score, negative next-day return (correct)
  Top-left or bottom-right: model wrong — sentiment didn't predict correctly

Data:
  X axis = FinBERT sentiment_score (P_pos - P_neg) from EARNINGS_TEXTS
  Y axis = actual next trading day return for that stock from factor_residuals.csv
  Each point = one earnings announcement (10 total)
"""

def build_sentiment_return_data():
    """
    Match each of the 10 historical earnings texts to the next trading day's
    return for that stock using factor_residuals.csv.

    Why factor_residuals not raw prices?
      Residuals = return AFTER removing Fama-French factor exposure.
      This isolates the stock-specific reaction to the earnings news,
      stripping out the market-wide move on that day (e.g. if the whole
      market rallied 2%, we don't want to credit FinBERT for that).
    """
    residuals = pd.read_csv(DATA_DIR / "factor_residuals.csv",
                            index_col=0, parse_dates=True)

    # Hardcoded earnings dates + tickers (from EARNINGS_TEXTS in finbert_sentiment.py)
    # We load the static texts here to get the dates + manually assign scores
    # (scores from the last static run — stale Q2 FY25 scores)
    STATIC_EVENTS = [
        {"ticker": "BAJFINANCE", "date": "2024-10-18", "sentiment_score":  0.923},
        {"ticker": "HDFCBANK",   "date": "2024-10-19", "sentiment_score":  0.920},
        {"ticker": "ICICIBANK",  "date": "2024-10-26", "sentiment_score":  0.169},
        {"ticker": "INFY",       "date": "2024-10-17", "sentiment_score":  0.939},
        {"ticker": "TCS",        "date": "2024-10-10", "sentiment_score":  0.915},
        {"ticker": "SUNPHARMA",  "date": "2024-11-07", "sentiment_score":  0.942},
        {"ticker": "MARUTI",     "date": "2024-10-25", "sentiment_score": -0.966},
        {"ticker": "DRREDDY",    "date": "2024-11-06", "sentiment_score": -0.957},
        {"ticker": "ONGC",       "date": "2024-11-12", "sentiment_score": -0.967},
        {"ticker": "ITC",        "date": "2024-10-28", "sentiment_score":  0.939},
    ]

    records = []
    for ev in STATIC_EVENTS:
        ticker = ev["ticker"]
        ev_date = pd.Timestamp(ev["date"])

        if ticker not in residuals.columns:
            continue

        # Find next trading day after the announcement
        future_dates = residuals.index[residuals.index > ev_date]
        if len(future_dates) == 0:
            continue
        next_day = future_dates[0]
        next_return = residuals.loc[next_day, ticker]

        records.append({
            "ticker":          ticker,
            "earnings_date":   ev_date,
            "next_trading_day": next_day,
            "sentiment_score": ev["sentiment_score"],
            "next_day_return": next_return,
            "next_day_return_pct": next_return * 100,
        })

    return pd.DataFrame(records)


def plot_sentiment_scatter(df: pd.DataFrame):
    """
    Scatter: sentiment_score (X) vs next-day idiosyncratic return (Y).

    Adds:
      - OLS regression line with 95% confidence band
      - Pearson r and p-value in the title
      - Quadrant labels (True Positive / False Positive etc.)
      - Per-point labels with ticker names
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Quadrant shading
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5,
               xmin=0.5, alpha=0.07, color="#22c55e")   # top-right: True Positive
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -5, 0,
               xmin=0, xmax=0.5, alpha=0.07, color="#ef4444")  # bottom-left: True Negative

    # Reference lines
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")

    # Scatter points — colour by quadrant
    colors = []
    for _, row in df.iterrows():
        if row["sentiment_score"] > 0 and row["next_day_return_pct"] > 0:
            colors.append("#22c55e")   # True Positive — green
        elif row["sentiment_score"] < 0 and row["next_day_return_pct"] < 0:
            colors.append("#22c55e")   # True Negative — green
        else:
            colors.append("#ef4444")   # Wrong — red

    ax.scatter(df["sentiment_score"], df["next_day_return_pct"],
               c=colors, s=120, zorder=5, edgecolors="white", linewidths=0.5)

    # Ticker labels
    for _, row in df.iterrows():
        ax.annotate(row["ticker"],
                    (row["sentiment_score"], row["next_day_return_pct"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color="white", alpha=0.85)

    # OLS regression line
    x = df["sentiment_score"].values
    y = df["next_day_return_pct"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#60a5fa", linewidth=1.5,
            label=f"OLS: y={slope:.3f}x+{intercept:.3f}")

    # Confidence band (95%)
    n = len(x)
    x_mean = x.mean()
    se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    t_crit = stats.t.ppf(0.975, df=n-2)
    ax.fill_between(x_line, y_line - t_crit*se_line, y_line + t_crit*se_line,
                    alpha=0.15, color="#60a5fa", label="95% CI")

    # Accuracy count
    correct = sum(
        (r["sentiment_score"] > 0 and r["next_day_return_pct"] > 0) or
        (r["sentiment_score"] < 0 and r["next_day_return_pct"] < 0)
        for _, r in df.iterrows()
    )
    accuracy = correct / len(df) * 100

    # Labels
    p_label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.set_title(
        f"FinBERT Sentiment Score vs Next-Day Idiosyncratic Return\n"
        f"Pearson r={r:.3f}  {p_label}  |  Directional Accuracy: {accuracy:.0f}%  ({correct}/{len(df)})",
        color="white", fontsize=12, pad=12
    )
    ax.set_xlabel("FinBERT Sentiment Score  [P(pos) − P(neg)]", color="#aaa", fontsize=10)
    ax.set_ylabel("Next-Day Idiosyncratic Return (%)", color="#aaa", fontsize=10)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Quadrant annotations
    ylim = ax.get_ylim()
    ax.text( 0.7,  ylim[1]*0.85 if ylim[1] > 1 else 1.5,
             "[+] True\nPositive", color="#22c55e", fontsize=8, alpha=0.6, ha="center")
    ax.text(-0.7,  ylim[0]*0.85 if ylim[0] < -1 else -1.5,
             "[+] True\nNegative", color="#22c55e", fontsize=8, alpha=0.6, ha="center")
    ax.text(-0.7,  ylim[1]*0.85 if ylim[1] > 1 else 1.5,
             "[-] False\nPositive", color="#ef4444", fontsize=8, alpha=0.6, ha="center")
    ax.text( 0.7,  ylim[0]*0.85 if ylim[0] < -1 else -1.5,
             "[-] False\nNegative", color="#ef4444", fontsize=8, alpha=0.6, ha="center")

    ax.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
              edgecolor="#333", loc="upper left")

    out = FIG_DIR / "m6_sentiment_vs_return.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return r, p, accuracy, slope


# ═══════════════════════════════════════════════════════════════
# M9: DRAWDOWN ANALYTICS
# ═══════════════════════════════════════════════════════════════
"""
M9 computes drawdown analytics on the simulated strategy equity curve.

What is drawdown?
  At each point in time, drawdown = (current value - peak so far) / peak so far
  It's always ≤ 0. -0.20 = you're 20% below your prior high.

Why it matters more than Sharpe in quant interviews:
  Sharpe = return/vol. Vol penalises upside AND downside equally.
  Drawdown measures ONLY the bad part — sustained losses from peak.
  Risk managers, PMs, and CIOs care about max drawdown:
  "Your fund dropped 30% peak to trough in 2022. How long to recover?"

Metrics we compute:
  1. Max Drawdown            = worst single trough from any prior peak
  2. Average Drawdown        = mean of all drawdown troughs (not just max)
  3. Max Drawdown Duration   = longest consecutive days below prior high
  4. Calmar Ratio            = annualised return / |max drawdown|
                               > 1.0 = good (you earn more than your worst loss/year)
  5. Pain Index              = average drawdown depth × duration (area under curve)

Data:
  Simulated equity curve = SUNPHARMA's cumulative return (the only
  active factor bet from M5). For a full strategy, you'd sum all
  gated position returns. Here we use SUNPHARMA as representative.
"""

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute the drawdown series from a daily return series.
    DD_t = (Equity_t - Peak_t) / Peak_t   where Peak_t = max(Equity_0..t)
    """
    equity = (1 + returns).cumprod()
    rolling_peak = equity.cummax()
    drawdown = (equity - rolling_peak) / rolling_peak
    return equity, rolling_peak, drawdown


def drawdown_stats(returns: pd.Series) -> dict:
    """Compute full drawdown analytics table."""
    equity, rolling_peak, dd = compute_drawdown_series(returns)
    n = len(returns)

    # Max drawdown
    max_dd = dd.min()

    # Average drawdown (only periods when underwater)
    avg_dd = dd[dd < 0].mean() if (dd < 0).any() else 0

    # Drawdown duration — find longest consecutive below-zero streak
    in_dd = dd < 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    # Calmar ratio
    ann_return = returns.mean() * 252
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # Pain index = mean |drawdown| across all days
    pain_index = dd.abs().mean()

    # Recovery from max drawdown
    max_dd_idx = dd.idxmin()
    post_max = equity[max_dd_idx:]
    peak_val = rolling_peak[max_dd_idx]
    recovered = post_max[post_max >= peak_val]
    recovery_days = (recovered.index[0] - max_dd_idx).days if len(recovered) > 0 else None

    return {
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "avg_drawdown_pct":   round(avg_dd * 100, 2),
        "max_dd_duration_days": max_dur,
        "recovery_days":      recovery_days,
        "calmar_ratio":       round(calmar, 3) if not np.isnan(calmar) else "N/A",
        "pain_index_pct":     round(pain_index * 100, 4),
        "ann_return_pct":     round(ann_return * 100, 2),
        "ann_vol_pct":        round(returns.std() * np.sqrt(252) * 100, 2),
        "sharpe":             round(ann_return / (returns.std() * np.sqrt(252) + 1e-9), 3),
        "n_days":             n,
    }


def plot_drawdown_analytics(returns: pd.Series, ticker: str = "SUNPHARMA"):
    """
    3-panel drawdown dashboard:
      Panel 1: Equity curve with rolling peak (peak = high water mark)
      Panel 2: Underwater curve (drawdown series) — always ≤ 0
      Panel 3: Rolling 60-day Sharpe ratio — shows regime of risk-adjusted returns
    """
    equity, rolling_peak, dd = compute_drawdown_series(returns)
    stats = drawdown_stats(returns)

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 2], hspace=0.35)

    colors = {"equity": "#60a5fa", "peak": "#94a3b8",
              "dd_fill": "#ef4444", "sharpe_pos": "#22c55e", "sharpe_neg": "#ef4444"}

    # ── Panel 1: Equity curve ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1a1d27")
    ax1.plot(equity.index, equity.values, color=colors["equity"],
             linewidth=1.2, label=f"{ticker} Equity Curve")
    ax1.plot(rolling_peak.index, rolling_peak.values, color=colors["peak"],
             linewidth=0.8, linestyle="--", alpha=0.6, label="Rolling Peak (HWM)")
    ax1.fill_between(equity.index, equity.values, rolling_peak.values,
                     where=equity < rolling_peak, alpha=0.25, color="#ef4444",
                     label="Underwater")
    ax1.set_title(f"{ticker} Strategy — Drawdown Analytics (M9)\n"
                  f"Max DD: {stats['max_drawdown_pct']:.1f}%  |  "
                  f"Calmar: {stats['calmar_ratio']}  |  "
                  f"Sharpe: {stats['sharpe']}  |  "
                  f"Ann Return: {stats['ann_return_pct']:.1f}%",
                  color="white", fontsize=11, pad=8)
    ax1.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#333", loc="upper left")
    ax1.tick_params(colors="#aaa")
    ax1.set_ylabel("Equity (₹1 invested)", color="#aaa", fontsize=9)
    for spine in ax1.spines.values(): spine.set_edgecolor("#333")

    # ── Panel 2: Drawdown (underwater curve) ──────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1a1d27")
    ax2.fill_between(dd.index, dd.values * 100, 0,
                     where=dd < 0, alpha=0.6, color="#ef4444", label="Drawdown")
    ax2.plot(dd.index, dd.values * 100, color="#ef4444", linewidth=0.8)
    ax2.axhline(stats["max_drawdown_pct"], color="#f97316", linewidth=1,
                linestyle=":", label=f"Max DD = {stats['max_drawdown_pct']:.1f}%")
    ax2.axhline(stats["avg_drawdown_pct"], color="#fbbf24", linewidth=0.8,
                linestyle=":", alpha=0.7, label=f"Avg DD = {stats['avg_drawdown_pct']:.1f}%")
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#333", loc="lower right")
    ax2.set_ylabel("Drawdown (%)", color="#aaa", fontsize=9)
    ax2.tick_params(colors="#aaa")
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")

    # ── Panel 3: Rolling 60-day Sharpe ────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#1a1d27")
    roll_sharpe = (returns.rolling(60).mean() /
                   (returns.rolling(60).std() + 1e-9)) * np.sqrt(252)
    pos_mask = roll_sharpe >= 0
    ax3.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                     where=pos_mask, alpha=0.4, color=colors["sharpe_pos"])
    ax3.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                     where=~pos_mask, alpha=0.4, color=colors["sharpe_neg"])
    ax3.plot(roll_sharpe.index, roll_sharpe.values, color="white", linewidth=0.7)
    ax3.axhline(0, color="#555", linewidth=0.8)
    ax3.axhline(1, color="#22c55e", linewidth=0.6, linestyle="--", alpha=0.5,
                label="Sharpe=1 (target)")
    ax3.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#333", loc="upper right")
    ax3.set_ylabel("Rolling 60d Sharpe", color="#aaa", fontsize=9)
    ax3.set_xlabel("Date", color="#aaa", fontsize=9)
    ax3.tick_params(colors="#aaa")
    for spine in ax3.spines.values(): spine.set_edgecolor("#333")

    out = FIG_DIR / f"m9_drawdown_{ticker.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return stats


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("M6 SCATTER + M9 DRAWDOWN ANALYTICS")
    print("=" * 60)

    # ── M6 Scatter ─────────────────────────────────────────────
    print("\n[M6] Building sentiment vs return scatter...")
    scatter_df = build_sentiment_return_data()
    print(scatter_df[["ticker", "earnings_date", "sentiment_score",
                       "next_trading_day", "next_day_return_pct"]].to_string(index=False))

    r, p, accuracy, slope = plot_sentiment_scatter(scatter_df)
    print(f"\nPearson r = {r:.3f}  |  p = {p:.3f}  |  Directional Accuracy = {accuracy:.0f}%")
    print(f"Slope = {slope:.4f}: 1 unit increase in sentiment_score → "
          f"{slope:.3f}% next-day return")

    # ── M9 Drawdown ─────────────────────────────────────────────
    print("\n[M9] Computing drawdown analytics...")
    # Use raw stock returns (log returns from vajra_returns) — NOT residuals.
    # Residuals are zero-mean by construction (Fama-French strips out market return),
    # so Calmar/Sharpe on residuals = -0. Raw returns show the actual equity journey.
    vajra_path = BASE_DIR.parent / "indian-risk-engine" / "data" / "vajra_returns.csv"
    if not vajra_path.exists():
        # fallback: use factor_returns MKT as proxy
        vajra_path = DATA_DIR / "factor_returns.csv"
    price_rets = pd.read_csv(vajra_path, index_col=0, parse_dates=True)

    for ticker in ["SUNPHARMA", "BAJFINANCE", "ICICIBANK"]:
        if ticker not in price_rets.columns:
            continue
        rets = price_rets[ticker].dropna()
        stats_out = drawdown_stats(rets)
        plot_drawdown_analytics(rets, ticker)
        print(f"\n{ticker} Drawdown Stats:")
        for k, v in stats_out.items():
            print(f"  {k:<28} {v}")

    print("\nAll figures saved to:", FIG_DIR)
