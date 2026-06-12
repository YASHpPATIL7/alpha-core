"""
FinBERT Sentiment Gating — M6
==============================

Model: ProsusAI/finbert  (BERT fine-tuned on Financial PhraseBank)
Task:  Score NSE earnings text → positive/neutral/negative + confidence
Gate:  Multiply M5 Kelly positions by sentiment multiplier before execution

Why ProsusAI/finbert and not vanilla BERT or GPT?
--------------------------------------------------
Vanilla BERT was trained on Wikipedia + BookCorpus.
Financial language is totally different:
  "The company revised guidance downward" → general NLP: neutral
                                          → FinBERT: NEGATIVE
  "Beat expectations by a thin margin"    → general NLP: positive
                                          → FinBERT: NEUTRAL (thin = uncertain)

ProsusAI/finbert was fine-tuned on 4500 sentences from the Financial PhraseBank
(Malo et al., 2014) — sentences annotated by finance domain experts.
It's the industry standard for earnings call/news sentiment.

Why sentiment gating matters for pairs trading:
-----------------------------------------------
Johansen says BAJFINANCE/HDFCBANK spread mean-reverts. Kelly sizes the bet.
But if BAJFINANCE just reported a profit warning → the spread WILL dislocate
even further before reverting. Entering a SHORT on BAJFINANCE now will bleed.

FinBERT catches this: negative earnings sentiment on BAJFINANCE
→ delay the pairs trade entry → avoid the initial shock drawdown.

Sentiment → position multiplier mapping:
  positive → 1.0  (full Kelly position)
  neutral  → 0.7  (reduce — uncertainty)
  negative → 0.0  (flat — don't fight bad news)

Output: finbert_sentiment.csv + kelly_positions_gated.csv
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

def sentiment_to_multiplier(sentiment_score: float) -> float:
    """
    Continuous confidence-weighted multiplier from sentiment_score.

    sentiment_score = P(positive) - P(negative)  ∈ [-1, +1]

    OLD approach (step function):
      positive → 1.0, neutral → 0.7, negative → 0.0
      Problem: SUNPHARMA (score=+0.94) and AXISBANK (score=+0.05) both get 1.0.
      That's identical treatment despite totally different confidence levels.

    NEW approach (linear map):
      multiplier = (sentiment_score + 1) / 2

      score = +1.0  → multiplier = 1.00  (maximum confidence positive)
      score = +0.94 → multiplier = 0.97  (SUNPHARMA — strong positive)
      score = +0.17 → multiplier = 0.58  (ICICIBANK — weak positive)
      score =  0.0  → multiplier = 0.50  (perfectly neutral = bet 50%)
      score = -0.95 → multiplier = 0.02  (INFY — strong negative, near zero)
      score = -1.0  → multiplier = 0.00  (maximum confidence negative)

    Why not hard-zero at negative?
      A score of -0.3 is 'somewhat negative' not 'certain disaster'.
      Hard-zeroing at any negative score throws away the nuance.
      0.35× multiplier on a -0.3 score is more honest than 0×.
      For very strong negative (score < -0.8), multiplier < 0.1 = effectively flat.
    """
    return round((sentiment_score + 1) / 2, 4)

# ── NSE tickers → Yahoo Finance suffix ──────────────────────────────────────
NSE_TICKERS = {
    "RELIANCE":   "RELIANCE.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "INFY":       "INFY.NS",
    "TCS":        "TCS.NS",
    "AXISBANK":   "AXISBANK.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "DRREDDY":    "DRREDDY.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC":        "ITC.NS",
    "MARUTI":     "MARUTI.NS",
    "ONGC":       "ONGC.NS",
    "SUNPHARMA":  "SUNPHARMA.NS",
    "WIPRO":      "WIPRO.NS",
}

# ── STEP 0: Live news fetcher ─────────────────────────────────────────────────
def fetch_live_news(max_per_ticker: int = 3) -> list:
    """
    Pull real-time news headlines + summaries from Yahoo Finance for all 14 NSE tickers.

    Why yfinance news?
      - Free, no API key required
      - Returns structured JSON: title + summary + pubDate per article
      - Updates multiple times per day (Reuters, ET, Mint, MoneyControl articles)
      - Already installed (used in M1/M3 for price data)

    Why title + summary and not just title?
      FinBERT was trained on full sentences. A title like "Bajaj Finance Q4 results"
      gives FinBERT no sentiment signal — it needs at least one clause with an
      outcome ("profit rose", "margin compressed", "NPA increased").
      Concatenating title + summary gives 100-300 tokens — enough context.

    Why max 3 articles per ticker?
      FinBERT inference takes ~0.05s/text on MPS. 14 tickers × 3 = 42 texts = 2s.
      More articles = better coverage but diminishing returns. 3 captures:
        - Most recent earnings result
        - Any analyst upgrade/downgrade
        - Any recent event (product launch, regulatory news)
      We then AVERAGE the 3 sentiment scores per ticker for a stable signal.

    Fallback:
      If yfinance returns empty (network issue, ticker delisted), the pipeline
      falls back to EARNINGS_TEXTS (curated static data from Q2 FY25).
    """
    import yfinance as yf

    live_texts = []
    failed_tickers = []

    logger.info("Fetching live news from Yahoo Finance for %d tickers...", len(NSE_TICKERS))

    for nse_ticker, yf_ticker in NSE_TICKERS.items():
        try:
            ticker_obj = yf.Ticker(yf_ticker)
            news_items = ticker_obj.news or []

            if not news_items:
                logger.warning("  %s — no news returned, will use static fallback", nse_ticker)
                failed_tickers.append(nse_ticker)
                continue

            count = 0
            for item in news_items[:max_per_ticker]:
                content = item.get("content", {})
                title   = content.get("title", "").strip()
                summary = content.get("summary", "").strip()
                pub_date = content.get("pubDate", "")[:10]  # YYYY-MM-DD

                if not title:
                    continue

                # Combine title + summary for richer context
                combined = f"{title}. {summary}" if summary else title
                combined = combined[:600]   # trim to ~120 tokens max

                live_texts.append({
                    "ticker":  nse_ticker,
                    "date":    pub_date,
                    "source":  f"Yahoo Finance (live) — {yf_ticker}",
                    "text":    combined,
                    "article_index": count,
                })
                count += 1

            logger.info("  %-12s → %d articles fetched (latest: %s)",
                        nse_ticker, count,
                        live_texts[-1]["date"] if live_texts else "N/A")

        except Exception as e:
            logger.warning("  %s — fetch error: %s", nse_ticker, e)
            failed_tickers.append(nse_ticker)

    logger.info("Live fetch complete: %d articles for %d tickers | %d failed",
                len(live_texts), len(NSE_TICKERS) - len(failed_tickers), len(failed_tickers))

    return live_texts, failed_tickers


def aggregate_ticker_sentiments(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    When multiple articles exist per ticker, average their sentiment scores.

    Why average and not take the most recent?
      A single article can be anomalously negative (e.g. a short-seller report)
      or anomalously positive (a sponsored article). Averaging 3 articles
      smooths out noise and gives a more stable signal.

    Output: one row per ticker (same format as single-article output).
    """
    agg = (
        scored_df
        .groupby("ticker")
        .agg(
            p_positive     = ("p_positive",      "mean"),
            p_neutral      = ("p_neutral",       "mean"),
            p_negative     = ("p_negative",      "mean"),
            sentiment_score= ("sentiment_score", "mean"),
            n_articles     = ("ticker",          "count"),
            latest_date    = ("date",            "max"),
            source         = ("source",          "first"),
        )
        .reset_index()
    )

    # Re-derive label from averaged scores
    agg["sentiment"] = agg.apply(
        lambda r: "positive" if r["p_positive"] >= r["p_negative"] and r["p_positive"] >= r["p_neutral"]
                  else ("negative" if r["p_negative"] >= r["p_neutral"] else "neutral"),
        axis=1
    )
    agg["confidence"] = agg.apply(
        lambda r: r[f"p_{r['sentiment']}"], axis=1
    )
    agg["multiplier"] = agg["sentiment_score"].apply(sentiment_to_multiplier)
    agg["text_preview"] = "(aggregated from " + agg["n_articles"].astype(str) + " articles)"

    return agg

# ── Static fallback texts (Q2 FY25 — used only when live fetch fails) ────────
# These are curated real-world earnings summaries from Oct-Nov 2024.
# They are stale but accurate for the specific quarter they cover.
# Priority: LIVE news > these static texts (only used for failed tickers).
EARNINGS_TEXTS = [
    {
        "ticker": "BAJFINANCE",
        "date":   "2024-10-18",
        "source": "Q2 FY25 earnings call",
        "text":   (
            "Bajaj Finance reported a net interest income of ₹8,838 crore, "
            "up 23% year-on-year, driven by strong AUM growth of 29%. "
            "However, gross NPA rose to 1.06% from 0.91% in the prior quarter, "
            "reflecting stress in the consumer durable and SME segments. "
            "Management guided for moderation in growth to 25-26% for FY25."
        ),
    },
    {
        "ticker": "HDFCBANK",
        "date":   "2024-10-19",
        "source": "Q2 FY25 results",
        "text":   (
            "HDFC Bank posted net profit of ₹16,821 crore, broadly in line with estimates. "
            "Net interest margin compressed to 3.46% from 3.65% a year ago, "
            "as the merged entity continues to digest higher-cost HDFC Ltd deposits. "
            "The bank reiterated its strategy of prioritising margin recovery over loan growth, "
            "keeping credit growth at a measured 7% this quarter."
        ),
    },
    {
        "ticker": "ICICIBANK",
        "date":   "2024-10-26",
        "source": "Q2 FY25 analyst briefing",
        "text":   (
            "ICICI Bank delivered a stellar quarter with net profit surging 14.5% to ₹11,792 crore, "
            "ahead of street estimates. Core operating profit grew 14% with ROE at 18.6%, "
            "the highest in five years. Asset quality improved with gross NPA falling to 2.15%, "
            "the lowest level since FY2017. Management expressed confidence in sustaining "
            "above-industry growth through digital acquisition channels."
        ),
    },
    {
        "ticker": "INFY",
        "date":   "2024-10-17",
        "source": "Q2 FY25 earnings release",
        "text":   (
            "Infosys narrowed its revenue growth guidance to 3.75-4.5% for FY25 in constant currency, "
            "raising the lower end from 3%. Revenue grew 5.1% YoY in CC terms. "
            "Large deal wins totalled $2.4 billion, the highest in six quarters, "
            "led by financial services and manufacturing verticals. "
            "Operating margin improved 50 basis points to 21.1%."
        ),
    },
    {
        "ticker": "TCS",
        "date":   "2024-10-10",
        "source": "Q2 FY25 press release",
        "text":   (
            "TCS reported revenue of ₹63,973 crore, growing 8.9% in rupee terms. "
            "Deal wins of $8.6 billion represented the best quarter in two years, "
            "though management flagged that decision-making cycles in BFSI remained elongated. "
            "EBIT margin held at 24.1%. The company declared a special dividend of ₹10 per share "
            "in addition to the interim dividend."
        ),
    },
    {
        "ticker": "SUNPHARMA",
        "date":   "2024-11-07",
        "source": "Q2 FY25 investor presentation",
        "text":   (
            "Sun Pharmaceutical reported 12.3% revenue growth driven by its specialty business, "
            "which now contributes 18% of US revenues. The US generics business remained resilient "
            "despite pricing pressure. EBITDA margin expanded 180 basis points to 28.4%. "
            "The company received USFDA approval for Leqselvi, a new dermatology product, "
            "ahead of the competition. Management raised FY25 guidance."
        ),
    },
    {
        "ticker": "MARUTI",
        "date":   "2024-10-25",
        "source": "Q2 FY25 results call",
        "text":   (
            "Maruti Suzuki posted a 17% decline in net profit to ₹3,069 crore, "
            "missing estimates significantly. Volumes were flat at 5.84 lakh units "
            "while realisations fell due to a product mix shift toward entry-level vehicles. "
            "Input costs remained elevated on account of commodity headwinds. "
            "Management warned of continued margin pressure through Q3 FY25."
        ),
    },
    {
        "ticker": "DRREDDY",
        "date":   "2024-11-06",
        "source": "Q2 FY25 earnings commentary",
        "text":   (
            "Dr Reddy's Laboratories reported modest 4% revenue growth, "
            "weighed down by erosion in gRevlimid exclusivity income. "
            "North America generics revenue fell 8% sequentially. "
            "The company launched 13 new products in the US, providing some offset. "
            "EBITDA margin contracted 320 basis points to 24.6%, "
            "below the management's stated 25%+ target range."
        ),
    },
    {
        "ticker": "ONGC",
        "date":   "2024-11-12",
        "source": "Q2 FY25 results",
        "text":   (
            "ONGC net profit fell 22% year-on-year to ₹10,142 crore on lower crude oil realisations "
            "and higher subsidy burden. The government's decision to cap domestic gas prices "
            "weighed on revenue. Production volumes from ageing fields continue to decline. "
            "Capital expenditure guidance was maintained but project execution delays persist "
            "at the KG-DWN basin."
        ),
    },
    {
        "ticker": "ITC",
        "date":   "2024-10-28",
        "source": "Q2 FY25 analyst meet",
        "text":   (
            "ITC reported steady 7% revenue growth with cigarette volumes growing 4.5%, "
            "ahead of estimates. The FMCG business continued its profitability improvement journey "
            "with EBIT margin reaching 11.2%, up from 9.8% a year ago. "
            "The demerger of the hotels business is progressing on schedule. "
            "Management guided for sustained double-digit EPS growth over the medium term."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load FinBERT
# ═══════════════════════════════════════════════════════════════
def load_finbert():
    """
    Load ProsusAI/finbert via HuggingFace pipeline.

    Why pipeline() and not manual tokeniser + model?
      pipeline() handles: tokenisation → forward pass → softmax → label mapping
      in one call. For inference-only (no training), it's cleaner and faster.
      Manual approach is needed only if you want to extract hidden states
      (e.g. for embedding-based similarity search — not needed here).

    Device selection:
      MPS  = Apple Silicon GPU (M1/M2/M3 Mac) — fastest on your machine
      CUDA = NVIDIA GPU — fastest on cloud/Linux
      CPU  = fallback — works everywhere, slower

    truncation=True, max_length=512:
      BERT has a 512-token limit (hard architectural constraint — positional
      embeddings only go to 512). Earnings texts are often 200-400 tokens.
      truncation=True silently cuts anything beyond 512. In practice, the
      first 512 tokens carry the most important information in earnings texts
      (headline metrics always come first).
    """
    from transformers import pipeline
    import torch

    if torch.backends.mps.is_available():
        device_id = "mps"
        device_label = "Apple MPS (GPU)"
    elif torch.cuda.is_available():
        device_id = "cuda"
        device_label = "NVIDIA CUDA (GPU)"
    else:
        device_id = "cpu"
        device_label = "CPU"

    logger.info("Loading ProsusAI/finbert on %s...", device_label)

    pipe = pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=device_id,
        truncation=True,
        max_length=512,
        top_k=None,          # return scores for ALL 3 labels (not just top)
    )

    logger.info("FinBERT loaded. Labels: positive / neutral / negative")
    return pipe


# ═══════════════════════════════════════════════════════════════
# STEP 2: Score texts
# ═══════════════════════════════════════════════════════════════
def score_texts(pipe, texts: list) -> pd.DataFrame:
    """
    Run FinBERT on all earnings texts. Returns one row per text with:
      - raw scores for all 3 labels (softmax probabilities, sum to 1.0)
      - winning label + confidence
      - sentiment_score: continuous [-1, +1]
            = P(positive) - P(negative)
            Useful for ranking stocks by sentiment intensity.

    Why top_k=None?
      Without it, pipeline returns only the top label.
      We want all 3 scores because P(neutral)=0.6 vs P(neutral)=0.3
      are very different — high neutral confidence = genuinely ambiguous,
      low neutral = the model was split between positive/negative.

    Why sentiment_score = P(pos) - P(neg)?
      Converts the 3-class output to a single scalar on [-1, +1].
      P(pos)=0.8, P(neg)=0.05 → score = +0.75 (strong positive)
      P(pos)=0.1, P(neg)=0.75 → score = -0.65 (strong negative)
      P(pos)=0.35, P(neg)=0.30 → score = +0.05 (nearly neutral)
      This scalar plugs directly into position multiplier logic.
    """
    results = []
    raw_texts = [t["text"] for t in texts]

    logger.info("Scoring %d texts through FinBERT...", len(raw_texts))
    outputs = pipe(raw_texts)  # list of list of dicts: [[{label, score}, ...], ...]

    for meta, label_scores in zip(texts, outputs):
        # label_scores = [{'label': 'positive', 'score': 0.82},
        #                 {'label': 'negative', 'score': 0.11},
        #                 {'label': 'neutral',  'score': 0.07}]
        score_dict = {d["label"]: d["score"] for d in label_scores}

        p_pos = score_dict.get("positive", 0.0)
        p_neu = score_dict.get("neutral",  0.0)
        p_neg = score_dict.get("negative", 0.0)

        top_label = max(score_dict, key=score_dict.get)
        confidence = score_dict[top_label]
        sentiment_score = p_pos - p_neg   # continuous [-1, +1]

        results.append({
            "ticker":          meta["ticker"],
            "date":            meta["date"],
            "source":          meta["source"],
            "text_preview":    meta["text"][:80] + "...",
            "p_positive":      round(p_pos, 4),
            "p_neutral":       round(p_neu, 4),
            "p_negative":      round(p_neg, 4),
            "sentiment":       top_label,
            "confidence":      round(confidence, 4),
            "sentiment_score": round(sentiment_score, 4),
            "multiplier":      sentiment_to_multiplier(sentiment_score),
        })

    df = pd.DataFrame(results)
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3: Log results
# ═══════════════════════════════════════════════════════════════
def log_sentiment_results(df: pd.DataFrame) -> None:
    logger.info("\n── FinBERT Sentiment Results ─────────────────────────────────────")
    logger.info("  %-12s  %-10s  %-8s  %-8s  %-8s  %-10s  %-6s  %s",
                "Ticker", "Sentiment", "P(pos)", "P(neu)", "P(neg)", "Score", "Mult", "Source")
    logger.info("  " + "─" * 88)

    for _, row in df.sort_values("sentiment_score", ascending=False).iterrows():
        sentiment_icon = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(row["sentiment"], "⚪")
        logger.info("  %-12s  %s %-8s  %-8.3f  %-8.3f  %-8.3f  %-10.3f  %-6.1f  %s",
                    row["ticker"],
                    sentiment_icon, row["sentiment"],
                    row["p_positive"], row["p_neutral"], row["p_negative"],
                    row["sentiment_score"], row["multiplier"],
                    row["source"])

    logger.info("\n  Distribution: %s positive, %s neutral, %s negative",
                (df["sentiment"] == "positive").sum(),
                (df["sentiment"] == "neutral").sum(),
                (df["sentiment"] == "negative").sum())


# ═══════════════════════════════════════════════════════════════
# STEP 4: Gate M5 Kelly positions
# ═══════════════════════════════════════════════════════════════
def gate_kelly_positions(sentiment_df: pd.DataFrame) -> dict:
    """
    Apply FinBERT sentiment multipliers to M5 Kelly positions.

    For pairs trades:
      The pair's sentiment = minimum of the two stocks' multipliers.
      Why minimum? If BAJFINANCE is negative (mult=0.0) and HDFCBANK is
      positive (mult=1.0), the pairs trade is still risky — bad news on
      one leg disrupts the spread reversion. Take the conservative view.

    For factor bets (single stock):
      Directly multiply Kelly position by the stock's sentiment multiplier.
    """
    # Build ticker → multiplier lookup
    sentiment_map = dict(zip(sentiment_df["ticker"], sentiment_df["multiplier"]))
    sentiment_score_map = dict(zip(sentiment_df["ticker"], sentiment_df["sentiment_score"]))

    results = {"pairs": [], "factor": []}

    # ── Gate pairs ─────────────────────────────────────────────
    pairs_path = DATA_DIR / "kelly_positions_pairs.csv"
    if pairs_path.exists():
        pairs_df = pd.read_csv(pairs_path)
        logger.info("\n── Pairs Position Gating ────────────────────────────────────────")
        logger.info("  %-28s  %-8s  %-10s  %-10s  %-10s  %s",
                    "Pair", "Signal", "Base A%", "Gated A%", "Multiplier", "Reason")
        logger.info("  " + "─" * 80)

        for _, row in pairs_df.iterrows():
            mult_a = sentiment_map.get(row["stock_a"], 0.7)   # default neutral
            mult_b = sentiment_map.get(row["stock_b"], 0.7)
            pair_mult = min(mult_a, mult_b)   # conservative: worst leg dominates

            gated_a = round(row["pos_a_pct"] * pair_mult, 3)
            gated_b = round(row["pos_b_pct"] * pair_mult, 3)

            reason = (
                f"{row['stock_a']}={mult_a:.1f}, {row['stock_b']}={mult_b:.1f} "
                f"→ min={pair_mult:.1f}"
            )

            logger.info("  %-28s  %-8s  %-10.3f  %-10.3f  %-10.1f  %s",
                        f"{row['stock_a']}/{row['stock_b']}",
                        row["signal"], row["pos_a_pct"], gated_a, pair_mult, reason)

            results["pairs"].append({
                **row.to_dict(),
                "sentiment_mult": pair_mult,
                "gated_pos_a_pct": gated_a,
                "gated_pos_b_pct": gated_b,
            })

    # ── Gate factor bets ────────────────────────────────────────
    factor_path = DATA_DIR / "kelly_positions_factor.csv"
    if factor_path.exists():
        factor_df = pd.read_csv(factor_path)
        logger.info("\n── Factor Position Gating ───────────────────────────────────────")
        logger.info("  %-12s  %-8s  %-10s  %-10s  %-10s  %s",
                    "Ticker", "Action", "Base %", "Gated %", "Sentiment", "Score")
        logger.info("  " + "─" * 70)

        for _, row in factor_df.iterrows():
            mult  = sentiment_map.get(row["ticker"], 0.7)
            score = sentiment_score_map.get(row["ticker"], 0.0)
            gated = round(row["pos_pct"] * mult, 3)

            sentiment_label = {1.0: "positive", 0.7: "neutral", 0.0: "negative"}.get(mult, "neutral")
            icon = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(sentiment_label, "⚪")

            logger.info("  %-12s  %-8s  %-10.3f  %-10.3f  %s %-8s  %.3f",
                        row["ticker"], row.get("action", "SKIP"),
                        row["pos_pct"], gated,
                        icon, sentiment_label, score)

            results["factor"].append({
                **row.to_dict(),
                "sentiment_mult":   mult,
                "sentiment_label":  sentiment_label,
                "sentiment_score":  score,
                "gated_pos_pct":    gated,
            })

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 5: Final portfolio view
# ═══════════════════════════════════════════════════════════════
def final_portfolio_view(gated: dict, regime_name: str) -> None:
    logger.info("\n── FINAL GATED PORTFOLIO ─────────────────────────────────────────")
    logger.info("  Three-layer gate: Regime(%s) → Kelly → FinBERT", regime_name)
    logger.info("")

    active = []

    for row in gated["pairs"]:
        if row["signal"] not in ("FLAT", "EXIT") and (
            row["gated_pos_a_pct"] > 0 or row["gated_pos_b_pct"] > 0
        ):
            active.append(f"  PAIRS  {row['stock_a']:12s} {row['signal']:10s} "
                          f"A={row['gated_pos_a_pct']:.2f}%  B={row['gated_pos_b_pct']:.2f}%")

    for row in gated["factor"]:
        if row.get("action") in ("LONG", "SHORT") and row["gated_pos_pct"] > 0:
            active.append(f"  FACTOR {row['ticker']:12s} {row['action']:10s} "
                          f"{row['gated_pos_pct']:.2f}%  "
                          f"(sentiment: {row['sentiment_label']})")

    if active:
        for line in active:
            logger.info(line)
    else:
        logger.info("  No active positions — all signals flat or zeroed by gate.")
        logger.info("  Engine is monitoring. Next check: tomorrow 09:15 IST.")

    all_factor = [r["gated_pos_pct"] for r in gated["factor"]
                  if r.get("action") in ("LONG", "SHORT")]
    total = sum(all_factor)
    logger.info("")
    logger.info("  Total deployed: %.2f%%  |  Cash: %.2f%%", total, 100 - total)


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_finbert_pipeline() -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("FINBERT SENTIMENT GATING — M6")
    logger.info("=" * 70)

    # Get regime for context
    regime_name = "Sideways"
    try:
        reg = pd.read_csv(DATA_DIR / "regime_labels.csv", index_col=0, parse_dates=True)
        regime_name = reg["regime_name"].iloc[-1]
    except Exception:
        pass

    # STEP 0: Fetch live news — fall back to static for failed tickers
    live_texts, failed_tickers = fetch_live_news(max_per_ticker=3)

    # For tickers that failed live fetch, use static fallback texts
    fallback_texts = [t for t in EARNINGS_TEXTS if t["ticker"] in failed_tickers]
    if fallback_texts:
        logger.info("Using static fallback for %d tickers: %s",
                    len(failed_tickers), failed_tickers)

    all_texts = live_texts + fallback_texts

    if not all_texts:
        logger.error("No texts available (live + fallback). Aborting.")
        return pd.DataFrame()

    # Load model
    pipe = load_finbert()

    # Score ALL texts (live articles + fallback)
    raw_scored = score_texts(pipe, all_texts)

    # Aggregate multiple articles per ticker → one row per ticker
    sentiment_df = aggregate_ticker_sentiments(raw_scored)

    # Fill in any tickers not covered at all
    covered = set(sentiment_df["ticker"])
    universe = set(NSE_TICKERS.keys())
    missing  = universe - covered
    if missing:
        logger.warning("No sentiment data for: %s — defaulting to neutral (0.7×)", missing)
        neutral_rows = pd.DataFrame([{
            "ticker": t, "p_positive": 0.33, "p_neutral": 0.34, "p_negative": 0.33,
            "sentiment_score": 0.0, "n_articles": 0, "latest_date": str(date.today()),
            "source": "default neutral", "sentiment": "neutral",
            "confidence": 0.34, "multiplier": 0.7, "text_preview": "(no data)"
        } for t in missing])
        sentiment_df = pd.concat([sentiment_df, neutral_rows], ignore_index=True)

    # Save raw per-article scores too
    raw_scored.to_csv(DATA_DIR / "finbert_articles_raw.csv", index=False)

    # Log aggregated results
    log_sentiment_results(sentiment_df)

    # Gate positions
    gated = gate_kelly_positions(sentiment_df)

    # Final view
    final_portfolio_view(gated, regime_name)

    # Save
    sentiment_df.to_csv(DATA_DIR / "finbert_sentiment.csv", index=False)

    if gated["pairs"]:
        pd.DataFrame(gated["pairs"]).to_csv(
            DATA_DIR / "kelly_positions_pairs_gated.csv", index=False)
    if gated["factor"]:
        pd.DataFrame(gated["factor"]).to_csv(
            DATA_DIR / "kelly_positions_factor_gated.csv", index=False)

    logger.info("\n── SAVED ────────────────────────────────────────────────────────")
    logger.info("  finbert_articles_raw.csv        — %d articles scored", len(raw_scored))
    logger.info("  finbert_sentiment.csv           — %d tickers (aggregated)", len(sentiment_df))
    logger.info("  kelly_positions_pairs_gated.csv — sentiment-adjusted pairs")
    logger.info("  kelly_positions_factor_gated.csv— sentiment-adjusted factor bets")
    logger.info("  → Downstream: M10 (Alpaca execution)")
    logger.info("=" * 70)

    return sentiment_df


def apply_gate_only(max_age_days: int = 5):
    """
    Bug 21 root-cause fix (2026-06-12).

    The daily cron runs `main.py --skip-finbert`, which previously meant the
    gated position files were NEVER regenerated — M10 consumed a gated file
    frozen at the last manual FinBERT run while M5's Kelly sizing moved on
    daily. This function decouples the CHEAP step (applying the sentiment
    gate to today's Kelly output) from the EXPENSIVE step (fetching news and
    running the FinBERT model).

    It reuses the last saved sentiment scores. If they are older than
    `max_age_days`, every multiplier is reset to a conservative neutral so
    stale opinions cannot tilt fresh positions — the gate still runs, the
    file is still fresh, and the M10 staleness guard passes for the right
    reason.
    """
    sent_path = DATA_DIR / "finbert_sentiment.csv"
    if not sent_path.exists():
        logger.error("apply_gate_only: finbert_sentiment.csv missing — "
                     "run the full FinBERT pipeline once first.")
        return None

    sentiment_df = pd.read_csv(sent_path)
    age_days = (pd.Timestamp.now()
                - pd.Timestamp.fromtimestamp(sent_path.stat().st_mtime)).days
    if age_days > max_age_days:
        logger.warning("apply_gate_only: sentiment is %dd old (> %dd) — "
                       "neutralising multipliers (0.85).", age_days, max_age_days)
        sentiment_df["multiplier"] = 0.85
        sentiment_df["sentiment"]  = "stale-neutral"

    gated = gate_kelly_positions(sentiment_df)

    if gated["pairs"]:
        pd.DataFrame(gated["pairs"]).to_csv(
            DATA_DIR / "kelly_positions_pairs_gated.csv", index=False)
    if gated["factor"]:
        pd.DataFrame(gated["factor"]).to_csv(
            DATA_DIR / "kelly_positions_factor_gated.csv", index=False)

    logger.info("apply_gate_only: gated files regenerated from today's Kelly "
                "output using %dd-old sentiment.", age_days)
    return gated


if __name__ == "__main__":
    df = run_finbert_pipeline()
    print("\n── Sentiment Summary ──")
    print(df[["ticker", "sentiment", "confidence", "sentiment_score", "multiplier"
              ]].to_string(index=False))
