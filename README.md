# Alpha-Core — Factor Signal Engine

**Part 2 of 3 in a connected quant system: Indian Risk Engine → Alpha-Core → Portfolio Optimizer**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Alpaca](https://img.shields.io/badge/Alpaca-Paper%20Trading-green)](https://alpaca.markets)

---

## What it does

Ten-module factor engine that generates daily trading signals for 14 NSE stocks and routes them to a live Alpaca paper trading account via US ETF proxies.

```
Fama-French alpha (M1)
    → HMM regime detection (M4)
    → Kelly position sizing (M5)
    → FinBERT sentiment gate (M6)
    → XGBoost IC direction (M7)
    → NSE→ETF proxy mapping
    → Alpaca paper execution (M10)
```

## Modules

| Module | File | What it does |
|---|---|---|
| M1 | `fama_french.py` | FF3 factor regression, daily alpha + t-stat per stock |
| M4 | `hmm_regime.py` | GaussianHMM → Bull / Bear / Sideways regime |
| M5 | `kelly_sizing.py` | Kelly criterion, t-stat ≥ 1.5 filter, halved in Bear |
| M6 | `finbert_sentiment.py` | FinBERT news sentiment gate on XGBoost signal |
| M7 | `xgb_predictor.py` | XGBoost residual predictor, IC computation, SHAP |
| M10 | `alpaca_gate.py` | NSE signal → US ETF proxy → live Alpaca paper order |

## NSE → US ETF Proxy Mapping

| NSE Stock | ETF Proxy | Rationale |
|---|---|---|
| HDFCBANK, ICICIBANK, AXISBANK | XLF | US Financials |
| INFY, TCS, WIPRO | QQQ | Nasdaq-100 (IT) |
| RELIANCE, ONGC | XLE | US Energy |
| SUNPHARMA, DRREDDY | XLV | US Healthcare |
| MARUTI | XLY | Consumer Discretionary |
| HINDUNILVR, ITC | XLP | Consumer Staples |

*Documented limitation: Alpaca supports US equities only. Proxies validate the execution infrastructure, not direct NSE trading.*

## Run

```bash
python main.py                    # full M1-M10 pipeline
python main.py --skip-finbert     # skip live FinBERT fetch
python -m alpha_core.alpaca_gate  # execution gate only
```

## Live output (May 26, 2026)

```
HMM Regime:   Bear
Active trade: BUY 23 × XLV (SUNPHARMA proxy)
              order_id: d3aaf07f — IC=0.185, t-stat=1.65 ✓
All others:   FLAT — Bear regime, t-stat < 1.5
```

## System architecture

```
Indian Risk Engine (indian-risk-engine)  → DCC covariance, CVaR signals
Alpha-Core (this repo)                   → factor signals, regime, IC, Alpaca execution
Portfolio Optimizer (ml-portfolio-optimizer) → Black-Litterman allocation, backtester
main.py (Portfolio Optimizer)            → orchestrates all three: one command, live report
```
