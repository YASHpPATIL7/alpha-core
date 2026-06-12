# Regime-Conditional Factor Premia and Institutional Flow Dynamics in Indian Equities

**Working Paper — SSRN Submission Draft**
Yash Patil | June 2026
*Preliminary — comments welcome*

---

## Abstract

We document strongly regime-conditional factor premia in Indian equity markets
and examine whether daily foreign institutional investor (FII) flows explain
the underlying dynamics. Using the IIM Ahmedabad survivorship-bias-adjusted
factor library (Agarwalla, Jacob, and Varma, 2013) spanning 2019–2025, we
identify Bull, Sideways, and Bear market states via a Hidden Markov Model and
characterise factor performance within each state. The momentum premium (WML)
concentrates in Sideways regimes (Sharpe ratio 1.667, t = 3.24), while the
market premium concentrates in Bull regimes (Sharpe ratio 1.485, t = 2.28).
Value (HML) is directionally positive in Bear regimes (Sharpe ratio 2.26),
consistent with flight-to-quality rotation, though statistical inference is
limited by the sparse Bear sample (n = 121 days). We next test whether daily
NSDL FII net flows Granger-cause individual factor premia across 15 regime-factor
pairs, applying Benjamini-Hochberg false discovery rate correction. No cell
survives — FII aggregate flows do not predict factor premia in any regime. A
buy-sell decomposition reveals that gross FII volume (both purchases and sales)
predicts aggregate market returns specifically in Bear regimes (F ≈ 5, p ≈ 0.009),
consistent with FII acting as the marginal price setter when domestic liquidity
thins. Realized pairwise correlation (DCC) rises significantly in Bear (mean 0.30
vs. 0.25 in Bull; t = 22.9), confirming diversification failure coincides with
regime transitions rather than leading them. A regime-aware factor allocation
strategy achieves Sharpe 0.794 versus 0.733 for buy-and-hold over the 2021–2026
walk-forward period.

**JEL Classification:** G11, G12, G15, G23
**Keywords:** factor investing, regime-conditional returns, Hidden Markov Model,
Granger causality, FII flows, Indian equities, momentum, DCC correlation,
institutional investors

---

## 1. Introduction

Factor investing — the systematic harvesting of return premia associated with
market beta, size, value, momentum, profitability, and investment — is
extensively documented in developed markets (Fama and French, 1993, 2015;
Carhart, 1997) but evidence for Indian equities remains limited in two respects.
First, most Indian studies use monthly data, precluding analysis of regime-specific
dynamics through which premia are transmitted. Second, the role of foreign
institutional investors (FIIs) — who account for roughly 20–25% of NSE cash
market turnover — has not been examined at the factor level.

A growing practitioner literature documents that factor premia vary substantially
across market regimes. Naik, Devarajan, Nowobilski, Page, and Pedersen (2016)
identify regime-dependency as a first-order concern for systematic strategies:
correlation structure and premia concentration shift materially between bull,
sideways, and bear environments, and static factor allocations that ignore the
business cycle leave substantial risk-adjusted return on the table. Systematic
approaches that condition factor exposure on estimated market state have been
shown to improve risk-adjusted performance relative to static
factor allocations. Our paper brings this framework to the Indian market using a
rigorous, returns-based regime identification approach.

The motivation for examining FII flows is both theoretical and practical.
Theoretically, if FIIs trade on factor-level signals, their daily flows should
carry incremental predictive power for specific factor premia beyond the factor's
own history. Practically, if the predictability concentrates in Bear regimes —
when systematic strategies are most exposed and premia are most volatile —
a flow-based early warning system has direct value for Indian AMCs and systematic
portfolio managers. We test this hypothesis rigorously using gross flow data from
the NSDL FPI archive, which allows decomposition into purchases and sales
separately — a distinction most prior literature cannot make.

Our main contributions are as follows. First, we provide the most comprehensive
regime-conditional factor analysis for Indian equities to date, using the
canonical IIM Ahmedabad factor library at daily frequency. Second, we conduct a
rigorous 15-test Granger causality analysis between FII flows and factor premia,
applying Benjamini-Hochberg false discovery rate correction — a methodological
standard rarely applied in Indian market microstructure research. Third, our
buy-sell decomposition reveals a regime-asymmetric market predictability finding
that net-flow studies cannot uncover. Fourth, we validate all factor results
against own-constructed daily factors from the NSE universe and quantify the
sensitivity to factor construction methodology.

The remainder of the paper is organized as follows. Section 2 describes the data.
Section 3 presents the HMM regime identification. Section 4 documents
regime-conditional factor premia. Section 5 tests FII flows as factor predictors.
Section 6 presents the buy-sell decomposition. Section 7 describes the
regime-aware strategy. Section 8 reports robustness checks. Section 9 discusses
limitations. Section 10 concludes.

---

## 2. Data

### 2.1 Factor Returns — Primary Dataset

Our primary factor return data are the IIM Ahmedabad Fama-French and Momentum
factor series (Agarwalla, Jacob, and Varma, 2013), downloaded from
https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/. We use the
daily, survivorship-bias-adjusted series (release 2025-12), which covers
October 1993 through December 2025. The IIMA library provides four factors:
MKT (market excess return over the risk-free rate), SMB (size), HML (value),
and WML (momentum). Factors are constructed from the full BSE/NSE universe
with micro-cap and penny-stock filters and point-in-time accounting data from
CMIE Prowess. The IIMA construction methodology is peer-reviewed, publicly
documented, and constitutes the standard benchmark for Indian factor research.

We restrict our analysis to the period January 2019–December 2025, yielding
1,669 common trading days after alignment with regime labels and FII data.
Full-sample factor returns are reported in Table 1.

### 2.2 Factor Returns — Robustness Dataset

As a secondary dataset, we construct own daily factors from a 14-stock NSE
blue-chip universe (Nifty 50 constituents selected for data completeness) using
constituent price data and annual-report fundamentals sourced from yfinance, with
a six-month post-fiscal-year-end lag for point-in-time compliance. This dataset
provides MKT, SMB, HML, RMW (profitability), and CMA (investment) at daily
frequency — enabling a five-factor comparison not available in the IIMA library.
Cross-validation against IIMA shows high correlation for MKT (r = 0.907), but
near-zero correlation for SMB (r = −0.18) and HML (r = 0.01). We attribute this
divergence to the absence of true size and value dispersion in a 14-stock
large-capitalisation universe; no stock in this set qualifies as genuinely small
or value relative to the full BSE universe from which IIMA constructs its
long-short SMB and HML portfolios. We treat own-constructed factors as
illustrative of daily factor dynamics and report regime results in Appendix A2;
IIMA factors constitute the primary analysis.

### 2.3 FII Daily Flow

We obtain daily FII net equity investment from the NSDL FPI investment activity
archive (https://www.fpi.nsdl.co.in), spanning January 2019 to June 2026 —
1,939 trading days. Data cover gross purchases, gross sales, and net investment
for equity cash markets, in ₹ crore (1 crore = ₹10 million). We normalise
daily net flow by a 252-day rolling mean of absolute flow, producing a
dimensionless signal comparable across time as market capitalisation grew. We
verify stationarity via augmented Dickey-Fuller tests. Gross flow decomposition
(purchases and sales separately) is preserved for Section 6.

### 2.4 Market Index and VIX

For HMM estimation and the MKT factor, we use Nifty 50 daily log returns.
India VIX daily close data is sourced from NSE India (2019–2026), providing
1,734 trading days. We use log-differenced VIX for Granger tests following
ADF confirmation of non-stationarity in levels.

### 2.5 Sample Alignment

After aligning all series on common trading days, the primary analysis covers
1,669 days (January 2019 – December 2025). Regime distribution: Bull 593 days
(35.5%), Sideways 955 days (57.2%), Bear 121 days (7.3%).

---

## 3. HMM Regime Identification

We model the daily return-volatility state as a latent variable following a
three-state first-order Hidden Markov process. Observable emissions are daily
Nifty 50 log return, India VIX, and 20-day momentum (z-scored), with Gaussian
emissions within each state. We label states post-hoc by state-conditional means:
the high-return, low-VIX state is Bull; the high-VIX, low-return state is Bear;
the intermediate state is Sideways.

**Causal identification.** We use filtered (forward-pass only) regime
probabilities throughout to avoid look-ahead bias inherent in full-sample
Viterbi smoothing. Regime assignments condition on prior-day labels for all
regressions. Verification checks confirm: COVID-19 crash (March 23, 2020) is
correctly labeled Bear; 2021 bull market is predominantly Bull/Sideways; Bear
VIX (mean 28.3) exceeds Bull VIX (mean 14.5); regime persistence diagonals
exceed 0.85 for all three states.

Realized pairwise correlation from a DCC(1,1) model (Engle, 2002) estimated
on GARCH(1,1)-standardized returns of 14 NSE stocks confirms the regime
identification: Bear-regime average pairwise DCC correlation (mean = 0.297)
is significantly higher than Bull (mean = 0.248; t = 22.9, p < 0.001),
consistent with the "all correlations go to 1 in a crash" phenomenon. Notably,
India VIX Granger-causes DCC average correlation (F = 6.99, p < 0.001) but
not vice versa (F = 1.05, p = 0.31), confirming that the options market prices
correlation stress before it appears in realized return comovement. Practitioners
using realized correlation as a real-time risk trigger are therefore systematically
late relative to the implied volatility signal.

---

## 4. Regime-Conditional Factor Premia

**[Table 1 — Full-Sample IIMA Factor Premia]**

Over the full 2019–2025 sample, the market (MKT, 12.05% annualised, SR = 0.726†)
and value (HML, 10.28%, SR = 0.781*) factors carry the largest risk-adjusted
premia. The momentum factor (WML, 14.72%, SR = 1.080**) is the sole factor
achieving conventional statistical significance at the 5% level in the full
sample. SMB is economically small and insignificant (−0.39%, SR = −0.032).

**[Table 2 — Regime-Conditional IIMA Factor Premia]**

Conditioning on HMM regime reveals substantial heterogeneity suppressed by
full-sample averages.

**Bull regime (n = 593).** The market premium concentrates in Bull: MKT earns
23.59% annualised (SR = 1.485, t = 2.28*). Momentum continues to earn
(WML: 10.60%, SR = 0.852) but is not statistically significant within regime.
Value (HML) and size (SMB) premia are economically moderate and insignificant.

**Sideways regime (n = 955).** The momentum premium dominates: WML earns
20.77% annualised (SR = 1.667, t = 3.24**) — the strongest and most
statistically significant regime-factor result in our analysis.¹ This finding
is economically intuitive: momentum strategies rely on trend persistence
without reversal, which characterises directionless but trending markets.
The result survives exclusion of the COVID-19 window (estimated SR = 1.59,
t = 2.89**). MKT is near zero (1.65%) and insignificant in Sideways, consistent
with the absence of a broad directional trend.

¹ *Note on identification.* The HMM emission variables include Nifty 50 20-day
time-series momentum (the index's own trailing return), used to distinguish
Sideways from Bull/Bear. WML is the IIMA cross-sectional long-short momentum
factor, constructed from individual stock relative returns. These are distinct
objects: the emission indexes the market's direction; WML captures dispersion in
stock-level return continuation. Regime labeling is driven primarily by VIX level
and the market return component, not WML return itself. The WML-Sideways result
is therefore not a mechanical artifact of the regime definition.

**Bear regime (n = 121).** Statistical inference is limited by the sparse Bear
sample; no factor achieves significance within regime. Directionally, HML is
positive (45.29% annualised, SR = 2.256), consistent with institutional
rotation toward cheap, low-valuation stocks during drawdowns — a flight-to-quality
into value. SMB is directionally negative (−32.91%, SR = −1.520), consistent
with flight from small-capitalisation names to liquid large-caps. MKT
(37.55%, SR = 1.003) is positive but driven by volatile recovery days within
the Bear window; the Bear classification captures high-VIX periods including
both crash and early-recovery phases. WML is directionally negative
(−12.89%, SR = −0.536), consistent with momentum crashes during market
dislocations documented by Daniel and Moskowitz (2016).

**Summary.** The largest regime-conditional Sharpe ratio spread across all
factor-regime pairs is for WML (spread = 2.20, Sideways high vs. Bear low)
and MKT (spread = 1.35, Bull high vs. Sideways low). Factor premia are not
stationary over the business cycle — they concentrate in specific regime
states in a pattern that is both statistically detectable and economically
interpretable.

---

## 5. FII Flows and Factor Premia: Granger Causality

### 5.1 Framework

We test Granger causality in bivariate VAR systems. For each factor F and
regime R, we estimate:

F_t = α + Σ β_k F_{t-k} + Σ γ_k FII_{t-k} + ε_t

restricted to observations within regime R. The null H₀: γ₁ = ⋯ = γ_p = 0.
We use SSR F-tests with Newey-West HAC standard errors (bandwidth = 5) to
account for autocorrelation and heteroskedasticity. Lag order p is AIC-selected
within each pair (p_max = 5). Minimum 80 observations required per regime-factor
cell.

**Multiple testing.** The Panel B grid is 15 simultaneous tests (5 factors ×
3 regimes). We apply Benjamini-Hochberg FDR correction jointly at q = 0.05.
Only FDR-surviving cells constitute confirmed findings.

### 5.2 Panel A — FII Net Flow vs. Aggregate Market Return

FII net flow does not Granger-cause aggregate MKT returns in any regime or
in the pooled sample. Pooled: F = 1.07, p = 0.30. Within regimes: Bull
(F = 0.47, p = 0.49), Sideways (F = 1.00, p = 0.37), Bear (F = 1.43,
p = 0.23). This null eliminates the possibility that Panel B results
reflect a broad market-level flow relationship.

### 5.3 Panel B — FII Net Flow vs. Factor Premia (3 × 5 Grid)

**[Table 3 — FII→Factor Granger Causality Matrix]**

Across 15 tests, Bear × SMB (F = 8.21, p_raw = 0.005) and Bear × CMA
(F = 6.98, p_raw = 0.009) exhibit the strongest raw signals. After
Benjamini-Hochberg correction, neither survives (BH threshold for rank-1:
0.05 × 1/15 = 0.0033). All 15 cells are null after FDR adjustment.

This complete null is a substantive finding. The large regime-conditional
factor patterns documented in Section 4 are not FII-flow driven. If foreign
institutional trading drove regime-conditional premia, FII flows would
Granger-cause the corresponding factor returns. The data reject this
hypothesis across all 15 tests. The mechanism behind regime-conditional
factor dynamics in India operates through channels beyond daily aggregate
FII net investment — most likely domestic participation structure, liquidity
segmentation, and macroeconomic regime shifts that affect all market
participants simultaneously.

### 5.4 Panel C — FII as Bear Regime Early Warning

A logistic model using lagged FII flows (lags 1–5) to predict Bear regime
onset within a 5-day forward window produces AUC-ROC = 0.548 and McFadden
R² = 0.004. No individual lag coefficient is statistically significant.
FII flows do not function as an early warning system for Bear onset at the
1–5 day horizon. This null complements Panel B: FII flows neither predict
individual factor premia nor signal imminent regime transitions.

---

## 6. Buy-Sell Decomposition: Gross Flow as Bear-Regime Price Impact

### 6.1 Motivation

Net FII flow aggregates gross purchases and sales, potentially discarding
information. We separately test gross purchases and gross sales as predictors
of MKT returns within each regime.

### 6.2 Results

**[Table 3, Panel D — Gross Buy vs. Gross Sell → MKT by Regime]**

In Bear regimes, both gross FII purchases (F = 5.06, p = 0.008, lag = 2)
and gross FII sales (F = 4.89, p = 0.009, lag = 2) Granger-cause aggregate
market returns. No such predictability exists in Bull (both p > 0.16) or
Sideways (both p > 0.05). The pooled full-sample tests are insignificant
(both p > 0.10), explaining why net-flow studies miss this result.

Three aspects of this finding deserve emphasis.

First, the predictability is Bear-specific. In calm regimes, domestic
participation absorbs FII order flow without systematic price impact. In Bear,
domestic liquidity withdrawal concentrates price discovery in the FII segment.
Large FII trades move prices and the impact persists for two trading days —
consistent with evidence on institutional price impact during dislocations
(Brunnermeier and Pedersen, 2009).

Second, the symmetry between gross purchase and gross sale predictability
argues against information asymmetry (Myers and Majluf, 1984) as the
mechanism. Informed exits would produce sell-dominant predictability; instead,
both sides carry equal predictive content. The informative signal is FII
trading volume — not direction. This is consistent with FII acting as the
marginal price setter under stress rather than as an informed trader.

Third, this finding would be invisible using net flow data alone. Net flow
cancels the gross components; by aggregating, the volume signal in Panel D
is averaged away. Our gross flow decomposition — using NSDL's published
purchase/sale breakdown — is what makes this result detectable.

---

## 7. Regime-Aware Factor Allocation Strategy

A walk-forward strategy allocating to MVO (Max-Sharpe) weights in Bull,
HRP weights in Sideways, and equal weights in Bear achieves the following
performance over 2021–2026 (Table 4):

| Strategy | Total Ret% | Ann Ret% | Sharpe | MaxDD% | Calmar |
|----------|-----------|---------|--------|--------|--------|
| Regime-Aware | 65.7 | 10.75 | 0.794 | −17.5% | 0.615 |
| Buy-and-Hold | 57.7 | 9.75 | 0.733 | −18.0% | 0.543 |
| Static HRP | 53.4 | 9.15 | 0.714 | −18.6% | 0.492 |
| MVO MaxSharpe | 102.4 | 14.78 | 1.018 | −15.0% | 0.988 |

Transaction costs of 0.25% one-way are applied at each monthly rebalance.
The regime-aware strategy improves Sharpe by 6.1 basis points over
buy-and-hold and reduces maximum drawdown by 0.5 percentage points. The
MVO strategy dominates over this period — suggesting that, conditional on
the 2021–2026 sample, the mean-variance optimization benefits from the
specific return structure more than the regime signal.

Regime-aware turnover (317% annualised) is high and represents a practical
constraint for strategies with significant AUM. This finding motivates lower-
frequency regime conditioning (monthly rather than daily), which we consider
an extension of the current work.

---

## 8. Robustness

**COVID exclusion.** We replicate all regime-conditional factor tests
excluding the COVID-19 crisis window (2020-02-01 to 2020-06-30). The
MOM-Sideways result survives (SR = 1.59, t = 2.89**). All Panel B Granger
tests remain null after FDR correction. Panel D Bear gross-flow finding
holds (both p < 0.02 post-COVID exclusion). Results are not crisis-driven.

**Alternative factor data.** Appendix A1 reports regime-conditional results
using own-constructed daily factors (MKT, SMB, HML, RMW, CMA) from the
NSE 200 universe. MKT results are consistent with IIMA (r = 0.907 between
series). SMB and HML diverge substantially (r = −0.18 and 0.01 respectively),
attributable to the absence of true small-cap and value-dispersion in the
blue-chip universe. We use IIMA as primary and flag own-constructed SMB/HML
results as unreliable in Appendix A2.

**HAC standard errors.** All F-statistics throughout use Newey-West HAC
covariance (bandwidth = 5). Full-sample results assuming homoskedastic errors
are reported in Appendix A4 for comparison; the correction is material in
several high-autocorrelation factor series.

**Alternative regime definition.** As a robustness check, we define Bull
(Bear) as periods where the 63-day rolling Nifty return exceeds (falls below)
+10% (−10%), with Sideways as the remainder. The MOM-Sideways direction
persists; the Bear sample under this definition is n = 89, further limiting
inference on Bear-specific results.

---

## 9. Limitations

**Short sample and Bear sparsity.** Our primary sample spans 2019–2025
(six years). The Bear regime covers only 121 trading days, concentrated
in the COVID-19 crash (2020) and the 2022 global tightening episode.
Bear-regime results lack statistical power; effect size estimates (Sharpe
ratios) are unstable and should be interpreted directionally, not as
precise parameter estimates.

**Factor construction.** IIMA factors use point-in-time accounting data
and a broad BSE/NSE universe with micro-cap filters. Our own-constructed
factors use yfinance fundamentals and a 14-stock NSE blue-chip universe,
producing near-zero SMB and HML correlation with IIMA equivalents. This
divergence underscores the importance of the robustness check and the
hazard of constructing size/value factors from a restricted large-cap
universe. IIMA does not provide RMW or CMA; we cannot benchmark the
profitability and investment factors against a canonical Indian source.

**Aggregate flow data.** We observe total daily FII net equity investment,
not individual fund or stock-level flows. Attribution of the Bear gross-flow
predictability to specific investment behaviors (momentum rebalancing, risk
parity deleveraging, mandate-driven redemptions) is interpretive. Verification
using quarterly NSDL FPI stock-level holdings data is a natural extension.

**Survivorship bias.** Our NSE universe excludes stocks delisted over
2019–2026 (YES Bank restructuring, IL&FS subsidiaries). This bias inflates
value and profitability premia directionally, making our null results for
these factors conservative and any positive findings potentially overstated.

**DCC as concurrent indicator.** The DCC average pairwise correlation rises
significantly in Bear regimes (mean 0.30 vs. 0.25 in Bull), confirming
realized diversification failure. However, DCC is a concurrent rather than
leading indicator — India VIX Granger-causes DCC but not vice versa,
indicating options markets price stress before realized correlation reflects
it. Practitioners relying on realized correlation as a real-time risk
trigger face a systematic lag relative to implied volatility signals.

**Regime-strategy turnover.** The regime-aware strategy generates 317%
annualised turnover at monthly rebalancing. In live implementation, this
constrains scalable AUM and requires execution infrastructure that the
current framework does not model.

---

## 10. Conclusion

We examine factor premia dynamics across HMM-identified market regimes in
Indian equities and the role of FII flows in explaining them.

Three findings emerge. First, factor premia are strongly regime-conditional
using the canonical IIM Ahmedabad factor library. The momentum premium
(WML) concentrates in Sideways regimes (Sharpe 1.667, t = 3.24**) — the
strongest statistically significant regime-factor result in our analysis and
the most actionable for practitioners with systematic momentum strategies.
The market premium concentrates in Bull (Sharpe 1.485, t = 2.28*). Value
(HML) is directionally positive in Bear, consistent with flight-to-quality
rotation, but Bear sample size limits formal inference.

Second, aggregate FII net flows do not Granger-cause individual factor
premia in any of 15 regime-factor tests after Benjamini-Hochberg FDR
correction. This rules out daily foreign institutional trading as the
mechanism behind the regime-conditional patterns documented above. We
explicitly feature this null as a contribution: rigorously executed negative
results are informative and reduce the risk of spurious factor-flow
attribution that characterises much of the practitioner literature on
foreign flows in India.

Third, a gross flow decomposition reveals that both FII purchases and sales
Granger-cause aggregate market returns specifically in Bear regimes
(F ≈ 5, p ≈ 0.009), with symmetry between buy and sell sides pointing to
volume-driven price impact rather than information asymmetry. This
Bear-specific market predictability is invisible when using net flow data
alone — it requires the gross flow decomposition that NSDL data supports
but most aggregate studies do not exploit.

A regime-aware factor allocation strategy achieves Sharpe 0.794 versus 0.733
for buy-and-hold over a 2021–2026 walk-forward period, with lower maximum
drawdown, providing an implementable application of the regime identification.

Future work should examine the FII-DII flow interaction channel — whether
the growth in domestic SIP and DII flows post-2020 has structurally reduced
Indian equities' fragility to FII outflow shocks — and use quarterly NSDL
FPI stock-level holdings to verify whether gross flow price impact operates
through specific factor-bucket concentration.

---

## References

Agarwalla, S.K., Jacob, J., and Varma, J.R. (2013). Four factor model in
Indian equities market. IIM Ahmedabad Working Paper No. 2013-09-05.

Asness, C., Moskowitz, T., and Pedersen, L. (2013). Value and momentum
everywhere. *Journal of Finance*, 68(3), 929–985.

Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate:
a practical and powerful approach to multiple testing. *Journal of the Royal
Statistical Society: Series B*, 57(1), 289–300.

Brunnermeier, M.K. and Pedersen, L.H. (2009). Market liquidity and funding
liquidity. *Review of Financial Studies*, 22(6), 2201–2238.

Carhart, M.M. (1997). On persistence in mutual fund performance. *Journal of
Finance*, 52(1), 57–82.

Naik, N., Devarajan, M., Nowobilski, A., Page, S., and Pedersen, N. (2016).
*Factor Investing and Asset Allocation: A Business Cycle Perspective*.
CFA Institute Research Foundation.

Daniel, K. and Moskowitz, T. (2016). Momentum crashes. *Journal of Financial
Economics*, 122(2), 221–247.

DeLong, E.R., DeLong, D.M., and Clarke-Pearson, D.L. (1988). Comparing the
areas under two or more correlated receiver operating characteristic curves.
*Biometrics*, 44(3), 837–845.

Engle, R.F. (2002). Dynamic conditional correlation: a simple class of
multivariate generalized autoregressive conditional heteroskedasticity models.
*Journal of Business and Economic Statistics*, 20(3), 339–350.

Fama, E.F. and French, K.R. (1993). Common risk factors in the returns on
stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.

Fama, E.F. and French, K.R. (2015). A five-factor asset pricing model.
*Journal of Financial Economics*, 116(1), 1–22.

Granger, C.W.J. (1969). Investigating causal relations by econometric models
and cross-spectral methods. *Econometrica*, 37(3), 424–438.

Grinblatt, M. and Titman, S. (1989). Mutual fund performance: an analysis of
quarterly portfolio holdings. *Journal of Business*, 62(3), 393–416.

Hamilton, J.D. (1989). A new approach to the economic analysis of
nonstationary time series and the business cycle. *Econometrica*, 57(2),
357–384.

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
Wiley. [For HRP methodology used in Table 4.]

Myers, S.C. and Majluf, N.S. (1984). Corporate financing and investment
decisions when firms have information that investors do not have. *Journal of
Financial Economics*, 13(2), 187–221.

Newey, W.K. and West, K.D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
*Econometrica*, 55(3), 703–708.

Rouwenhorst, K.G. (1998). International momentum strategies. *Journal of
Finance*, 53(1), 267–284.

Sehgal, S. and Balakrishnan, I. (2013). Robustness of Fama-French three
factor model: further evidence for Indian stock market. *Vision*, 17(2),
119–127.

Sharpe, W.F. (1964). Capital asset prices: a theory of market equilibrium
under conditions of risk. *Journal of Finance*, 19(3), 425–442.

---

## Appendix

### A1: Regime-Conditional IIMA Factor Premia — Full Table

*(Table A1 output — see outputs/table_a1_iima_factors.csv)*

Reported for MKT, SMB, HML, WML across Bull, Sideways, and Bear regimes
with n, annualised return, annualised volatility, Sharpe ratio, t-statistic,
and significance level. All factors from Agarwalla, Jacob & Varma (2013),
survivorship-bias adjusted, daily frequency.

### A2: Own-Constructed Five-Factor Regime Results

*(Table A2 output — see outputs/table2_regime_factor_matrix.csv)*

Own-constructed MKT, SMB, HML, RMW, CMA from a 14-stock NSE blue-chip universe. MKT
series has r = 0.907 correlation with IIMA MKT (high confidence). SMB
(r = −0.18) and HML (r = 0.01) diverge substantially from IIMA equivalents
due to absence of true size/value dispersion in the 14-stock blue-chip
universe. SMB Bear (SR = +1.853) and HML Bear (SR = −1.135) results from
own-constructed factors should not be relied upon as this construction cannot
identify the size or value premium reliably. RMW and CMA own-constructed
results are reported for completeness; no IIMA benchmark is available.

### A3: FII Granger Causality — Full Matrix

*(Panel B output — see outputs/table3_panel_b.csv)*

All 15 regime-factor cells. Raw p-values and BH-FDR adjusted p-values
reported. Lag selection by AIC (p_max = 5). HAC standard errors throughout.

### A4: COVID-Exclusion Robustness

*(Outputs available in outputs/table3_panel_*.csv)*

All Granger tests replicated excluding 2020-02-01 to 2020-06-30.
MOM-Sideways: SR = 1.59, t = 2.89**. Panel B: all cells null after FDR.
Panel D Bear gross-flow: both p < 0.02. Results are not COVID-driven.

---

*Code, data, and replication scripts available at:
[github.com/yashpatil/alpha-core/paper_analysis]*

*Contact: theyashh.patil7@gmail.com*
