# BCB aggregate small-scale model — specification, data and estimation

This document records, step by step, how the model of the Banco Central do Brasil (BCB) box
**"Atualização dos modelos semiestruturais de pequeno porte"** (Relatório de Inflação, June 2024, pp. 97–107;
file `refs/ri202406b12p.pdf`) is reconstructed from public data and re-estimated. The June-2024 box is the
latest published re-estimation of the model: the Monetary Policy Reports of 2025 and 2026 (`refs/full/`) keep
citing it as the operative specification (e.g. RPM March 2026, box on exchange-rate pass-through, footnote 5;
RPM September 2026, box on El Niño/La Niña, footnote 6), and the June-2025 RPM box on administered prices
(`refs/rpm202506b9p.pdf`) describes only the companion administered-price model. Earlier vintages used for
context: September 2020 (first Bayesian version, `ri202009b7p.pdf`), December 2021 (`ri202112b7p.pdf`,
sample 2003Q4–2019Q4, parameter table) and June 2017 (GMM version, `ri201706b7p.pdf`).

## 1. Equations (aggregate version)

Quarterly data. Inflation rates are quarterly log changes in percent; interest rates are annualised (the box:
"as variáveis são representadas em logaritmo"; we use $100\log(1+x/100)$ for annual rates).

**Phillips curve, free prices** (eq. 1)

$$\pi^{L,sa}_t=\alpha_{1L}\pi^{L,sa}_{t-1}+\alpha_{1I}\frac{\sum_{i=1}^{4}\pi^{IPCA,sa}_{t-i}}{4}
+(1-\alpha_{1L}-\alpha_{1I})\,\pi^e_{t,t+4|t}+\alpha_2\hat\pi^*_t+\alpha_3\widehat{\Delta e}_{t-1}+\alpha_4 h_t
+\frac{\sum_{i=0}^{2}(\alpha_5 d^{el}_{t-i}+\alpha_6 d^{la}_{t-i})Clima^2_{t-i}}{3}
-\frac{\sum_{i=3}^{5}(\alpha_5 d^{el}_{t-i}+\alpha_6 d^{la}_{t-i})Clima^2_{t-i}}{3}+\epsilon^{\pi L}_t$$

* $\hat\pi^*_t$: IC-Br (BRL) quarterly inflation minus the target, $\hat\pi^*=w_a\hat\pi^{*agri}+w_m\hat\pi^{*metal}+w_e\hat\pi^{*energia}$.
* $\widehat{\Delta e}_t$: exchange-rate change minus the PPP drift $(\pi^{meta}-\pi^{*ss})/4$, $\pi^{*ss}=2\%$.
* $Clima^2$: squared ONI anomaly; $d^{el}$ ($d^{la}$) = 1 when the anomaly is positive (negative).
* Only change versus 2021: the exchange rate enters with one lag instead of two.
* Long-run verticality: the coefficients on the three inflation terms sum to one; $\pi^e$ must therefore be
  expressed as a quarterly rate (annual expectation converted with $100\log(1+\pi^e/100)/4$).

**IS curve** (eq. 2)

$$h_t=\beta_1 h_{t-1}-\beta_2\frac{\hat r_{t-1}}{4}-\beta_3\widetilde{rp}_t+\beta_4 h^*_t+s^h_t+\epsilon^{h,crise2008}_t+\epsilon^{h,crise2020}_t,\qquad
\hat r_t=i^e_{t,t+4|t}-\pi^e_{t,t+4|t}-\overline{rr}^{IS}_t$$
$$s^h_t=\beta_5 s^h_{t-1}+\epsilon^h_t,\qquad \overline{rr}^{IS}_t=\overline{rr}^{trend}_t+\widetilde{rr}^{IS}_t,\qquad
\widetilde{rr}^{IS}_t=\widetilde{rr}^{IS}_{t-1}+\epsilon^{rr,IS}_t$$

* $i^e$: Focus expectation of the Selic "ao longo de um horizonte de doze meses"; $\pi^e$: Focus inflation
  expectation for the same horizon; $\overline{rr}^{trend}$: HP trend of $i^e-\pi^e$.
* $\widetilde{rp}$: central-government primary result corrected for outliers and cyclically adjusted, as deviation
  from trend. $h^*$: trade-weighted world output gap.
* Crisis shocks: 2008Q4–2009Q4 and 2020Q1–2020Q4, with high calibrated variances. The s.d. of $\epsilon^{rr,IS}$ is
  calibrated so that $\overline{rr}^{IS}$ has a variance similar to the mean/median of neutral-rate indicators.

**Taylor rule** (eq. 3)

$$i_t=\theta_1 i_{t-1}+\theta_2 i_{t-2}+(1-\theta_1-\theta_2)\left[\overline{rr}^{taylor}_t+\pi^{meta}_t+\theta_3(\pi^e_{t,t+4|t}-\pi^{meta}_t)\right]+\epsilon^i_t,\quad
\overline{rr}^{taylor}_t=\overline{rr}^{trend}_t+\widetilde{rr}^{taylor}_t,\ \widetilde{rr}^{taylor}_t=\widetilde{rr}^{taylor}_{t-1}+\epsilon^{rr,taylor}_t$$

No output-gap term (footnote 11 of the box).

**UIP** (eq. 4)

$$\Delta e_t=\Delta e^{ppc}_t-\delta\,(i^{dif}_t-i^{dif}_{t-1})+\epsilon^e_t,\qquad i^{dif}_t=i_t-(i^*_t+CDS_t),\qquad \Delta e^{ppc}_t=(\pi^{meta}_t-\pi^{*ss})/4$$

**Expectations** (eq. 5)

$$\pi^e_{t,t+4|t}=\varphi_1\pi^e_{t-1,t+3|t-1}+\varphi_2E_t\pi_{t,t+4}+\varphi_3\sum_{i=1}^{4}\pi^{IPCA}_{t-i}+(1-\varphi_1-\varphi_2-\varphi_3)\pi^{meta}_t+\epsilon^e_t$$

**Output-gap measurement** (eqs. 6–9), with cyclical components $f x_t$:

$$fpib_t=h_t+\sigma^h\epsilon^{pib}_t,\quad fnuci_t/\gamma_{nuci}=h_t+\sigma^h\epsilon^{nuci}_t,\quad
femp_t/\gamma_{emp}=h_{t-1}+\sigma^h\epsilon^{emp}_t,\quad fcaged_t/\gamma_{caged}=h_{t-1}+\sigma^h\epsilon^{caged}_t$$

GDP and Caged cycles from an HP filter ($\lambda=1600$); NUCI and unemployment as deviations from their
pre-Covid (to 2019Q4) means.

**Estimation in the box**: Bayesian, quarterly data 2003Q4–2023Q4, uniform priors on the supports (Beta for
$\beta_3$), higher shock variances in the Phillips curve in 2020Q2–2022Q4, Kalman filter with the gap and the
neutral rates as unobserved states. Table 1 of the box reports posterior modes and 90% credible intervals.

## 2. Data sources (all public, downloaded by `src/fetch.py`)

| Variable | Source | Identifier |
|---|---|---|
| IPCA, free prices, administered prices (monthly %) | BCB SGS | 433, 11428, 4449 |
| Inflation target | CMN resolutions (hard-coded) | 2003/2004 adjusted targets 8.5%/5.5% |
| Selic target (daily) | BCB SGS | 432 |
| Focus: IPCA 12 months ahead (smoothed/raw), Selic per Copom meeting, Selic monthly (2001–05), Selic end-year | BCB Olinda OData | `ExpectativasMercadoInflacao12Meses`, `ExpectativasMercadoSelic`, `ExpectativaMercadoMensais`, `ExpectativasMercadoAnuais` |
| Copom meeting dates | BCB site API | `historicotaxasjuros` |
| PTAX BRL/USD monthly average (sell) | BCB SGS | 3698 |
| IC-Br and sub-indices (BRL) | BCB SGS | 27574, 27575, 27576, 27577 |
| GDP, chained volume index SA / current prices | IBGE SIDRA | tables 1621 and 1846 (variable 584 / 585, PIB a preços de mercado) |
| NUCI FGV (SA, monthly) | BCB SGS | 28561 |
| Unemployment: PNAD Contínua (monthly rolling quarter) and PME | BCB SGS | 24369, 10777 |
| Stock of formal jobs (Caged) | BCB SGS | 28763 |
| Primary result, central government (monthly, BRL mn) and monthly nominal GDP | BCB SGS | 4639 (+ = deficit), 4380 |
| Risk premium: EMBI+ Brazil (bps) | Ipeadata | `JPM366_EMBI366` (Brazil 5y CDS is not freely available; EMBI+ is the proxy, as in the 2017 box "Embi ou CDS") |
| Fed funds effective rate | OECD SDMX (MEI financial indicators) | `USA.M.IRSTCI.PA` |
| Trading-partner real GDP (quarterly, SA, volume index) | OECD SDMX QNA | 16 countries; China built from q/q and y/y growth |
| Oceanic Niño Index | NOAA CPC | `oni.ascii.txt` |

## 3. Construction of each model variable (`src/build_dataset.py`)

| Model object | Construction | Judgement calls |
|---|---|---|
| $\pi^{L,sa},\pi^{IPCA,sa}$ | quarterly sum of monthly log changes; seasonal adjustment by robust STL (period 4) on the quarterly series | the box does not name the SA method (BCB normally uses X-13); STL is used because no X-13 binary is installed; a quarter-dummy alternative is kept for sensitivity |
| $\pi^e_{t,t+4|t}$ | quarterly mean of the daily Focus median, 12-month horizon, smoothed series, full respondent base (`baseCalculo`=0) | smoothed vs raw tested in the sensitivity block |
| $i^e_{t,t+4|t}$ | for each survey day, expected Selic path from the per-meeting medians (step function from the current target; decisions effective the next day; meeting dates from the Copom history, future meetings at the usual calendar month) averaged over the following 365 days; quarterly mean. Before Nov-2004 (no per-meeting data) the same object is built from end-of-month Selic expectations (horizon filled with the end-year expectation) and level-spliced on the 2004Q4–2005Q4 overlap (mean difference 0.08 pp, correlation 0.94 on daily data) | the box only says "expectativa da taxa Selic ao longo de um horizonte de doze meses"; the 2021 box formula (trapezoid of quarterly expected rates) is the model counterpart |
| $\hat r_t$, $\overline{rr}^{trend}_t$ | $i^e-\pi^e$ with both in log form (i.e. the log Fisher real rate); HP(1600) trend over the full available sample (2001Q4–2026Q3) | log rates follow "variáveis em logaritmo"; raw-difference version kept (`r_focus_raw`) |
| $i_t$ | quarterly mean of the daily Selic target, log form | |
| $\pi^{meta}_t$ | Taylor rule / expectations: month-weighted average of this and next year's targets (horizon of 12 months); PPP and commodity terms: calendar-year target | |
| $\hat\pi^*_t$ | $100\Delta\log$(IC-Br quarterly mean) $-\pi^{meta}/4$ | the aggregate IC-Br is used; a regression of its changes on the three sub-indices gives weights summing to ≈1 with $R^2\approx1$, so the aggregate equals the weighted combination of the box |
| $\widehat{\Delta e}_t$ | $100\Delta\log$(PTAX quarterly mean) $-(\pi^{meta}-2)/4$, lagged one quarter in the Phillips curve | |
| $Clima^2$, dummies | ONI anomaly of the season centred on the quarter (JFM, AMJ, JAS, OND), **in tenths of a degree**, squared; sign dummies; Phillips terms = mean of $t..t-2$ minus mean of $t-3..t-5$ | the box's prior support U[0, 0.01] and the published modes (0.0012, 0.0007) are incompatible with $Clima^2$ in squared degrees (contributions would be ≈0.01 pp, whereas the 2017 box shows up to ≈1.5 pp in 2015–16); unrestricted least squares on the degree scale returns coefficients ≈100× the BCB values, so the anomaly is scaled by 10 (see `build_climate`) |
| $fpib_t$ | HP(1600) cycle of $100\log$(GDP SA index), full sample 1996Q1–2026Q2 | the BCB extends GDP with Focus forecasts before filtering in its gap boxes; using the full sample to 2026 plays the same role for the 2003–2023 estimation window |
| $fnuci_t$ | FGV NUCI (SA) quarterly mean minus the 2003Q4–2019Q4 mean | |
| $femp_t$ | $-(u^{sa}_t-\bar u_{2003Q4-2019Q4})$: PNAD Contínua calendar-quarter rate (value of the rolling quarter ending in the quarter's last month) spliced backwards with PME quarterly changes, seasonally adjusted (STL); sign flipped so that a positive value means a tight labour market (the box's $\gamma_{emp}$ prior is on [0,3]) | the pre-2012 splice is a judgement call; a variant with the PNADC-only sample (missing before 2012) is estimated for comparison |
| $fcaged_t$ | HP(1600) cycle of $100\log$(quarterly mean of the formal-job stock) | |
| $\widetilde{rp}_t$ | central-government primary surplus (sign flipped from the NFSP convention), net of one-off items (2010 Petrobras capitalisation, 2016–17 repatriation, 2019 cessão onerosa auction, 2023 precatórios), 4-quarter flow in % of 4-quarter nominal GDP (IBGE), cyclically adjusted with a semi-elasticity of 0.22 to the GDP HP gap, minus its HP(1600) trend | the box does not detail the correction; the one-off list and the semi-elasticity are documented in `build_dataset.py` (the coefficient $\beta_3$ is tightly pinned by its Beta prior at 0.03, so the other parameters are insensitive to this construction) |
| $h^*_t$ | export-share-weighted average of HP(1600) cycles of partners' real GDP (OECD QNA; China level index rebuilt from growth rates) | fixed approximate export shares 2010–2023 |
| $i^{dif}_t$ | $\log$-Selic minus $100\log(1+(i^*+EMBI)/100)$ | |
| crisis / high-variance dummies | 2008Q4–2009Q4, 2020Q1–2020Q4 (IS); 2020Q2–2022Q4 (Phillips) | multiples of the base s.d. calibrated at 3 |

## 4. Estimation strategy (two layers, both from scratch in numpy/scipy)

* **Layer A – blocks** (`notebooks/03_gap_is.py`, `04_single_equations.py`): the gap block (IS curve + four
  indicators + random-walk neutral-rate component) is a small state-space model estimated by posterior-mode
  optimisation and random-walk Metropolis-Hastings with the box's priors; the smoothed gap then feeds
  single-equation Bayesian regressions (Phillips, UIP), a Kalman-filter Taylor rule with the random-walk
  equilibrium-rate component, and a 2SLS estimate of the expectations equation (the model-consistent expectation
  instrumented with date-t information).
* **Layer B – full system** (`notebooks/05_system.py`): all equations, including the model-consistent expectation
  $E_t\pi_{t,t+4}$ and the Focus expected-Selic equation of the 2021 box, are written as a linear rational-
  expectations model, solved with a gensys port (`src/re_solver.py`), and estimated with the Kalman filter
  (`src/ssm.py`) by posterior-mode optimisation and MCMC (`src/bayes.py`). Exogenous observed drivers follow
  AR(1) processes pre-estimated by OLS; the target and the HP trend of the real rate are random walks. The
  commodity term is split into an exogenous USD component and the exchange-rate change
  ($\hat\pi^*_t=\hat\pi^{*,USD}_t+\Delta e_t$, footnote 4 of the box), so that a depreciation raises imported
  inflation one-for-one in the simulations; the IPCA identity uses the sample-mean free-price weight.
  Impulse responses replicate the three experiments of the box (Selic +1 pp for four quarters, permanent 10%
  depreciation with the exchange-rate path imposed, one-off +1 pp gap shock); the administered-price model that the
  BCB combines with the small model is outside the scope, so IPCA responses are reported for the free-price channel
  and with a calibrated administered-price add-on taken from the box (1.65 pp over four quarters per 10% depreciation).

Calibrated items not reported by the BCB (values in `src/config.py`):

* s.d. of the neutral-rate random walks: 0.15 pp per quarter (the smoothed random-walk components stay small, so
  the neutral rate is essentially the HP trend of the Focus real rate; a 0.30 variant changes nothing material);
* crisis shocks in the IS curve (2008Q4–2009Q4, 2020Q1–2020Q4): s.d. multiple **20** of the IS-shock s.d. The box
  only says the variances are "relativamente elevadas". The sweep in `notebooks/03c` (output
  `output/03c_crisis_sweep.csv`) shows why this matters: with a small multiple (3) the 2009 and 2020 collapses are
  attributed to the world gap ($\beta_4\approx0.41$) and the gap's persistence falls ($\beta_1\approx0.56$); with
  multiple 20 the block reproduces the BCB values ($\beta_1=0.84$, $\beta_2=0.43$, $\beta_4=0.16$, $\beta_5=0.85$),
  and the results are flat beyond that (multiple 40: 0.86/0.44/0.14/0.86);
* Phillips-curve shock in 2020Q2–2022Q4: s.d. multiple 3 (sensitivity to 1, √3 and 5 reported in `output/04_phillips_sensitivity.csv`);
* IPCA weight of free prices: sample mean of the implicit weight ≈ 0.73 (used only in the system's IPCA identity).
