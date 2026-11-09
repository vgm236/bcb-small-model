# The administered-price model of the BCB: reconstruction, re-estimation, integration and backtest

Source: Monetary Policy Report of June 2025, box "Atualização do modelo para projeção de médio prazo dos preços
administrados" (`refs/rpm202506b9p.pdf`, pp. 79-90; the equations are in the summary table at the end of the box,
read from the PDF since the text extraction loses the formulas). Code: `src/administered.py` (equations, data,
estimation, simulation), `src/adm_projection.py` (real-time truncation, scenario drivers, feedback with the small
model), `src/fetch_adm.py` (data), `notebooks/10_administered.py` (estimation, fit, projection, report
`results_administered.md`), `notebooks/11_backtest_administered.py` (comparison with the Copom's administered
projections at each meeting). The app's sidebar selects the administered source and the tariff-flag assumption, and
the page "Administered prices" shows the coefficients, the fit and the backtest.

## 1. The model as published

Twenty-four quarterly equations, one per administered item of the IPCA (26% of the index), "em geral não
estimadas, mas calibradas a partir do arcabouço institucional vigente". In the notation of the box (ΔIPCA_t is the
quarterly headline inflation, d_j,t the quarter dummies):

* **Fuels** (gasolina 5.24%, óleo diesel 0.25%, gás de botijão 1.25%): the consumer price is a weighted sum of
  the Petrobras price, the biofuel added (follows Petrobras), the ICMS (adjusted in the first quarter by last
  year's IPCA), fixed federal taxes and the margin (indexed to inflation; for bottled gas only in the third
  quarter). The Petrobras price follows oil in reais with distributed lags: gasoline and diesel c = (0.4, 0.4, 0.2)
  and (0.3, 0.4, 0.3) over lags 0-2, bottled gas 0.2 over lags 1-5; price shares p: gasoline (0.35 Petrobras,
  0.13 ethanol, 0.24 ICMS, 0.11 federal, 0.17 margin), diesel (0.468, 0.125, 0.173, 0.05, 0.184), bottled gas
  (0.32 Petrobras, 0.17 ICMS, 0, 0.51 margin). The long-run pass-through of oil in reais is therefore 0.48 for
  gasoline, 0.59 for diesel and 0.32 for bottled gas.
* **Electricity** (3.78%): π = π^ex-band + π^band. The ex-flag tariff is indexed to the previous four quarters'
  IPCA with seasonal weights c = (0.08, 0.40, 0.42, 0.10) for the share (1 − α_Itaipu α_energia), α_Itaipu = 0.086,
  α_energia = 0.6, and the Itaipu share follows the US CPI (annual adjustment in the first quarter) and the exchange
  rate with weights (0.10, 0.35, 0.42, 0.13). The flag component is the specialist's short-term path, then the
  pattern of the last four quarters repeated with the opposite sign so that four-quarter sums are not affected.
* **Pharmaceuticals** (3.46%): the CMED readjustment of April = IPCA over four quarters − Fator X + Fator Y +
  Fator Z, with X = Z = 0 in the medium term and Fator Y = 0.15 (exchange rate + US CPI − IPCA over four quarters)
  + 0.07 (electricity ex-flag − IPCA over four quarters); pass-through 0.8 in the second quarter, 0.05 in the third
  and fourth, 0.1 in the first quarter of the next year.
* **Health plans** (4.06%): the ANS readjustment R_A = 0.8 IVDA_A + 0.2 IPCA^{ex plano}_{A−1}, applied as R/4 per
  quarter from the second quarter; IVDA_A = 0.6 IVDA_{A−1} + 0.4 IPCA_{A−1} beyond the specialist's horizon.
* **Emplacamento e licença** (2.69%) and **conselho de classe**: last year's IPCA/4 in the first quarter, repeated
  in the following three.
* **Items with seasonal readjustments** (água e esgoto, ônibus urbano/intermunicipal/interestadual, táxi, pedágio,
  metrô, correio): π_t = Σ_j d_j,t c_j Σ_{i=1..4} ΔIPCA_{t−i}, with item-specific c_j summing to one (e.g. ônibus
  urbano 0.75/0.20/0.05/0, água 0.19/0.50/0.09/0.22).
* **Items without seasonality** (jogos de azar, telefonia fixa, multa): (1/4) Σ_{i=1..4} ΔIPCA_{t−i}; **first-
  quarter items** (integração, trem, cartório): the full four-quarter IPCA in the first quarter.
* **Gás encanado** and **gás veicular**: regressions on past inflation, the bottled-gas or gasoline item and oil in
  reais (β = 0.55/0.25/0.05 and 0.5/0.1/0.2 with oil lags 0.1/0.05/0.05).

The box's own impulse responses: a permanent 10% depreciation raises administered inflation by 1.8 pp after four
quarters (IPCA 1.0 pp); a permanent 10% rise in Brent raises it by 1.3 pp (gasoline pass-through about 50% in the
fourth quarter, bottled gas about 20%); a 1 pp shock to free prices raises administered inflation with a lag and
more persistently; a 1 pp shock to the IVDA raises health plans by 0.8 pp and administered inflation by 0.14 pp.

The box is explicit that the model is a medium-term tool: short-term projections of administered prices (four to
five quarters) come from specialists with institutional information, and the medium-term model takes over
afterwards; the projections of the semi-structural models are formed "considerando o conjunto de equações do
modelo de administrados", so that the headline projections incorporate the feedback between free and
administered prices.

## 2. Implementation

Every equation is written as π_item,t = θ' z_t with z_t built from the drivers, so that the BCB's calibration
(`BCB_THETA`) and OLS estimates on the same regressors are directly comparable and either can drive the
simulation. Units are quarterly log-% (equivalent to the box's percentage variations to first order).

Data, all public and downloaded at each run (`fetch_adm.py`):

| Input | Source |
|---|---|
| Monthly variation and weight of the 24 subitems and the general index, 2012- | IBGE SIDRA tables 1419 (2012-2019) and 7060 (2020-), variables 63 and 66, subitem codes of the box |
| Tariff flag in force each month and its surcharge (R$/MWh) | ANEEL open data, dataset "bandeiras tarifárias" (acionamento, adicional) |
| Brent (US$) | EIA daily spot (quarterly means) |
| Exchange rate | BCB SGS 3698 (PTAX monthly, quarterly means) |
| US CPI | BLS public API, CUUR0000SA0 |
| Headline IPCA, administered index (check) | BCB SGS 433, 4449 |

Construction choices: the flag's contribution to the electricity item is 100·Δlog(1 + S_m/T_m), with S the
surcharge and T the reference residential tariff (750 R$/MWh in 2024 prices, chosen as the value that minimises
the variance of the item's inflation net of the flag effect, and scaled with the item's own index); the ex-flag
electricity series is the item minus this contribution. The ANS readjustment of each year is inferred from the
health-plan item (twelve times the monthly rate observed from June: IBGE applies the readjustment linearly over
twelve months), and the IVDA from the ANS rule; the CMED readjustment is inferred from the second-quarter
pharmaceutical inflation divided by the 0.8 pass-through. Both are therefore self-updating, and the rules of the
box project the years not yet observed. The 24 items with their monthly weights reproduce the official
administered index (quarterly correlation 0.999, mean difference −0.06 pp).

Estimation (`estimate`): HAC OLS of each item's inflation on its regressors, from 2012Q2 for the fuel, electricity,
pharmaceutical and health-plan equations and from 2020Q1 for the items indexed to past inflation (the box
calibrates their seasonal weights "com amostra desde 2020"); the table reports the BCB value, the estimate, its
standard error and the distance in standard errors.

Integration (`adm_projection.run`, `projection.project(adm_source="model")`): the item data are truncated at the
meeting's information set; the drivers over the horizon are the reference-scenario paths (oil in reais from the
Brent and PTAX paths, US CPI at its recent average, the tariff flag held through the current quarter and then at
the December assumption of the Copom); the current quarter keeps the Focus headline nowcast and the annual
administered figure (the specialists' territory), and `adm_handover` can keep the annual-figure path for further
quarters before the model takes over; the simulation runs from the last complete quarter, the small model is
conditioned on the resulting administered path, the headline it produces (free and administered through the
identity) re-indexes the administered items, and the loop repeats until the administered path changes by less
than 0.005 pp (three or four iterations).

## 3. Re-estimation versus the BCB's calibration

See `results_administered.md` (generated) for the full table. Main findings:

* **Health plans and pharmaceuticals** follow the rules closely: the pass-through weights of the inferred
  readjustments are estimated at 1.005/0.973 (BCB 1/1) and 0.11/0.80/0.06/0.06 (BCB 0.1/0.8/0.05/0.05), R² 0.68 and
  0.95. The rule for the CMED readjustment without the X and Z factors overstates it in some years (2025: rule 7.1,
  observed 4.3 log-%), which the specialist handles in the BCB's process.
* **Fuels** are where the calibration departs from the data. The estimated pass-through of oil in reais to the
  gasoline item is 0.21 on impact and nil afterwards (BCB: 0.19, 0.19, 0.10, cumulated 0.48); for diesel 0.26 then
  0.06/0.04 (BCB 0.18/0.24/0.18, 0.59); for bottled gas 0.08 on impact and about 0.08 cumulated (BCB 0.32). The
  sample (2012-2026) contains the years of Petrobras price controls (2012-2016), the import-parity policy (2016-2022)
  and the current policy "levando em consideração fatores locais" (since May 2023); the box itself notes that the
  Petrobras price no longer passes international fluctuations directly, and leaves the convergence of the Petrobras
  price to the specialist. The ICMS coefficients are estimated well above the calibrated shares (0.85 versus 0.24),
  picking up the first-quarter readjustments of fuel prices in general.
* **Electricity**: the ex-flag item is not explained by the indexation rule in quarterly data (R² 0.01): the
  timing of the annual tariff readjustments by region, the Itaipu bonus (2024, 2025, 2026) and extraordinary
  reviews dominate. The calibrated seasonal weights (0.08/0.40/0.42/0.10) put the readjustments in the second and
  third quarters, the estimates spread them more evenly (0.36/0.24/0.14/0.17, with wide standard errors).
* **Items indexed to past inflation**: the seasonal weights estimated since 2020 are mostly within two standard
  errors of the BCB's (água e esgoto 0.25/0.47/0.08/0.22 versus 0.19/0.50/0.09/0.22; ônibus intermunicipal,
  táxi, pedágio, correio, metrô similar); ônibus urbano has a lower first-quarter weight than calibrated (0.45
  versus 0.75) with offsetting negative weights later (fare freezes and cuts in several capitals since 2023);
  jogos de azar, telefonia fixa and multa show weaker indexation than one-for-one (0.54, 0.30, 0; the BCB
  calibrates 1).

Of the 83 parameters, 17 differ from the BCB's by more than two standard errors; the aggregate implications are
discussed with the backtest in Section 4 and in `results_administered.md`.

## 4. Backtest against the Copom's administered projections

`notebooks/11_backtest_administered.py` runs the projections at the ten meetings with the administered path from
the Copom's annual figures (the outside input of the earlier backtest), from the model with the BCB's calibration,
from the model with the re-estimated parameters, and from the model taking over after four quarters (the
specialists' horizon), with the tariff flag the Copom assumed at each meeting. The comparison with the Copom's
administered projections and the resulting IPCA projections is in `results_administered.md` and in Section 5 below.

## 5. Results and the correction adopted

Administered-price projections, model minus Copom, percentage points, ten meetings (current year, next year
where published, relevant horizon); the IPCA columns are the projections of the full system with that
administered source (`output/11_backtest_administered_summary.csv`):

| administered source | MAE adm. current year | MAE adm. next year | MAE adm. horizon | MAE IPCA next year | MAE IPCA horizon |
|---|---|---|---|---|---|
| Copom's annual figures (outside input) | 0 | 0 | 0 | 0.13 | 0.22 |
| model from the next quarter, BCB calibration | 0.75 | 1.97 | 1.14 | 0.81 | 0.67 |
| model from the next quarter, re-estimated parameters | 1.09 | 0.57 | 0.65 | 0.26 | 0.40 |
| model after four quarters, BCB calibration | 0 | 0.20 | 0.32 | 0.15 | 0.29 |
| model after four quarters, re-estimated parameters | 0 | 0.07 | 0.56 | 0.19 | 0.37 |

* **Started from the next quarter, the calibrated model is far from the Copom's administered numbers** (mean
  −1.5 pp for the next year). The reason is the fuel block: with the oil price in reais falling through 2025
  (−14% in 2025Q2, −9% in 2025Q4) the calibrated pass-through of 0.48 takes gasoline down by 2-3% a quarter, which
  neither Petrobras nor the Copom's specialists delivered (September 2025: model 0.05% for administered prices in
  2026 against the Copom's 3.8%); in 2026 the opposite happens with the oil spike (June 2026: 6.2% against 4.7%).
  The re-estimated parameters (pass-through about 0.2) remove most of this (next-year MAE 0.57) but add noise in
  the current year, where the quarterly timing of tariffs and flags dominates.
* **The model's proper use is the medium term.** Keeping the annual figures for four quarters, as the BCB keeps
  its specialists' projections, and letting the model take over afterwards brings the administered projections
  within 0.2 pp of the Copom's for the next year and 0.3 pp at the relevant horizon with the BCB's own calibration,
  and the IPCA projections within 0.15-0.29 pp, close to the outside-input case (0.13-0.22). With the re-estimated
  parameters the next-year figure is marginally closer (0.07) but the horizon is worse (0.56): the lower
  estimated indexation of several items (fuels through the ICMS and margin terms, buses, games, telephony) pulls
  the medium-term path below the Copom's, whose model keeps the calibrated one-for-one indexation.
* **Adopted** (`projection.ADOPTED_ADM_MODEL`): when the administered model is used, the BCB's calibration with the
  handover after four quarters; the annual figures (the Copom's projections in the backtest, the Focus medians for
  a future meeting) remain the default input for the first year. The December tariff flag follows the Copom's
  stated assumption; the ANS and CMED readjustments of the current year are read from the data as soon as they
  appear in the IPCA and the rules of the box project the following years.
* What cannot be corrected from public data: the specialists' bottom-up projections (fuel price decisions,
  tariff reviews, the Itaipu bonus, the X and Z factors of the CMED rule), which is why the first four quarters
  are kept as an input, exactly as in the BCB's process.
