# Conditional projections in the Copom's format and the backtest against the last ten meetings

This note documents the projection feature (`src/projection.py`, `src/vintage.py`, `src/copom_meetings.py`,
`notebooks/07_projection.py`, `notebooks/08_backtest_copom.py`): how the estimated model is run under the Copom's
reference-scenario assumptions, how the information set of a past meeting is reconstructed, and how the results
compare with the projections published by the Copom. The results section is written from the outputs of
notebook 08 (`output/08_backtest_*.csv`, `output/figs/08_*.png`).

## 1. What the Copom publishes and what is used as input

Since 2025 the Copom statement and the minutes (section "Cenários e análise de riscos", Table 1 of the minutes)
report, for the reference scenario, the projections of the IPCA, of free prices and of administered prices for the
calendar years ahead and for the "relevant horizon" (six quarters ahead; accumulated in four quarters). The
conditioning assumptions are stated in the same paragraph:

* the Selic path "extraída da pesquisa Focus" (medians of the survey of the Friday before the meeting week);
* the exchange rate "parte de R$ x/US$ e evolui segundo a paridade do poder de compra (PPC)", where the starting
  value is the average PTAX of the ten business days ending on the last day of the week before the meeting, rounded
  to R$0.05 (footnote of the minutes; adopted at the 258th meeting). The reconstruction in
  `src/copom_meetings.py` reproduces the published value for all ten meetings;
* oil "segue aproximadamente a curva futura pelos próximos seis meses e passa a aumentar 2% ao ano
  posteriormente";
* the electricity tariff flag assumed for December;
* a neutral real interest rate of 5.00% (stated in the "Projeções condicionais" chapter of every Monetary Policy
  Report of 2025-2026);
* since 2026, the climate term uses NOAA's Relative Oceanic Niño Index (RONI) and, for the projection, NOAA's
  official RONI outlook (Monetary Policy Report of September 2026, box on El Niño and footnote 46).

The ten meetings used (July 2025 to September 2026, 272nd to 281st) and their published numbers are transcribed in
`src/copom_meetings.py` from the minutes (`refs/copom/atas/*.pdf`, text in `refs/copom/atas/txt`);
`check_against_minutes()` re-reads the minutes and verifies the transcription (the statements carry the same
figures, but the minutes also give the free/administered split and the exchange-rate footnote, so they are the
source). The administered-price projections of the Copom are used as an outside input, because the small model
only has a Phillips curve for free prices (the BCB combines it with a separate administered-price model).

## 2. Information set of a meeting (`src/vintage.py`)

For a meeting decided on date D with Focus reference date F, the quarterly dataset is rebuilt from the cached raw
series using only what was public at D, with these publication-lag rules (conservative approximations of the
official calendars):

| Series | Available at D if |
|---|---|
| IPCA (headline, free, administered) of month m | D ≥ 12th of month m+1 |
| IC-Br of month m | D ≥ 10th of m+1 |
| FGV capacity utilisation of month m | D ≥ last day of m |
| PNAD unemployment (rolling quarter ending in m), Caged stock, central-government primary result of m | D ≥ 1st of m+2 |
| IBGE quarterly GDP of quarter q | D ≥ end of q + 65 days |
| partners' GDP (OECD) of quarter q | D ≥ end of q + 60 days |
| ONI season centred on quarter q | D ≥ 10th of the month after the quarter |
| Focus survey | survey of F |
| Selic target | decisions up to and including D |
| PTAX | through F (the scenario path applies afterwards) |

Data revisions cannot be undone (the current vintage of each series is truncated); GDP, Caged and PNAD are the
only inputs subject to revision. Filtered objects are recomputed on the truncated sample: STL seasonal factors of
the quarterly inflation series, HP cycles of GDP, Caged, partners' GDP and the cyclically adjusted primary balance,
where the level series are extended by eight quarters before filtering (GDP with the Focus growth medians of F,
Caged and partners' GDP with their recent average growth, the fiscal balance kept constant) to limit the end-point
bias. The neutral real rate follows the Copom's conditioning value (5.00%, log form) from 2024Q1, the first
quarter after the model's estimation sample; in-sample, the model uses the HP trend of the ex-ante real rate as
estimated. (The HP trend extended through 2026 reaches about 9%, because the two-sided filter follows the
restrictive policy stance of 2024-2026; the June-2024 box's calibration of the random-walk component cannot offset
that, and no estimate of the neutral rate supports it. A variant with the HP trend is reported in the backtest.)

**Current quarter (nowcast).** The quarter of D is included as an observed quarter: the months whose IPCA is not
yet published get the Focus median of the monthly IPCA (survey of F) as headline inflation and the administered
assumption as administered inflation, so free-price inflation follows from the IPCA identity with the latest
implicit weight of free prices; the Selic is the realized target through D and the Focus path afterwards; the
exchange rate is the realized PTAX through F and the scenario afterwards; the Focus 12-month inflation expectation
and the expected Selic over the next twelve months are the values at F; IC-Br is extended with the oil assumption
(energy sub-index, elasticity to Brent estimated on monthly data, about 0.5) and flat agricultural and metal
sub-indices in US dollars; the gap indicators use the months already published (partial-quarter averages) or are
missing; the fiscal and world-gap drivers are missing (their AR(1) processes carry them); the ONI uses the monthly
Niño 3.4 readings already published. The Kalman filter then delivers the state at the origin (output gap, neutral-
rate component, expectations, persistent IS shock).

## 3. Reference-scenario paths (`src/projection.py`)

* **Selic**: daily step function from the realized target, the Focus per-meeting medians at their meeting dates
  (decisions effective the next day), then a linear glide to the Focus end-year medians of later years, then
  constant; quarterly averages in log form give the model's `i`. The expected Selic over twelve months (`ie`) at the
  origin is the average of this path over the 365 days after F, the same construction used in estimation.
* **Exchange rate**: realized PTAX through F, then the start value growing with the PPP drift (target minus 2%
  foreign inflation); quarterly averages give `de`.
* **Oil**: the Brent spot price at F (five-day average, EIA series) and, for the next six months, the shape of the
  Brent path of the latest EIA Short-Term Energy Outlook published before F (the STEO's near-term path tracks the
  futures curve; no public history of the Brent futures curve exists), then +2% a year. The energy sub-index of
  the IC-Br in dollars follows Brent with the estimated elasticity; the other sub-indices are flat in dollars; the
  BRL index follows the exchange-rate path. This gives the path of the model's `pistar_usd`.
* **Administered prices**: monthly path consistent with the Copom's calendar-year projections (taking the months
  already published into account) and, for the year beyond the last calendar year given, with the four-quarter
  projection at the relevant horizon; a flat monthly profile is used (the Copom publishes only annual figures and
  the monthly seasonality of administered prices is dominated by one-off tariff events; a seasonal profile is
  available as a variant). The quarterly sums enter the model as `piA`.
* **Climate**: the ONI after the last published season follows the monthly Niño 3.4 readings already published
  for the current quarter and then an AR(1) decay (coefficient 0.79, estimated on 1950-2026) towards zero; the
  Phillips-curve terms are rebuilt from the full ONI path. Alternatives: an explicit path (e.g. NOAA's outlook),
  the realized ONI (perfect foresight) or no climate effects after the current quarter.
* **Target**: 3% (log form) for the twelve-month-ahead target; **neutral rate**: constant at the origin value.
* The other drivers (world gap, fiscal impulse, external rate) follow their estimated AR(1) processes from the
  filtered state.

## 4. Conditional forecast with anticipated shocks

The solved rational-expectations system (gensys) is augmented with news shocks: for each conditioned shock
(policy-rate, exchange-rate, administered-price, USD-commodity and the two climate shocks) anticipated components
known k quarters in advance (k = 1..12) are added as predetermined states, and the augmented system is solved again.
The news vector at the first projection quarter is then chosen so that the Selic, the exchange-rate change,
administered inflation, commodity inflation and the climate terms follow their paths over the twelve quarters
(a square linear system). Because the shocks are anticipated, the model-consistent component of inflation
expectations and the expected-Selic equation are consistent with the published Selic path and with the
administered-price assumption, as in the Copom's exercise, in which the Focus path is the market's expectation.
The Taylor rule is switched off over the conditioning horizon (it is what the policy news shocks do) and resumes
afterwards.

Uncertainty bands come from stochastic simulations: the initial state is drawn from the filtered distribution, the
non-conditioned shocks are drawn with their estimated standard deviations, and the conditioning paths are
re-imposed period by period; the band is centred on the deterministic path. They cover the model's shocks only,
with the administered path, the Selic path and the exchange-rate path taken as given.

## 5. From model units to the Copom's table

Free-price inflation (seasonally adjusted, log-% per quarter) is converted to calendar inflation with the
quarterly STL seasonal factors of the last year of the vintage, combined with the administered path through the
IPCA identity with the latest implicit weight of free prices (about 0.74), and accumulated over calendar years and
over four quarters (the current year combines the published months, the nowcast months and the projected
quarters). The relevant horizon is the quarter six quarters after the current one.

## 6. Backtest design (`notebooks/08_backtest_copom.py`)

For each of the ten meetings the baseline projection (Section 3) is compared with the Copom's Table 1 for the
calendar years and the relevant horizon; the comparison is reported for the IPCA, free prices and administered
prices (the last one matches by construction). Because the administered path is common, the headline difference is
the free-price difference times the free-price weight, so the free-price comparison is the informative one. The
realized outcomes known at the time of writing (calendar year 2025) are also shown. Four variants isolate the
assumptions on which the model's output depends most: climate effects switched off after the current quarter,
climate with perfect foresight of the ONI, the HP-trend neutral rate, and the seasonal monthly profile for
administered prices.

## 7. Results and discussion

The tables are in `results_backtest.md` (generated by `notebooks/08b_backtest_report.py`); the figures are
`output/figs/08_backtest_panels.png`, `08_backtest_scatter.png` and `08_backtest_horizon.png`. Numbers below are
model minus Copom, in percentage points of the IPCA (free prices in brackets).

**Overall.** Across the 26 published targets (10 current-year, 6 next-year, 10 relevant-horizon figures) the model
is on average 0.07 pp above the Copom [0.09], with a mean absolute difference of 0.33 pp [0.43] and a maximum of
0.71 pp [0.91]. By type of target: current year −0.08 (MAE 0.33), next year +0.30 (0.35), relevant horizon +0.08
(0.31). The Copom's horizon projection falls inside the model's 50% band in all ten cases (the band's half-width
at six quarters is 0.9 pp; 2.1 pp for the 90% band). The size of the differences is of the order of the Copom's own
revisions between consecutive meetings (mean absolute revision of the 2026 projection: 0.24 pp for the Copom,
0.45 pp for the model); the revisions of the two move together (correlation 0.75 for the 2026 target over ten
meetings, 0.54 for 2025 over four).

**Pattern over time.** The differences are concentrated at turning points and change sign:

* July-September 2025: the model is 0.5-0.8 pp above the Copom for 2025, 2026 and 2027Q1 [0.6-0.8]. Free-price
  inflation collapsed in 2025H2 (0.19 and 0.85 log-% in Q3 and Q4, after 2.1 in Q1) with food deflation and a 7%
  appreciation of the real over two quarters. The Copom's projections for 2025 fell from 4.9 (July) to 4.4
  (December); the model's fell from 5.4 to 4.4, but only once the Q3 data were in (November). Against the realized
  2025 inflation of 4.26% [3.90] the Copom's errors were +0.6, +0.5, +0.3 and +0.1 at the four meetings and the
  model's +1.2, +1.1, +0.3 and +0.1.
* November-December 2025: within ±0.2 pp for every target.
* January-March 2026: the model is 0.5-0.9 pp below the Copom [0.6-0.9]. After two quarters of very low free-price
  inflation, the model's inertia terms and a gap estimated near zero at the origin (0.2% in January) pushed the
  2026 projection to 2.8% (Copom 3.4) and the 2027Q3 projection to 2.6% (3.2); the Copom's numbers then rose quickly
  with the oil supply shock (Brent from 67 US$ in January to 110 in April) and the realized data (IPCA of 1.9 log-%
  in 2026Q1), which the model only incorporated at the April meeting (4.5 versus 4.6 for 2026).
* April-June 2026: within ±0.1 pp at the horizon; 0.5 pp below for the 2026 figure at the June meeting.
* August-September 2026: 0.3-0.5 pp above for 2027 and for 2028Q1 [0.4-0.6], with the model's output gap back to
  1.1% at the origin and the El Niño term adding to free-price inflation into 2027.

The model is more momentum-driven than the Copom's projection process: a quarterly aggregate model with an
inertia weight of about 0.5 on past inflation, whose current quarter is nowcast with the Focus monthly medians,
adjusts to a change in the inflation regime one or two meetings after the Copom, whose short-term projections
come from disaggregated monthly models and high-frequency information and which applies judgement.

**What the variants show.**

* *Climate.* The El Niño/La Niña term matters for the 2026 meetings: without it the model's 2028Q1 projections at
  the August and September meetings would be 4.1-4.5% instead of 3.5-3.7% (Copom 3.2%), because the term reverses
  sign three quarters after the event and lowers inflation in 2027H2-2028Q1. Perfect foresight of the ONI makes
  the 2025 projections far worse (+0.8 pp at the horizon), since the 2026 El Niño was not foreseeable in 2025. With
  the NOAA outlook quoted by the BCB in September 2026 (RONI peaking at 2.7 in 2026Q4 and vanishing by 2027Q2,
  expressed in ONI units) the horizon projections coincide with the Copom's (3.1 versus 3.2 at both meetings) but
  2027 overshoots (4.7-4.9 versus 3.8-3.9), so the model front-loads the climate effect more than the Copom's
  projections do. Two differences in the BCB's current practice explain part of this: the BCB switched to the
  RONI in 2026, whose squared values are about 40% smaller than the ONI's at these levels, and the model's
  coefficients were estimated on the ONI through 2023 (the June-2024 box) and are kept here for consistency.
* *Neutral rate.* With the HP trend of the ex-ante real rate instead of the Copom's 5.00%, projections are
  0.4-0.5 pp higher at the horizon (the trend reaches 8-9% in 2025-2026 and the stance looks neutral); the 5%
  assumption is necessary to reproduce the Copom's numbers and consistent with the Reports.
* *Administered profile.* The monthly distribution of the administered path affects only the current-year
  figure (−0.2 pp on average with the seasonal profile), through the split of the nowcast months between
  free and administered prices.

**Caveats.** (i) The data vintages are approximated by truncating the current vintage with publication-lag rules
(GDP, Caged and PNAD revisions are not undone). (ii) The futures-curve assumption for Brent is proxied by the
shape of the EIA STEO path; for the model this matters little (the commodity coefficient is small and
administered fuel prices are an input), but it is not the Copom's curve. (iii) The risk-premium input (EMBI+)
ends in July 2024, so the external block runs on its AR(1) process afterwards; the exchange-rate path is imposed,
so the effect on inflation is negligible. (iv) The Copom's statements could not be downloaded through the site's
API; the minutes' Table 1, which reports the same projections with the free/administered split, is the source.
(v) The uncertainty bands cover the model's shocks only, with the Selic, exchange-rate and administered paths
taken as given.

**Live projection (first version).** `notebooks/07_projection.py` runs the same machinery for a future meeting
(by default the meeting of 3-4 November 2026 with the Focus survey of 23 October 2026, exchange rate from the PTAX
convention, administered prices from the Focus medians). With the first configuration it gave IPCA 5.2% for 2026,
4.4% for 2027, 3.1% for 2028 and 3.3% for the four quarters to 2028Q2 (free prices 5.3, 4.6, 2.9 and 3.1), with the
output gap at 0.9%; Section 8 revises the configuration and Section 9 adds the adjusted output.

## 8. Where the differences come from, and the configuration adopted (`notebooks/09_diagnostics.py`)

The variants of Section 7 switch ingredients off; the diagnostics switch them to what the Copom plausibly used,
one at a time and in combination, over the ten meetings (unadjusted output, model minus Copom, pp of the IPCA):

| variant | mean | MAE | MAE current year | MAE next year | MAE horizon | gap at origin, Sep-2026 |
|---|---|---|---|---|---|---|
| baseline (Section 3) | +0.07 | 0.33 | 0.33 | 0.35 | 0.32 | 1.07 |
| output gap anchored on the BCB's published estimates | −0.11 | **0.25** | 0.35 | 0.13 | 0.22 | 0.54 |
| BCB coefficients (June-2024 box) | +0.17 | 0.40 | 0.32 | 0.48 | 0.42 | 1.11 |
| RONI instead of ONI | −0.14 | 0.33 | 0.37 | 0.13 | 0.41 | 1.13 |
| perfect nowcast of the current quarter | −0.03 | 0.25 | 0.26 | 0.25 | 0.23 | 1.06 |
| gap anchor + BCB coefficients | −0.03 | 0.26 | 0.34 | 0.21 | 0.22 | 0.45 |
| gap anchor + perfect nowcast | −0.20 | 0.27 | 0.30 | 0.24 | 0.26 | 0.53 |
| gap anchor + RONI | −0.31 | 0.40 | 0.40 | 0.25 | 0.49 | 0.56 |

* **The output gap at the origin is the main identified source.** The Monetary Policy Reports publish the gap
  the Copom conditions on (0.5-0.9% in 2025Q1-Q2, 0.2-0.5% from 2025Q3 to 2026Q3); the model's real-time gap ran
  about 0.5 pp higher in mid-2025 and in 2026H2 (1.1-1.2%), because the labour-market indicator (unemployment 5 pp
  below its 2003-2019 mean, measured as a deviation from a constant mean as in the box) signals a very positive
  gap. Anchoring the filtered gap on the published estimates (an extra observation of the gap state with a
  standard deviation of 0.1 pp for the quarters covered by the reports available at the meeting; function
  `rpm_gap_estimates`, which parses the reports) removes the next-year bias (+0.31 to +0.02), cuts the horizon
  MAE from 0.32 to 0.22 and the overall MAE from 0.33 to 0.25, and brings the August-September 2026 projections for
  2027 and 2028Q1 within 0.0-0.3 pp of the Copom's. This is conditioning on published information, as the Copom
  itself does, not a fit to its projections.
* **Short-term information** is the second source: with the published IPCA of the current quarter in place of the
  Focus medians (an upper bound, not available in real time) the current-year MAE falls from 0.33 to 0.26; the
  remaining large errors at the January-March 2026 meetings (−0.6 to −0.8 for 2026) are of this kind, since the
  Q1-2026 IPCA surprised upwards before the oil shock.
* **Coefficients.** The modes published in the June-2024 box do more harm than good (MAE 0.40): the Copom's
  projections do not come from that single model with those coefficients. The project's estimates are kept.
* **Climate index.** The RONI (the BCB's index since 2026) lowers the horizon projections of the 2025 meetings
  and worsens the overall fit; the coefficients were estimated on the ONI, so the ONI is kept. The strong 2026 El
  Niño and its reversal remain the largest source of uncertainty for 2027-2028.

Adopted configuration (`projection.ADOPTED`, the default of the notebooks and the app): neutral rate 5.00%,
output gap anchored on the BCB's published estimates, ONI with the real-time rule, estimated coefficients, flat
administered profile, Focus nowcast.

## 9. Two outputs: unadjusted and adjusted by the previous meeting's error (`src/adjust.py`)

For every meeting the backtest stores the comparison with the Copom (`output/backtest_cache/<meeting>.json`). The
**adjusted** projection at meeting k is the unadjusted projection minus the model's error (model minus Copom) at
meeting k−1 for the same type of target (current year, next year, relevant horizon; when the previous meeting did
not publish the next-year figure, the horizon error is used); the quarterly four-quarter path is adjusted with the
same errors interpolated between the target quarters. The weight of the correction can be set below one
(`damping`); the backtest reports the mean absolute difference for weights 0 to 1. For a future meeting, the
adjustment uses the latest meeting with stored results, so it moves forward automatically once the minutes of a
new meeting are parsed and the backtest re-run (the app does both with one button).

## 10. Interface (`app/app.py`, `run_app.bat`)

A Streamlit application in the project's virtual environment: the next scheduled meeting (calendar in
`copom_meetings.CALENDAR`, completed by the BCB history API), the Focus survey, the PTAX convention and the Focus
administered-price medians are loaded automatically; the relevant horizon is six quarters after the meeting's
quarter. Pages: next meeting (unadjusted and adjusted outputs, bands, assumptions), scenario builder (own Selic
path by quarter, own BRL/USD path, administered inflation by year, 12-month expectation at the origin; compared with
the reference scenario), backtest (tables and figures; one button updates the Copom history from new minutes and
re-runs the backtest, which refreshes the stored errors) and data updates (re-download of all inputs). Model
settings are in the sidebar. The minutes parser (`copom_meetings.parse_minutes`) reproduces the transcribed table
for the ten known meetings and handles both layouts of Table 1 used by the Copom in 2025-2026.

## 10b. Shocks to inflation expectations (scenario builder)

Two kinds of shock, both in percentage points relative to the Focus survey (`projection.project`, arguments
`pie_shock_origin` and `pie_shock_path`): a shock at the origin shifts the 12-month expectation observed in the
current quarter, so the Kalman filter starts from a higher (lower) expectations state and the expectations
equation drives it back over time (persistence about 0.7 a quarter); a shock path keeps the model's expectations
the stated number of points above (below) their own baseline path in the chosen projection quarters. The path
shock is implemented by conditioning the expectations variable on its baseline path plus the shock with
anticipated expectation shocks (the same news-shock machinery as the policy and exchange-rate paths), so that
the model-consistent component of expectations, the Phillips curve and the IS curve respond consistently. With
the November 2026 origin, a +0.5 pp shock at the origin raises the 2027 IPCA projection from 4.2 to 4.5% and
the 2028Q2 figure from 3.1 to 3.4%; +0.5 pp for four quarters raises them to 4.5 and 3.3%. The desktop
application exposes both (tab Scenario builder) and its tab "How it works" documents every tab.

## 10c. Output gap: estimate and projection

Every projection run also returns the output gap (`out['gap']`, tab "Output gap" of the desktop application,
expander on the web page, `output/07_projection_<date>_gap.csv` and `output/figs/07_gap_<date>.png` from
notebook 07). History: the smoothed estimate (Durbin-Koopman smoother on the real-time dataset through the
current quarter, with its 90% band), the filtered estimate (past data only), the four indicator-implied gaps
(GDP HP cycle; capacity utilisation, unemployment and formal jobs divided by their estimated loadings, the
last two shifted one quarter because they load on the lagged gap) and the estimates published in the
Monetary Policy Reports available at the date (red squares; the filter is anchored on them under the adopted
configuration). Projection: the gap path of the conditional forecast (IS curve under the scenario's real-rate
path) with 50% and 90% bands from the stochastic simulations. For the November 2026 origin the gap is 0.35%
in 2026Q4 (BCB 0.4% in 2026Q3) and closes gradually to about 0.1% by 2028Q2, with a 90% band of about
±2.4 pp at that horizon; the unemployment-implied gap alone would be above 4%, which is why the anchor matters.

## 11. Backtest results with the adopted configuration

Tables in `results_backtest.md`; figures `output/figs/08_backtest_panels.png` (unadjusted and adjusted paths with
bands) and `08_backtest_scatter.png`. Model minus Copom, percentage points of the IPCA, 26 targets at ten meetings:

| | mean | MAE | MAE current year | MAE next year | MAE horizon |
|---|---|---|---|---|---|
| unadjusted (adopted configuration) | −0.11 | 0.25 | 0.35 | 0.13 | 0.22 |
| adjusted by the previous meeting's error (weight 1) | +0.02 | 0.24 | 0.37 | 0.16 | 0.17 |
| first version (Section 7, no gap anchor) | +0.07 | 0.33 | 0.33 | 0.35 | 0.32 |

* The gap anchor halves the next-year and horizon differences; the current-year figure, dominated by the nowcast,
  is unchanged. The remaining large differences are the January and March 2026 meetings (−0.7 and −0.8 for 2026,
  −0.6 for 2027Q3) and the June 2026 meeting (−0.6 for 2026); elsewhere the model is within 0.3 pp of the Copom.
* The adjustment with the previous meeting's error removes the bias and improves the horizon projections (MAE
  0.22 to 0.17, nine of ten adjusted horizon figures within 0.3 pp of the Copom's), but adds noise to the current-
  year figure when the sign of the error changes between meetings (April 2026: +0.6 after −0.8 in March). Over the
  weights 0 to 1, the mean absolute difference is lowest at 0.25 overall (0.22) and at 1 for the horizon (0.18);
  the full adjustment is the default of the "adjusted" output and the weight can be set in the app.
* Realized 2025 inflation (4.26%): the model's July and September 2025 projections for 2025 were 5.3% (Copom 4.9
  and 4.8); from November both were within 0.3 pp of the outcome.

**Live projection with the adopted configuration** (meeting of 3-4 November 2026, Focus of 23 October 2026,
exchange rate 5.05, Focus administered medians 4.8/4.0/3.6, output gap 0.35% and neutral rate 5.5% at the origin;
`notebooks/07_projection.py`, outputs `output/07_projection_20261104_table*.csv`):

| IPCA, % | 2026 | 2027 | 2028 | 2028Q2 (relevant horizon) |
|---|---|---|---|---|
| unadjusted | 5.2 (free 5.3) | 4.2 (4.3) | 2.8 (2.5) | 3.1 (2.8) |
| adjusted by the error at the September 2026 meeting (+0.12 / +0.31 / +0.17) | 5.1 | 3.9 | 2.5 | 2.9 |
| Focus medians of 23 October 2026 | 5.2 | 4.3 | 3.75 | |

Under the Focus Selic path (13.5% in November 2026 falling to 10% by end-2028) the model has inflation below the
target from 2028, with the reversal of the El Niño term in 2027H2-2028H1 contributing to the decline.
