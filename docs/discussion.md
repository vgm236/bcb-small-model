# Discussion: how the reconstruction was iterated towards the BCB estimates

This note records the discrepancies found at each stage, the diagnosis, and what was changed (or deliberately
not changed). Numbers are posterior modes; the BCB values are from Table 1 of the June-2024 box. The final
tables are in `results.md`; the full history of the runs is in `output/*.log`.

## 1. First pass (default calibrations) and what was fixed

* **IS curve.** With a crisis-shock s.d. multiple of 3, the gap block returned $\beta_1=0.56$ and $\beta_4=0.41$
  against 0.85 and 0.054. Diagnosis: the 2009 and 2020 collapses of activity were being "explained" by the world
  gap (which falls by 8–9% in 2020Q2) instead of by the crisis shocks, so the world-gap coefficient absorbed them
  and the persistence of the gap was pulled down. The box states that the crisis shocks have "variâncias
  calibradas relativamente elevadas" precisely to avoid this. A sweep over the multiple (3, 5, 10, 20, 40) moves
  the block monotonically onto the BCB values and flattens out from 20 onwards
  (`output/03c_crisis_sweep.csv`); 20 was adopted. The other parameters of the block ($\beta_2$, $\beta_3$,
  $\beta_5$, the loadings, $\sigma^h$) were already inside the BCB bands at the first pass, and the robustness
  table (`output/03b_gap_variants.csv`) shows the adopted baseline is insensitive to the unemployment splice,
  the neutral-rate random-walk variance and the demeaning window.
* **Climate terms.** Unrestricted least squares on the Phillips curve returned $\alpha_5=0.13$, $\alpha_6=0.19$ with
  the ONI in degrees, two orders of magnitude above the BCB support U[0, 0.01], and the Bayesian posterior was flat
  on the support (uninformative). Diagnosis: a scale mismatch; the BCB support and modes, together with the
  contributions shown in the 2017 box (up to ≈1.5 pp in 2015–16), require $Clima^2$ to be of order $10^2$, i.e.
  the anomaly in tenths of a degree. With that scaling the estimates are $\alpha_5=0.0013$ (vs 0.0012) and
  $\alpha_6=0.0019$–0.0028 (vs 0.0007, band up to 0.0021).
* **Commodity prices in the system simulations.** The first full-system run treated IC-Br inflation in reais as a
  purely exogenous process, so a depreciation did not raise imported inflation and the exchange-rate pass-through
  was far too low (0.22 pp instead of ≈0.9). Footnote 4 of the box says the imported-inflation terms incorporate
  the exchange-rate variation, so the term was split into an exogenous USD component plus $\Delta e_t$ and the
  system was re-estimated (the coefficient estimates barely moved; the first run is archived in
  `output/run1_without_commodity_link/`).
* **Gap-shock experiment.** Using the persistent IS shock $s_t$ for the "+1 pp gap" experiment produced an
  implausibly long-lived gap; the box's chart shows a one-off shock decaying at $\beta_1$, which the crisis-type
  white-noise shock reproduces.
* **Expectations equation.** The first 2SLS run failed on a duplicated regressor column (a code bug, fixed).

## 2. Remaining differences after the iteration

* **Inertia in the Phillips curve.** $\alpha_{1I}$ is 0.22 (Layer A) and 0.24 (system) against the BCB's 0.38,
  although inside its 90% band [0.13; 0.69]; the implied weight on expectations is 0.49–0.60 versus 0.38. The
  sensitivity table shows the main driver is the treatment of 2020Q2–2022Q4: without down-weighting,
  $\alpha_{1I}$ rises to 0.33 (and $\alpha_2$ to 0.030); with a variance multiple of 3 instead of an s.d. multiple of 3
  it is 0.26. The BCB does not report its multiple. Seasonal-adjustment method, smoothed vs raw Focus
  expectations and the exchange-rate lag move the coefficient by at most ±0.05. Rather than tuning the multiple
  to hit the published number, the baseline keeps the s.d. multiple of 3 and documents the range.
* **La Niña coefficient** ($\alpha_6$ 0.0023–0.0028 vs 0.0007, band up to 0.0021): the point estimate sits just above
  the BCB band; the posterior is wide because the sample contains few strong La Niña episodes.
* **Taylor rule.** Single-equation: $(\theta_1,\theta_2,\theta_3)=(1.45,-0.65,2.03)$; system: $(1.35,-0.49,2.20)$;
  BCB: $(1.48,-0.58,2.03)$. The overall smoothing $\theta_1+\theta_2$ is 0.80–0.86 versus 0.90, so the
  reconstructed rule adjusts slightly faster; the response to the expectations gap ($\theta_3\approx2$) is the same.
  Candidate explanations: the 12-month-ahead target (month-weighted) and the log transformation of the Selic.
* **Expectations, model-consistent weight.** $\varphi_2$ is 0.10 by 2SLS (vs 0.11) but 0.16 in the system (band
  [0.06; 0.13]); the system's expectations are somewhat more forward-looking than the BCB's, which together with
  the lower inertia makes the propagation of shocks shorter (see the impulse responses).
* **Caged loading** ($\gamma_{caged}$ 0.82–0.83 vs 0.69): the spliced Caged stock (old Caged until 2019, Novo
  Caged after) has a somewhat larger cyclical amplitude than the BCB's series apparently has.
* **World-gap coefficient** ($\beta_4$ 0.16–0.22 vs 0.054): inside the band but higher; the trade weights and the
  country coverage of the world GDP index differ from the BCB's time-varying export shares.

## 3. Checks that passed without adjustment

* Real-rate coefficient $\beta_2$ (0.41–0.43 vs 0.44), fiscal coefficient $\beta_3$ (0.030, dominated by its prior),
  IS-shock persistence $\beta_5$ (0.85–0.86 vs 0.84), NUCI and unemployment loadings (1.96–1.97/1.17 vs 1.87/1.10),
  measurement-error s.d. (1.13–1.14 vs 1.09).
* Phillips curve: gap coefficient 0.11 vs 0.12, commodity coefficient 0.012–0.020 vs 0.023, exchange-rate
  coefficient 0.009–0.019 vs 0.011, El Niño 0.0013 vs 0.0012, free-price inertia 0.18–0.27 vs 0.24.
* UIP coefficient 1.7–1.9 vs 1.9; expectations equation (2SLS) within 0.05 on every coefficient.
* Neutral real rate implied by the IS block of the system: 3.3% in 2021Q4 (the December-2021 box reported 3.6%),
  5.2% in 2023Q2 and 5.8% in 2023Q4 (the June-2024 neutral-rate box reports 5.5% for the aggregate model in
  2023Q2 and 2024Q2).
* Output gap: ≈−7% in 2020Q2, +0.5 to +1% in 2023, consistent with the range of the BCB's multivariate measures
  (RPM June 2025 box); the system and Layer-A gaps have correlation 0.998.
* Impulse responses: a 1 pp Selic increase held for four quarters lowers 4-quarter IPCA inflation by 0.21 pp at
  the peak (box: 0.27); a permanent 10% depreciation raises it by 0.80 pp with the administered-price add-on
  (box: 0.96), free prices by 0.49 pp (box: 0.72); a one-off 1 pp gap shock raises it by 0.36 pp (box: 0.49).
  The ordering, timing and magnitude of the channels match; the responses are 20–30% smaller, consistent with
  the lower estimated inertia.
