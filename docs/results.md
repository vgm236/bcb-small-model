# Results: reconstruction vs the BCB June-2024 box

Sample 2003Q4–2023Q4, quarterly. BCB values are the posterior modes and 90% credible intervals of Table 1 of the box (aggregate model); the December-2021 vintage (sample 2003Q4–2019Q4) is shown for context. Layer A = block-by-block estimation (gap block, then single equations given the smoothed gap); Layer B = full-system Bayesian estimation with the Kalman filter and rational expectations.

| Parameter | Description | BCB 2024 mode [90%] | BCB 2021 | Layer A mode [90%] | Layer B mode [90%] |
|---|---|---|---|---|---|
| a1L | Phillips: inertia of free-price inflation | 0.240 [0.020; 0.380] | 0.238 | 0.178 [0.029; 0.340] | 0.271 [0.005; 0.439] |
| a1I | Phillips: inertia of IPCA inflation | 0.380 [0.130; 0.690] | 0.256 | 0.222 [0.033; 0.408] | 0.237 [0.051; 0.465] |
| a2 | Phillips: imported (commodity) inflation | 0.023 [0.006; 0.039] | 0.018 | 0.020 [0.004; 0.034] | 0.012 [0.003; 0.026] |
| a3 | Phillips: exchange-rate change (lag 1) | 0.011 [0.000; 0.025] | 0.017 | 0.009 [0.001; 0.018] | 0.019 [0.005; 0.030] |
| a4 | Phillips: output gap | 0.120 [0.072; 0.198] | 0.139 | 0.112 [0.041; 0.172] | 0.112 [0.065; 0.174] |
| a5 | Phillips: climate, El Nino | 0.0012 [0.0004; 0.0019] | 0.0012 | 0.0013 [0.0004; 0.0020] | 0.0013 [0.0004; 0.0019] |
| a6 | Phillips: climate, La Nina | 0.0007 [0.0000; 0.0021] | 0.0010 | 0.0023 [0.0007; 0.0037] | 0.0028 [0.0010; 0.0041] |
| b1 | IS: autoregressive | 0.850 [0.700; 0.950] | 0.739 | 0.840 [0.695; 0.952] | 0.777 [0.650; 0.932] |
| b2 | IS: real interest-rate gap | 0.440 [0.210; 0.660] | 0.549 | 0.425 [0.140; 0.716] | 0.411 [0.173; 0.741] |
| b3 | IS: primary result (Beta prior) | 0.030 [0.027; 0.032] | 0.030 | 0.030 [0.027; 0.033] | 0.030 [0.027; 0.033] |
| b4 | IS: world gap | 0.054 [0.000; 0.230] | 0.043 | 0.159 [0.027; 0.275] | 0.221 [0.041; 0.347] |
| b5 | IS: AR of the IS shock | 0.840 [0.590; 0.990] |  | 0.850 [0.685; 0.970] | 0.862 [0.703; 0.957] |
| th1 | Taylor: smoothing, lag 1 | 1.480 [1.410; 1.540] | 1.457 | 1.452 [1.351; 1.579] | 1.346 [1.281; 1.421] |
| th2 | Taylor: smoothing, lag 2 | -0.580 [-0.630; -0.520] | -0.544 | -0.651 [-0.761; -0.561] | -0.491 [-0.555; -0.435] |
| th3 | Taylor: response to expected inflation gap | 2.030 [1.470; 2.640] | 1.300 | 2.028 [1.358; 2.735] | 2.201 [1.667; 2.695] |
| phi1 | Expectations: inertia | 0.750 [0.680; 0.820] | 0.733 | 0.799 [0.697; 0.901] | 0.695 [0.644; 0.782] |
| phi2 | Expectations: model-consistent | 0.110 [0.060; 0.130] | 0.123 | 0.096 [0.054; 0.138] | 0.157 [0.109; 0.202] |
| phi3 | Expectations: past inflation | 0.021 [0.000; 0.049] | 0.044 | 0.025 [-0.014; 0.063] | 0.024 [0.001; 0.044] |
| delta | UIP: interest differential | 1.900 [0.770; 3.220] | 1.718 | 1.729 [0.364; 2.770] | 1.853 [0.660; 2.884] |
| g_nuci | Gap: loading of NUCI | 1.870 [1.650; 2.120] | 2.089 | 1.962 [1.725; 2.248] | 1.970 [1.745; 2.290] |
| g_emp | Gap: loading of unemployment | 1.100 [0.960; 1.250] | 1.084 | 1.175 [1.018; 1.354] | 1.168 [1.032; 1.342] |
| g_caged | Gap: loading of Caged | 0.690 [0.610; 0.790] | 0.780 | 0.824 [0.721; 0.940] | 0.828 [0.731; 0.960] |
| sig_h | Gap: measurement-error s.d. | 1.090 [0.980; 1.210] |  | 1.139 [1.034; 1.294] | 1.133 [1.025; 1.255] |

* Layer A: 20 of 23 modes fall inside the BCB 90% interval; BCB's mode falls inside our 90% interval for 22 parameters.
* Layer B: 18 of 23 modes fall inside the BCB 90% interval; BCB's mode falls inside our 90% interval for 19 parameters.

## Impulse responses (full system, posterior mode)

| Experiment | BCB (box) | This model |
|---|---|---|
| Selic +1 pp for 4 quarters, then Taylor rule: max effect on IPCA accumulated in 4 quarters | −0.27 pp at the 4th quarter | -0.21 pp at quarter 3 (free-price channel only: -0.14 pp at quarter 4) |
| Permanent 10% depreciation: max effect on IPCA (4q) / free prices (4q) | 0.96 pp (with the administered-price model) / 0.72 pp | 0.80 pp (with the calibrated administered add-on) / 0.49 pp |
| One-off +1 pp output-gap shock: max effect on IPCA (4q) | 0.49 pp (7th quarter) | 0.36 pp (quarter 3) |

The IPCA responses combine the model's free-price response (weight 0.73) with the administered-price pass-through published in the box (1.65 pp over four quarters for a 10% depreciation) because the administered-price model is a separate BCB model. The smaller and earlier-peaking responses relative to the box are consistent with the lower estimated inertia (a1I 0.24 vs 0.38) and the larger weight on model-consistent expectations (phi2 0.16 vs 0.11), both of which shorten the propagation of shocks.

## Robustness of the gap block (posterior modes)

|                                     |    b1 |    b2 |   b3 |    b4 |    b5 |   g_nuci |   g_emp |   g_caged |   sig_h |   logpost |   gap_2023Q4 |   gap_2020Q2 |   rr_is_2023Q4 |
|:------------------------------------|------:|------:|-----:|------:|------:|---------:|--------:|----------:|--------:|----------:|-------------:|-------------:|---------------:|
| 1 baseline (crisis x20, sd_rr 0.15) | 0.84  | 0.425 | 0.03 | 0.159 | 0.85  |    1.962 |   1.175 |     0.824 |   1.139 |  -597.806 |        0.977 |       -7.168 |          6.768 |
| 2 PNADC only (u from 2012Q1)        | 0.83  | 0.43  | 0.03 | 0.174 | 0.855 |    1.955 |   1.141 |     0.826 |   1.079 |  -524.007 |        0.984 |       -7.239 |          6.746 |
| 3 no unemployment                   | 0.8   | 0.477 | 0.03 | 0.177 | 0.843 |    1.968 |   1.5   |     0.861 |   1.039 |  -439.92  |        0.651 |       -8.198 |          6.62  |
| 4 sd_rr = 0.30                      | 0.834 | 0.42  | 0.03 | 0.16  | 0.853 |    1.961 |   1.174 |     0.824 |   1.139 |  -598.046 |        0.996 |       -7.165 |          7.097 |
| 5 crisis multiple 3                 | 0.564 | 0.441 | 0.03 | 0.414 | 0.899 |    1.978 |   1.173 |     0.824 |   1.142 |  -594.368 |        0.858 |       -6.586 |          6.344 |
| 6 crisis multiple 5                 | 0.665 | 0.419 | 0.03 | 0.33  | 0.884 |    1.976 |   1.174 |     0.824 |   1.139 |  -594.505 |        0.885 |       -6.715 |          6.369 |
| 7 full-sample means                 | 0.84  | 0.426 | 0.03 | 0.16  | 0.85  |    1.954 |   1.174 |     0.823 |   1.137 |  -597.079 |        1.045 |       -7.116 |          6.829 |
| BCB June 2024 (mode)                | 0.85  | 0.44  | 0.03 | 0.054 | 0.84  |    1.87  |   1.1   |     0.69  |   1.09  |   nan     |      nan     |      nan     |        nan     |

## Phillips curve: sensitivity of the mode to data choices

|                                        |    a1L |    a1I |     a2 |     a3 |     a4 |     a5 |     a6 |
|:---------------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| no HV weighting                        | 0.211  | 0.3256 | 0.0295 | 0.0118 | 0.0914 | 0.0013 | 0.0024 |
| HV variance x3 (s.d. x1.73)            | 0.1954 | 0.255  | 0.0231 | 0.01   | 0.1044 | 0.0013 | 0.0024 |
| HV s.d. x5                             | 0.17   | 0.2114 | 0.0186 | 0.0089 | 0.1154 | 0.0013 | 0.0023 |
| seasonal adjustment by quarter dummies | 0.2199 | 0.1763 | 0.0186 | 0.0144 | 0.1153 | 0.0014 | 0.0027 |
| raw (non-smoothed) Focus               | 0.1714 | 0.2085 | 0.0195 | 0.0089 | 0.1113 | 0.0013 | 0.0023 |
| exchange-rate lag 2 (2021 spec)        | 0.1645 | 0.2609 | 0.0221 | 0.021  | 0.1192 | 0.001  | 0.0023 |
| sample 2003Q4-2019Q4 (2021 box)        | 0.1444 | 0.2236 | 0.018  | 0.0083 | 0.1348 | 0.0014 | 0.0022 |
| BCB 2024 (2003Q4-2023Q4)               | 0.24   | 0.38   | 0.023  | 0.011  | 0.12   | 0.0012 | 0.0007 |
| BCB 2021 (2003Q4-2019Q4)               | 0.2376 | 0.2557 | 0.0183 | 0.0173 | 0.1387 | 0.0012 | 0.001  |
