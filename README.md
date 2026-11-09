# Reconstruction of the Banco Central do Brasil small-scale aggregate model

A from-scratch Python reconstruction of the BCB's **modelo semiestrutural agregado de pequeno porte** in the
specification of the Inflation Report box of June 2024 ("Atualização dos modelos semiestruturais de pequeno
porte"), the version that the 2025–2026 Monetary Policy Reports still reference. Every input series is
downloaded from public sources (BCB SGS and Focus, IBGE, Ipeadata, OECD, NOAA), the derived variables (output
gap indicators, ex-ante real rate and its trend, fiscal impulse, world gap, climate terms) are built here, the
model is estimated by Bayesian methods (posterior mode + MCMC, Kalman filter, rational-expectations solution),
and the estimates are compared with the coefficients published by the BCB.

## Headline results

Sample 2003Q4–2023Q4 (the box's sample). Of the 23 parameters published by the BCB, 20 (block-by-block
estimation) and 18 (full-system estimation) posterior modes fall inside the BCB's 90% credible intervals, and the
BCB's mode falls inside ours for 22 and 19 parameters respectively. Selected comparisons (ours, system / BCB):
IS curve 0.78, 0.41, 0.030, 0.22, 0.86 / 0.85, 0.44, 0.030, 0.054, 0.84; Phillips curve gap 0.11 / 0.12,
commodities 0.012 / 0.023, exchange rate 0.019 / 0.011, El Niño 0.0013 / 0.0012; Taylor rule 1.35, −0.49, 2.20 /
1.48, −0.58, 2.03; UIP 1.85 / 1.90; expectations 0.69, 0.16, 0.02 / 0.75, 0.11, 0.02. The main remaining gap is
the lower inertia of IPCA inflation in the Phillips curve (0.24 vs 0.38), which also makes the impulse responses
20–30% smaller than the box's. Two calibration findings were needed to get there: the ONI climate anomaly must be
in tenths of a degree, and the 2008/2020 crisis shocks need a large variance (s.d. multiple ≈20). See
`docs/results.md` and `docs/discussion.md`.

## Projections in the Copom's format and backtest

The model can be run as the BCB runs its reference scenario (`notebooks/07_projection.py`): Selic path from the
Focus survey, exchange rate from the 10-business-day PTAX average (rounded to R$0.05) with PPP drift, Brent from
the spot price and a futures-curve proxy for six months then +2% a year, administered prices as an outside input
(the Copom's projection or the Focus medians), neutral real rate of 5%, climate term from the latest Niño 3.4
readings; the current quarter is nowcast with the Focus monthly medians and the paths are imposed through
anticipated shocks. The backtest at the Copom meetings since July 2025 (ten meetings to September 2026, extended automatically as new
minutes are published; `notebooks/08_backtest_copom.py`) compares the output with Table 1 of the minutes. The
diagnostics of `notebooks/09_diagnostics.py` locate the main source of the differences in the output gap at the
origin (about 0.5 pp above the estimate the Copom publishes); anchoring the gap on the BCB's published estimates is
therefore part of the adopted configuration. With it, across the 26 published targets the model is on average
0.11 pp below the Copom's IPCA projections with a mean absolute difference of 0.25 pp (0.32 pp for free prices;
0.22 pp at the relevant horizon). A second output adjusts each projection by the model's error at the previous
meeting: it removes the bias (+0.02 pp) and lowers the horizon difference to 0.17 pp. The larger remaining
differences (0.6-0.8 pp for the current year at the January-March 2026 meetings) come from short-term information
the quarterly model does not have. See `docs/projection.md` and `docs/results_backtest.md`.

## Interface (desktop application)

Double-click **`BCB model app.bat`**: it opens a native window (Tkinter, no browser, no console) from the project's
virtual environment. Tabs: **Next meeting** (reference scenario for the next scheduled Copom meeting: Focus survey,
PTAX convention, Brent, administered prices from the annual figures or from the BCB's administered-price model;
unadjusted output and output adjusted by the model's error at the previous meeting, bands, assumptions),
**Output gap** (smoothed and filtered estimates, indicator-implied gaps, the BCB's published estimates and the projection ahead with bands, chart and table), **Scenario builder** (own Selic path by quarter, own BRL/USD path, administered inflation by year, the 12-month
expectation, and shocks to inflation expectations relative to the Focus survey: a shock at the origin and/or
quarter-by-quarter shocks over the projection, imposed as anticipated expectation shocks; compared with the
reference scenario), **Administered prices** (the BCB's administered-price model:
coefficients versus the BCB's, fit), **Backtest** (model versus the Copom at every meeting since July 2025,
unadjusted and adjusted; a button updates the Copom history from new minutes and re-runs the backtest) and
**Data and updates** (refresh the inputs) and **How it works** (what each tab does, why, and what to watch). The next meeting, the relevant horizon (six quarters ahead) and the
previous-meeting error move forward automatically with the calendar and with the minutes published by the BCB.
Model settings (neutral rate, climate assumption, gap anchor, coefficients, administered source, tariff flag,
bands, weight of the adjustment) are on the first tab. Long computations run in the background and the window
stays responsive; errors are written to `app/desktop_app.log`. `python app/desktop_app.py --selftest` exercises
every tab and saves screenshots in `output/figs/desktop_*.png`.

A web version of the same interface (Streamlit) is available with `run_app.bat`, which opens the browser.

## Source package (to rebuild elsewhere)

`python tools/make_package.py` writes `BCB_model_source.zip` (about 0.3 MB): the code, the scripts, the interfaces,
the documentation, `requirements.txt` and the setup scripts, without the downloaded data, the reference PDFs, the
results or the virtual environment. On another computer, unzip and run `setup.bat` (environment, references, data,
dataset; about 10 minutes) and then `build_model.bat` (estimation, backtests, projections; about two hours, or
`build_model.bat 10000` for a faster MCMC). `INSTALL.md` inside the package has the details.

To send it by e-mail, `python tools/make_email_package.py` writes `email_package/`: the same zip with the launchers
stored as `*.bat.txt` (mail servers block `.bat` and, in Outlook, `.py` attachments, and look inside zip archives
for them), encoded as a plain-text attachment `BCB_model_source.b64.txt` (0.3 MB; base64, decoded on the other
side with the built-in `certutil -decode`, or `base64 -d`, or Python) plus `DECODE_INSTRUCTIONS.txt` to paste into
the message, with the SHA-256 of the zip. When the text is larger than 10 MB it is split into parts (`--part-mb`
sets the cap), to be sent in several messages and joined before decoding; `--include-data` also packs the
downloaded data and the results (29 MB zip, 4 parts), so that the recipient does not have to download anything.
After unzipping, `python tools/restore_launchers.py` gives the launchers their `.bat` names back.

The complete project, with the downloaded data, the reference documents and the results (about 140 MB), is in the
GitHub repository https://github.com/vgm236/bcb-small-model (`git clone https://github.com/vgm236/bcb-small-model.git`).
Only the virtual environment, the caches and the derived packages are left out, so after cloning it is enough to
create the environment (`python -m venv .venv` and `.venv\Scripts\python.exe -m pip install -r requirements.txt`,
step 1 of `setup.bat`): the interface, the notebooks and the stored results work without downloading or
re-estimating anything.

## Layout

```
README.md                      this file
BCB model app.bat              launcher of the desktop interface (app/desktop_app.py, Tkinter, run in .venv without a console)
app/desktop_app.py             the desktop interface: next meeting, scenario builder, administered prices, backtest, data updates
run_app.bat                    launcher of the web version of the interface (Streamlit app in app/app.py)
app/app.py                     the web interface (same pages)
.venv/                         virtual environment with pinned packages (see Requirements)
docs/model_specification.md    equations, data sources, construction of every variable, estimation strategy
docs/results.md                estimation results and the comparison with the BCB (written by notebooks/06)
docs/discussion.md             what differed at first, why, and what was changed
docs/projection.md             projection method (Copom conventions, information sets, conditioning) and backtest design
docs/results_backtest.md       backtest tables (written by notebooks/08b)
docs/administered.md           the administered-price model: specification, data, estimates versus the BCB, backtest, correction
docs/results_administered.md   coefficient table, fit and projection of the administered model (written by notebooks/10)
refs/                          BCB boxes and reports used as references (PDF), refs/txt extracted text
refs/copom/atas                minutes of the Copom meetings 270-281 (PDF and extracted text)
src/config.py                  paths, series codes, sample settings
src/fetch.py                   download layer with caching (BCB SGS/Focus/Copom, SIDRA, Ipeadata, NOAA, OECD)
src/tsutil.py                  quarterly aggregation, HP filter, STL seasonal adjustment, splicing
src/build_dataset.py           construction of the quarterly model dataset
src/ssm.py                     Kalman filter and smoother
src/bayes.py                   priors, posterior-mode optimisation, Laplace covariance, adaptive RW-MH
src/re_solver.py               gensys (linear rational-expectations solver) with self-tests
src/gap_filter.py              Layer A: output-gap / IS block as an unobserved-components model
src/single_eq.py               Layer A: Phillips curve, Taylor rule, UIP, expectations (single equations)
src/model_system.py            Layer B: full model as a state-space RE system
src/copom_meetings.py          the last ten Copom meetings: assumptions and published projections (from the minutes)
src/vintage.py                 real-time dataset for an information date (publication lags, nowcast of the current quarter)
src/projection.py              conditional projections: reference-scenario paths, news-shock conditioning, Copom-style tables, adopted settings
src/adjust.py                  adjustment of the projections by the model's error at the previous meeting
src/backtest.py                backtest runner (unadjusted and adjusted outputs, stored per meeting), used by notebook 08 and the app
src/administered.py            the BCB's administered-price model (June-2025 box): 24 item equations, data, estimation, simulation
src/adm_projection.py          administered model inside the reference scenario: real-time data, drivers, flag path, feedback with free prices
src/fetch_adm.py               data for the administered model: IBGE subitems (SIDRA), ANEEL tariff flags, BLS US CPI
notebooks/01_download.py       VS Code interactive scripts (# %% cells), to be run in order
notebooks/02_build_dataset.py
notebooks/03_gap_is.py, 03b_gap_variants.py (robustness), 03c sweep is inline in output/03c_crisis_sweep.*
notebooks/04_single_equations.py
notebooks/05_system.py, 05b_system_states.py
notebooks/06_compare.py
notebooks/07_projection.py     conditional projection for a Copom meeting (parameters at the top; REFRESH=True re-downloads)
notebooks/08_backtest_copom.py backtest at all meetings since July 2025 versus the Copom's projections (unadjusted and adjusted); 08b writes docs/results_backtest.md
notebooks/09_diagnostics.py    variants that locate the sources of the differences from the Copom
notebooks/10_administered.py   administered-price model: re-estimation versus the BCB's calibration, fit, projection (writes docs/results_administered.md)
notebooks/11_backtest_administered.py  backtest of the administered model against the Copom's administered projections
data/raw, data/processed       cached downloads and the model dataset (model_data.csv + dictionary)
output/                        estimation outputs (csv, npy chains, logs, figures in output/figs)
src/download_refs.py           downloads the BCB boxes and reports into refs/ and extracts their text; fetches the known minutes
tools/make_package.py          builds BCB_model_source.zip (code, scripts, docs, no data)
tools/make_email_package.py    builds email_package/: the zip as a plain-text (base64) attachment in parts under 10 MB, with decode instructions
tools/restore_launchers.py     after unzipping an e-mail package, renames the launchers *.bat.txt -> *.bat
email_package/                 output of make_email_package.py (the text attachment(s) and DECODE_INSTRUCTIONS.txt)
setup.bat, build_model.bat     one-time setup on a new computer; full rebuild of the results
INSTALL.md, requirements.txt   installation notes and the pinned environment
```

## Requirements

Python 3.8+, numpy, pandas, scipy, statsmodels, matplotlib, requests, xlrd, openpyxl, pdfminer.six (reference text
extraction and the minutes parser); the interface needs streamlit and plotly. No API keys are needed. The project
ships a virtual environment `.venv` (Python 3.8, numpy 1.24.4, pandas 2.0.3, scipy 1.10.1, statsmodels 0.14.1,
matplotlib 3.7.5, streamlit 1.40.1, plotly 5.24.1) used by `run_app.bat`; the notebooks run both in it and in the
Anaconda base environment (numpy 1.20 / pandas 1.2) with identical results. To recreate the environment:

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install numpy==1.24.4 pandas==2.0.3 scipy==1.10.1 statsmodels==0.14.1 matplotlib==3.7.5 streamlit==1.40.1 plotly==5.24.1 requests xlrd openpyxl pdfminer.six
```

## Running

```bash
python notebooks/01_download.py
python notebooks/02_build_dataset.py
python notebooks/03_gap_is.py
python notebooks/04_single_equations.py
python notebooks/05_system.py 30000         # argument: number of MCMC draws (about 45 minutes)
python notebooks/05b_system_states.py
python notebooks/06_compare.py
python notebooks/09_diagnostics.py          # variants that locate the sources of the differences from the Copom (~10 minutes)
python notebooks/08_backtest_copom.py 300   # backtest at all meetings since July 2025, unadjusted and adjusted outputs (argument: draws; ~8 minutes with the variants)
python notebooks/08b_backtest_report.py     # writes docs/results_backtest.md
python notebooks/07_projection.py           # projection for the next Copom meeting, unadjusted and adjusted (parameters at the top)
python notebooks/10_administered.py refresh # administered-price model: download, re-estimate, compare with the BCB, project (~1 minute)
python notebooks/11_backtest_administered.py  # administered model versus the Copom's administered projections at every meeting (~10 minutes)
"BCB model app.bat"                         # the desktop interface (run_app.bat: web version)
```

Each script can also be executed cell by cell in VS Code (Python interactive window).
