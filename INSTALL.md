# Installing and rebuilding the project from the source package

`BCB_model_source.zip` (built by `python tools/make_package.py`) contains everything needed to rebuild the whole
project on another computer: the code (`src/`), the step-by-step scripts (`notebooks/`), the interfaces (`app/`,
`BCB model app.bat`, `run_app.bat`), the documentation (`docs/`, `README.md`), the pinned environment
(`requirements.txt`) and the setup scripts. It does **not** contain the downloaded data (`data/`), the reference
PDFs (`refs/`), the results (`output/`) or the virtual environment (`.venv/`); all of them are recreated from
public sources by the steps below.

The GitHub repository https://github.com/vgm236/bcb-small-model holds the complete project, data, references and
results included (about 140 MB); after `git clone` only step 1 of `setup.bat` (the virtual environment) is needed.

## 0. If the package arrived by e-mail

Mail servers block `.bat` (and, in Outlook, `.py`) attachments and look for them inside zip archives, so the package
travels as a plain-text attachment, `BCB_model_source.b64.txt` (the zip encoded in base64; built by
`python tools/make_email_package.py`), possibly in several parts, with the steps in the message body
(`DECODE_INSTRUCTIONS.txt`). In short: save the attachment(s) in one folder; if there are several parts, join them in
order (`cmd /c copy /b part01+part02+... BCB_model_source.zip.b64.txt`); decode with
`certutil -decode <text file> BCB_model_source.zip` (Windows; `base64 -d` on macOS/Linux; or Python); check the
SHA-256 given in the message with `certutil -hashfile BCB_model_source.zip SHA256`; unzip; and in the folder
`BCB model` run `python tools/restore_launchers.py`, which renames `*.bat.txt` back to `*.bat`. Then continue below.

## 1. Requirements

* Windows (the launchers are .bat files; the Python code itself also runs on Linux/macOS), Python 3.8 or newer on
  the PATH (the project was built with Python 3.8.8 and the versions in `requirements.txt`), internet access.
* About 300 MB for the virtual environment and 150 MB for the data, references and results.

## 2. Setup (once)

Unzip, open the folder `BCB model` and double-click **`setup.bat`**. It creates `.venv`, installs the packages,
downloads the BCB boxes and reports (`src/download_refs.py`), the data (`notebooks/01_download.py`: BCB SGS and
Focus, IBGE SIDRA, OECD, EIA, NOAA, ANEEL, BLS; no API keys) and builds the quarterly dataset
(`notebooks/02_build_dataset.py`) and the administered-price data. About 10 minutes.

If a download fails (a site temporarily unavailable), run the step again later; the downloads are cached and
only missing files are fetched. The published output-gap estimates and the Copom projections used by the
projection engine are also hard-coded in the source as a fallback, so the projections work even if the report
texts could not be downloaded.

## 3. Rebuilding the results

Double-click **`build_model.bat`** (or run the scripts in `notebooks/` in numerical order). It re-estimates the
small model block by block and as a full system (Bayesian estimation with MCMC; about 45 minutes for the system
with 30,000 draws; `build_model.bat 10000` is faster and adequate for the projections), compares the coefficients
with the BCB's table, runs the diagnostics and the backtests at every Copom meeting, estimates the administered-
price model and produces the projection for the next meeting. Results are written to `output/` and the generated
documentation tables to `docs/results*.md`.

The interface does not need the full rebuild: after `setup.bat`, the projections only require the posterior mode
of the small model (`output/05_system_mode.csv`, produced by `notebooks/05_system.py`; run
`python notebooks/05_system.py 2000` for a quick version) and the stored backtest for the adjusted output
(`notebooks/08_backtest_copom.py 300 novariants`, about three minutes).

## 4. Using it

* **`BCB model app.bat`**: the desktop interface (next meeting, output gap, scenario builder with own Selic/BRL
  paths and expectation shocks, administered prices, backtest, data updates, how it works).
* **`run_app.bat`**: the same interface in the browser (Streamlit).
* Before a new Copom meeting: tab "Data and updates" (refresh inputs), and after the minutes come out, tab
  "Backtest" (update history and re-run), so that the adjusted output uses the latest meeting's error.

## 5. What each folder is for

| folder / file | content |
|---|---|
| `src/` | data download (`fetch.py`, `fetch_adm.py`), dataset (`build_dataset.py`, `vintage.py`), estimation (`ssm.py`, `bayes.py`, `re_solver.py`, `gap_filter.py`, `single_eq.py`, `model_system.py`), projections (`projection.py`, `adjust.py`, `backtest.py`), administered prices (`administered.py`, `adm_projection.py`), Copom history (`copom_meetings.py`), references (`download_refs.py`) |
| `notebooks/` | the pipeline as VS Code cell scripts, numbered in run order |
| `app/` | desktop interface (`desktop_app.py`, `gap_tab.py`, `help_text.py`) and web interface (`app.py`) |
| `docs/` | specification, estimation results, projection method and backtests, administered-price model |
| `tools/make_package.py` | rebuilds this source package |
| `tools/make_email_package.py`, `tools/restore_launchers.py` | e-mail form of the package (plain-text attachment, parts under 10 MB, decode instructions) and the launcher restore after unzipping |

See `README.md` for the results and the methodological summary.
