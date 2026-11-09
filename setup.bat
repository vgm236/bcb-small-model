@echo off
rem One-time setup on a new computer: creates the virtual environment, installs the pinned packages, downloads the
rem reference documents and the data, and builds the quarterly dataset. Needs Python 3.8+ on the PATH and internet.
cd /d "%~dp0"
echo === 1/5 virtual environment
if not exist ".venv\Scripts\python.exe" python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo === 2/5 reference documents (BCB reports and Copom minutes)
".venv\Scripts\python.exe" src\download_refs.py
echo === 3/5 data download (BCB SGS and Focus, IBGE, OECD, EIA, NOAA, ANEEL, BLS)
".venv\Scripts\python.exe" notebooks\01_download.py
if errorlevel 1 goto :fail
echo === 4/5 quarterly dataset
".venv\Scripts\python.exe" notebooks\02_build_dataset.py
if errorlevel 1 goto :fail
echo === 5/5 administered-price data
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, 'src'); import administered; administered.build_data(refresh=True); print('administered data ready')"
echo.
echo Setup complete. Next: build_model.bat (estimation, backtests, projections; about two hours), then "BCB model app.bat".
pause
exit /b 0
:fail
echo.
echo Setup stopped because of an error (see the messages above).
pause
exit /b 1
