@echo off
rem Rebuilds every result from the downloaded data: estimation of the small model (two layers, MCMC), comparison
rem with the BCB's table, diagnostics, backtests, the administered-price model and the projection for the next
rem meeting. Run setup.bat first. Total time about two hours (the MCMC of step 05 is the longest part; pass a
rem smaller number of draws as the first argument to shorten it, e.g. build_model.bat 10000).
cd /d "%~dp0"
set DRAWS=%1
if "%DRAWS%"=="" set DRAWS=30000
set PY=".venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
for %%S in (03_gap_is 03b_gap_variants 04_single_equations) do (
    echo === %%S
    %PY% notebooks\%%S.py || goto :fail
)
echo === 05_system (MCMC draws: %DRAWS%)
%PY% notebooks\05_system.py %DRAWS% || goto :fail
for %%S in (05b_system_states 06_compare 09_diagnostics 08_backtest_copom 08b_backtest_report 10_administered 11_backtest_administered 07_projection) do (
    echo === %%S
    %PY% notebooks\%%S.py || goto :fail
)
echo.
echo Build complete: results in output\, documentation tables in docs\. Start the interface with "BCB model app.bat".
pause
exit /b 0
:fail
echo.
echo Build stopped because of an error (see the messages above).
pause
exit /b 1
