@echo off
rem Alternative web interface (Streamlit): starts a local server and opens the default browser.
rem The native desktop interface is "BCB model app.bat".
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app\app.py --browser.gatherUsageStats false
) else (
    python -m streamlit run app\app.py --browser.gatherUsageStats false
)
pause
