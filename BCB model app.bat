@echo off
rem Opens the native desktop interface (no browser, no console). Uses the project's virtual environment when present.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "app\desktop_app.py"
) else (
    start "" pythonw "app\desktop_app.py"
)
