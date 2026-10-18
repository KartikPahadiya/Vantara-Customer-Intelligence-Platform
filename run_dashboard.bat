@echo off
REM Vantara dashboard launcher — always uses the project's Python 3.12 venv.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Run: Python312\python.exe -m venv .venv  then install requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run frontend\dashboard.py
