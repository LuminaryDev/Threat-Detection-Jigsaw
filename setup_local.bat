@echo off
REM ============================================================
REM setup_local.bat — Windows Local Environment Setup
REM Run this ONCE to create a virtual environment and install deps.
REM
REM USAGE:
REM   Double-click this file OR run in PowerShell:
REM   .\setup_local.bat
REM ============================================================

echo.
echo === NLP Project: Local Setup ===
echo.

REM Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/4] Creating virtual environment (.venv)...
python -m venv .venv

REM Activate it
echo [2/4] Activating .venv...
call .venv\Scripts\activate.bat

REM Upgrade pip silently
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip -q

REM Install dependencies
echo [4/4] Installing dependencies (requirements_training.txt)...
pip install -r requirements_training.txt

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To activate the environment next time, run:
echo    .venv\Scripts\activate
echo.
echo  To run the API server:
echo    uvicorn app.main:app --reload --port 8000
echo.
echo  To run the Gradio UI:
echo    python app\demo.py
echo.
echo  To run training notebook:
echo    python notebook\train.py
echo    OR open it in VS Code with the Jupyter extension
echo ============================================================
pause
