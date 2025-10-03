```batch
@echo off
ECHO "Setting up Trading Bot environment..."

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO "Python is not installed or not in PATH. Please install Python 3.11+ and try again."
    pause
    exit /b
)

REM Create a virtual environment if it doesn't exist
IF NOT EXIST venv (
    ECHO "Creating virtual environment..."
    python -m venv venv
)

ECHO "Activating environment and installing required packages..."
CALL .\venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt

ECHO "Setup complete. Launching application..."
python main.py

ECHO "Application closed."
pause

