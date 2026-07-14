@echo off
setlocal
rem ---------------------------------------------------------------
rem  PCB checking tool - install prerequisites (Windows)
rem  Installs the python packages and verifies the setup.
rem ---------------------------------------------------------------
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then re-run.
    pause
    exit /b 1
)

python --version
echo.
echo Installing packages from requirements.txt
echo (torch is ~200 MB - the download can take a few minutes) ...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package installation FAILED - see the messages above.
    pause
    exit /b 1
)

echo.
python tools\check_install.py
echo.
pause
