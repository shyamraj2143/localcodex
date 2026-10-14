@echo off
setlocal EnableDelayedExpansion
set "EMICODER_ROOT=%~dp0"
if exist "%EMICODER_ROOT%.venv\Scripts\python.exe" (
  "%EMICODER_ROOT%.venv\Scripts\python.exe" "%EMICODER_ROOT%main.py" %*
  exit /b !errorlevel!
)
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3 "%EMICODER_ROOT%main.py" %*
  exit /b !errorlevel!
)
python -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install Python 3.10 or newer and add it to PATH.
  exit /b 1
)
python "%EMICODER_ROOT%main.py" %*
exit /b !errorlevel!
