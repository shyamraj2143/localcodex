@echo off
setlocal EnableDelayedExpansion
set "CODEX_ROOT=%~dp0"
if exist "%CODEX_ROOT%.venv\Scripts\python.exe" (
  "%CODEX_ROOT%.venv\Scripts\python.exe" "%CODEX_ROOT%main.py" %*
  exit /b !errorlevel!
)
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3 "%CODEX_ROOT%main.py" %*
  exit /b !errorlevel!
)
python -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install Python 3.10 or newer and add it to PATH.
  exit /b 1
)
python "%CODEX_ROOT%main.py" %*
exit /b !errorlevel!
