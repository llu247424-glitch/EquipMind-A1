@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [EquipMind A1] Python 3 was not found.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [EquipMind A1] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)
set "PY=%CD%\.venv\Scripts\python.exe"

"%PY%" -c "import streamlit, sklearn, dotenv, PIL, pandas, requests, altair, pypdf, docx" >nul 2>nul
if errorlevel 1 (
  echo [EquipMind A1] Installing Python dependencies...
  "%PY%" -m pip install --upgrade pip
  if exist "wheels" (
    "%PY%" -m pip install --no-index --find-links wheels -r requirements.txt
  ) else (
    "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 "%PY%" -m pip install -r requirements.txt
  )
  if errorlevel 1 exit /b 1
)

"%PY%" scripts\prepare_demo.py
if exist ".env" (
  "%PY%" scripts\test_llm.py
  if errorlevel 1 echo [EquipMind A1] LLM unavailable; continuing with local evidence-summary mode.
) else (
  echo [EquipMind A1] No .env found; local evidence-summary mode is enabled.
)

echo [EquipMind A1] Starting at http://127.0.0.1:8501
"%PY%" -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
endlocal
