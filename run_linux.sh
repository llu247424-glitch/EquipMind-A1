#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY_BOOTSTRAP="${PYTHON:-}"
if [ -z "$PY_BOOTSTRAP" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BOOTSTRAP=python3
  elif command -v python >/dev/null 2>&1; then
    PY_BOOTSTRAP=python
  else
    echo "[EquipMind A1] Python 3 was not found." >&2
    exit 1
  fi
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[EquipMind A1] Creating virtual environment..."
  "$PY_BOOTSTRAP" -m venv .venv
fi
PY="$(pwd)/.venv/bin/python"

if ! "$PY" -c "import streamlit, sklearn, dotenv, PIL, pandas, requests, altair, pypdf, docx" >/dev/null 2>&1; then
  echo "[EquipMind A1] Installing Python dependencies..."
  "$PY" -m pip install --upgrade pip
  if [ -d wheels ] && find wheels -maxdepth 1 -type f | grep -q .; then
    "$PY" -m pip install --no-index --find-links wheels -r requirements.txt
  else
    "$PY" -m pip install -r requirements.txt -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" || \
      "$PY" -m pip install -r requirements.txt
  fi
fi

"$PY" scripts/loongarch_check.py || true
"$PY" scripts/prepare_demo.py

if [ "${EQUIPMIND_FULL_CHECKS:-0}" = "1" ]; then
  "$PY" scripts/run_all_checks.py
fi

if [ -f .env ] || [ "${EQUIPMIND_TEST_LLM:-0}" = "1" ]; then
  "$PY" scripts/test_llm.py || echo "[EquipMind A1] LLM unavailable; continuing with local evidence-summary mode."
else
  echo "[EquipMind A1] No .env found; local evidence-summary mode is enabled."
fi

echo "[EquipMind A1] Starting at http://127.0.0.1:${EQUIPMIND_PORT:-8501}"
exec "$PY" -m streamlit run app.py \
  --server.address "${EQUIPMIND_HOST:-0.0.0.0}" \
  --server.port "${EQUIPMIND_PORT:-8501}" \
  --server.headless true
