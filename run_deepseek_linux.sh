#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .env ]; then
  echo "[EquipMind A1] .env is missing. Run: python scripts/init_deepseek.py" >&2
fi
EQUIPMIND_TEST_LLM=1 exec bash run_linux.sh
