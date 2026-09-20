"""Test the configured LLM endpoint without starting Streamlit.

Usage:
    python scripts/test_llm.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import llm_config_summary, test_llm_connection  # noqa: E402


if __name__ == "__main__":
    print("Current LLM configuration:")
    for key, value in llm_config_summary().items():
        print(f"- {key}: {value}")
    ok, message = test_llm_connection()
    print("\nResult:", "OK" if ok else "FAILED")
    print(message)
    raise SystemExit(0 if ok else 1)
