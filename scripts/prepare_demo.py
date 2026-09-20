from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "scripts/build_index.py"])
    run([sys.executable, "scripts/check_data.py"])
    run([sys.executable, "tests_smoke.py"])
    run([sys.executable, "scripts/quality_gate.py"])
    print("\n演示环境准备完成。下一步运行：streamlit run app.py")


if __name__ == "__main__":
    main()
