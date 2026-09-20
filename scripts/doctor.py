from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.platform_check import platform_report
CHECKS = [
    ("app.py", ROOT / "app.py"),
    (".env.example", ROOT / ".env.example"),
    ("knowledge index", ROOT / "storage" / "knowledge_index.pkl"),
    ("manuals", ROOT / "data" / "manuals"),
    ("workflows", ROOT / "data" / "workflows"),
    ("approved cases", ROOT / "data" / "cases" / "approved"),
]
PACKAGES = ["streamlit", "requests", "dotenv", "sklearn", "PIL", "pandas", "altair", "pypdf"]


def main() -> None:
    print("EquipMind A1 环境检查")
    ok = True
    for name, path in CHECKS:
        exists = path.exists()
        ok = ok and exists
        print(("[OK] " if exists else "[MISS] ") + f"{name}: {path}")
    for package in PACKAGES:
        exists = importlib.util.find_spec(package) is not None
        ok = ok and exists
        print(("[OK] " if exists else "[MISS] ") + f"python package: {package}")

    env = platform_report()
    print(f"[INFO] architecture: {env['architecture']}")
    print(f"[INFO] operating system: {env['os']}")
    print(f"[INFO] python: {env['python']}")
    print(("[OK] " if env["competition_ready"] else "[RECHECK] ") + env["status"])
    print("结果：" + ("可运行" if ok else "存在缺失，请先执行 pip install -r requirements.txt 和 python scripts/build_index.py"))


if __name__ == "__main__":
    main()
