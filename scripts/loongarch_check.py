from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.platform_check import platform_report


def main() -> None:
    parser = argparse.ArgumentParser(description="检查龙芯/银河麒麟运行环境")
    parser.add_argument("--strict", action="store_true", help="不满足 LoongArch + 银河麒麟时返回非零退出码")
    args = parser.parse_args()
    report = platform_report()
    print(f"Architecture: {report['architecture']}")
    print(f"OS: {report['os']}")
    print(f"Python: {report['python']}")
    print(f"Status: {report['status']}")
    if args.strict and not report["competition_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
