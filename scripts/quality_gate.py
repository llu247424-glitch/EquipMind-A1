from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "docs"
FORBIDDEN = [
    "MVP", "正式参赛时可替换", "占位", "没做完", "TODO", "半成品",
    "冲奖版", "高分看点", "评审友好",
]
REQUIRED_DOCX_NAMES = {
    "软件功能需求分析文档.docx",
    "软件功能设计文档.docx",
    "软件产品说明书.docx",
    "软件功能测试报告.docx",
    "软件安装包及部署文档.docx",
}


def main() -> None:
    bad: list[str] = []
    for path in [ROOT / "app.py", ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for word in FORBIDDEN:
            if word in text:
                bad.append(f"{path.relative_to(ROOT)} contains {word}")

    actual_files = {p.name for p in DOC_DIR.iterdir() if p.is_file()}
    missing = REQUIRED_DOCX_NAMES - actual_files
    extra = actual_files - REQUIRED_DOCX_NAMES
    for name in sorted(missing):
        bad.append(f"missing required document: docs/{name}")
    for name in sorted(extra):
        bad.append(f"unexpected file in docs: docs/{name}")
    for name in REQUIRED_DOCX_NAMES & actual_files:
        path = DOC_DIR / name
        if path.stat().st_size < 10_000:
            bad.append(f"invalid required document: docs/{name}")

    if bad:
        print("质量检查未通过：")
        for item in bad:
            print("-", item)
        sys.exit(1)
    print("quality gate passed")


if __name__ == "__main__":
    main()
