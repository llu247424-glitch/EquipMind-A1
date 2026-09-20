from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest

PAGES = [
    "首页 / 系统概览",
    "智能检修对话",
    "智能检修问答",
    "图文联合诊断",
    "标准化作业指引",
    "智能工单生成",
    "案例上传与审核",
    "人工反馈与纠错",
    "知识图谱与统计",
    "大模型配置",
    "知识库管理",
]


def main() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
    app.run()
    failures: list[str] = []
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        if app.exception:
            failures.append(page + ": " + " | ".join(exc.message for exc in app.exception))
    if failures:
        raise SystemExit("UI smoke test failed:\n" + "\n".join(failures))
    print(f"UI smoke test passed: {len(PAGES)} pages")


if __name__ == "__main__":
    main()
