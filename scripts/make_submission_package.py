"""Build a clean release zip package.

By default the package is fully offline-safe and excludes .env/API keys. The
application still runs through the local evidence-summary fallback. Teams that
explicitly need a cloud demo key may pass --include-env, preferably with a
low-quota and revocable key.

Usage:
    python scripts/make_submission_package.py
    python scripts/make_submission_package.py --include-env
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "equipmind_a1_release.zip"
ENV_PATH = ROOT / ".env"

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "dist", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".bak"}


def read_env_value(key: str) -> str:
    if not ENV_PATH.exists():
        return ""
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return ""


def include_file(path: Path, include_env: bool) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.name == ".env" and not include_env:
        return False
    if rel.parts and rel.parts[0] == "storage" and path.name not in {"knowledge_index.pkl", "demo_work_order.md"}:
        return False
    return True


def validate_env_for_packaging() -> None:
    if not ENV_PATH.exists():
        raise SystemExit("使用 --include-env 时必须先配置 .env。")
    key = read_env_value("LLM_API_KEY") or read_env_value("DEEPSEEK_API_KEY")
    if not key or "请替换" in key or "你的" in key:
        raise SystemExit(".env 中未检测到可用 API Key；请修正或移除 --include-env。")
    if " " in key or len(key) < 12:
        raise SystemExit(".env 中的 API Key 格式异常。")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 EquipMind A1 发布包")
    parser.add_argument(
        "--include-env",
        action="store_true",
        help="显式把本机 .env 打包（存在密钥泄露风险，默认不包含）",
    )
    args = parser.parse_args()

    if args.include_env:
        validate_env_for_packaging()

    # Validate the application and the five retained Word documents before packaging.
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all_checks.py")], cwd=ROOT, check=True)

    DIST.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob("*"):
            if path.is_dir() or not include_file(path, args.include_env):
                continue
            arc = Path("equipmind_a1") / path.relative_to(ROOT)
            zf.write(path, arc.as_posix())

    print(f"已生成发布包：{OUT}")
    if args.include_env:
        print("警告：发布包包含 .env。请使用低额度可撤销密钥，提交后及时轮换。")
    else:
        print("安全模式：未包含 .env/API Key；系统可使用本地证据摘要模式运行。")


if __name__ == "__main__":
    main()
