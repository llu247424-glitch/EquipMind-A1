from __future__ import annotations

import platform
from pathlib import Path
from typing import Any


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def platform_report() -> dict[str, Any]:
    arch = platform.machine() or "unknown"
    release = _os_release()
    os_name = release.get("PRETTY_NAME") or platform.platform()
    arch_lower = arch.lower()
    os_lower = os_name.lower()
    is_loongarch = any(token in arch_lower for token in ("loongarch", "loong64"))
    is_kylin = any(token in os_lower for token in ("kylin", "麒麟"))
    return {
        "architecture": arch,
        "os": os_name,
        "python": platform.python_version(),
        "is_loongarch": is_loongarch,
        "is_kylin": is_kylin,
        "competition_ready": is_loongarch and is_kylin,
        "status": "龙芯麒麟环境已识别" if is_loongarch and is_kylin else "当前为开发/演示环境，提交前需在龙芯+银河麒麟虚机复验",
    }
