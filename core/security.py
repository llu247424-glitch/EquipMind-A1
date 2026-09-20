from __future__ import annotations

from pathlib import Path
import re
import time


_SAFE_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


def sanitize_filename(name: str, *, allowed_suffixes: set[str] | None = None, default_stem: str = "upload") -> str:
    """Return a basename safe for local storage.

    Both POSIX and Windows separators are handled so an uploaded name cannot
    escape the intended data directory on either platform.
    """
    raw = (name or "").replace("\\", "/").split("/")[-1].strip()
    raw = raw.lstrip(".")
    path = Path(raw)
    suffix = path.suffix.lower()
    if allowed_suffixes is not None and suffix not in allowed_suffixes:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}")
    stem = _SAFE_CHARS.sub("_", path.stem).strip("._-") or default_stem
    stem = stem[:80]
    return f"{stem}{suffix}"


def unique_target(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return directory / f"{candidate.stem}_{stamp}{candidate.suffix}"


def validate_upload_size(data: bytes, *, max_mb: int) -> None:
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"文件超过 {max_mb} MB 限制，请压缩或拆分后再上传。")
