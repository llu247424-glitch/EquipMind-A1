from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .config import PENDING_CASE_DIR, STORAGE_DIR
from .security import sanitize_filename, unique_target

FEEDBACK_PATH = STORAGE_DIR / "model_feedback.jsonl"


@dataclass
class FeedbackRecord:
    feedback_id: str
    created_at: str
    question: str
    answer: str
    rating: str
    correction: str
    reviewer: str
    device: str
    source_page: str
    references: list[dict[str, Any]]
    status: str = "待审核"


def _clean_text(value: str, limit: int) -> str:
    value = re.sub(r"\x00", "", value or "").strip()
    return value[:limit]


def save_feedback(
    *,
    question: str,
    answer: str,
    rating: str,
    correction: str = "",
    reviewer: str = "匿名用户",
    device: str = "",
    source_page: str = "智能检修问答",
    references: list[dict[str, Any]] | None = None,
    path: Path = FEEDBACK_PATH,
) -> FeedbackRecord:
    allowed_ratings = {"准确有用", "部分有用", "需要纠正"}
    if rating not in allowed_ratings:
        raise ValueError("反馈评价不合法。")
    if rating == "需要纠正" and len((correction or "").strip()) < 6:
        raise ValueError("选择“需要纠正”时，请填写具体修正内容。")
    refs = []
    for row in (references or [])[:6]:
        refs.append({
            "title": _clean_text(str(row.get("title", "")), 160),
            "source": _clean_text(str(row.get("source", "")), 220),
            "source_type": _clean_text(str(row.get("source_type", "")), 40),
            "score": round(float(row.get("score", 0)), 4),
        })
    record = FeedbackRecord(
        feedback_id=f"FB-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}",
        created_at=datetime.now().isoformat(timespec="seconds"),
        question=_clean_text(question, 2000),
        answer=_clean_text(answer, 8000),
        rating=rating,
        correction=_clean_text(correction, 5000),
        reviewer=_clean_text(reviewer, 80) or "匿名用户",
        device=_clean_text(device, 80),
        source_page=_clean_text(source_page, 80),
        references=refs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return record


def load_feedback(*, path: Path = FEEDBACK_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("feedback_id"):
            rows.append(item)
    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return rows[:limit] if limit else rows


def feedback_stats(*, path: Path = FEEDBACK_PATH) -> dict[str, Any]:
    rows = load_feedback(path=path)
    ratings = {"准确有用": 0, "部分有用": 0, "需要纠正": 0}
    for row in rows:
        rating = row.get("rating", "")
        ratings[rating] = ratings.get(rating, 0) + 1
    corrected = ratings.get("需要纠正", 0)
    return {
        "count": len(rows),
        "ratings": ratings,
        "correction_rate": corrected / max(len(rows), 1),
    }


def promote_feedback_to_pending_case(record: dict[str, Any]) -> Path:
    correction = (record.get("correction") or "").strip()
    if not correction:
        raise ValueError("该反馈没有人工修正内容，不能转为待审核案例。")
    device = (record.get("device") or "通用设备").strip()
    safe_device = sanitize_filename(f"{device}.md", allowed_suffixes={".md"}, default_stem="device").removesuffix(".md")
    filename = f"feedback_{record.get('feedback_id', 'unknown')}_{safe_device}.md"
    target = unique_target(PENDING_CASE_DIR, filename)
    refs = record.get("references", [])
    ref_lines = "\n".join(
        f"- {r.get('title', '未命名资料')} | {r.get('source', '')} | 得分 {r.get('score', 0)}"
        for r in refs
    ) or "- 无"
    content = f"""# {device} 人工纠错知识候选

- 来源反馈：{record.get('feedback_id', '')}
- 提交人：{record.get('reviewer', '匿名用户')}
- 提交时间：{record.get('created_at', '')}
- 设备：{device}
- 原问题：{record.get('question', '')}

## 系统原回答
{record.get('answer', '')}

## 人工修正
{correction}

## 原引用资料
{ref_lines}

## 审核建议
请由设备工程师核对人工修正与现场事实、手册条款和复测数据，审核通过后纳入知识库。
"""
    target.write_text(content, encoding="utf-8")
    return target
