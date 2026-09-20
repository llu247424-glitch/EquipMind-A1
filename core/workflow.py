from __future__ import annotations

import json
from typing import Any
from .config import WORKFLOW_DIR


def list_workflows() -> list[dict[str, Any]]:
    rows = []
    for p in sorted(WORKFLOW_DIR.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            obj["_file"] = p.name
            rows.append(obj)
        except Exception:
            continue
    return rows


def get_workflow(workflow_id: str) -> dict[str, Any] | None:
    for wf in list_workflows():
        if wf.get("id") == workflow_id:
            return wf
    return None


def best_workflow(device: str, severity: str = "一级") -> dict[str, Any] | None:
    rows = list_workflows()
    for wf in rows:
        if device and device in wf.get("device", "") and severity in wf.get("level", ""):
            return wf
    for wf in rows:
        if device and device in wf.get("device", ""):
            return wf
    return rows[0] if rows else None


def compliance_check(text: str, workflow: dict[str, Any]) -> list[dict[str, str]]:
    text = text or ""
    rules = workflow.get("compliance_rules", [])
    results = []
    for rule in rules:
        keywords = rule.get("keywords", [])
        passed = any(k in text for k in keywords)
        results.append({
            "规则": rule.get("name", "未命名规则"),
            "状态": "通过" if passed else "需补充",
            "说明": rule.get("description", ""),
            "建议关键词": "、".join(keywords),
        })
    return results


def completion_rate(results: list[dict[str, str]]) -> float:
    if not results:
        return 1.0
    return sum(1 for r in results if r.get("状态") == "通过") / len(results)
