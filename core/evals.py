from __future__ import annotations

import json
import time
from typing import Any

from .config import EVALUATION_DIR
from .retriever import SYNONYM_GROUPS


def load_eval_set() -> list[dict[str, Any]]:
    p = EVALUATION_DIR / "retrieval_eval.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _expected_variants(expected: str) -> list[str]:
    variants = [expected]
    low = expected.lower()
    for group in SYNONYM_GROUPS:
        if any(term.lower() in low or low in term.lower() for term in group):
            variants.extend(group)
    # Common compressed wording used by technicians versus formal manual titles.
    manual_aliases = {
        "油分压差高": ["油气分离器压差高", "油气分离器压差报警"],
        "母排过热": ["母排连接点过热", "母排接头过热"],
        "联轴器不对中": ["对中偏差", "联轴器对中"],
    }
    variants.extend(manual_aliases.get(expected, []))
    return list(dict.fromkeys(v for v in variants if v))


def _is_hit(expected: str, text: str) -> bool:
    compact = text.replace(" ", "").lower()
    return any(v.replace(" ", "").lower() in compact for v in _expected_variants(expected))


def run_retrieval_eval(retriever, top_k: int = 3) -> dict[str, Any]:
    items = load_eval_set()
    rows = []
    hit1 = hitk = 0
    reciprocal_rank = 0.0
    total_ms = 0.0
    confidences = {"高": 0, "中": 0, "低": 0}

    for item in items:
        t0 = time.perf_counter()
        results = retriever.search(item["query"], device=item.get("device", "全部"), top_k=top_k)
        cost_ms = (time.perf_counter() - t0) * 1000
        total_ms += cost_ms
        expected = item.get("expected_keyword", "")
        hit_positions = [
            idx for idx, result in enumerate(results, start=1)
            if _is_hit(expected, result["title"] + " " + result["content"])
        ]
        ok1 = bool(hit_positions and hit_positions[0] == 1)
        okk = bool(hit_positions)
        hit1 += int(ok1)
        hitk += int(okk)
        if hit_positions:
            reciprocal_rank += 1.0 / hit_positions[0]
        top_conf = results[0].get("confidence", "低") if results else "低"
        confidences[top_conf] = confidences.get(top_conf, 0) + 1
        rows.append({
            "问题": item["query"],
            "场景": item.get("scenario", ""),
            "期望关键词": expected,
            "Top1命中": "是" if ok1 else "否",
            f"Top{top_k}命中": "是" if okk else "否",
            "首条置信度": top_conf,
            "首条得分": round(results[0]["score"], 3) if results else 0,
            "耗时ms": round(cost_ms, 2),
            "Top1资料": results[0]["title"] if results else "无",
        })
    n = max(len(items), 1)
    return {
        "count": len(items),
        "hit1": hit1 / n,
        "hitk": hitk / n,
        "mrr": reciprocal_rank / n,
        "avg_ms": total_ms / n,
        "confidence_distribution": confidences,
        "rows": rows,
    }
