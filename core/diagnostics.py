from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DiagnosticResult:
    fault_type: str
    severity: str
    risk_score: int
    priority: str
    recommended_actions: list[str]
    safety_notices: list[str]


KEYWORD_RULES = [
    ("温度异常", ["温度", "发热", "过热", "温升", "高温"], ["核对冷却系统", "检查润滑油位/油质", "测量轴承与电机壳温度", "复测负载电流"]),
    ("振动异常", ["振动", "抖动", "异响", "噪声"], ["测量振动值", "检查联轴器同轴度", "检查轴承游隙", "确认基础螺栓紧固状态"]),
    ("泄漏异常", ["漏", "泄漏", "渗油", "渗水", "密封"], ["确认泄漏介质与位置", "检查密封件/垫片", "检查压力波动", "必要时停机更换密封组件"]),
    ("电气告警", ["过载", "过流", "跳闸", "绝缘", "告警", "电流", "电压"], ["核对报警代码", "测量三相电流/电压", "检查接线端子温升", "执行绝缘电阻测试"]),
]

CRITICAL_SIGNALS = ["起火", "冒烟", "爆炸", "电弧", "触电", "人员受伤", "剧烈振动", "大量泄漏", "无法停机"]
HIGH_SIGNALS = ["跳闸", "过流", "短路", "绝缘低", "接地故障", "温度高", "高温", "过热", "泄漏", "卡滞", "堵转", "超压"]
MEDIUM_SIGNALS = ["异常", "偏高", "告警", "波动", "噪声", "振动", "发热", "异响", "压力低"]


def _matched_signals(text: str, signals: list[str]) -> list[str]:
    return [signal for signal in signals if signal in text]


def diagnose(query: str, retrieved: list[dict[str, Any]] | None = None) -> DiagnosticResult:
    query_text = (query or "").strip()
    evidence_text = " ".join(r.get("content", "") for r in (retrieved or [])[:3])
    classification_text = f"{query_text} {evidence_text}"

    matched: list[str] = []
    actions: list[str] = []
    for fault_type, kws, acts in KEYWORD_RULES:
        if any(k in classification_text for k in kws):
            matched.append(fault_type)
            actions.extend(acts)
    if not matched:
        matched = ["一般设备异常"]
        actions = ["确认设备型号与告警时间", "查看最近检修记录", "按点检清单逐项排查", "将新增现象沉淀为案例"]

    # Risk level must be grounded in the user's observed condition. Retrieved
    # manuals contain generic safety words and must not inflate every query to P1.
    critical = _matched_signals(query_text, CRITICAL_SIGNALS)
    high = _matched_signals(query_text, HIGH_SIGNALS)
    medium = _matched_signals(query_text, MEDIUM_SIGNALS)
    score = 30
    score += 45 * min(len(critical), 1)
    score += 25 * min(len(high), 2)
    score += 10 * min(len(medium), 2)
    score = min(score, 100)

    if critical or score >= 80:
        severity, priority = "二级/紧急", "P1"
    elif score >= 55:
        severity, priority = "一级/重点", "P2"
    else:
        severity, priority = "一级/常规", "P3"

    safety = ["检修完成后必须复测并记录关键参数"]
    if any(k in classification_text for k in ["电气", "过流", "跳闸", "绝缘", "电压", "电流", "配电柜", "变频器"]):
        safety.insert(0, "执行停机、断电、挂牌和验电确认，确认无残压后作业")
        safety.append("佩戴绝缘手套、护目镜等电气作业 PPE")
    elif any(k in classification_text for k in ["空压机", "压力", "泄漏", "泵"]):
        safety.insert(0, "执行停机、隔离和泄压确认，防止残压或介质喷出")
        safety.append("佩戴安全帽、护目镜和与介质相适配的防护用品")
    else:
        safety.insert(0, "执行停机、断电或机械隔离确认，防止设备意外启动")
        safety.append("佩戴安全帽、护目镜等规定 PPE")

    return DiagnosticResult(
        fault_type="、".join(dict.fromkeys(matched)),
        severity=severity,
        risk_score=score,
        priority=priority,
        recommended_actions=list(dict.fromkeys(actions))[:8],
        safety_notices=list(dict.fromkeys(safety)),
    )


def result_dict(result: DiagnosticResult) -> dict[str, Any]:
    return asdict(result)
