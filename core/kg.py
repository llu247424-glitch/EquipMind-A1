from __future__ import annotations

from typing import Iterable


DEVICE_TO_FAULTS = {
    "空压机": ["排气温度高", "油分压差高", "供气压力低"],
    "离心泵": ["机械密封泄漏", "振动异常", "轴承温升"],
    "配电柜": ["过载告警", "母排过热", "绝缘异常"],
    "变频器": ["OC 过流", "过热降额", "接地故障"],
    "风机": ["振动偏大", "皮带打滑", "叶轮积灰"],
}

FAULT_TO_PARTS = {
    "排气温度高": ["冷却器", "润滑油", "温控阀"],
    "油分压差高": ["油分芯", "压差传感器"],
    "供气压力低": ["进气阀", "管网泄漏"],
    "机械密封泄漏": ["机械密封", "轴套", "入口压力"],
    "振动异常": ["联轴器", "轴承", "基础螺栓"],
    "轴承温升": ["轴承", "润滑脂", "冷却水"],
    "过载告警": ["断路器", "负载电流", "电缆接头"],
    "母排过热": ["母排接头", "红外测温", "紧固力矩"],
    "绝缘异常": ["绝缘电阻", "电缆", "电机绕组"],
    "OC 过流": ["电机绝缘", "负载卡滞", "加减速时间"],
    "过热降额": ["散热风道", "散热风扇", "环境温度"],
    "接地故障": ["输出电缆", "电机绕组", "接地线"],
    "振动偏大": ["叶轮", "轴承", "地脚螺栓"],
    "皮带打滑": ["皮带张力", "皮带轮", "防护罩"],
    "叶轮积灰": ["叶轮", "滤网", "动平衡"],
}

PART_TO_ACTIONS = {
    "冷却器": "清洁冷却器并复测温度",
    "润滑油": "检查油位油质",
    "温控阀": "确认温控阀动作",
    "机械密封": "检查并更换密封组件",
    "联轴器": "复核同轴度",
    "轴承": "检查游隙与润滑",
    "断路器": "核查保护定值",
    "母排接头": "断电后紧固并测温",
    "电机绝缘": "测量绝缘电阻",
    "负载卡滞": "脱开负载试运行",
    "散热风道": "清理散热通道",
    "输出电缆": "检查电缆绝缘与接地",
    "叶轮": "清理积灰并做动平衡检查",
    "皮带张力": "调整张紧度",
}


def build_edges(chunks: Iterable) -> list[tuple[str, str, str]]:
    evidence = "\n".join(getattr(c, "content", "") + " " + getattr(c, "title", "") for c in chunks)
    devices_in_kb = sorted({getattr(c, "device", "设备") for c in chunks})
    edges: set[tuple[str, str, str]] = set()

    for device in devices_in_kb:
        faults = DEVICE_TO_FAULTS.get(device, [])
        for fault in faults:
            if fault in evidence or device in evidence:
                edges.add((device, "关联故障", fault))
                for part in FAULT_TO_PARTS.get(fault, []):
                    if part in evidence or fault in evidence:
                        rel = "检查部件" if part not in {"红外测温", "绝缘电阻", "负载电流"} else "检测参数"
                        edges.add((fault, rel, part))
                        action = PART_TO_ACTIONS.get(part)
                        if action:
                            edges.add((part, "处置动作", action))
        if device in {"配电柜", "变频器", "空压机", "离心泵", "风机"}:
            edges.add((device, "安全措施", "停机/断电/挂牌"))

    tag_rel = {
        "温度": "关联故障", "振动": "关联故障", "泄漏": "关联故障", "漏油": "关联故障", "过载": "关联告警",
        "绝缘": "检测参数", "润滑": "维护项目", "冷却": "维护项目", "轴承": "关键部件", "密封": "关键部件",
        "电流": "检测参数", "电压": "检测参数", "过流": "关联告警", "安全": "安全措施",
    }
    for c in chunks:
        device = getattr(c, "device", "设备")
        for tag in getattr(c, "tags", []):
            if tag in tag_rel:
                edges.add((device, tag_rel[tag], tag))

    return sorted(edges)[:90]


def to_dot(edges: list[tuple[str, str, str]]) -> str:
    lines = [
        "digraph G {",
        "rankdir=LR;",
        "graph [bgcolor=transparent, pad=0.25, nodesep=0.45, ranksep=0.8];",
        'node [shape=box, style="rounded,filled", fillcolor="#F8FAFC", color="#CBD5E1", fontname="Microsoft YaHei,Noto Sans CJK SC,Arial", fontsize=12, margin=0.12];',
        'edge [color="#64748B", fontname="Microsoft YaHei,Noto Sans CJK SC,Arial", fontsize=10, arrowsize=0.7];',
    ]
    for s, r, o in edges:
        lines.append(f'"{s}" -> "{o}" [label="{r}"];')
    lines.append("}")
    return "\n".join(lines)


from pathlib import Path
import json


def load_curated_kg() -> dict:
    """Load the curated maintenance knowledge graph used for the evaluation-facing page."""
    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "assets" / "kg_relations.json"
    if not p.exists():
        return {"nodes": [], "edges": [], "chains": [], "schema": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def kg_metrics() -> dict[str, int]:
    data = load_curated_kg()
    node_types: dict[str, int] = {}
    for n in data.get("nodes", []):
        node_types[n.get("type", "未知")] = node_types.get(n.get("type", "未知"), 0) + 1
    return {
        "nodes": len(data.get("nodes", [])),
        "edges": len(data.get("edges", [])),
        "chains": len(data.get("chains", [])),
        "node_types": node_types,
    }


def typed_dot(data: dict | None = None, limit_edges: int = 55) -> str:
    data = data or load_curated_kg()
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    colors = {
        "设备": "#DBEAFE",
        "子系统/部件": "#E0F2FE",
        "故障现象": "#FEE2E2",
        "检测参数": "#FEF3C7",
        "可能原因": "#F5E8FF",
        "处置动作": "#DCFCE7",
        "安全措施": "#FFE4E6",
        "SOP": "#F1F5F9",
    }
    lines = [
        "digraph G {",
        "rankdir=LR;",
        "graph [bgcolor=transparent, pad=0.25, nodesep=0.42, ranksep=0.65];",
        'node [shape=box, style="rounded,filled", color="#94A3B8", fontname="Microsoft YaHei,Noto Sans CJK SC,Arial", fontsize=11, margin=0.12];',
        'edge [color="#64748B", fontname="Microsoft YaHei,Noto Sans CJK SC,Arial", fontsize=9, arrowsize=0.65];',
    ]
    for node_id, n in nodes.items():
        fill = colors.get(n.get("type", ""), "#F8FAFC")
        border = "#DC2626" if n.get("risk") == "高" else "#94A3B8"
        lines.append(f'"{node_id}" [fillcolor="{fill}", color="{border}", label="{node_id}\n{n.get("type", "")}"];')
    for s, r, o in data.get("edges", [])[:limit_edges]:
        lines.append(f'"{s}" -> "{o}" [label="{r}"];')
    lines.append("}")
    return "\n".join(lines)
