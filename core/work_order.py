from __future__ import annotations

from datetime import datetime
from typing import Any

from .retriever import assess_evidence


def _line_items(items: list[str]) -> str:
    if not items:
        return "- 暂无"
    return "\n".join(f"- {x}" for x in items)


def create_work_order(device: str, model: str, fault: str, diag: dict[str, Any], workflow: dict[str, Any] | None, retrieved: list[dict[str, Any]]) -> str:
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    order_id = f"EM-A1-{now}"
    wf_name = workflow.get("name", "未匹配 SOP") if workflow else "未匹配 SOP"
    wf_level = workflow.get("level", "常规检修") if workflow else "常规检修"
    steps = workflow.get("steps", []) if workflow else []
    tools = workflow.get("tools", []) if workflow else []
    refs = [
        f"- {r['title']} | {r['source_type']} | {r['source']} | 综合得分 {r['score']:.3f} | 置信度 {r.get('confidence', '未评估')}"
        for r in retrieved[:6]
    ]
    evidence = assess_evidence(f"{model} {fault}", retrieved)
    severity = diag.get("severity", "未评估")
    priority = diag.get("priority", "未评估")
    risk_score = diag.get("risk_score", "-")

    md = f"""# EquipMind A1 设备检修作业单

| 字段 | 内容 |
|---|---|
| 工单编号 | {order_id} |
| 设备名称 | {device or '未填写'} |
| 设备型号 | {model or '未填写'} |
| 故障现象 | {fault or '未填写'} |
| 风险等级 | {severity} / {priority} / 风险分 {risk_score} |
| 推荐 SOP | {wf_name}（{wf_level}） |
| 证据充分度 | {evidence['grade']}（{evidence['score']}/100），证据类型 {evidence['source_types']} 类 |
| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| 文档版本 | V1.0 / AI 生成草案，须经班组长复核 |

## 一、智能诊断结论

- 故障类型：{diag.get('fault_type', '未识别')}
- 处理优先级：{priority}
- 风险评分：{risk_score}

### 推荐排查动作
{_line_items(diag.get('recommended_actions', []))}

## 二、安全与作业前置条件

> 本作业单为辅助决策草案，不替代企业作业票、停送电票、动火票、受限空间票或现场负责人指令。

{_line_items(diag.get('safety_notices', []))}

### 作业许可与隔离确认

- [ ] 作业负责人已确认设备编号、位置和故障边界
- [ ] 已执行停机、能源隔离、上锁挂牌（LOTO）
- [ ] 已完成验电、泄压、降温或机械防转确认
- [ ] 已辨识交叉作业、高处、动火、受限空间等附加风险
- [ ] 作业票证、监护人和应急措施已落实

### 工具与防护用品
{_line_items(tools)}

## 三、标准化作业步骤

"""
    if steps:
        for idx, step in enumerate(steps, start=1):
            md += f"{idx}. **{step.get('title','步骤')}**：{step.get('detail','')}\n"
            if step.get('risk'):
                md += f"   - 风险提醒：{step.get('risk')}\n"
    else:
        md += "- 未匹配到 SOP，请由班组长确认作业票、隔离措施和复测要求。\n"

    md += """
## 四、证据与信息完整性

- 证据评估：{evidence['grade']}（{evidence['score']}/100）
- 评估依据：{'；'.join(evidence['reasons'])}
- 建议补充：{'、'.join(evidence['missing']) if evidence['missing'] else '无'}
- 决策要求：证据不足时不得仅依据 AI 建议直接拆检，应补充现场测量或由设备工程师复核。

## 五、验收与复测标准

| 验收项 | 记录要求 |
|---|---|
| 安全隔离 | 记录断电、挂牌、泄压、验电或防护确认情况 |
| 故障复测 | 记录温度、振动、电流、电压、压力或泄漏复测值 |
| 试运行 | 记录空载/带载试运行时间、告警状态和稳定性 |
| 遗留风险 | 如存在遗留问题，记录风险等级、临时措施和责任人 |
| 知识沉淀 | 将真实原因、处理过程和复测结果提交案例审核入库 |

## 六、引用资料与溯源

""" + ("\n".join(refs) if refs else "未命中资料")

    md += """

## 七、现场检修记录填写区

- 实施人：
- 监护人：
- 开始时间：
- 结束时间：
- 实际原因：
- 实施过程：
- 备件/耗材：
- 复测结果：
- 是否恢复运行：
- 遗留问题：
- 班组长确认：
- 设备工程师复核：
- AI 建议采纳情况（全部/部分/未采纳）：
- 未采纳或人工修正说明：
"""
    return md
