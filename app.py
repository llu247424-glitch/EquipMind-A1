from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from core.config import MANUAL_DIR, PENDING_CASE_DIR, APPROVED_CASE_DIR, INDEX_PATH, ASSET_DIR, STORAGE_DIR
from core.retriever import KnowledgeRetriever, assess_evidence
from core.llm import answer_question, chat_with_retrieval, describe_image_with_llm, llm_config_summary, llm_status, test_llm_connection
from core.image_search import search_similar_image, load_image_cases
from core.workflow import list_workflows, get_workflow, best_workflow, compliance_check, completion_rate
from core.kg import build_edges, to_dot, load_curated_kg, typed_dot, kg_metrics
from core.diagnostics import diagnose, result_dict
from core.work_order import create_work_order
from core.security import sanitize_filename, unique_target, validate_upload_size
from core.feedback import save_feedback, load_feedback, feedback_stats, promote_feedback_to_pending_case

ROOT_DIR = Path(__file__).resolve().parent
DOC_DIR = ROOT_DIR / "docs"

st.set_page_config(page_title="EquipMind A1", page_icon="🛠️", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 2.5rem;}
.equipmind-card {border: 1px solid #E2E8F0; border-radius: 16px; padding: 1rem; background: #FFFFFF;}
.equipmind-muted {color: #64748B; font-size: 0.95rem;}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def get_retriever() -> KnowledgeRetriever:
    r = KnowledgeRetriever()
    r.load_or_build()
    return r


def rebuild_index() -> None:
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    get_retriever.clear()
    get_retriever().build()
    st.success("索引已重建。")


def show_header() -> None:
    st.title("🛠️ EquipMind A1 设备检修知识检索与作业辅助系统")
    st.caption("混合知识检索 · 图片案例匹配 · SOP 作业指引 · 工单与知识维护")


def load_assets() -> list[dict]:
    p = ASSET_DIR / "equipment_registry.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def _retrieval_label(item: dict) -> str:
    return (
        f"{item.get('title', '未命名资料')} | {item.get('source_type', '文档')} | "
        f"置信度 {item.get('confidence', '未评估')} | 综合得分 {item.get('score', 0):.3f}"
    )


def _render_retrieval_detail(item: dict, *, max_chars: int | None = None) -> None:
    st.caption(f"来源：{item.get('source', '')}")
    matched = item.get("matched_terms", [])
    if matched:
        st.caption("命中要点：" + "、".join(matched))
    st.caption(
        f"语义分 {item.get('semantic_score', 0):.3f} · "
        f"BM25 {item.get('lexical_score', 0):.3f} · "
        f"关键词覆盖 {item.get('keyword_coverage', 0):.0%} · "
        f"标题覆盖 {item.get('title_coverage', 0):.0%} · "
        f"检索意图 {item.get('intent', '综合检索')}"
    )
    content = item.get("content", "")
    st.write(content[:max_chars] if max_chars else content)


def _render_feedback_form(*, question: str, answer: str, rows: list[dict], device: str, key: str, source_page: str) -> None:
    with st.expander("✍️ 人工反馈与纠错（提交后进入审核）"):
        st.caption("反馈会进入待审核队列；只有设备工程师审核通过后，才会转为知识案例并参与后续检索。")
        rating = st.radio("回答评价", ["准确有用", "部分有用", "需要纠正"], horizontal=True, key=f"{key}_rating")
        correction = st.text_area(
            "人工修正内容",
            placeholder="例如：实际原因是输出电缆绝缘下降，应先做绝缘测试，再检查负载卡滞。",
            key=f"{key}_correction",
        )
        reviewer = st.text_input("标注人/班组", value="检修一班", key=f"{key}_reviewer")
        if st.button("保存反馈", key=f"{key}_save"):
            try:
                record = save_feedback(
                    question=question,
                    answer=answer,
                    rating=rating,
                    correction=correction,
                    reviewer=reviewer,
                    device=device if device != "全部" else "",
                    source_page=source_page,
                    references=rows,
                )
                st.success(f"反馈已保存：{record.feedback_id}。可在“人工反馈与纠错”页面审核并转为知识案例。")
            except ValueError as exc:
                st.error(str(exc))


def page_home():
    show_header()
    r = get_retriever()
    stats = r.stats()
    assets = load_assets()
    kgm = kg_metrics()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("知识片段", stats["chunk_count"])
    c2.metric("设备类型", len(stats["devices"]))
    c3.metric("SOP 流程", len(list_workflows()))
    c4.metric("图谱关系", kgm["edges"])
    c5.metric("设备台账", len(assets))
    c6.metric("人工反馈", feedback_stats()["count"])

    st.markdown("### 项目定位")
    st.success("EquipMind A1 将设备手册、检修案例、SOP、故障图片和关系数据集中到一个 Web 系统中，用于资料查询、故障初筛、作业指引、工单生成和案例维护。")
    st.info(llm_status())

    st.markdown("### 系统价值")
    value_rows = pd.DataFrame([
        {"对象": "一线检修人员", "价值": "根据设备型号、故障现象和现场图片快速查找资料，并获得可执行的排查步骤。"},
        {"对象": "班组长与设备工程师", "价值": "通过资料引用、SOP 合规检查和知识关系链核对检修方案。"},
        {"对象": "设备管理人员", "价值": "统一维护设备手册、检修案例、工单和设备台账，减少知识分散。"},
        {"对象": "知识库管理员", "价值": "审核一线案例和人工修正，重建索引后供后续检索与问答使用。"},
    ])
    st.dataframe(value_rows, width="stretch", hide_index=True)

    st.markdown("### 技术实现")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.container(border=True).markdown("**RAG 检修问答**  \n先检索设备手册、案例和 SOP，再生成带引用的诊断建议，避免大模型凭空回答。")
    with h2:
        st.container(border=True).markdown("**GraphRAG 业务图谱**  \n将设备、部件、故障、参数、安全措施组织为可解释关系链，辅助多跳排查。")
    with h3:
        st.container(border=True).markdown("**工单与知识更新**  \n检修建议可生成工单，人工修正和新案例经审核后进入知识库。")

    st.markdown("### 检修任务流程")
    st.graphviz_chart("""
digraph G {
rankdir=LR;
graph [bgcolor=transparent, pad=0.25, nodesep=0.45, ranksep=0.8];
node [shape=box, style="rounded,filled", fillcolor="#F8FAFC", color="#CBD5E1", fontname="Microsoft YaHei,Noto Sans CJK SC,Arial", fontsize=12, margin=0.15];
edge [color="#64748B", fontname="Microsoft YaHei,Noto Sans CJK SC,Arial", fontsize=10];
现场故障 -> 多模态输入;
多模态输入 -> 文本RAG检索;
多模态输入 -> 图片案例检索;
文本RAG检索 -> 大模型增强诊断;
图片案例检索 -> 相似案例建议;
大模型增强诊断 -> SOP作业指引;
相似案例建议 -> SOP作业指引;
SOP作业指引 -> 合规检查;
合规检查 -> 智能工单;
智能工单 -> 案例审核入库;
案例审核入库 -> 知识库更新;
知识库更新 -> 文本RAG检索;
知识库更新 -> 知识图谱;
知识图谱 -> 大模型增强诊断;
}
""")



def page_chat():
    show_header()
    st.subheader("💬 智能检修多轮对话")
    st.caption("本页面是真正的智能对话入口：每轮自动检索设备手册、案例与 SOP，并结合对话上下文生成结构化检修建议。")
    r = get_retriever()
    cfg = llm_config_summary()
    devices = ["全部"] + sorted(r.stats()["devices"].keys())

    with st.sidebar:
        st.markdown("### 对话设置")
        device = st.selectbox("设备范围", devices, key="chat_device")
        top_k = st.slider("每轮引用资料数", 3, 8, 5, key="chat_top_k")
        show_refs = st.toggle("显示每轮命中资料", value=True, key="chat_show_refs")
        if st.button("清空对话", width="stretch"):
            st.session_state["equipmind_chat"] = []
            st.rerun()

    if cfg["enabled"]:
        st.success(f"大模型已启用：{cfg['provider']} / {cfg['model']}。回答会结合本地知识库生成。")
    else:
        st.warning("当前使用本地知识库摘要模式。配置模型服务后可启用增强回答。")

    if "equipmind_chat" not in st.session_state:
        st.session_state["equipmind_chat"] = [
            {
                "role": "assistant",
                "content": "你好，我是 EquipMind 检修助手。你可以描述设备型号、故障现象、报警代码或现场数据，我会结合本地手册、案例和 SOP 给出排查建议。",
                "refs": [],
            }
        ]

    for msg_idx, msg in enumerate(st.session_state["equipmind_chat"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_refs and msg.get("refs"):
                with st.expander("本轮引用资料"):
                    for item in msg["refs"]:
                        st.markdown(f"**{_retrieval_label(item)}**")
                        _render_retrieval_detail(item, max_chars=600)
            if msg.get("role") == "assistant" and msg.get("question") and msg_idx == len(st.session_state["equipmind_chat"]) - 1:
                _render_feedback_form(
                    question=msg["question"],
                    answer=msg["content"],
                    rows=msg.get("refs", []),
                    device=msg.get("device", "全部"),
                    key=f"chat_feedback_{msg_idx}",
                    source_page="智能多轮对话",
                )

    prompt = st.chat_input("例如：变频器 OC 过流，复位后又跳闸，应该怎么排查？")
    if prompt:
        st.session_state["equipmind_chat"].append({"role": "user", "content": prompt, "refs": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                rows = r.search(prompt, device=device, top_k=top_k)
                # Pass only role/content to the LLM history, not internal references.
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["equipmind_chat"][:-1]
                    if m.get("role") in {"user", "assistant"}
                ]
                response = chat_with_retrieval(prompt, rows, history=history)
                evidence = assess_evidence(prompt, rows)
            st.caption(f"证据充分度：{evidence['grade']}（{evidence['score']}/100）；检索意图：{rows[0].get('intent', '综合检索') if rows else '未识别'}")
            st.markdown(response)
            if show_refs and rows:
                with st.expander("本轮引用资料"):
                    for item in rows:
                        st.markdown(f"**{_retrieval_label(item)}**")
                        _render_retrieval_detail(item, max_chars=600)
        st.session_state["equipmind_chat"].append({
            "role": "assistant",
            "content": response,
            "refs": rows,
            "question": prompt,
            "device": device,
        })
        st.rerun()


def page_search():
    show_header()
    st.subheader("🔎 智能检修问答：文本 / 故障现象 / 设备型号")
    r = get_retriever()
    devices = ["全部"] + sorted(r.stats()["devices"].keys())
    col1, col2, col3 = st.columns([3, 1, 1])
    query = col1.text_input("请输入故障现象、设备型号或检修问题", value="空压机排气温度高，可能是什么原因？")
    device = col2.selectbox("设备类型", devices)
    top_k = col3.slider("返回资料数", 3, 8, 5)

    if st.button("开始检索并生成建议", type="primary"):
        with st.spinner("正在执行混合检索、证据评估并生成检修建议..."):
            rows = r.search(query, device=device, top_k=top_k)
            ans = answer_question(query, rows)
            diag = result_dict(diagnose(query, rows))
            evidence = assess_evidence(query, rows)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("故障类型", diag["fault_type"])
        c2.metric("优先级", diag["priority"])
        c3.metric("风险分", diag["risk_score"])
        c4.metric("证据充分度", f"{evidence['grade']} / {evidence['score']}")
        c5.metric("证据类型", evidence["source_types"])
        c6.metric("检索耗时", f"{r.last_query_ms:.1f} ms")
        if evidence["grade"] == "不足":
            st.warning("当前证据不足，不建议直接拆检。建议补充：" + "、".join(evidence["missing"] or ["现场信息"]))
        elif evidence["missing"]:
            st.info("为了进一步提高诊断可靠性，可补充：" + "、".join(evidence["missing"]))
        st.caption("证据评估依据：" + "；".join(evidence["reasons"]))
        st.markdown("### 智能检修建议")
        st.info(ans)
        st.markdown("### 推荐排查动作")
        st.write("\n".join(f"- {x}" for x in diag["recommended_actions"]))
        st.markdown("### 命中的知识片段")
        for item in rows:
            with st.expander(_retrieval_label(item)):
                _render_retrieval_detail(item)
        _render_feedback_form(
            question=query,
            answer=ans,
            rows=rows,
            device=device,
            key="search_feedback",
            source_page="智能检修问答",
        )


def page_image():
    show_header()
    st.subheader("🖼️ 图文联合故障检索与诊断")
    st.write("上传现场图片，并补充设备类型或故障现象。系统会融合图像特征、文本语义和设备约束召回相似案例，再联动手册、SOP 与图谱生成可追溯建议。")

    r = get_retriever()
    cases = load_image_cases()
    device_options = ["全部"] + sorted({c.device for c in cases})
    tab1, tab2 = st.tabs(["图文联合检索", "案例库预览"])
    uploaded = None
    symptom_hint = ""
    device_hint = "全部"
    with tab1:
        left, right = st.columns([1, 1])
        with left:
            uploaded = st.file_uploader("上传故障图片", type=["png", "jpg", "jpeg"])
            st.caption("建议上传报警屏、泄漏点、轴承温升点、接线端子或设备局部清晰图片。")
            if uploaded:
                st.image(uploaded, caption="待分析现场图片", width="stretch")
        with right:
            device_hint = st.selectbox("设备类型提示（可选）", device_options, key="image_device_hint")
            symptom_hint = st.text_area(
                "现场现象补充（建议填写）",
                value="",
                placeholder="例如：变频器显示 OC，复位后再次跳闸，电机有异响。",
                key="image_symptom_hint",
            )
            st.info("融合检索会同时计算图像相似度、故障描述相似度和设备类型匹配，不再只依赖图片外观。")
            sample_names = [f"{c.device} - {c.fault}" for c in cases]
            sample_choice = st.selectbox("也可以选择内置案例图片", ["不使用示例"] + sample_names)
            if sample_choice != "不使用示例":
                sample = cases[sample_names.index(sample_choice)]
                st.image(sample.image_path, caption=sample_choice, width="stretch")
                if not symptom_hint:
                    symptom_hint = sample.description
                if device_hint == "全部":
                    device_hint = sample.device
                with open(sample.image_path, "rb") as f:
                    class _LocalUpload:
                        def __init__(self, data: bytes):
                            self._data = data
                            self.type = "image/png"
                        def getvalue(self):
                            return self._data
                    uploaded = _LocalUpload(f.read())
                st.caption("已载入案例图片和对应现场描述，可直接执行联合检索。")

    with tab2:
        st.markdown("#### 设备故障图像案例库")
        cols = st.columns(3)
        for i, c in enumerate(cases):
            with cols[i % 3]:
                st.image(c.image_path, caption=f"{c.device}｜{c.fault}", width="stretch")
                st.caption(c.description)

    if uploaded and st.button("执行图文联合诊断", type="primary", key="image_search_btn"):
        image_bytes = uploaded.getvalue()
        try:
            validate_upload_size(image_bytes, max_mb=10)
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner("正在融合图片特征、现场描述与知识库证据..."):
            vision_text = describe_image_with_llm(image_bytes, getattr(uploaded, "type", "image/png") or "image/png")
            image_rows = search_similar_image(
                uploaded,
                text_hint=symptom_hint,
                device_hint=device_hint,
                top_k=3,
            )
            top_case = image_rows[0] if image_rows else {}
            multimodal_query = " ".join(
                x for x in [device_hint if device_hint != "全部" else "", symptom_hint, top_case.get("fault", ""), top_case.get("description", "")]
                if x
            )
            knowledge_rows = r.search(multimodal_query, device=device_hint, top_k=5)
            joint_answer = answer_question(multimodal_query, knowledge_rows)
            diag = result_dict(diagnose(multimodal_query, knowledge_rows))
            evidence = assess_evidence(multimodal_query, knowledge_rows)
        if vision_text:
            st.markdown("### 视觉模型初筛")
            if vision_text.startswith("大模型接口调用失败"):
                st.warning(vision_text)
            else:
                st.info(vision_text)

        st.markdown("### 跨模态案例召回")
        for row in image_rows:
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                c1.image(row["image_path"], caption=row["fault"], width="stretch")
                c2.write(f"**设备：**{row['device']}")
                c2.write(f"**融合相似度：**{row['score']}（图像 {row.get('image_score', 0)} / 文本 {row.get('text_score', 0)} / 设备匹配 {row.get('device_match', 0)}）")
                c2.write(f"**融合方式：**{row.get('fusion_mode', '图像特征')}；置信度：{row.get('confidence', '未评估')}")
                c2.write(f"**案例说明：**{row['description']}")
                c2.success(f"案例处置建议：{row['suggestion']}")

        st.markdown("### 图文联合诊断结论")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("疑似故障", top_case.get("fault", "未识别"))
        c2.metric("作业优先级", diag["priority"])
        c3.metric("证据充分度", f"{evidence['grade']} / {evidence['score']}")
        c4.metric("知识证据类型", evidence["source_types"])
        st.info(joint_answer)
        with st.expander("查看联合诊断引用资料"):
            for item in knowledge_rows:
                st.markdown(f"**{_retrieval_label(item)}**")
                _render_retrieval_detail(item, max_chars=600)
        _render_feedback_form(
            question=multimodal_query,
            answer=joint_answer,
            rows=knowledge_rows,
            device=device_hint,
            key="image_feedback",
            source_page="图文联合诊断",
        )


def page_workflow():
    show_header()
    st.subheader("📋 标准化作业指引与合规检查")
    workflows = list_workflows()
    names = {f"{w['device']} - {w['name']}（{w['level']}）": w["id"] for w in workflows}
    if not names:
        st.warning("暂无 SOP 流程。")
        return
    selected = st.selectbox("选择设备与检修等级", list(names.keys()))
    wf = get_workflow(names[selected])
    if not wf:
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 作业步骤")
        for idx, step in enumerate(wf.get("steps", []), start=1):
            st.markdown(f"**{idx}. {step['title']}**")
            st.write(step.get("detail", ""))
            if step.get("risk"):
                st.warning("风险提醒：" + step["risk"])
    with c2:
        st.markdown("### 工具与防护")
        st.write("、".join(wf.get("tools", [])))
        st.markdown("### 适用条件")
        st.write(wf.get("condition", ""))

    st.markdown("### 检修记录合规检查")
    record = st.text_area("粘贴检修记录，系统检查是否包含关键合规项", value="已断电挂牌，检查润滑油位，完成泄压，试运行并复测温度。")
    if st.button("执行合规检查"):
        results = compliance_check(record, wf)
        rate = completion_rate(results)
        st.metric("合规完成率", f"{rate:.0%}")
        st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)


def page_order():
    show_header()
    st.subheader("🧾 智能工单生成")
    r = get_retriever()
    assets = load_assets()
    asset_names = [f"{a['设备']} / {a['型号']} / {a['位置']}" for a in assets]
    selected = st.selectbox("选择设备台账", asset_names)
    asset = assets[asset_names.index(selected)] if asset_names else {"设备":"", "型号":""}
    fault = st.text_area("故障现象", value=f"{asset.get('设备','设备')}出现异常告警，需要生成检修作业单。")
    top_k = st.slider("引用资料数量", 3, 8, 5, key="order_topk")
    if st.button("生成作业单", type="primary"):
        rows = r.search(f"{asset.get('型号','')} {fault}", device=asset.get("设备"), top_k=top_k)
        diag = result_dict(diagnose(fault, rows))
        wf = best_workflow(asset.get("设备", ""))
        md = create_work_order(asset.get("设备", ""), asset.get("型号", ""), fault, diag, wf, rows)
        STORAGE_DIR.mkdir(exist_ok=True)
        p = STORAGE_DIR / f"work_order_{int(time.time())}.md"
        p.write_text(md, encoding="utf-8")
        st.success("作业单已生成，已包含风险等级、SOP 步骤、验收标准和引用依据。")
        st.download_button("下载 Markdown 作业单", data=md.encode("utf-8"), file_name=p.name, mime="text/markdown", width="stretch")
        st.markdown(md)

        st.markdown("### 工单使用说明")
        st.info("生成的作业单可用于现场检修记录。检修完成后，将实际原因、处理过程和复测结果提交到‘案例上传与审核’，即可沉淀为新的知识片段。")


def page_cases():
    show_header()
    st.subheader("🧠 一线案例上传、审核与知识沉淀")
    tab1, tab2, tab3 = st.tabs(["上传案例", "审核入库", "已入库案例"])

    with tab1:
        device = st.text_input("设备名称", "空压机")
        fault = st.text_input("故障现象", "排气温度持续偏高")
        cause = st.text_area("原因分析", "冷却器积灰，润滑油位偏低，导致散热能力下降。")
        action = st.text_area("处理过程", "停机断电后清洁冷却器，补充润滑油，复测排气温度恢复正常。")
        author = st.text_input("提交人/班组", "检修一班")
        if st.button("提交待审核", type="primary"):
            safe_device = sanitize_filename(f"{device}.md", allowed_suffixes={".md"}, default_stem="device").removesuffix(".md")
            ts = int(time.time())
            p = unique_target(PENDING_CASE_DIR, f"case_{ts}_{safe_device}.md")
            p.write_text(f"# {device} {fault}\n\n- 提交人：{author}\n- 设备：{device}\n- 故障现象：{fault}\n\n## 原因分析\n{cause}\n\n## 处理过程\n{action}\n\n## 经验总结\n建议纳入类似设备点检清单，出现同类告警时优先检查。\n", encoding="utf-8")
            st.success(f"案例已提交待审核：{p.name}")

    with tab2:
        pending = sorted(PENDING_CASE_DIR.glob("*.md"))
        if not pending:
            st.info("暂无待审核案例。")
        else:
            names = [p.name for p in pending]
            chosen = st.selectbox("待审核案例", names)
            p = PENDING_CASE_DIR / chosen
            content = st.text_area("案例内容", p.read_text(encoding="utf-8"), height=280)
            col1, col2 = st.columns(2)
            if col1.button("通过并入库"):
                APPROVED_CASE_DIR.mkdir(parents=True, exist_ok=True)
                target = APPROVED_CASE_DIR / p.name
                target.write_text(content, encoding="utf-8")
                p.unlink()
                rebuild_index()
                st.success("已审核入库，并重建知识索引。")
            if col2.button("驳回删除"):
                p.unlink()
                st.warning("已删除待审核案例。")
    with tab3:
        approved = sorted(APPROVED_CASE_DIR.glob("*.md"))
        st.write(f"已入库案例数：{len(approved)}")
        for p in approved[:20]:
            with st.expander(p.name):
                st.write(p.read_text(encoding="utf-8")[:1200])


def page_feedback():
    show_header()
    st.subheader("🧑‍🔧 人工反馈、标注与模型纠错")
    st.write("一线人员可以评价回答并提交修正。修正内容先进入审核队列，设备工程师确认后才能转为正式案例并参与后续检索。")
    stats = feedback_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("反馈总数", stats["count"])
    c2.metric("准确有用", stats["ratings"].get("准确有用", 0))
    c3.metric("部分有用", stats["ratings"].get("部分有用", 0))
    c4.metric("需要纠正", stats["ratings"].get("需要纠正", 0))

    rows = load_feedback()
    if not rows:
        st.info("暂无反馈。可先在“智能检修问答”或“图文联合诊断”页面生成回答并提交评价。")
        return

    overview = pd.DataFrame([
        {
            "反馈编号": row.get("feedback_id", ""),
            "时间": row.get("created_at", ""),
            "设备": row.get("device", "") or "未指定",
            "评价": row.get("rating", ""),
            "标注人": row.get("reviewer", ""),
            "来源页面": row.get("source_page", ""),
            "问题摘要": row.get("question", "")[:80],
        }
        for row in rows
    ])
    st.dataframe(overview, width="stretch", hide_index=True)

    ids = [row.get("feedback_id", "") for row in rows]
    chosen_id = st.selectbox("选择反馈进行审核", ids)
    record = next(row for row in rows if row.get("feedback_id") == chosen_id)
    st.markdown("### 原问题")
    st.write(record.get("question", ""))
    st.markdown("### 系统原回答")
    st.info(record.get("answer", ""))
    st.markdown("### 人工修正")
    correction = record.get("correction", "")
    if correction:
        st.success(correction)
    else:
        st.caption("该反馈未提交修正内容，仅用于质量统计。")
    with st.expander("原回答引用证据"):
        refs = record.get("references", [])
        if refs:
            st.dataframe(pd.DataFrame(refs), width="stretch", hide_index=True)
        else:
            st.caption("无引用资料记录。")

    already = any(chosen_id in p.name for p in PENDING_CASE_DIR.glob("*.md"))
    if already:
        st.info("该反馈已转为待审核案例，请到“案例上传与审核”页面继续审核。")
    elif correction and st.button("转为待审核知识案例", type="primary"):
        try:
            target = promote_feedback_to_pending_case(record)
            st.success(f"已生成待审核案例：{target.name}。审核通过并重建索引后，人工修正才会参与后续问答。")
        except ValueError as exc:
            st.error(str(exc))

    st.markdown("### 审核与入库规则")
    st.warning("人工反馈不会直接污染正式知识库。所有修正必须经过设备工程师审核，并保留原问题、原回答、引用证据和标注人信息。")


def page_kg():
    show_header()
    st.subheader("🕸️ GraphRAG 知识图谱与统计看板")
    st.write("知识图谱把检修知识从“文档片段”升级为“可解释的业务关系”：设备连接部件，故障连接原因、检测参数、处置动作和安全措施。问答时可利用这些关系进行多跳排查和引用解释。")

    r = get_retriever()
    stats = r.stats()
    data = load_curated_kg()
    kgm = kg_metrics()
    device_rows = sorted(
        [{"设备": k, "知识片段数": v} for k, v in stats["devices"].items()],
        key=lambda x: x["知识片段数"],
        reverse=True,
    )
    node_type_rows = [{"节点类型": k, "数量": v} for k, v in kgm["node_types"].items()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("图谱节点", kgm["nodes"])
    c2.metric("关系边", kgm["edges"])
    c3.metric("知识片段", stats["chunk_count"])
    c4.metric("典型链路", kgm["chains"])

    st.markdown("### 图谱 Schema")
    schema = data.get("schema", {})
    st.caption(schema.get("description", ""))
    sc1, sc2 = st.columns(2)
    sc1.container(border=True).markdown("**实体类型**  \n" + "、".join(schema.get("entities", [])))
    sc2.container(border=True).markdown("**关系类型**  \n" + "、".join(schema.get("relations", [])))

    st.markdown("### 设备—故障—处置关系图")
    st.caption("红色边框表示高风险节点；黄色节点表示检测参数；绿色节点表示处置动作。")
    st.graphviz_chart(typed_dot(data, limit_edges=55))

    st.markdown("### 典型检修知识链路")
    chains = data.get("chains", [])
    for row in chains:
        with st.container(border=True):
            st.markdown(f"**{row['name']}**")
            st.write(row["chain"])
            st.caption("业务价值：" + row.get("value", ""))

    st.markdown("### 图谱节点类型分布")
    if node_type_rows:
        df_type = pd.DataFrame(node_type_rows)
        chart = (
            alt.Chart(df_type)
            .mark_bar()
            .encode(
                x=alt.X("数量:Q", title="数量", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("节点类型:N", title="节点类型", sort="-x"),
                tooltip=["节点类型", "数量"],
            )
            .properties(height=max(260, 36 * len(df_type)))
        )
        st.altair_chart(chart, width="stretch")

    st.markdown("### 设备知识覆盖情况")
    if device_rows:
        df = pd.DataFrame(device_rows)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("知识片段数:Q", title="知识片段数", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("设备:N", title="设备", sort="-x"),
                tooltip=["设备", "知识片段数"],
            )
            .properties(height=max(220, 42 * len(df)))
        )
        st.altair_chart(chart, width="stretch")
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("### 图谱关系明细")
    rel_df = pd.DataFrame(data.get("edges", []), columns=["主体", "关系", "客体"])
    st.dataframe(rel_df, width="stretch", hide_index=True)


def page_ai_config():
    show_header()
    st.subheader("🤖 大模型接入与连通性检查")
    cfg = llm_config_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Provider", str(cfg["provider"]))
    c2.metric("模型", str(cfg["model"]))
    c3.metric("文本大模型", "已启用" if cfg["enabled"] else "未启用")
    c4.metric("视觉大模型", "已启用" if cfg["vision_enabled"] else "未启用")

    st.markdown("### 当前配置")
    st.dataframe(pd.DataFrame([cfg]), width="stretch", hide_index=True)

    st.markdown("### 推荐 .env 配置")
    provider = st.selectbox("选择配置模板", ["deepseek", "qwen", "ollama", "openai_compatible"] )
    examples = {
        "deepseek": "LLM_PROVIDER=deepseek\nLLM_BASE_URL=https://api.deepseek.com\nLLM_API_KEY=sk-你的DeepSeekKey\nLLM_MODEL=deepseek-v4-flash\nLLM_DEEPSEEK_THINKING=false\nLLM_REASONING_EFFORT=high",
        "qwen": "LLM_PROVIDER=qwen\nLLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\nLLM_API_KEY=sk-你的百炼Key\nLLM_MODEL=qwen-plus",
        "ollama": "LLM_PROVIDER=ollama\nLLM_BASE_URL=http://localhost:11434/v1\nLLM_API_KEY=ollama\nLLM_MODEL=qwen3:8b",
        "openai_compatible": "LLM_PROVIDER=openai_compatible\nLLM_BASE_URL=https://你的服务地址/v1\nLLM_API_KEY=你的Key\nLLM_MODEL=你的模型名",
    }
    st.code(examples[provider], language="bash")
    st.caption("修改 .env 后请重启 Streamlit。真实 API Key 不要写进 README、PPT、视频画面或公开仓库。")

    if st.button("测试大模型连接", type="primary"):
        with st.spinner("正在发送一条最小测试请求..."):
            ok, msg = test_llm_connection()
        if ok:
            st.success(f"连接成功：{msg}")
        else:
            st.error(msg)

    st.markdown("### 接入逻辑")
    st.write("系统先从本地设备手册、SOP 和案例库中查找相关资料，再将资料与问题发送给已配置的大模型。模型服务不可用时，页面直接显示本地摘要，其他功能不受影响。")


def page_admin():
    show_header()
    st.subheader("⚙️ 知识库管理")
    uploaded = st.file_uploader("上传检修手册/制度文档（md/txt/pdf）", type=["md", "txt", "pdf"])
    if uploaded and st.button("导入知识库"):
        try:
            data = uploaded.getvalue()
            validate_upload_size(data, max_mb=20)
            safe_name = sanitize_filename(uploaded.name, allowed_suffixes={".md", ".txt", ".pdf"}, default_stem="manual")
            target = unique_target(MANUAL_DIR, safe_name)
            target.write_bytes(data)
            rebuild_index()
            st.success(f"已安全导入：{target.name}")
        except ValueError as exc:
            st.error(str(exc))
    if st.button("手动重建索引"):
        rebuild_index()
    st.markdown("### 设备台账")
    st.dataframe(pd.DataFrame(load_assets()), width="stretch", hide_index=True)
    st.caption("支持导入设备手册、SOP 和故障案例，导入后可一键重建索引并立即用于检修问答。")


pages = {
    "首页 / 系统概览": page_home,
    "智能检修对话": page_chat,
    "智能检修问答": page_search,
    "图文联合诊断": page_image,
    "标准化作业指引": page_workflow,
    "智能工单生成": page_order,
    "案例上传与审核": page_cases,
    "人工反馈与纠错": page_feedback,
    "知识图谱与统计": page_kg,
    "大模型配置": page_ai_config,
    "知识库管理": page_admin,
}

choice = st.sidebar.radio("功能菜单", list(pages.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("EquipMind A1 · 设备检修知识检索与作业辅助系统")
st.sidebar.caption("检修资料查询、故障诊断、作业执行与知识维护")
pages[choice]()
