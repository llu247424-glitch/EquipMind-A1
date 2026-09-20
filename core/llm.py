from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any
import requests

from .retriever import assess_evidence
from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_DEEPSEEK_THINKING,
    LLM_ENABLE_VISION,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_REASONING_EFFORT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    LLM_VISION_MODEL,
)

SYSTEM_PROMPT = """你是设备检修知识助手。你必须优先基于给定资料回答，不能把通用常识伪装成已检索到的事实。
回答请面向一线检修人员，结构固定为：可能原因、检查步骤、风险提醒、建议处置、引用资料。
若资料不足，请明确说明缺少哪些资料，并给出安全保守的下一步建议。"""

VISION_PROMPT = """你是设备检修图片初筛助手。请根据图片做谨慎描述，输出：
1. 可见异常现象；2. 可能关联设备/部件；3. 建议补充拍摄角度；4. 安全提醒。
如果图片无法判断，请明确说明不能仅凭图片确认故障。"""


@dataclass(frozen=True)
class ProviderProfile:
    provider: str
    base_url: str
    model: str
    api_key_required: bool
    note: str


PROFILES: dict[str, ProviderProfile] = {
    "none": ProviderProfile("none", "", "", False, "不开启大模型，仅使用本地 RAG 摘要。"),
    "deepseek": ProviderProfile("deepseek", "https://api.deepseek.com", "deepseek-flash", True, "DeepSeek OpenAI-compatible Chat API。"),
    "qwen": ProviderProfile("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", True, "阿里云百炼/通义千问 OpenAI 兼容接口。"),
    "ollama": ProviderProfile("ollama", "http://localhost:11434/v1", "qwen3:8b", False, "Ollama 本地 OpenAI 兼容接口，API Key 会被忽略。"),
    "openai_compatible": ProviderProfile("openai_compatible", "", "", True, "任意兼容 OpenAI /chat/completions 的服务。"),
}


def _active_profile() -> ProviderProfile:
    key = LLM_PROVIDER or "none"
    profile = PROFILES.get(key, PROFILES["openai_compatible"])
    return ProviderProfile(
        provider=profile.provider,
        base_url=LLM_BASE_URL or profile.base_url,
        model=LLM_MODEL or profile.model,
        api_key_required=profile.api_key_required,
        note=profile.note,
    )


def _redact(value: str) -> str:
    if not value:
        return "未填写"
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


def _chat_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _headers(profile: ProviderProfile) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    # Ollama ignores the key but accepts any Authorization value through the
    # OpenAI-compatible endpoint. Keep the header only when a key is supplied or
    # the provider formally requires one.
    if LLM_API_KEY or profile.api_key_required:
        headers["Authorization"] = f"Bearer {LLM_API_KEY or 'ollama'}"
    return headers


def llm_config_summary() -> dict[str, str | bool]:
    profile = _active_profile()
    enabled = profile.provider != "none" and bool(profile.base_url and profile.model) and (bool(LLM_API_KEY) or not profile.api_key_required)
    return {
        "provider": profile.provider,
        "base_url": profile.base_url or "未配置",
        "model": profile.model or "未配置",
        "api_key": _redact(LLM_API_KEY),
        "enabled": enabled,
        "vision_enabled": bool(LLM_ENABLE_VISION and (LLM_VISION_MODEL or profile.model)),
        "vision_model": LLM_VISION_MODEL or profile.model or "未配置",
        "deepseek_thinking": bool(profile.provider == "deepseek" and LLM_DEEPSEEK_THINKING),
        "reasoning_effort": LLM_REASONING_EFFORT,
        "note": profile.note,
    }


def llm_status() -> str:
    cfg = llm_config_summary()
    if cfg["enabled"]:
        return f"已启用大模型：{cfg['provider']} / {cfg['model']}。若接口异常会自动降级为本地 RAG 摘要。"
    return "未启用大模型接口，当前使用本地 RAG 摘要模式；填写 .env 后可启用 DeepSeek、通义千问、Ollama 或其他 OpenAI 兼容模型。"


def _call_openai_compatible(messages: list[dict[str, Any]], temperature: float | None = None, model: str | None = None) -> str | None:
    profile = _active_profile()
    if profile.provider == "none":
        return None
    if not profile.base_url or not (model or profile.model):
        return None
    if profile.api_key_required and not LLM_API_KEY:
        return None

    payload = {
        "model": model or profile.model,
        "messages": messages,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
    }
    # DeepSeek V4 supports an optional thinking mode. Keep it off by default
    # because the demo system values stable latency and low cost.
    if profile.provider == "deepseek" and LLM_DEEPSEEK_THINKING:
        payload["reasoning_effort"] = LLM_REASONING_EFFORT or "high"
        payload["thinking"] = {"type": "enabled"}
    try:
        r = requests.post(
            _chat_url(profile.base_url),
            headers=_headers(profile),
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"大模型接口调用失败，已降级为本地检索摘要。错误：{exc}"


def _format_context(retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "无检索资料。"
    blocks: list[str] = []
    for idx, r in enumerate(retrieved, start=1):
        blocks.append(
            f"[资料{idx}]\n"
            f"标题：{r.get('title', '')}\n"
            f"设备：{r.get('device', '')}\n"
            f"来源：{r.get('source', '')}\n"
            f"相似度：{r.get('score', 0):.3f}\n"
            f"内容：{r.get('content', '')}"
        )
    return "\n\n".join(blocks)


def answer_question(question: str, retrieved: list[dict[str, Any]]) -> str:
    context = _format_context(retrieved)
    evidence = assess_evidence(question, retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"已从本地知识库召回的资料：\n{context}\n\n"
                f"证据充分度：{evidence['grade']}（{evidence['score']}/100）；建议补充：{'、'.join(evidence['missing']) or '无'}。\n\n"
                "请严格基于资料回答。输出 Markdown，包含：\n"
                "## 可能原因\n## 检查步骤\n## 风险提醒\n## 建议处置\n## 引用资料\n"
                "引用资料请写成“资料1、资料2”的形式。"
            ),
        },
    ]
    llm_answer = _call_openai_compatible(messages)
    if llm_answer and not llm_answer.startswith("大模型接口调用失败"):
        return llm_answer

    if not retrieved:
        suffix = f"\n\n> {llm_answer}" if llm_answer else ""
        return "未检索到足够资料。建议补充设备手册、故障案例或选择更明确的设备型号。" + suffix

    def clean_line(value: str) -> str:
        value = re.sub(r"^[#>*\-\d.、)（(\s]+", "", value.strip())
        value = re.sub(r"\s+", " ", value)
        return value.strip("；。 ")

    def section_items(text: str, headings: tuple[str, ...]) -> list[str]:
        lines = text.splitlines()
        active = False
        items: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            heading = line.lstrip("#").strip()
            if line.startswith("#"):
                active = any(h in heading for h in headings)
                continue
            if active:
                item = clean_line(line)
                if len(item) >= 8:
                    items.append(item)
            elif any(h in line for h in headings) and "：" in line:
                item = clean_line(line.split("：", 1)[1])
                if len(item) >= 8:
                    items.append(item)
        return items

    def dedupe(items: list[str], limit: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = re.sub(r"[，。；：、\s]", "", item)
            if not key or any(key in old or old in key for old in seen):
                continue
            seen.add(key)
            out.append(item[:220])
            if len(out) >= limit:
                break
        return out

    causes: list[str] = []
    steps: list[str] = []
    risks: list[str] = []
    refs: list[str] = []
    for idx, row in enumerate(retrieved[:5], start=1):
        text = row.get("content", "")
        refs.append(
            f"资料{idx}《{row.get('title', '未命名资料')}》"
            f"（{row.get('source_type', '文档')}，置信度{row.get('confidence', '未评估')}）"
        )
        causes.extend(f"{x} [资料{idx}]" for x in section_items(text, ("原因分析", "可能原因", "故障原因")))
        steps.extend(f"{x} [资料{idx}]" for x in section_items(text, ("处理过程", "检查步骤", "标准作业步骤", "处置步骤")))
        risks.extend(f"{x} [资料{idx}]" for x in section_items(text, ("安全注意", "安全措施", "风险提醒")))

    if not causes:
        causes = [f"{clean_line(retrieved[0].get('content', ''))[:200]} [资料1]"]
    if not steps:
        for idx, row in enumerate(retrieved[:3], start=1):
            for sentence in re.split(r"[。；\n]", row.get("content", "")):
                if any(k in sentence for k in ("检查", "测量", "确认", "更换", "复测", "试运行")):
                    item = clean_line(sentence)
                    if item:
                        steps.append(f"{item} [资料{idx}]")
    if not risks:
        risks = ["作业前根据设备类型执行停机、断电、挂牌、验电或泄压确认；涉及带电、高温和旋转部件时必须使用相应 PPE。"]

    top_confidence = retrieved[0].get("confidence", "未评估")
    evidence_note = f"证据充分度：**{evidence['grade']}（{evidence['score']}/100）**。"
    missing_note = (" 建议补充：" + "、".join(evidence["missing"]) + "。") if evidence["missing"] else ""
    degraded_reason = ""
    if llm_answer and llm_answer.startswith("大模型接口调用失败"):
        degraded_reason = f"\n\n> {llm_answer}"
    return (
        f"> 当前为本地证据摘要模式；首条证据置信度：**{top_confidence}**；{evidence_note}{missing_note}所有结论均标注资料编号。"
        + degraded_reason
        + "\n\n## 可能原因\n"
        + "\n".join(f"- {x}" for x in dedupe(causes, 4))
        + "\n\n## 检查步骤\n"
        + "\n".join(f"{i}. {x}" for i, x in enumerate(dedupe(steps, 6), start=1))
        + "\n\n## 风险提醒\n"
        + "\n".join(f"- {x}" for x in dedupe(risks, 4))
        + "\n\n## 建议处置\n"
        + "- 优先执行高置信度资料中的 SOP；处理后记录温度、电流、振动、压力或绝缘等复测值。\n"
        + "- 若首条证据置信度为低，先补充设备型号、报警代码、发生工况和现场测量值，再决定拆检。"
        + "\n\n## 引用资料\n- "
        + "；".join(refs)
    )



def chat_with_retrieval(question: str, retrieved: list[dict[str, Any]], history: list[dict[str, str]] | None = None) -> str:
    """Multi-turn maintenance chat powered by RAG + optional DeepSeek/OpenAI-compatible LLM.

    The service-side chat API is stateless, so the Streamlit page passes recent
    conversation turns in every request. Retrieved local knowledge is still the
    highest-priority source; conversation history is used only to resolve follow-up
    references such as “继续”, “上一步”, “这个设备”.
    """
    context = _format_context(retrieved)
    evidence = assess_evidence(question, retrieved)
    recent_history = (history or [])[-8:]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + "\n你正在进行多轮检修对话。必须优先使用本轮召回资料，其次使用最近对话来理解指代。"
                + "\n不要编造不存在的设备参数、标准编号或现场数据；缺资料时要说清楚。"
                + "\n回答要像检修班组长，先给结论，再给可执行步骤。"
            ),
        }
    ]
    for item in recent_history:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            # Keep each turn compact so the retrieved evidence remains prominent.
            messages.append({"role": role, "content": content[:1600]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"本轮用户问题：{question}\n\n"
                f"本轮从本地知识库召回的资料：\n{context}\n\n"
                f"证据充分度：{evidence['grade']}（{evidence['score']}/100）；建议补充：{'、'.join(evidence['missing']) or '无'}。\n\n"
                "请输出 Markdown，格式为：\n"
                "### 结论\n"
                "### 排查步骤\n"
                "### 安全注意\n"
                "### 需要补充的信息\n"
                "### 引用资料\n"
                "引用资料请写成“资料1、资料2”的形式。"
            ),
        }
    )
    llm_answer = _call_openai_compatible(messages)
    if llm_answer and not llm_answer.startswith("大模型接口调用失败"):
        return llm_answer

    degraded = answer_question(question, retrieved)
    if llm_answer and llm_answer.startswith("大模型接口调用失败") and llm_answer not in degraded:
        return degraded + "\n\n" + f"> {llm_answer}"
    return degraded


def describe_image_with_llm(image_bytes: bytes, mime_type: str = "image/png") -> str | None:
    profile = _active_profile()
    model = LLM_VISION_MODEL or profile.model
    if not LLM_ENABLE_VISION or profile.provider == "none" or not model or not profile.base_url:
        return None
    if profile.api_key_required and not LLM_API_KEY:
        return None
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请对这张设备故障图片做检修初筛说明。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    answer = _call_openai_compatible(messages, temperature=0.1, model=model)
    if answer and not answer.startswith("大模型接口调用失败"):
        return answer
    return answer


def test_llm_connection() -> tuple[bool, str]:
    profile = _active_profile()
    if profile.provider == "none":
        return False, "LLM_PROVIDER=none，未启用大模型。"
    if not profile.base_url or not profile.model:
        return False, "缺少 LLM_BASE_URL 或 LLM_MODEL。"
    if profile.api_key_required and not LLM_API_KEY:
        return False, "当前服务需要 LLM_API_KEY。"
    response = _call_openai_compatible(
        [
            {"role": "system", "content": "你是接口连通性测试助手。"},
            {"role": "user", "content": "请只回复：连接成功"},
        ],
        temperature=0,
    )
    if response and not response.startswith("大模型接口调用失败"):
        return True, response
    return False, response or "未得到模型响应。"
