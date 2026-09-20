from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import re
import time
from typing import Any, Dict, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    APPROVED_CASE_DIR,
    ASSET_DIR,
    INDEX_PATH,
    MANUAL_DIR,
    STORAGE_DIR,
    WORKFLOW_DIR,
)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


INDEX_VERSION = 4


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    title: str
    device: str
    content: str
    tags: list[str]
    source_type: str = "文档"
    updated_at: str = "2026-07-17"


# Lightweight domain expansion keeps the default installation fully local and
# LoongArch-friendly while correcting common shop-floor abbreviations/variants.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("油分", "油气分离器", "油分芯"),
    ("压差高", "压差报警", "压差异常"),
    ("母排接头", "母排连接点", "母排接点"),
    ("振动大", "振动异常", "振动偏大", "振动升高"),
    ("滴漏", "泄漏", "渗漏"),
    ("OC", "过流", "过电流"),
    ("GF", "接地故障", "对地故障"),
    ("温升高", "温度高", "过热", "发热"),
    ("供气不足", "供气压力低", "压力不足"),
    ("联轴器偏心", "联轴器不对中", "对中偏差"),
    ("卡住", "卡滞", "机械堵转"),
    ("地脚松动", "地脚螺栓松动", "基础螺栓松动"),
)

DOMAIN_TERMS: tuple[str, ...] = (
    "空压机", "离心泵", "配电柜", "变频器", "风机", "电机", "轴承", "联轴器",
    "机械密封", "油气分离器", "母排", "端子", "叶轮", "冷却器", "润滑油",
    "排气温度高", "油分压差高", "供气压力低", "振动异常", "机械密封泄漏",
    "汽蚀", "过载告警", "母排过热", "OC 过流", "接地故障", "轴承温升",
    "叶轮积灰", "绝缘电阻偏低", "油压偏低", "联轴器不对中", "端子松动",
    "加速时间过短", "地脚螺栓松动", "跳闸", "泄压", "断电", "验电", "挂牌",
)

STOP_TERMS = {"怎么", "如何", "什么", "应该", "可能", "需要", "下一步", "处理", "检查", "排查", "设备"}


def _read_pdf(path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
            continue
        if buf:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n" + p).strip()
        else:
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                chunks.append(p[start:end])
                if end == len(p):
                    break
                start = max(0, end - overlap)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def infer_device_and_tags(path: Path, text: str) -> tuple[str, list[str]]:
    name = path.stem
    candidates = ["空压机", "离心泵", "配电柜", "变频器", "风机", "电机", "轴承", "冷却塔", "锅炉给水泵"]
    device = next((c for c in candidates if c in name or c in text[:800]), name)
    tag_pool = [
        "温度", "振动", "漏油", "泄漏", "噪声", "过载", "绝缘", "点检", "一级检修", "二级检修",
        "安全", "润滑", "冷却", "电流", "电压", "过流", "联轴器", "轴承", "密封", "工单", "泄压",
        "断电", "验电", "挂牌", "SOP", "母排", "压差", "汽蚀",
    ]
    tags = [t for t in tag_pool if t in text]
    return device, tags[:14]


def _extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title
    return path.stem.replace("_", " ")


def _source_name(path: Path) -> str:
    try:
        return str(path.relative_to(path.parents[2]))
    except Exception:
        return path.name


def _workflow_to_text(path: Path) -> tuple[str, str, str, list[str]] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    device = str(obj.get("device", "通用设备"))
    title = f"{device} {obj.get('name', '标准作业流程')} {obj.get('level', '')}".strip()
    lines = [
        f"# {title}",
        f"适用条件：{obj.get('condition', '')}",
        "工具与防护：" + "、".join(obj.get("tools", [])),
        "## 标准作业步骤",
    ]
    for idx, step in enumerate(obj.get("steps", []), start=1):
        lines.append(f"{idx}. {step.get('title', '')}：{step.get('detail', '')}")
        if step.get("risk"):
            lines.append(f"风险提醒：{step['risk']}")
    rules = obj.get("compliance_rules", [])
    if rules:
        lines.append("## 合规校验")
        for rule in rules:
            lines.append(
                f"- {rule.get('name', '')}：{rule.get('description', '')}；关键词：{'、'.join(rule.get('keywords', []))}"
            )
    text = "\n".join(lines)
    tags = ["SOP", "标准作业", "合规检查"] + [str(x) for x in obj.get("tools", [])[:5]]
    return title, device, text, tags


def _kg_to_text(path: Path) -> list[tuple[str, str, str, list[str]]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows: list[tuple[str, str, str, list[str]]] = []
    for idx, chain in enumerate(obj.get("chains", []), start=1):
        name = str(chain.get("name", f"知识链路{idx}"))
        chain_text = str(chain.get("chain", ""))
        value = str(chain.get("value", ""))
        device = next((d for d in ["空压机", "离心泵", "配电柜", "变频器", "风机"] if d in name + chain_text), "通用设备")
        rows.append((name, device, f"# {name}\n知识链路：{chain_text}\n业务价值：{value}", ["知识图谱", "GraphRAG", "关系链路"]))
    return rows


def _iter_source_files() -> Iterable[Path]:
    for folder in (MANUAL_DIR, APPROVED_CASE_DIR, WORKFLOW_DIR):
        if folder.exists():
            yield from sorted(p for p in folder.glob("**/*") if p.is_file())
    kg_path = ASSET_DIR / "kg_relations.json"
    if kg_path.exists():
        yield kg_path


def source_fingerprint() -> str:
    h = sha256()
    for path in _iter_source_files():
        stat = path.stat()
        try:
            rel = path.relative_to(path.parents[2]).as_posix()
        except Exception:
            rel = path.as_posix()
        h.update(rel.encode("utf-8", errors="ignore"))
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()


def load_documents() -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    document_files = list(MANUAL_DIR.glob("**/*")) + list(APPROVED_CASE_DIR.glob("**/*"))
    document_files = [p for p in document_files if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}]
    for path in sorted(document_files):
        text = read_text_file(path)
        if not text.strip():
            continue
        device, tags = infer_device_and_tags(path, text)
        title = _extract_title(path, text)
        source_type = "检修案例" if APPROVED_CASE_DIR in path.parents else "设备手册"
        for idx, part in enumerate(split_text(text)):
            chunks.append(DocumentChunk(
                chunk_id=f"{path.name}-{idx}",
                source=_source_name(path),
                title=title,
                device=device,
                content=part,
                tags=tags,
                source_type=source_type,
            ))

    for path in sorted(WORKFLOW_DIR.glob("*.json")):
        parsed = _workflow_to_text(path)
        if not parsed:
            continue
        title, device, text, tags = parsed
        for idx, part in enumerate(split_text(text)):
            chunks.append(DocumentChunk(
                chunk_id=f"{path.name}-{idx}",
                source=_source_name(path),
                title=title,
                device=device,
                content=part,
                tags=tags,
                source_type="SOP流程",
            ))

    kg_path = ASSET_DIR / "kg_relations.json"
    for idx, (title, device, text, tags) in enumerate(_kg_to_text(kg_path), start=1):
        chunks.append(DocumentChunk(
            chunk_id=f"kg-chain-{idx}",
            source=_source_name(kg_path),
            title=title,
            device=device,
            content=text,
            tags=tags,
            source_type="知识图谱",
        ))
    return chunks


def expand_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    additions: list[str] = []
    lower = normalized.lower()
    for group in SYNONYM_GROUPS:
        if any(term.lower() in lower for term in group):
            additions.extend(term for term in group if term.lower() not in lower)
    return " ".join([normalized] + list(dict.fromkeys(additions))).strip()


def _extract_terms(text: str) -> list[str]:
    text = (text or "").strip()
    found = [term for term in DOMAIN_TERMS if term.lower() in text.lower()]
    codes = re.findall(r"\b[A-Za-z]{1,5}[A-Za-z0-9_-]{1,15}\b|\b\d+(?:\.\d+)?\s*(?:°C|℃|A|V|kW|MPa|mm/s)\b", text, flags=re.I)
    phrases = re.findall(r"[\u4e00-\u9fff]{2,8}", re.sub(r"[，。；：、？！,.!?()（）/\\]", " ", text))
    for phrase in phrases:
        if phrase not in STOP_TERMS and len(phrase) <= 8:
            found.append(phrase)
    return list(dict.fromkeys(found + codes))[:24]


def _confidence(score: float, keyword_coverage: float) -> str:
    if score >= 0.55 or (score >= 0.40 and keyword_coverage >= 0.30):
        return "高"
    if score >= 0.28 or keyword_coverage >= 0.25:
        return "中"
    return "低"




def _lexical_tokens(text: str) -> list[str]:
    """Create lightweight Chinese/industrial tokens without external NLP models."""
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    tokens: list[str] = []
    tokens.extend(term.lower().replace(" ", "") for term in DOMAIN_TERMS if term.lower() in normalized)
    tokens.extend(re.findall(r"[a-z]{1,6}[a-z0-9_-]{1,20}|\d+(?:\.\d+)?(?:℃|°c|a|v|kw|mpa|mm/s)?", normalized, flags=re.I))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        if len(segment) <= 8:
            tokens.append(segment)
        for n in (2, 3):
            if len(segment) >= n:
                tokens.extend(segment[i:i+n] for i in range(len(segment)-n+1))
    return [t for t in tokens if t and t not in STOP_TERMS]


def _query_intent(query: str) -> str:
    q = query or ""
    if any(k in q for k in ("安全", "风险", "检修前", "注意", "断电", "验电", "泄压", "挂牌", "合规")):
        return "安全作业"
    if any(k in q for k in ("步骤", "流程", "怎么处理", "如何处理", "怎么检查", "如何排查", "下一步")):
        return "处置步骤"
    if any(k in q for k in ("案例", "经验", "曾经", "历史", "类似")):
        return "案例经验"
    if any(k in q for k in ("关系", "链路", "图谱", "关联")):
        return "知识关系"
    if any(k in q for k in ("原因", "为什么", "可能是什么", "故障")):
        return "故障诊断"
    return "综合检索"


_SOURCE_INTENT_BOOSTS: dict[str, dict[str, float]] = {
    "安全作业": {"SOP流程": 0.075, "设备手册": 0.025, "知识图谱": 0.015},
    "处置步骤": {"SOP流程": 0.065, "检修案例": 0.035, "设备手册": 0.025},
    "案例经验": {"检修案例": 0.080, "设备手册": 0.015},
    "知识关系": {"知识图谱": 0.090, "设备手册": 0.010},
    "故障诊断": {"检修案例": 0.045, "设备手册": 0.035, "知识图谱": 0.020},
    "综合检索": {"设备手册": 0.020, "检修案例": 0.020, "SOP流程": 0.015},
}


def _extract_model_codes(query: str) -> list[str]:
    codes = re.findall(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9_-]{2,20}\b", query or "", flags=re.I)
    blocked = {"oc", "gf", "ac", "dc", "sop", "rag"}
    return [c for c in dict.fromkeys(codes) if c.lower() not in blocked]


def assess_evidence(query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess whether retrieved evidence is sufficient for a maintenance decision."""
    if not rows:
        return {
            "score": 0,
            "grade": "不足",
            "source_types": 0,
            "term_coverage": 0.0,
            "reasons": ["未召回可用资料"],
            "missing": ["设备型号", "报警代码", "发生工况", "现场测量值"],
        }
    top = rows[0]
    source_types = len({r.get("source_type", "") for r in rows[:5] if r.get("source_type")})
    source_files = len({r.get("source", "") for r in rows[:5] if r.get("source")})
    terms = _extract_terms(expand_query(query))
    matched = {t for r in rows[:5] for t in r.get("matched_terms", [])}
    term_coverage = len(matched) / max(len(terms), 1)
    top_score = float(top.get("score", 0))
    margin = max(0.0, top_score - float(rows[1].get("score", 0))) if len(rows) > 1 else top_score
    score = round(min(100.0, 48 * min(top_score / 0.65, 1.0) + 20 * min(term_coverage / 0.55, 1.0) + 18 * min(source_types / 3, 1.0) + 8 * min(source_files / 3, 1.0) + 6 * min(margin / 0.18, 1.0)))
    grade = "充足" if score >= 75 else "一般" if score >= 50 else "不足"
    reasons: list[str] = []
    if top_score >= 0.5:
        reasons.append("首条证据相关度较高")
    if source_types >= 3:
        reasons.append("手册、案例或SOP等多类证据互相印证")
    elif source_types == 1:
        reasons.append("证据类型较单一")
    if term_coverage >= 0.45:
        reasons.append("问题关键术语覆盖较完整")
    elif term_coverage < 0.25:
        reasons.append("问题关键术语覆盖不足")
    missing: list[str] = []
    q = query or ""
    if not _extract_model_codes(q):
        missing.append("设备型号")
    if not re.search(r"\b(?:OC|GF|E\d+|F\d+|[A-Z]{1,4}\d{1,4})\b", q, flags=re.I):
        missing.append("报警代码")
    if not re.search(r"\d+(?:\.\d+)?\s*(?:℃|°C|A|V|kW|MPa|mm/s|Hz|%)", q, flags=re.I):
        missing.append("现场测量值")
    if not any(k in q for k in ("启动", "运行", "满载", "空载", "停机", "升速", "降速", "复位")):
        missing.append("发生工况")
    return {
        "score": score,
        "grade": grade,
        "source_types": source_types,
        "source_files": source_files,
        "term_coverage": term_coverage,
        "reasons": reasons or ["已完成基础证据检索"],
        "missing": missing[:4],
    }


class KnowledgeRetriever:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.fine_vectorizer: TfidfVectorizer | None = None
        self.fine_matrix = None
        self.chunks: list[DocumentChunk] = []
        self.lexical_doc_tokens: list[list[str]] = []
        self.lexical_df: dict[str, int] = {}
        self.lexical_avgdl: float = 0.0
        self.last_query_ms: float = 0.0
        self.index_fingerprint: str = ""

    def clear(self) -> None:
        self.vectorizer = None
        self.matrix = None
        self.fine_vectorizer = None
        self.fine_matrix = None
        self.chunks = []
        self.lexical_doc_tokens = []
        self.lexical_df = {}
        self.lexical_avgdl = 0.0
        self.last_query_ms = 0.0
        self.index_fingerprint = ""
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()

    def _build_lexical_index(self, texts: list[str]) -> None:
        self.lexical_doc_tokens = [_lexical_tokens(text) for text in texts]
        df: Counter[str] = Counter()
        for tokens in self.lexical_doc_tokens:
            df.update(set(tokens))
        self.lexical_df = dict(df)
        self.lexical_avgdl = sum(len(tokens) for tokens in self.lexical_doc_tokens) / max(len(self.lexical_doc_tokens), 1)

    def build(self) -> None:
        self.chunks = load_documents()
        texts = [f"{c.title} {c.device} {' '.join(c.tags)} {c.content}" for c in self.chunks]
        if not texts:
            texts = ["空知识库"]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.fine_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1, sublinear_tf=True)
        self.fine_matrix = self.fine_vectorizer.fit_transform(texts)
        self._build_lexical_index(texts)
        self.index_fingerprint = source_fingerprint()
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        with INDEX_PATH.open("wb") as f:
            pickle.dump({
                "version": INDEX_VERSION,
                "fingerprint": self.index_fingerprint,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "fine_vectorizer": self.fine_vectorizer,
                "fine_matrix": self.fine_matrix,
                "chunks": self.chunks,
                "lexical_doc_tokens": self.lexical_doc_tokens,
                "lexical_df": self.lexical_df,
                "lexical_avgdl": self.lexical_avgdl,
            }, f)

    def load_or_build(self) -> None:
        current_fingerprint = source_fingerprint()
        if INDEX_PATH.exists():
            try:
                obj = pickle.loads(INDEX_PATH.read_bytes())
                if obj.get("version") != INDEX_VERSION or obj.get("fingerprint") != current_fingerprint:
                    raise ValueError("index version or data fingerprint changed")
                self.vectorizer = obj["vectorizer"]
                self.matrix = obj["matrix"]
                self.fine_vectorizer = obj["fine_vectorizer"]
                self.fine_matrix = obj["fine_matrix"]
                self.chunks = obj["chunks"]
                self.lexical_doc_tokens = obj["lexical_doc_tokens"]
                self.lexical_df = obj["lexical_df"]
                self.lexical_avgdl = obj["lexical_avgdl"]
                self.index_fingerprint = obj["fingerprint"]
                return
            except Exception:
                pass
        self.build()

    def _bm25_scores(self, query: str) -> list[float]:
        q_tokens = list(dict.fromkeys(_lexical_tokens(query)))
        n_docs = max(len(self.lexical_doc_tokens), 1)
        k1, b = 1.5, 0.72
        scores: list[float] = []
        for tokens in self.lexical_doc_tokens:
            counts = Counter(tokens)
            dl = max(len(tokens), 1)
            value = 0.0
            for token in q_tokens:
                tf = counts.get(token, 0)
                if not tf:
                    continue
                df = self.lexical_df.get(token, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * dl / max(self.lexical_avgdl, 1.0))
                value += idf * tf * (k1 + 1) / denom
            scores.append(value)
        max_score = max(scores, default=0.0)
        return [s / max_score if max_score > 0 else 0.0 for s in scores]

    def _diverse_top(self, scored: list[tuple[float, int, dict[str, Any]]], top_k: int) -> list[tuple[float, int, dict[str, Any]]]:
        pool = scored[: max(top_k * 5, 20)]
        selected: list[tuple[float, int, dict[str, Any]]] = []
        while pool and len(selected) < top_k:
            best_pos = 0
            best_value = -1e9
            for pos, item in enumerate(pool):
                relevance, idx, details = item
                chunk = self.chunks[idx]
                same_source = sum(1 for _, j, _ in selected if self.chunks[j].source == chunk.source)
                same_title = sum(1 for _, j, _ in selected if self.chunks[j].title == chunk.title)
                new_type = 1 if chunk.source_type not in {self.chunks[j].source_type for _, j, _ in selected} else 0
                value = relevance - 0.055 * same_source - 0.025 * same_title + 0.012 * new_type
                if value > best_value:
                    best_value = value
                    best_pos = pos
            selected.append(pool.pop(best_pos))
        return selected

    def search(self, query: str, device: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        if self.vectorizer is None or self.matrix is None or self.fine_vectorizer is None or self.fine_matrix is None:
            self.load_or_build()
        if not self.chunks or not (query or "").strip():
            return []

        t0 = time.perf_counter()
        expanded = expand_query(query)
        q = f"{device or ''} {expanded}".strip()
        coarse = cosine_similarity(self.vectorizer.transform([q]), self.matrix).ravel()
        fine = cosine_similarity(self.fine_vectorizer.transform([q]), self.fine_matrix).ravel()
        semantic = 0.62 * coarse + 0.38 * fine
        lexical = self._bm25_scores(q)

        query_terms = _extract_terms(expanded)
        requested_device = (device or "").strip()
        intent = _query_intent(query)
        models = _extract_model_codes(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for i, chunk in enumerate(self.chunks):
            haystack = f"{chunk.title} {chunk.device} {' '.join(chunk.tags)} {chunk.content}"
            haystack_lower = haystack.lower()
            title_lower = chunk.title.lower()
            matched_terms = [term for term in query_terms if term.lower() in haystack_lower]
            title_terms = [term for term in query_terms if term.lower() in title_lower]
            coverage = len(matched_terms) / max(len(query_terms), 1)
            title_coverage = len(title_terms) / max(len(query_terms), 1)
            device_match = 1.0 if requested_device and requested_device != "全部" and (
                requested_device in chunk.device or requested_device in chunk.title or requested_device in chunk.content
            ) else 0.0
            if requested_device and requested_device != "全部" and device_match == 0:
                continue

            phrase_bonus = 0.0
            clean_query = re.sub(r"[\s，。；：、？！,.!?()（）/\\]", "", query)
            clean_haystack = re.sub(r"\s+", "", haystack)
            if len(clean_query) >= 4 and clean_query in clean_haystack:
                phrase_bonus = 0.09
            elif title_terms:
                phrase_bonus = min(0.075, 0.022 * len(title_terms))

            model_match = 1.0 if models and any(code.lower() in haystack_lower for code in models) else 0.0
            source_boost = _SOURCE_INTENT_BOOSTS.get(intent, {}).get(chunk.source_type, 0.0)
            final = (
                0.56 * float(semantic[i])
                + 0.17 * float(lexical[i])
                + 0.13 * coverage
                + 0.055 * title_coverage
                + 0.035 * device_match
                + 0.04 * model_match
                + phrase_bonus
                + source_boost
            )
            final = min(max(final, 0.0), 1.0)
            scored.append((final, i, {
                "semantic_score": float(semantic[i]),
                "lexical_score": float(lexical[i]),
                "keyword_coverage": coverage,
                "title_coverage": title_coverage,
                "device_match": device_match,
                "model_match": model_match,
                "source_boost": source_boost,
                "intent": intent,
            }))

        scored.sort(key=lambda x: (x[0], x[2]["keyword_coverage"], x[2]["title_coverage"]), reverse=True)
        selected = self._diverse_top(scored, max(top_k, 1))
        rows: list[dict[str, Any]] = []
        for final, i, details in selected:
            c = self.chunks[i]
            rows.append({
                **asdict(c),
                "score": float(final),
                "confidence": _confidence(final, details["keyword_coverage"]),
                "matched_terms": [term for term in query_terms if term.lower() in f"{c.title} {c.content}".lower()][:8],
                **details,
            })
        self.last_query_ms = (time.perf_counter() - t0) * 1000
        return rows

    def stats(self) -> dict[str, Any]:
        if not self.chunks:
            self.load_or_build()
        devices: Dict[str, int] = {}
        tags: Dict[str, int] = {}
        source_types: Dict[str, int] = {}
        for c in self.chunks:
            devices[c.device] = devices.get(c.device, 0) + 1
            source_types[c.source_type] = source_types.get(c.source_type, 0) + 1
            for tag in c.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return {
            "chunk_count": len(self.chunks),
            "devices": devices,
            "tags": tags,
            "source_types": source_types,
            "index_version": INDEX_VERSION,
            "retrieval_engine": "双路字符TF-IDF + BM25 + 意图加权 + 多样性重排",
            "fingerprint": self.index_fingerprint or source_fingerprint(),
        }
