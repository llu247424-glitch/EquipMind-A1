from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from core.retriever import KnowledgeRetriever, assess_evidence
from core.workflow import list_workflows, best_workflow, compliance_check, completion_rate
from core.image_search import load_image_cases
from core.llm import answer_question
from core.evals import run_retrieval_eval
from core.security import sanitize_filename, validate_upload_size
from core.platform_check import platform_report
from core.diagnostics import diagnose
from core.feedback import save_feedback, load_feedback, feedback_stats

ROOT = Path(__file__).resolve().parent


def test_retriever():
    r = KnowledgeRetriever(); r.build()
    assert len(r.chunks) >= 30, f"knowledge chunks too few: {len(r.chunks)}"
    stats = r.stats()
    assert {"设备手册", "检修案例", "SOP流程", "知识图谱"}.issubset(stats["source_types"]), stats["source_types"]
    rows = r.search("空压机排气温度高怎么处理", top_k=3)
    assert rows, "retriever returned empty result"
    assert rows[0]["score"] >= 0
    assert rows[0]["confidence"] in {"高", "中", "低"}
    assert "semantic_score" in rows[0] and "lexical_score" in rows[0] and "keyword_coverage" in rows[0]
    assert rows[0]["intent"] in {"安全作业", "处置步骤", "案例经验", "知识关系", "故障诊断", "综合检索"}
    evidence = assess_evidence("空压机排气温度高怎么处理", rows)
    assert evidence["grade"] in {"充足", "一般", "不足"} and 0 <= evidence["score"] <= 100


def test_hybrid_retrieval_aliases():
    r = KnowledgeRetriever(); r.load_or_build()
    rows = r.search("GA75 空压机油分压差高", device="空压机", top_k=3)
    assert rows and "油气分离器压差高" in (rows[0]["title"] + rows[0]["content"])
    rows = r.search("配电柜母排接头过热", device="配电柜", top_k=3)
    assert rows and "母排连接点过热" in (rows[0]["title"] + rows[0]["content"])


def test_workflow():
    assert len(list_workflows()) >= 5
    wf = best_workflow("离心泵")
    assert wf and wf.get("steps")
    results = compliance_check("已断电挂牌，泄压，检查轴承和润滑，试运行复测", wf)
    assert 0 <= completion_rate(results) <= 1


def test_image_cases():
    cases = load_image_cases()
    assert len(cases) >= 5
    for c in cases:
        assert Path(c.image_path).exists()
    class LocalUpload:
        def __init__(self, data: bytes):
            self._data = data
        def getvalue(self):
            return self._data
    sample = cases[3]
    rows = __import__("core.image_search", fromlist=["search_similar_image"]).search_similar_image(
        LocalUpload(Path(sample.image_path).read_bytes()), top_k=1
    )
    assert rows[0]["case_id"] == sample.case_id and rows[0]["score"] == 1.0
    fused = __import__("core.image_search", fromlist=["search_similar_image"]).search_similar_image(
        LocalUpload(Path(sample.image_path).read_bytes()), top_k=1, text_hint="变频器OC过流并跳闸", device_hint="变频器"
    )
    assert fused[0]["case_id"] == sample.case_id
    assert fused[0]["fusion_mode"] == "图像特征 + 故障描述 + 设备类型"
    assert fused[0]["text_score"] > 0 and fused[0]["confidence"] in {"高", "中", "低"}


def test_chat_fallback():
    r = KnowledgeRetriever(); r.load_or_build()
    rows = r.search("变频器 OC 过流", top_k=3)
    ans = answer_question("变频器 OC 过流", rows)
    assert "建议" in ans or "检查" in ans
    assert "引用资料" in ans


def test_eval():
    r = KnowledgeRetriever(); r.load_or_build()
    report = run_retrieval_eval(r, top_k=3)
    assert report["count"] >= 10
    assert report["hitk"] >= 0.9
    assert report["mrr"] >= 0.85


def test_risk_scoring_is_grounded():
    assert diagnose("设备有轻微噪声").priority == "P3"
    assert diagnose("空压机排气温度高").priority == "P2"
    assert diagnose("变频器冒烟并反复跳闸").priority == "P1"


def test_upload_security():
    safe = sanitize_filename("../../恶意/../manual.pdf", allowed_suffixes={".pdf"})
    assert safe == "manual.pdf"
    safe_win = sanitize_filename(r"..\..\设备手册.md", allowed_suffixes={".md"})
    assert safe_win == "设备手册.md"
    validate_upload_size(b"123", max_mb=1)
    try:
        validate_upload_size(b"x" * (1024 * 1024 + 1), max_mb=1)
        raise AssertionError("oversized upload was accepted")
    except ValueError:
        pass



def test_feedback_loop():
    with TemporaryDirectory() as td:
        path = Path(td) / "feedback.jsonl"
        record = save_feedback(
            question="变频器OC过流怎么处理",
            answer="先断电并检查电机绝缘。",
            rating="需要纠正",
            correction="应先核对报警历史，再执行断电验电并测量电机和输出电缆绝缘。",
            reviewer="测试班组",
            device="变频器",
            references=[{"title": "变频器检修手册", "source": "manual.md", "source_type": "设备手册", "score": 0.8}],
            path=path,
        )
        assert record.feedback_id.startswith("FB-")
        rows = load_feedback(path=path)
        assert len(rows) == 1 and rows[0]["rating"] == "需要纠正"
        stats = feedback_stats(path=path)
        assert stats["count"] == 1 and stats["correction_rate"] == 1.0

def test_platform_report():
    report = platform_report()
    assert report["architecture"]
    assert report["os"]
    assert isinstance(report["competition_ready"], bool)




def test_launch_scripts_are_clean():
    for name in ["run_windows.bat", "run_deepseek_windows.bat"]:
        data = (ROOT / name).read_bytes()
        bad = [b for b in data if b < 32 and b not in (9, 10, 13)]
        assert not bad, f"control characters found in {name}: {bad}"
        decoded = data.decode("utf-8")
        assert "Scripts\\python.exe" in decoded or "run_windows.bat" in decoded
    linux = (ROOT / "run_linux.sh").read_text(encoding="utf-8")
    assert ".venv/bin/python" in linux and "python scripts/loongarch_check.py" not in linux
    assert '"$PY" scripts/loongarch_check.py' in linux

def test_required_submission_docs():
    names = [
        "软件功能需求分析文档.docx",
        "软件功能设计文档.docx",
        "软件产品说明书.docx",
        "软件功能测试报告.docx",
        "软件安装包及部署文档.docx",
    ]
    for name in names:
        path = ROOT / "docs" / name
        assert path.exists() and path.stat().st_size > 10_000, f"required document missing: {name}"
        assert zipfile.is_zipfile(path), f"invalid docx package: {name}"

def test_ui_copy_is_formal():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    forbidden = ["MVP", "正式参赛时", "占位", "没做完", "TODO"]
    for word in forbidden:
        assert word not in text, f"forbidden UI word found: {word}"


if __name__ == "__main__":
    test_retriever(); test_hybrid_retrieval_aliases(); test_workflow(); test_image_cases()
    test_chat_fallback(); test_eval(); test_risk_scoring_is_grounded(); test_upload_security(); test_feedback_loop(); test_platform_report(); test_launch_scripts_are_clean(); test_required_submission_docs(); test_ui_copy_is_formal()
    print("smoke tests passed")
