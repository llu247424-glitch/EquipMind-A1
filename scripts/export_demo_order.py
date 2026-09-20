from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.retriever import KnowledgeRetriever
from core.diagnostics import diagnose, result_dict
from core.workflow import best_workflow
from core.work_order import create_work_order

if __name__ == "__main__":
    r = KnowledgeRetriever(); r.load_or_build()
    query = "空压机排气温度高，运行 20 分钟报警"
    rows = r.search(query, device="空压机", top_k=5)
    diag = result_dict(diagnose(query, rows))
    wf = best_workflow("空压机")
    md = create_work_order("空压机", "GA75", query, diag, wf, rows)
    out = ROOT / "storage" / "demo_work_order.md"
    out.write_text(md, encoding="utf-8")
    print(out)
