from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.retriever import KnowledgeRetriever

if __name__ == "__main__":
    r = KnowledgeRetriever()
    r.build()
    print(f"Index built: {len(r.chunks)} chunks")
