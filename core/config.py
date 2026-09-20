from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MANUAL_DIR = DATA_DIR / "manuals"
WORKFLOW_DIR = DATA_DIR / "workflows"
IMAGE_DIR = DATA_DIR / "images"
CASE_DIR = DATA_DIR / "cases"
PENDING_CASE_DIR = CASE_DIR / "pending"
APPROVED_CASE_DIR = CASE_DIR / "approved"
EVALUATION_DIR = DATA_DIR / "evaluation"
ASSET_DIR = DATA_DIR / "assets"
STORAGE_DIR = ROOT_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "knowledge_index.pkl"

load_dotenv(ROOT_DIR / ".env")

# LLM configuration. The application is designed to run without any API key.
# When LLM_PROVIDER is not "none", the RAG answer page calls an
# OpenAI-compatible /chat/completions endpoint and falls back to local summary
# if the request fails.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# DeepSeek V4 optional thinking mode. Keep disabled by default for faster, cheaper
# demo responses. Enable only when you need stronger reasoning in final presentation.
LLM_DEEPSEEK_THINKING = os.getenv("LLM_DEEPSEEK_THINKING", "false").strip().lower() in {"1", "true", "yes", "on"}
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "high").strip()

# Optional vision model used for uploaded fault images. Leave empty to keep the
# image page in local perceptual-hash mode only.
LLM_ENABLE_VISION = os.getenv("LLM_ENABLE_VISION", "false").strip().lower() in {"1", "true", "yes", "on"}
LLM_VISION_MODEL = os.getenv("LLM_VISION_MODEL", "").strip()
