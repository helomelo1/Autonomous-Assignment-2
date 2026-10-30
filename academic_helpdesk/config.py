from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
TRANSCRIPTIONS_DIR = DATA_DIR / "manual_transcriptions"
COURSES_CSV = DATA_DIR / "courses(1).csv"
FACULTY_CSV = DATA_DIR / "faculty_emails.csv"
CHROMA_DIR = ROOT / "chroma_db"
UNANSWERED_LOG = ROOT / "unanswered_questions.csv"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "2800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "350"))
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
