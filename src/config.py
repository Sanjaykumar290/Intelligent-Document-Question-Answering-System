"""Central settings for the Document QA system."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
INDEX_DIR = BASE_DIR / "data" / "indexes"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Language model (answers and question decomposition) ----------
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
# Comma-separated backups in .env, tried in order if the main model is overloaded.
LLM_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()
]
TEMPERATURE = 0.2          # low = factual and consistent, which suits QA

# ---------- Embeddings ----------
# "local"  = small ONNX model on your own CPU (no API quota)
# "gemini" = Gemini embedding API (uses free-tier quota)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

if EMBEDDING_PROVIDER == "local":
    EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIM = 384
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIM = 768

EMBED_BATCH_SIZE = 50      # texts per embedding batch

# ---------- Chunking ----------
CHUNK_SIZE = 1000          # target characters per chunk
CHUNK_OVERLAP = 150        # characters shared between neighbouring chunks

# ---------- Retrieval ----------
TOP_K = 5                  # passages retrieved per (sub)question
MAX_SUBQUESTIONS = 4       # cap on how many pieces a question is split into
MAX_CONTEXT_CHUNKS = 10    # cap on total passages sent to the LLM

# ---------- Upload limits and caching ----------
MAX_PDF_MB = 50
# Refuse PDFs that would create more passages than this (protects the API quota
# when using the Gemini embedding provider; generous for local embeddings).
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "3000" if EMBEDDING_PROVIDER == "local" else "600"))
USE_DISK_CACHE = os.getenv("DOCQA_DISK_CACHE", "1") == "1"   # reuse saved indexes for repeat uploads
USE_LLM_CACHE = os.getenv("DOCQA_LLM_CACHE", "1") == "1"     # reuse saved answers for identical requests


def get_api_key() -> str:
    """Return the Gemini API key from .env locally or Streamlit secrets when deployed."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Add it to your .env file (local) "
            "or to Streamlit secrets (deployed)."
        )
    return key