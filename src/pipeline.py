"""Orchestrates the full pipeline: PDF -> searchable index, and question -> cited answer."""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from typing import Callable

from src.chunking.chunker import build_chunks
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_DIR,
    MAX_CHUNKS,
    MAX_PDF_MB,
    USE_DISK_CACHE,
)
from src.embeddings.embedder import embed_documents
from src.generation.answer_generator import Answer, generate_answer
from src.ingestion.formula_handler import extract_formulas
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.table_extractor import extract_tables
from src.retrieval.query_decomposer import decompose_question
from src.retrieval.retriever import retrieve
from src.utils.logger import get_logger
from src.vectorstore.faiss_store import FaissStore

logger = get_logger(__name__)

MAX_CACHED_INDEXES = 10
META_FILE = "meta.json"

ProgressFn = Callable[[str, float], None]   # (message, fraction between 0 and 1)


@dataclass
class DocumentInfo:
    """Summary of the indexed document, for display in the UI."""
    filename: str
    n_pages: int
    n_chunks: int
    n_tables: int
    n_equations: int
    from_cache: bool


@dataclass
class QAResponse:
    """An answer plus how it was produced."""
    answer: Answer
    subquestions: list[str]
    n_passages: int
    seconds: float


# ------------------------------------------------------------------ helpers

def _cache_key(pdf_bytes: bytes) -> str:
    """Hash of the file AND the settings that shape the index.

    If you change the embedding model or chunk size, the key changes,
    so old cached indexes are never reused by mistake.
    """
    h = hashlib.sha256(pdf_bytes)
    h.update(f"|{EMBEDDING_MODEL}|{EMBEDDING_DIM}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:32]


def _safe_extract(fn, source: bytes, label: str) -> list:
    """Run an optional extractor. If it fails, continue with plain text only."""
    try:
        return fn(source)
    except Exception as exc:
        logger.warning("%s extraction failed (%s); continuing without it", label, exc)
        return []


def _prune_cache(keep: int = MAX_CACHED_INDEXES) -> None:
    """Keep only the most recently used cached indexes."""
    dirs = [d for d in INDEX_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for old in dirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def _validate_pdf(pdf_bytes: bytes) -> None:
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")
    if not pdf_bytes.lstrip()[:5] == b"%PDF-":
        raise ValueError("This doesn't look like a PDF file.")
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        raise ValueError(f"PDF is {size_mb:.0f} MB; the limit is {MAX_PDF_MB} MB.")


# ----------------------------------------------------------------- pipeline

class DocumentQA:
    """One indexed document that you can ask questions about."""

    def __init__(self) -> None:
        self.store: FaissStore | None = None
        self.info: DocumentInfo | None = None

    @property
    def is_ready(self) -> bool:
        return self.store is not None

    # ------------------------------------------------------------- indexing

    def index_pdf(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        progress: ProgressFn | None = None,
    ) -> DocumentInfo:
        """Parse, chunk, embed and index a PDF. Reuses a saved index if the same PDF was seen."""
        report = progress or (lambda message, fraction: None)
        _validate_pdf(pdf_bytes)

        key = _cache_key(pdf_bytes)
        cache_dir = INDEX_DIR / key

        # --- fast path: this exact PDF was indexed before
        if USE_DISK_CACHE and cache_dir.exists():
            try:
                report("Loading saved index", 0.5)
                store = FaissStore.load(cache_dir)
                meta = json.loads((cache_dir / META_FILE).read_text(encoding="utf-8"))
                cache_dir.touch()                     # mark as recently used
                self.store = store
                self.info = DocumentInfo(
                    filename=filename,
                    n_pages=meta["n_pages"],
                    n_chunks=store.size,
                    n_tables=meta["n_tables"],
                    n_equations=meta["n_equations"],
                    from_cache=True,
                )
                report("Ready", 1.0)
                return self.info
            except Exception as exc:
                logger.warning("Cached index unusable (%s); rebuilding", exc)
                shutil.rmtree(cache_dir, ignore_errors=True)

        # --- full path
        report("Reading pages", 0.05)
        try:
            pages = parse_pdf(pdf_bytes)
        except ValueError:
            raise                                       # e.g. scanned PDF: message is already clear
        except Exception as exc:
            raise ValueError("Could not read this file as a PDF.") from exc

        report("Finding tables", 0.12)
        tables = _safe_extract(extract_tables, pdf_bytes, "Table")

        report("Finding equations", 0.18)
        formulas = _safe_extract(extract_formulas, pdf_bytes, "Equation")

        report("Splitting into passages", 0.22)
        chunks = build_chunks(pages, tables, formulas)
        if not chunks:
            raise ValueError("No searchable content was found in this PDF.")
        if len(chunks) > MAX_CHUNKS:
            raise ValueError(
                f"This PDF produced {len(chunks)} passages, above the limit of {MAX_CHUNKS}. "
                "Try a shorter PDF or a section of it."
            )

        def on_embed(done: int, total: int) -> None:
            report(f"Embedding passages ({done}/{total})", 0.25 + 0.7 * done / total)

        vectors = embed_documents([c.text for c in chunks], progress_callback=on_embed)

        report("Building search index", 0.96)
        store = FaissStore()
        store.build(chunks, vectors)

        info = DocumentInfo(
            filename=filename,
            n_pages=len(pages),
            n_chunks=len(chunks),
            n_tables=len(tables),
            n_equations=len(formulas),
            from_cache=False,
        )

        if USE_DISK_CACHE:
            try:
                store.save(cache_dir)
                (cache_dir / META_FILE).write_text(
                    json.dumps({
                        "n_pages": info.n_pages,
                        "n_tables": info.n_tables,
                        "n_equations": info.n_equations,
                    }),
                    encoding="utf-8",
                )
                _prune_cache()
            except OSError as exc:                       # read-only or full disk: not fatal
                logger.warning("Could not save index cache: %s", exc)

        self.store, self.info = store, info
        report("Ready", 1.0)
        return info

    # ------------------------------------------------------------- answering

    def ask(self, question: str) -> QAResponse:
        """Answer a question about the indexed document."""
        if self.store is None:
            raise RuntimeError("Upload and index a PDF before asking questions.")

        question = question.strip()
        if not question:
            raise ValueError("Please type a question.")

        start = time.perf_counter()
        subquestions = decompose_question(question)
        results = retrieve(self.store, question, subquestions)
        answer = generate_answer(question, results)
        seconds = time.perf_counter() - start

        logger.info("Answered in %.1fs (%d passages)", seconds, len(results))
        return QAResponse(answer, subquestions, len(results), seconds)