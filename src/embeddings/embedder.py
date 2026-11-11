"""Convert text to vectors: a local ONNX model (default) or the Gemini embedding API."""

import logging
from typing import Callable

import numpy as np

from src.config import (
    BASE_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    get_api_key,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# bge models work best when SEARCH QUERIES (not documents) start with this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ====================================================== local (fastembed / ONNX)

_local_model = None


def _get_local_model():
    """Load the model once per process. The first run downloads it (about 130 MB)."""
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding

        logger.info("Loading local embedding model %s (first run downloads it)", EMBEDDING_MODEL)
        _local_model = TextEmbedding(
            model_name=EMBEDDING_MODEL,
            cache_dir=str(BASE_DIR / "data" / "models"),
        )
    return _local_model


def _embed_local(texts: list[str], task_type: str) -> np.ndarray:
    if task_type == "RETRIEVAL_QUERY":
        texts = [QUERY_PREFIX + t for t in texts]
    return np.array(list(_get_local_model().embed(texts, batch_size=32)), dtype=np.float32)


# ============================================================= Gemini embedding API

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(
            api_key=get_api_key(),
            http_options=types.HttpOptions(
                timeout=60_000,  # milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),  # our retry below logs each attempt
            ),
        )
    return _client


def _is_daily_quota(exc: BaseException) -> bool:
    """A per-day quota error: waiting a few minutes won't help, so don't retry."""
    from google.genai import errors

    return (
        isinstance(exc, errors.APIError)
        and exc.code == 429
        and "perday" in str(exc).lower().replace(" ", "")
    )


def _is_retryable(exc: BaseException) -> bool:
    from google.genai import errors

    return (
        isinstance(exc, errors.APIError)
        and exc.code in (429, 500, 502, 503, 504)
        and not _is_daily_quota(exc)
    )


def _wait_time(retry_state) -> float:
    """Rate limits need a long wait (the quota window is a minute); server errors a short one."""
    from google.genai import errors

    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    if isinstance(exc, errors.APIError) and exc.code == 429:
        return min(30 * attempt, 90)
    return min(2 ** attempt, 30)


def _embed_gemini(texts: list[str], task_type: str) -> np.ndarray:
    from google.genai import types
    from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=_wait_time,
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def call():
        response = _get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        return np.array([e.values for e in response.embeddings], dtype=np.float32)

    return call()


# ================================================================== shared logic

def _embed_batch(texts: list[str], task_type: str) -> np.ndarray:
    if EMBEDDING_PROVIDER == "local":
        vectors = _embed_local(texts, task_type)
    else:
        vectors = _embed_gemini(texts, task_type)

    if len(vectors) != len(texts):
        raise RuntimeError(f"Sent {len(texts)} texts but got {len(vectors)} embeddings back")
    if vectors.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding model returned {vectors.shape[1]} dimensions, expected {EMBEDDING_DIM}. "
            "Check EMBEDDING_PROVIDER and the model name in .env."
        )
    return vectors


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to length 1 so that dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(
    texts: list[str],
    task_type: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed many texts in batches. Returns a float32 array of shape (len(texts), EMBEDDING_DIM)."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    batches = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        batches.append(_embed_batch(batch, task_type))
        done = min(start + EMBED_BATCH_SIZE, total)
        logger.info("Embedded %d/%d texts", done, total)
        if progress_callback:
            progress_callback(done, total)

    return _normalize(np.vstack(batches))


def embed_documents(
    texts: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed document chunks (used when indexing an uploaded PDF)."""
    return embed_texts(texts, "RETRIEVAL_DOCUMENT", progress_callback)


def embed_query(text: str) -> np.ndarray:
    """Embed one question. Returns shape (1, EMBEDDING_DIM), ready for FAISS."""
    return embed_texts([text], "RETRIEVAL_QUERY")