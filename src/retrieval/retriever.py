# Multi-query retrieval, merge and dedupe evidence
"""Multi-query retrieval: search, merge, de-duplicate and select evidence."""

import re

from src.config import MAX_CONTEXT_CHUNKS, TOP_K
from src.chunking.chunker import Chunk
from src.embeddings.embedder import embed_texts
from src.utils.logger import get_logger
from src.vectorstore.faiss_store import FaissStore, SearchResult

logger = get_logger(__name__)

EXTRA_CANDIDATES = 2          # fetch a few spares per query, since some get dropped as duplicates
REDUNDANCY_THRESHOLD = 0.7    # share of a text chunk's tokens found in a table/equation chunk

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+")


# ----------------------------------------------------------- de-duplication

def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _containment(text_chunk: Chunk, structured_chunk: Chunk) -> float:
    """What fraction of the text chunk's tokens also appear in the structured chunk?"""
    small = _tokens(text_chunk.text)
    if not small:
        return 0.0
    return len(small & _tokens(structured_chunk.text)) / len(small)


def _is_redundant(candidate: Chunk, selected: list[SearchResult]) -> bool:
    """True if `candidate` is plain page text that a selected table/equation already covers."""
    if candidate.chunk_type != "text":
        return False
    for r in selected:
        s = r.chunk
        if (
            s.chunk_type != "text"
            and s.page_number == candidate.page_number
            and _containment(candidate, s) >= REDUNDANCY_THRESHOLD
        ):
            return True
    return False


# ---------------------------------------------------------------- selection

def select_chunks(
    ranked_lists: list[list[SearchResult]],
    top_k: int = TOP_K,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
) -> list[SearchResult]:
    """Merge one ranked list per query into a single, de-duplicated evidence set.

    Queries take turns (rank 1 of every query, then rank 2, ...) so each
    sub-question gets its best passages into the set before the cap is reached.
    """
    selected: dict[str, SearchResult] = {}
    covered = [0] * len(ranked_lists)       # how many passages each query has "got"
    depth = max((len(r) for r in ranked_lists), default=0)
    full = False

    for rank in range(depth):
        for qi, ranked in enumerate(ranked_lists):
            if rank >= len(ranked) or covered[qi] >= top_k:
                continue

            result = ranked[rank]
            cid = result.chunk.chunk_id

            if cid in selected:                      # another query already found it
                covered[qi] += 1
                if result.score > selected[cid].score:
                    selected[cid] = result
                continue

            if _is_redundant(result.chunk, list(selected.values())):
                continue

            if len(selected) >= max_chunks:
                full = True
                break

            selected[cid] = result
            covered[qi] += 1

            # A new table/equation may make already-selected page text redundant.
            if result.chunk.chunk_type != "text":
                for other_id, other in list(selected.items()):
                    if other.chunk.chunk_type == "text" and _is_redundant(other.chunk, [result]):
                        del selected[other_id]
        if full:
            break

    return sorted(selected.values(), key=lambda r: r.score, reverse=True)


# --------------------------------------------------------------- public API

def retrieve(
    store: FaissStore,
    question: str,
    subquestions: list[str] | None = None,
) -> list[SearchResult]:
    """Find the passages that best answer `question` (and its sub-questions)."""
    # Search the sub-questions AND the original, so a bad split can't lose the evidence.
    queries, seen = [], set()
    for q in (subquestions or []) + [question]:
        key = " ".join(q.lower().split()).rstrip("?. ")
        if key and key not in seen:
            seen.add(key)
            queries.append(q)

    vectors = embed_texts(queries, "RETRIEVAL_QUERY")     # one API call for all queries

    k = TOP_K + EXTRA_CANDIDATES
    ranked_lists = [store.search(vectors[i], k) for i in range(len(queries))]

    results = select_chunks(ranked_lists)
    logger.info(
        "Retrieved %d passages for %d quer%s",
        len(results), len(queries), "y" if len(queries) == 1 else "ies",
    )
    return results