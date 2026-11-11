# FAISS index: build, save, load, search
"""FAISS vector index: build, search, save and load."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from src.chunking.chunker import Chunk
from src.config import EMBEDDING_DIM
from src.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


@dataclass
class SearchResult:
    """One retrieved chunk and how similar it was to the query."""
    chunk: Chunk
    score: float   # cosine similarity: closer to 1.0 = more similar


class FaissStore:
    """Holds a FAISS index together with the chunks it was built from."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    @property
    def size(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ build

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Create a fresh index. vectors[i] must be the embedding of chunks[i]."""
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected vectors of shape (n, {EMBEDDING_DIM}), got {vectors.shape}"
            )
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors; they must match one-to-one."
            )

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Exact search using inner product. Our vectors are normalized,
        # so inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)
        self.chunks = list(chunks)
        logger.info("Built FAISS index with %d vectors", self.index.ntotal)

    # ----------------------------------------------------------------- search

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        """Return the k most similar chunks, best first."""
        if self.index is None or self.size == 0:
            raise RuntimeError("The index is empty. Call build() or load() first.")

        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Query vector must have {EMBEDDING_DIM} dimensions")

        k = max(1, min(k, self.size))
        scores, ids = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:   # FAISS pads with -1 when it has fewer than k results
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    # ------------------------------------------------------------ save / load

    def save(self, directory: str | Path) -> None:
        """Write the index and its chunks to a folder."""
        if self.index is None:
            raise RuntimeError("Nothing to save: the index has not been built.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / INDEX_FILE))
        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)
        logger.info("Saved index (%d chunks) to %s", self.size, directory)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissStore":
        """Load a previously saved index. Raises FileNotFoundError if it isn't there."""
        directory = Path(directory)
        index_path = directory / INDEX_FILE
        chunks_path = directory / CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"No saved index found in {directory}")

        store = cls()
        store.index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            store.chunks = [Chunk(**item) for item in json.load(f)]

        if store.index.ntotal != len(store.chunks):
            raise ValueError("Saved index and chunk list are out of sync; rebuild the index.")

        logger.info("Loaded index with %d chunks from %s", store.size, directory)
        return store