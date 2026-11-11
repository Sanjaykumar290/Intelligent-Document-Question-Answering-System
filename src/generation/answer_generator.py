# Build grounded answer with citations
"""Generate a grounded answer and verify its citations."""

import re
from dataclasses import dataclass

from src.generation.gemini_client import generate_text
from src.generation.prompts import ANSWER_SYSTEM, NOT_FOUND_MESSAGE, build_answer_prompt
from src.utils.logger import get_logger
from src.vectorstore.faiss_store import SearchResult

logger = get_logger(__name__)

# Matches [1], [2][4] (as two matches), [1, 2], [1-3]
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*[,;–-]\s*\d+)*)\s*\]")
_MAX_RANGE = 20


@dataclass
class Citation:
    """A passage the answer actually cites."""
    number: int          # the number shown in the final answer text
    chunk_id: str
    page_number: int
    chunk_type: str      # "text" | "table" | "equation"
    text: str            # full passage text (the UI can truncate it)
    score: float         # retrieval similarity


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    not_found: bool          # the document did not contain the answer
    dropped_citations: int   # invented citation numbers removed from the text

    @property
    def is_cited(self) -> bool:
        """True if this is a real answer backed by at least one verified citation."""
        return (not self.not_found) and len(self.citations) > 0


# ------------------------------------------------------------ citation checks

def _parse_numbers(group: str) -> list[int]:
    """'1, 3' -> [1, 3];  '2-4' -> [2, 3, 4]"""
    numbers: list[int] = []
    for part in re.split(r"[,;]", group):
        part = part.strip()
        span = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if span:
            lo, hi = int(span.group(1)), int(span.group(2))
            if lo <= hi and hi - lo <= _MAX_RANGE:
                numbers.extend(range(lo, hi + 1))
            else:
                numbers.extend([lo, hi])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def _rewrite_citations(text: str, n_passages: int) -> tuple[str, dict[int, int], int]:
    """Remove invented citation numbers and renumber the real ones in order of appearance.

    Returns (clean text, {original number: new number}, count of removed citations).
    """
    mapping: dict[int, int] = {}
    dropped = 0

    def replace(match: re.Match) -> str:
        nonlocal dropped
        new_numbers: list[int] = []
        for old in _parse_numbers(match.group(1)):
            if not 1 <= old <= n_passages:
                dropped += 1
                continue
            if old not in mapping:
                mapping[old] = len(mapping) + 1
            if mapping[old] not in new_numbers:
                new_numbers.append(mapping[old])
        return "".join(f"[{n}]" for n in new_numbers)

    cleaned = _CITATION_RE.sub(replace, text)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)   # tidy "word ." left by a removed citation
    return cleaned.strip(), mapping, dropped


def _is_refusal(text: str) -> bool:
    """True if the model replied with just the standard 'not found' sentence."""
    normalized = text.strip().strip("\"'").strip().rstrip(".").lower()
    return normalized == NOT_FOUND_MESSAGE.rstrip(".").lower()


# -------------------------------------------------------------- public API

def generate_answer(question: str, results: list[SearchResult]) -> Answer:
    """Answer `question` using only the retrieved passages, with verified citations."""
    if not results:
        return Answer(question, NOT_FOUND_MESSAGE, [], not_found=True, dropped_citations=0)

    chunks = [r.chunk for r in results]
    raw = generate_text(
        build_answer_prompt(question, chunks),
        system_instruction=ANSWER_SYSTEM,
    )

    if _is_refusal(raw):
        return Answer(question, NOT_FOUND_MESSAGE, [], not_found=True, dropped_citations=0)

    text, mapping, dropped = _rewrite_citations(raw, len(chunks))
    if dropped:
        logger.warning("Removed %d invented citation(s) from the answer", dropped)

    citations = []
    for old, new in sorted(mapping.items(), key=lambda item: item[1]):
        result = results[old - 1]            # passage [n] was results[n - 1]
        citations.append(
            Citation(
                number=new,
                chunk_id=result.chunk.chunk_id,
                page_number=result.chunk.page_number,
                chunk_type=result.chunk.chunk_type,
                text=result.chunk.text,
                score=result.score,
            )
        )

    if not citations:
        logger.warning("The answer contains no valid citations")

    return Answer(question, text, citations, not_found=False, dropped_citations=dropped)