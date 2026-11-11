# Split complex questions into subquestions
"""Split complex questions into sub-questions for multi-section retrieval."""

import re

from google.genai import errors

from src.config import MAX_SUBQUESTIONS
from src.generation.gemini_client import GenerationError, generate_json
from src.generation.prompts import build_decompose_prompt, decompose_system
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_SUBQUESTION_CHARS = 300
LONG_QUESTION_WORDS = 18

# Words that often signal a multi-part question.
_MULTI_PART_CUES = re.compile(
    r"\b(and|compare[sd]?|comparison|versus|vs\.?|difference|differences|differ|"
    r"both|as well as|respectively|also|then|each|between)\b",
    re.IGNORECASE,
)


def _might_be_multipart(question: str) -> bool:
    """Cheap check that avoids an API call for obviously simple questions."""
    if question.count("?") > 1:
        return True
    if _MULTI_PART_CUES.search(question):
        return True
    return len(question.split()) > LONG_QUESTION_WORDS


def _clean_subquestions(raw, original: str) -> list[str]:
    """Validate the model's JSON. Anything unusable falls back to the original question."""
    items = raw.get("subquestions") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [original]

    cleaned, seen = [], set()
    for item in items:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())                 # collapse stray whitespace/newlines
        if not text or len(text) > MAX_SUBQUESTION_CHARS:
            continue
        key = text.lower().rstrip("?. ")              # catch near-duplicates
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    return cleaned[:MAX_SUBQUESTIONS] or [original]


def decompose_question(question: str) -> list[str]:
    """Return a list of self-contained sub-questions (just [question] if it is simple).

    This function never fails because of the model: if the API call errors out
    or returns junk, the original question is used as-is.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question is empty.")

    if not _might_be_multipart(question):
        logger.info("Simple question; skipping decomposition")
        return [question]

    try:
        raw = generate_json(
            build_decompose_prompt(question),
            system_instruction=decompose_system(),
        )
    except (GenerationError, errors.APIError) as exc:
        logger.warning("Decomposition failed (%s); using the original question", exc)
        return [question]

    subquestions = _clean_subquestions(raw, question)
    logger.info("Decomposed into %d sub-question(s)", len(subquestions))
    return subquestions