# Split content into chunks, linking tables/equations to context
"""Turn parsed pages, tables and equations into searchable chunks."""

import re
from collections import defaultdict
from dataclasses import dataclass

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.formula_handler import FormulaContent
from src.ingestion.pdf_parser import PageContent
from src.ingestion.table_extractor import TableContent
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_TABLE_CHARS = 3000        # tables longer than this are split by rows
MAX_LINKED_SENTENCES = 3      # sentences from the page that mention a table/equation
MAX_SENTENCE_CHARS = 300
MIN_CHUNK_CHARS = 40          # drop fragments like a lone "Figure 3"

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CAPTION_START_RE = re.compile(r"^\s*(?:table|fig\.?|figure)\s+\d+\s*[:.\-–—]", re.IGNORECASE)
_TABLE_NUM_RE = re.compile(r"table\s+(\d+)", re.IGNORECASE)


@dataclass
class Chunk:
    """One searchable piece of the document."""
    chunk_id: str
    text: str
    page_number: int
    chunk_type: str      # "text" | "table" | "equation"

    @property
    def citation(self) -> str:
        suffix = "" if self.chunk_type == "text" else f", {self.chunk_type}"
        return f"p. {self.page_number}{suffix}"


# ---------------------------------------------------------------- text helpers

def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Last resort: split on word boundaries when there is no sentence break."""
    out, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > size:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def _split_long(paragraph: str, size: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries."""
    pieces, current = [], ""
    for sentence in _sentences(paragraph):
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(sentence, size))
        elif current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting at a word boundary (used as overlap)."""
    if n <= 0:
        return ""
    if len(text) <= n:
        return text
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def _chunk_paragraphs(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of about `size` characters with overlap."""
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_long(p, size) if len(p) > size else [p])

    chunks, current = [], ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > size:
            chunks.append(current)
            current = f"{_tail(current, overlap)} {unit}".strip()
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------ linking helpers

def _linked_sentences(sentences: list[str], pattern: re.Pattern, already_in: str) -> list[str]:
    """Sentences on the page that refer to a table/equation (e.g. 'Table 2 shows...')."""
    found = []
    for s in sentences:
        if not pattern.search(s):
            continue
        if _CAPTION_START_RE.match(s):      # the caption itself is already in the chunk
            continue
        if s[:60] in already_in:
            continue
        found.append(s[:MAX_SENTENCE_CHARS])
        if len(found) >= MAX_LINKED_SENTENCES:
            break
    return found


def _split_table(markdown: str, max_chars: int) -> list[str]:
    """Split a big Markdown table by rows, repeating the header in every part."""
    lines = markdown.split("\n")
    header, rows = lines[:2], lines[2:]
    header_text = "\n".join(header)

    parts, current, size = [], [], len(header_text)
    for row in rows:
        if current and size + len(row) + 1 > max_chars:
            parts.append("\n".join(header + current))
            current, size = [], len(header_text)
        current.append(row)
        size += len(row) + 1
    if current:
        parts.append("\n".join(header + current))
    return parts or [markdown]


# ---------------------------------------------------------------- main entry

def build_chunks(
    pages: list[PageContent],
    tables: list[TableContent] | None = None,
    formulas: list[FormulaContent] | None = None,
) -> list[Chunk]:
    """Create text, table and equation chunks from the ingestion output."""
    tables = tables or []
    formulas = formulas or []

    tables_by_page = defaultdict(list)
    for t in tables:
        tables_by_page[t.page_number].append(t)
    formulas_by_page = defaultdict(list)
    for f in formulas:
        formulas_by_page[f.page_number].append(f)

    chunks: list[Chunk] = []

    for page in pages:
        pn = page.page_number
        page_sentences = _sentences(page.text)

        # 1) Regular text chunks
        for i, text in enumerate(_chunk_paragraphs(_paragraphs(page.text), CHUNK_SIZE, CHUNK_OVERLAP)):
            if len(text) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(f"p{pn}-text-{i}", text, pn, "text"))

        # 2) Table chunks: caption + Markdown grid + sentences that discuss the table
        for t in tables_by_page.get(pn, []):
            num = _TABLE_NUM_RE.search(t.caption)
            discussion: list[str] = []
            if num:
                pattern = re.compile(rf"\btable\s+{num.group(1)}\b", re.IGNORECASE)
                discussion = _linked_sentences(page_sentences, pattern, t.caption)

            parts = _split_table(t.markdown, MAX_TABLE_CHARS)
            for j, part in enumerate(parts):
                title = t.caption or f"Table on page {pn}"
                if len(parts) > 1:
                    title += f" (part {j + 1} of {len(parts)})"
                body = f"{title}\n{part}"
                if discussion:
                    body += "\n\nDiscussed in the text: " + " ".join(discussion)
                chunks.append(Chunk(f"p{pn}-table-{t.table_index}-{j}", body, pn, "table"))

        # 3) Equation chunks: equation + context + sentences that cite it ("Eq. (3)")
        for k, f in enumerate(formulas_by_page.get(pn, [])):
            body = f.to_text()
            if f.equation_number:
                num = re.escape(f.equation_number)
                pattern = re.compile(
                    rf"\b(?:eq\.?|eqn\.?|equation)s?\s*\(?\s*{num}(?!\d)\s*\)?", re.IGNORECASE
                )
                cited = _linked_sentences(page_sentences, pattern, body)
                if cited:
                    body += "\n\nReferred to as: " + " ".join(cited)
            if len(body) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(f"p{pn}-eq-{k}", body, pn, "equation"))

    counts = {t: sum(c.chunk_type == t for c in chunks) for t in ("text", "table", "equation")}
    logger.info("Built %d chunks %s", len(chunks), counts)
    return chunks