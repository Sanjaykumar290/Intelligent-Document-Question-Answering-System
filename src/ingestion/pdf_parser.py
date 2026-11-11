# Extract text + page numbers from PDF
"""Extract clean text, page by page, from a PDF."""

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """Text of one PDF page."""
    page_number: int   # 1-based, matches what a human sees in a PDF viewer
    text: str


def _clean_text(text: str) -> str:
    """Tidy raw extracted text without destroying its structure."""
    # Re-join words that were hyphenated across a line break: "retri-\neval" -> "retrieval"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace single newlines inside a paragraph with a space; keep blank-line paragraph breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into a single paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(source: bytes | str | Path) -> list[PageContent]:
    """Parse a PDF and return one PageContent per page that contains text.

    `source` can be raw bytes (what Streamlit's uploader gives you)
    or a path to a file on disk.
    """
    if isinstance(source, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(source), filetype="pdf")
    else:
        doc = fitz.open(str(source))

    pages: list[PageContent] = []
    empty_pages = 0

    with doc:
        total_pages = doc.page_count
        for index, page in enumerate(doc):
            # sort=True orders text blocks top-to-bottom, left-to-right,
            # which helps with multi-column papers.
            raw = page.get_text("text", sort=True)
            text = _clean_text(raw)

            if not text:
                empty_pages += 1
                continue

            pages.append(PageContent(page_number=index + 1, text=text))

    logger.info(
        "Parsed %d/%d pages with text (%d empty)",
        len(pages), total_pages, empty_pages,
    )

    if not pages:
        raise ValueError(
            "No extractable text found. This PDF may be scanned images; "
            "OCR is not supported in this version."
        )

    if empty_pages > total_pages * 0.5:
        logger.warning(
            "More than half the pages had no text. The PDF may be partly scanned."
        )

    return pages