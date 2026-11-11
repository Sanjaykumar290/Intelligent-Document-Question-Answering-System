# Detect and extract tables from PDF pages
"""Detect tables in a PDF and convert them to Markdown."""

import io
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from src.utils.logger import get_logger

logger = get_logger(__name__)

CAPTION_SEARCH_HEIGHT = 50   # how far (in PDF points) above/below a table to look for a caption
MIN_ROWS = 2
MIN_COLS = 2
MIN_FILLED_RATIO = 0.3       # reject "tables" that are mostly empty cells (false positives)


@dataclass
class TableContent:
    """One table found in the PDF."""
    page_number: int   # 1-based
    table_index: int   # 0-based position among tables on that page
    markdown: str      # the table as a Markdown grid
    caption: str       # e.g. "Table 2: Accuracy by model" (empty if none found)
    n_rows: int
    n_cols: int


def _clean_cell(cell) -> str:
    """Make one cell safe to place inside a Markdown row."""
    if cell is None:
        return ""
    text = str(cell).replace("\n", " ").strip()
    return text.replace("|", "/")  # a literal pipe would break the Markdown grid


def _to_markdown(rows: list[list[str]]) -> str:
    """Turn a list of rows into a Markdown table. The first row is the header."""
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]  # pad ragged rows

    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _find_caption(page, bbox) -> str:
    """Look just above (then just below) the table for a line starting with 'Table'."""
    x0, top, x1, bottom = bbox
    regions = [
        (0, max(0, top - CAPTION_SEARCH_HEIGHT), page.width, top),                      # above
        (0, bottom, page.width, min(page.height, bottom + CAPTION_SEARCH_HEIGHT)),      # below
    ]
    for rx0, rtop, rx1, rbottom in regions:
        if rbottom - rtop < 1:
            continue
        try:
            text = page.within_bbox((rx0, rtop, rx1, rbottom)).extract_text() or ""
        except Exception:
            continue
        for line in reversed(text.splitlines()):   # closest line to the table first (for "above")
            if line.strip().lower().startswith("table"):
                return line.strip()
    return ""


def extract_tables(source: bytes | str | Path) -> list[TableContent]:
    """Return all tables found in the PDF. Returns an empty list if there are none."""
    if isinstance(source, (bytes, bytearray)):
        pdf_file = pdfplumber.open(io.BytesIO(bytes(source)))
    else:
        pdf_file = pdfplumber.open(str(source))

    tables: list[TableContent] = []

    with pdf_file as pdf:
        for page in pdf.pages:
            try:
                found = page.find_tables()
            except Exception as exc:  # one bad page shouldn't kill the whole upload
                logger.warning("Table detection failed on page %d: %s", page.page_number, exc)
                continue

            for idx, table in enumerate(found):
                raw_rows = table.extract()
                rows = [[_clean_cell(c) for c in row] for row in raw_rows]

                # Drop rows that are entirely empty.
                rows = [r for r in rows if any(r)]
                if len(rows) < MIN_ROWS:
                    continue

                n_cols = max(len(r) for r in rows)
                if n_cols < MIN_COLS:
                    continue

                total_cells = len(rows) * n_cols
                filled = sum(1 for r in rows for c in r if c)
                if filled / total_cells < MIN_FILLED_RATIO:
                    continue

                tables.append(
                    TableContent(
                        page_number=page.page_number,
                        table_index=idx,
                        markdown=_to_markdown(rows),
                        caption=_find_caption(page, table.bbox),
                        n_rows=len(rows),
                        n_cols=n_cols,
                    )
                )

    logger.info("Found %d tables", len(tables))
    return tables