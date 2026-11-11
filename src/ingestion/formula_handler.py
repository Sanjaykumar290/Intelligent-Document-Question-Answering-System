# Detect equations and keep them with surrounding text

"""Detect equations in a PDF and keep them attached to their explanatory text."""

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONTEXT_LINES_BEFORE = 3    # lines of text kept before an equation
CONTEXT_LINES_AFTER = 4     # after is longer: "where x is ..." definitions follow equations
MAX_LINE_LEN = 200          # longer lines are prose, not equations
MAX_EQUATION_LEN = 500

_MATH_CHARS = set("=+−×÷·∑∏∫√∂∇≈≠≤≥<>∈∉⊂⊆∪∩∞±→←↔∀∃∝^|‖⊗⊕∘")
_RELATION_CHARS = set("=≈≠≤≥∝")
# Equation numbers at the end of a line: "(3)", "(2.1)", "(4a)"
_EQ_NUMBER_RE = re.compile(r"\(\s*(\d+(?:\.\d+)*[a-z]?)\s*\)\s*$")


@dataclass
class FormulaContent:
    """One equation plus the text around it."""
    page_number: int         # 1-based
    equation: str
    equation_number: str     # "3" for "(3)", or "" if unnumbered
    context_before: str
    context_after: str

    def to_text(self) -> str:
        """Equation and its context as one block, ready to chunk and embed."""
        label = f"Equation ({self.equation_number})" if self.equation_number else "Equation"
        parts = [self.context_before, f"{label}: {self.equation}", self.context_after]
        return "\n".join(p for p in parts if p)


def _is_math_char(c: str) -> bool:
    code = ord(c)
    return (
        c in _MATH_CHARS
        or c in "²³¹"
        or 0x0370 <= code <= 0x03FF      # Greek letters
        or 0x2070 <= code <= 0x209F      # superscripts and subscripts
        or 0x1D400 <= code <= 0x1D7FF    # mathematical italic/bold letters
    )


def _looks_like_formula(line: str) -> bool:
    """Heuristic: does this line look like an equation rather than prose?"""
    s = line.strip()
    if not s or len(s) > MAX_LINE_LEN:
        return False

    chars = [c for c in s if not c.isspace()]
    ratio = sum(_is_math_char(c) for c in chars) / len(chars)
    long_words = len(re.findall(r"[A-Za-z]{4,}", s))
    has_relation = any(c in _RELATION_CHARS for c in s)
    has_number = bool(_EQ_NUMBER_RE.search(s))

    # A trailing "(3)" is a strong signal, so allow more words on that line.
    if has_number and (has_relation or ratio >= 0.08) and long_words <= 6:
        return True
    # Prose has many ordinary words; equations have few.
    if long_words >= 4:
        return False
    if has_relation and ratio >= 0.12:
        return True
    return ratio >= 0.35


def _find_blocks(lines: list[str]) -> tuple[list[tuple[int, int]], list[bool]]:
    """Group consecutive formula lines into blocks.

    Returns (blocks as (start, end) inclusive, per-line formula mask).
    """
    n = len(lines)
    is_f = [_looks_like_formula(line) for line in lines]
    mask = list(is_f)

    # PDFs often split an equation into tiny fragments ("2", "(1)", "x").
    # Absorb those when they sit right next to a formula line.
    for i, line in enumerate(lines):
        if is_f[i] or len(line) > 3 or i == n - 1:  # skip last line: probably a page number
            continue
        if (i > 0 and is_f[i - 1]) or (i < n - 1 and is_f[i + 1]):
            mask[i] = True

    blocks, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, n - 1))
    return blocks, mask


def extract_formulas(source: bytes | str | Path) -> list[FormulaContent]:
    """Return equations (with surrounding text) found in the PDF."""
    if isinstance(source, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(source), filetype="pdf")
    else:
        doc = fitz.open(str(source))

    formulas: list[FormulaContent] = []

    with doc:
        for index, page in enumerate(doc):
            raw = page.get_text("text", sort=True)
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            blocks, mask = _find_blocks(lines)

            for start, end in blocks:
                equation = " ".join(lines[start:end + 1])
                if len(equation) < 4 or len(equation) > MAX_EQUATION_LEN:
                    continue

                before = [
                    lines[i] for i in range(max(0, start - CONTEXT_LINES_BEFORE), start)
                    if not mask[i]
                ]
                after = [
                    lines[i]
                    for i in range(end + 1, min(len(lines), end + 1 + CONTEXT_LINES_AFTER))
                    if not mask[i]
                ]

                match = _EQ_NUMBER_RE.search(equation)
                formulas.append(
                    FormulaContent(
                        page_number=index + 1,
                        equation=equation,
                        equation_number=match.group(1) if match else "",
                        context_before=" ".join(before),
                        context_after=" ".join(after),
                    )
                )

    logger.info("Found %d equations", len(formulas))
    return formulas