# Measures accuracy, recall, citation rate
"""Simple evaluation of the Document QA pipeline on a hand-written question set.

Usage (from the project root, venv active):
    python evaluation/evaluate.py                                  # data/uploads/syllabus.pdf
    python evaluation/evaluate.py data/uploads/other.pdf           # a different PDF
    python evaluation/evaluate.py --check                          # verify keywords, no API calls
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from google.genai import errors  # noqa: E402

from src.chunking.chunker import build_chunks  # noqa: E402
from src.generation.gemini_client import GenerationError  # noqa: E402
from src.ingestion.pdf_parser import parse_pdf  # noqa: E402
from src.ingestion.table_extractor import extract_tables  # noqa: E402
from src.pipeline import DocumentQA  # noqa: E402

QUESTIONS_FILE = ROOT / "evaluation" / "test_questions.json"
RESULTS_FILE = ROOT / "evaluation" / "results.json"
DEFAULT_PDF = "data/uploads/syllabus.pdf"
CATEGORIES = {"text", "table", "formula", "multi_part", "unanswerable"}


# ------------------------------------------------------------------ helpers

def norm(text: str) -> str:
    """Lowercase and tidy text so keyword matching ignores case, spacing and markdown marks."""
    text = text.lower().replace("−", "-").replace("–", "-")
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def has_keyword(blob: str, keyword: str) -> bool:
    """True if `keyword` occurs in `blob` (both already normalized by norm()).

    Pure numbers must match as whole numbers, so "24" does not match "124" or "2.45".
    """
    kw = norm(keyword)
    if re.fullmatch(r"[\d.,]+", kw):
        return re.search(rf"(?<![\d.,]){re.escape(kw)}(?![\d])", blob) is not None
    return kw in blob


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def load_questions(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for i, q in enumerate(items, 1):
        if str(q.get("question", "")).startswith("REPLACE"):
            continue
        if q.get("category") not in CATEGORIES:
            sys.exit(f"Question {i}: category must be one of {sorted(CATEGORIES)}")
        if q["category"] != "unanswerable":
            if not q.get("keywords"):
                sys.exit(f"Question {i}: answerable questions need at least one keyword")
            for k in q["keywords"]:
                if len(norm(k)) < 3 and not re.fullmatch(r"\d+", norm(k)):
                    print(f"  Warning (question {i}): keyword {k!r} is very short and may match by accident")
        questions.append(q)
    if not questions:
        sys.exit("No questions yet: replace the REPLACE placeholders in evaluation/test_questions.json")
    return questions


# ---------------------------------------------------------------- check mode

def check_keywords(pdf_path: Path, questions: list[dict]) -> None:
    """Offline sanity check: does every keyword occur somewhere in the extracted PDF text?"""
    data = pdf_path.read_bytes()
    chunks = build_chunks(parse_pdf(data), extract_tables(data))
    blob = norm(" ".join(c.text for c in chunks))

    problems = 0
    for q in questions:
        for k in q.get("keywords", []):
            if not has_keyword(blob, k):
                problems += 1
                print(f"  Not found in the PDF text: {k!r}   (question: {q['question'][:60]})")

    if problems:
        print(f"\n{problems} keyword(s) not found. Check spelling, or whether the PDF writes it differently.")
    else:
        print("All keywords occur in the extracted text.")


# ------------------------------------------------------------------ summary

def print_summary(records: list[dict], n_questions: int) -> dict:
    by_cat = defaultdict(lambda: [0, 0])
    for r in records:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["correct"])

    total = len(records)
    right = sum(r["correct"] for r in records)
    answered = [r for r in records if r["answered"]]
    cited = sum(r["verified_citations"] for r in answered)

    print("\n| Category | Correct | Accuracy |")
    print("|---|---|---|")
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"| {cat} | {c}/{n} | {pct(c, n)} |")
    print(f"| **overall** | {right}/{total} | {pct(right, total)} |")
    print(f"\nAnswers with fully verified citations: {cited}/{len(answered)} ({pct(cited, len(answered))})")

    if total < n_questions:
        print(f"\nNote: only {total} of {n_questions} questions were scored, so these numbers are partial.")

    misses = [r for r in records if not r["correct"]]
    if misses:
        print(f"\n--- {len(misses)} miss(es): read these to see why ---")
        for r in misses:
            print(f"\n[{r['category']}] {r['question']}")
            if r["missing_keywords"]:
                print(f"  missing keywords: {r['missing_keywords']}")
            print(f"  answer: {r['answer'][:300]}")

    return {
        "scored": total,
        "correct": right,
        "accuracy_pct": round(100 * right / total) if total else None,
        "by_category": {k: {"correct": v[0], "total": v[1]} for k, v in by_cat.items()},
        "verified_citations": {"fully_verified": cited, "answered": len(answered)},
    }


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Document QA pipeline.")
    ap.add_argument("pdf", nargs="?", default=DEFAULT_PDF, help=f"PDF to evaluate on (default: {DEFAULT_PDF})")
    ap.add_argument("--check", action="store_true", help="only verify keywords exist in the PDF text (no API calls)")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds to wait between questions")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}\nCopy your syllabus into data/uploads/ or pass its path.")

    questions = load_questions(QUESTIONS_FILE)

    if args.check:
        check_keywords(pdf_path, questions)
        return

    qa = DocumentQA()
    info = qa.index_pdf(pdf_path.read_bytes(), pdf_path.name)
    print(
        f"Indexed {pdf_path.name}: {info.n_chunks} passages, {info.n_tables} tables | "
        f"{len(questions)} questions\n"
    )

    records = []
    for i, q in enumerate(questions, 1):
        try:
            answer = qa.ask(q["question"]).answer
        except (errors.APIError, GenerationError) as exc:
            if getattr(exc, "code", None) == 429:
                print("\nQuota reached. Rerun later: answers already generated are reused from the cache.")
                break
            print(f"[{i}/{len(questions)}] ERROR  {str(exc)[:80]}")
            continue

        if q["category"] == "unanswerable":
            correct, missing = answer.not_found, []
        else:
            blob = norm(answer.text)
            missing = [k for k in q["keywords"] if not has_keyword(blob, k)]
            correct = (not answer.not_found) and not missing

        records.append({
            "category": q["category"],
            "question": q["question"],
            "answer": answer.text,
            "correct": bool(correct),
            "missing_keywords": missing,
            "answered": not answer.not_found,
            "verified_citations": bool(answer.is_cited and answer.dropped_citations == 0),
        })
        print(f"[{i}/{len(questions)}] {'OK  ' if correct else 'MISS'} {q['category']:<12} {q['question'][:65]}")
        time.sleep(args.pause)

    if not records:
        sys.exit("No questions were scored.")

    summary = print_summary(records, len(questions))

    RESULTS_FILE.write_text(
        json.dumps(
            {"pdf": pdf_path.name, "date": time.strftime("%Y-%m-%d"), "summary": summary, "records": records},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {RESULTS_FILE}")


if __name__ == "__main__":
    main()