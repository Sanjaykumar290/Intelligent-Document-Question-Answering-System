from collections import Counter

from src.chunking.chunker import build_chunks
from src.ingestion.formula_handler import extract_formulas
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.table_extractor import extract_tables

f = "data/uploads/sample.pdf"
chunks = build_chunks(parse_pdf(f), extract_tables(f), extract_formulas(f))

print(len(chunks), "chunks", dict(Counter(c.chunk_type for c in chunks)))
print("Longest text chunk:", max(len(c.text) for c in chunks if c.chunk_type == "text"), "chars")

for kind in ("text", "table", "equation"):
    sample = next((c for c in chunks if c.chunk_type == kind), None)
    if sample:
        print("\n-----", sample.chunk_id, "|", sample.citation)
        print(sample.text[:600])