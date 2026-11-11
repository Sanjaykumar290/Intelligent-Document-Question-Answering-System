# Streamlit UI entry point
"""Streamlit UI for the Intelligent Document QA system."""

import hashlib

import streamlit as st
from google.genai import errors

from src.config import MAX_PDF_MB, get_api_key
from src.generation.gemini_client import GenerationError
from src.pipeline import DocumentQA, QAResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Document QA", page_icon="📄", layout="wide")

# Fail early with a clear message if the API key is missing.
try:
    get_api_key()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------- state

for key, default in {"qa": None, "file_hash": None, "history": [], "index_error": None}.items():
    st.session_state.setdefault(key, default)


def reset_document() -> None:
    st.session_state.qa = None
    st.session_state.file_hash = None
    st.session_state.history = []
    st.session_state.index_error = None


def short(exc: Exception) -> str:
    return str(exc)[:300]


# ---------------------------------------------------------------- indexing

def index_uploaded(data: bytes, name: str) -> None:
    """Index a new PDF, showing a progress bar. Errors are stored, not raised."""
    st.session_state.history = []
    st.session_state.qa = None
    st.session_state.index_error = None

    bar = st.progress(0.0, text="Starting...")

    def on_progress(message: str, fraction: float) -> None:
        bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    qa = DocumentQA()
    try:
        qa.index_pdf(data, name, progress=on_progress)
        st.session_state.qa = qa
    except ValueError as exc:                      # bad file, scanned PDF, too large...
        st.session_state.index_error = str(exc)
    except (GenerationError, errors.APIError) as exc:
        st.session_state.index_error = f"The Gemini API failed while indexing: {short(exc)}"
    except Exception:
        logger.exception("Unexpected error while indexing")
        st.session_state.index_error = "Something went wrong while processing this PDF."
    finally:
        bar.empty()


# --------------------------------------------------------------- rendering

def render_response(resp: QAResponse) -> None:
    answer = resp.answer
    st.markdown(answer.text)

    if not answer.not_found and not answer.is_cited:
        st.warning("This answer has no verified citations. Treat it with caution.")
    if answer.dropped_citations:
        st.caption(f"{answer.dropped_citations} invalid citation(s) were removed from this answer.")

    for c in answer.citations:
        label = f"[{c.number}] Page {c.page_number} · {c.chunk_type} · similarity {c.score:.2f}"
        with st.expander(label):
            if c.chunk_type == "table":
                st.markdown(c.text)        # renders the Markdown table as a real table
            else:
                st.text(c.text)

    if len(resp.subquestions) > 1:
        with st.expander("How this question was handled"):
            st.markdown("The question was split into sub-questions, and evidence was gathered for each:")
            for i, sq in enumerate(resp.subquestions, 1):
                st.markdown(f"{i}. {sq}")

    st.caption(f"{resp.seconds:.1f}s · {resp.n_passages} passages searched")


# -------------------------------------------------------------------- page

st.title("📄 Document Question Answering")

with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    st.caption(f"Text-based PDFs up to {MAX_PDF_MB} MB. Scanned images aren't supported.")

# Index when a new file appears (outside the sidebar block so the progress bar shows in the main area)
if uploaded is None:
    if st.session_state.file_hash is not None:
        reset_document()
else:
    data = uploaded.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()
    if file_hash != st.session_state.file_hash:
        st.session_state.file_hash = file_hash
        index_uploaded(data, uploaded.name)

qa: DocumentQA | None = st.session_state.qa

with st.sidebar:
    if qa is not None and qa.info is not None:
        info = qa.info
        st.success(f"Ready: {info.filename}")
        col1, col2 = st.columns(2)
        col1.metric("Pages", info.n_pages)
        col2.metric("Passages", info.n_chunks)
        col1.metric("Tables", info.n_tables)
        col2.metric("Equations", info.n_equations)
        if info.from_cache:
            st.caption("Loaded from a saved index (no embedding calls used).")
    if st.session_state.history and st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

if st.session_state.index_error:
    st.error(st.session_state.index_error)

if qa is None and not st.session_state.index_error:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.markdown(
        "**Try questions like:**\n"
        "- *What problem does this document address?*\n"
        "- *What does Table 2 show, and how does it compare to the baseline?*\n"
        "- *Explain Equation 3 and list the variables it uses.*"
    )

# Previous messages
for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        if item["error"]:
            st.error(item["error"])
        else:
            render_response(item["response"])

# New question
question = st.chat_input("Ask a question about your document", disabled=qa is None)

if question and qa is not None:
    with st.chat_message("user"):
        st.markdown(question)

    entry = {"question": question, "response": None, "error": None}
    with st.chat_message("assistant"):
        with st.spinner("Searching the document and writing an answer..."):
            try:
                entry["response"] = qa.ask(question)
            except ValueError as exc:
                entry["error"] = str(exc)
            except (GenerationError, errors.APIError) as exc:
                entry["error"] = (
                    "The AI service is busy or unavailable right now. "
                    f"Please try again in a minute. ({short(exc)})"
                )
            except Exception:
                logger.exception("Unexpected error while answering")
                entry["error"] = "Something went wrong while answering. Please try again."

        if entry["error"]:
            st.error(entry["error"])
        else:
            render_response(entry["response"])

    st.session_state.history.append(entry)