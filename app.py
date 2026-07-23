"""Minimal Streamlit shell for the SHSU transfer-student RAG assistant.

Wires the language selector (TR/EN) to both the UI labels and the system prompt used by
rag/generator.py, and drives a chat loop through rag/retriever.py + rag/generator.py.

This file is intentionally thin: all real logic (loading, chunking, embedding, storing,
retrieving, generating) lives in the plain-Python `rag` package so it stays importable and
testable independent of Streamlit (locked architectural rule, see PROJECT_PLAN.md).

Run with: streamlit run app.py
(Requires knowledge.db to already exist -- run `python scripts/ingest.py` first.)
"""
import streamlit as st

from rag import config, store
from rag.embedder import get_embedder
from rag.generator import answer_query
from rag.retriever import get_top_chunks

LABELS = {
    "tr": {
        "page_title": "SHSU AI Asistani",
        "title": "SHSU AI Asistani",
        "caption": "Firat -> SHSU transfer sureci hakkinda soru sorun.",
        "input_placeholder": "Sorunuzu yazin...",
        "empty_db_warning": (
            "Bilgi tabani bos gorunuyor. Once `python scripts/ingest.py` calistirarak "
            "docs/ klasorunu isleyin."
        ),
    },
    "en": {
        "page_title": "SHSU AI Assistant",
        "title": "SHSU AI Assistant",
        "caption": "Ask a question about the Firat -> SHSU transfer process.",
        "input_placeholder": "Ask a question...",
        "empty_db_warning": (
            "The knowledge base looks empty. Run `python scripts/ingest.py` first to "
            "process the docs/ folder."
        ),
    },
}


@st.cache_resource
def _get_embedder():
    return get_embedder()


@st.cache_resource
def _get_db_connection():
    return store.init_db(config.DB_PATH)


def main() -> None:
    st.set_page_config(page_title="SHSU AI Assistant")

    lang_choice = st.sidebar.radio("Language / Dil", ["Turkce", "English"])
    language = "tr" if lang_choice == "Turkce" else "en"
    labels = LABELS[language]

    st.title(labels["title"])
    st.caption(labels["caption"])

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

    question = st.chat_input(labels["input_placeholder"])
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    embedder = _get_embedder()
    conn = _get_db_connection()
    chunks = get_top_chunks(question, embedder, conn, k=config.TOP_K)

    with st.chat_message("assistant"):
        if not chunks:
            answer_text = labels["empty_db_warning"]
            st.warning(answer_text)
        else:
            result = answer_query(question, chunks, language=language)
            answer_text = result["answer"]
            st.markdown(answer_text)

    st.session_state.history.append({"question": question, "answer": answer_text})


if __name__ == "__main__":
    main()
