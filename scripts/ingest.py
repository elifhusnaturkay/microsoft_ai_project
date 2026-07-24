#!/usr/bin/env python3
"""CLI: loader -> chunker -> embedder -> store, run over everything in docs/.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --docs-dir docs --db-path knowledge.db --backend foundry --reset
"""
import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/ingest.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import chunker, config, loader, store  # noqa: E402
from rag.embedder import get_embedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs/ into the SQLite knowledge base.")
    parser.add_argument("--docs-dir", default=str(config.DOCS_DIR), help="Folder of .pdf/.md/.txt source docs.")
    parser.add_argument("--db-path", default=config.DB_PATH, help="SQLite DB file to write chunks into.")
    parser.add_argument(
        "--backend",
        default=config.EMBED_BACKEND,
        choices=["foundry", "ollama", "gemini"],
        help="Embedding backend to use (default from RAG_EMBED_BACKEND / config.py).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing chunks before ingesting (otherwise chunks are appended).",
    )
    args = parser.parse_args()

    print(f"[1/4] Loading documents from {args.docs_dir} ...")
    documents = loader.load_documents(args.docs_dir)
    print(f"      loaded {len(documents)} document(s)")
    if not documents:
        print("No documents found under docs-dir -- nothing to ingest.")
        return

    print("[2/4] Chunking (paragraph-based, ~200-500 words) ...")
    chunks = chunker.chunk_documents(documents)
    print(f"      produced {len(chunks)} chunk(s)")

    _embed_model_by_backend = {
        "foundry": config.LOCAL_EMBED_MODEL,
        "ollama": config.OLLAMA_EMBED_MODEL,
        "gemini": config.GEMINI_EMBED_MODEL,
    }
    print(f"[3/4] Embedding with backend={args.backend!r} (model={_embed_model_by_backend[args.backend]}) ...")
    embedder = get_embedder(args.backend)
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    print(f"[4/4] Writing {len(chunks)} chunk(s) to {args.db_path} ...")
    conn = store.init_db(args.db_path)
    try:
        if args.reset:
            store.clear_chunks(conn)
        store.insert_chunks(conn, chunks)
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    main()
