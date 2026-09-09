from pathlib import Path

import chromadb

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

_collection = None


def _get_collection():
    """Lazily build an in-memory Chroma collection from knowledge/*.txt.

    Uses Chroma's bundled default embedding model (a local ONNX MiniLM),
    so this needs no API key and makes no LLM call — embedding is a
    separate concern from generation, and here it's a local one.
    """
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.Client()
    collection = client.create_collection("lease_market_knowledge")

    paths = sorted(KNOWLEDGE_DIR.glob("*.txt"))
    collection.add(
        documents=[p.read_text().strip() for p in paths],
        ids=[p.stem for p in paths],
        metadatas=[{"source": p.name} for p in paths],
    )

    _collection = collection
    return collection


def search_market_context(query: str, k: int = 3) -> list[str]:
    """Return the top-k most relevant knowledge-base snippets for a query."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]
