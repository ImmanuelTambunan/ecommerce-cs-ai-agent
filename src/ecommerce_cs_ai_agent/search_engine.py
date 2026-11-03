"""
search_engine.py
----------------
Simple semantic search wrapper around the persistent ChromaDB collection
created by :pymod:`knowledge_base`.  It exposes two public helpers:

* ``embed_query(query: str)`` – returns the embedding vector for a raw query.
* ``search(query: str, top_k: int = 5, similarity_threshold: float = 0.55)`` –
  returns a list of the most similar chunks together with their metadata.

The module is deliberately lightweight – it lazily loads the embedding model
and the Chroma client only when a function is called.  This keeps start‑up
time low for the unit tests.
"""

import pathlib
from typing import List, Dict, Any

# Configuration – must match the paths used in ``knowledge_base.py``
DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data"
CHROMA_DB_DIR = DATA_ROOT / "chroma_db"
COLLECTION_NAME = "ecommerce_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Lazy imports – they are only required when we actually run a search.
def _load_embeddings_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _load_chroma_collection():
    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(CHROMA_DB_DIR),
            anonymized_telemetry=False,
        )
    )
    return client.get_collection(name=COLLECTION_NAME)


def embed_query(query: str):
    """Return the vector embedding for *query* using the same model as the index."""
    model = _load_embeddings_model()
    return model.encode([query])[0].tolist()


def search(query: str, top_k: int = 5, similarity_threshold: float = 0.55) -> List[Dict[str, Any]]:
    """Perform a similarity search against the persistent vector store.

    Parameters
    ----------
    query: str
        Raw user query.
    top_k: int, default 5
        Number of top results to return.
    similarity_threshold: float, default 0.55
        Minimum cosine similarity for a result to be considered *in‑domain*.
        Results below the threshold are omitted – this satisfies the OOD
        filtering requirement of the milestone.

    Returns
    -------
    List[Dict]
        Each entry contains ``id``, ``text`` (the chunk), ``metadata`` and the
        ``similarity`` score.
    """
    collection = _load_chroma_collection()
    query_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "ids"],
    )
    # ``distances`` are Euclidean distances – convert to cosine similarity approx.
    # Chromadb returns ``distances`` as 1 - cosine_similarity when the collection
    # uses the default ``cosine`` metric, so we can treat them directly.
    output = []
    for idx, sim in enumerate(results["distances"][0]):
        # In the default cosine metric, smaller distance = higher similarity;
        # we invert it to a similarity score between 0 and 1.
        similarity = 1 - sim
        if similarity < similarity_threshold:
            continue
        output.append(
            {
                "id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "similarity": similarity,
            }
        )
    # Sort by similarity descending for deterministic output.
    output.sort(key=lambda x: x["similarity"], reverse=True)
    return output

if __name__ == "__main__":
    # Simple demo when run directly.
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ecommerce_cs_ai_agent.search_engine \"your query\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    hits = search(query)
    if not hits:
        print("❌ No relevant results (below similarity threshold).")
    else:
        for hit in hits:
            print(f"[Score {hit['similarity']:.3f}] {hit['metadata']['doc_id']} – {hit['text'][:120]}…")
"