"""
search_engine.py
----------------
Modul pencarian semantik yang menggunakan koleksi ChromaDB persisten
yang dibuat oleh ``knowledge_base.py``.

Fungsi publik:
* ``search(query, top_k, similarity_threshold)`` — cari chunk paling relevan.
* ``embed_query(query)`` — dapatkan vektor embedding dari sebuah query.

Fitur OOD (Out-of-Domain) filtering:
  Jika skor cosine similarity semua hasil di bawah ``similarity_threshold``,
  maka query dianggap tidak relevan (out-of-domain) dan tidak ada hasil
  yang dikembalikan.
"""

import pathlib
from typing import List, Dict, Any

# ---------- Konfigurasi (harus sama dengan knowledge_base.py) ----------------
DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data"
CHROMA_DB_DIR = DATA_ROOT / "chroma_db"
COLLECTION_NAME = "ecommerce_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------- Lazy loaders -----------------------------------------------------

_MODEL_CACHE = None
_COLLECTION_CACHE = None

def _load_embeddings_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL_CACHE


def _load_chroma_collection():
    global _COLLECTION_CACHE
    if _COLLECTION_CACHE is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _COLLECTION_CACHE = client.get_collection(name=COLLECTION_NAME)
    return _COLLECTION_CACHE


# ---------- Fungsi publik ----------------------------------------------------

def embed_query(query: str):
    """Kembalikan vektor embedding untuk *query* menggunakan model yang sama."""
    model = _load_embeddings_model()
    return model.encode([query])[0].tolist()


def search(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """Lakukan pencarian semantik terhadap vector store.

    Parameters
    ----------
    query : str
        Query teks dari pengguna/pelanggan.
    top_k : int, default 5
        Jumlah hasil teratas yang dikembalikan.
    similarity_threshold : float, default 0.25
        Batas minimum cosine similarity agar hasil dianggap relevan.
        Hasil di bawah threshold ini dibuang (OOD filtering).

    Returns
    -------
    List[Dict]
        Tiap entri berisi ``id``, ``text``, ``metadata``, dan ``similarity``.
    """
    collection = _load_chroma_collection()
    query_vec = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB dengan metric "cosine" mengembalikan distance = 1 - similarity.
    # Jadi similarity = 1 - distance.
    output = []
    for idx, dist in enumerate(results["distances"][0]):
        similarity = 1.0 - dist
        if similarity < similarity_threshold:
            continue
        output.append(
            {
                "id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "similarity": round(similarity, 4),
            }
        )

    output.sort(key=lambda x: x["similarity"], reverse=True)
    return output


# ---------- Demo CLI ---------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: uv run src/ecommerce_cs_ai_agent/search_engine.py "query anda"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f'\n[SEARCH] Query: "{query}"\n')

    hits = search(query)

    if not hits:
        print("[X] Tidak ada hasil relevan (semua di bawah similarity threshold).")
        print("    Query ini kemungkinan out-of-domain (OOD).")
    else:
        print(f"Ditemukan {len(hits)} hasil:\n")
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. [Score {hit['similarity']:.4f}] {hit['metadata']['doc_id']}")
            print(f"     {hit['text'][:150]}...")
            print()
