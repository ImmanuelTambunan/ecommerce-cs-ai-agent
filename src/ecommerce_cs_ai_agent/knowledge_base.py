"""
knowledge_base.py
-----------------
Modul untuk membangun **persistent** vector store (ChromaDB) dari dokumen
SOP/FAQ bisnis ritel perlengkapan sekolah.

Alur kerja:
1. Membaca semua file .txt / .md di folder ``data/corpus/``.
2. Memotong (chunk) tiap dokumen menjadi potongan 500 karakter dengan
   overlap 50 karakter agar konteks antar potongan tidak terputus.
3. Meng-embed tiap chunk dengan model Sentence-Transformers
   ``paraphrase-multilingual-MiniLM-L12-v2`` (multilingual, mendukung
   Bahasa Indonesia).
4. Menyimpan chunk + embedding + metadata ke ChromaDB **persisten** di
   folder ``data/chroma_db/``.

Fungsi utama: ``build_vector_store()`` — idempotent (aman dipanggil ulang).
"""

import pathlib
from typing import List, Tuple

# ---------- Konfigurasi ------------------------------------------------------
DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data"
CORPUS_DIR = DATA_ROOT / "corpus"
CHROMA_DB_DIR = DATA_ROOT / "chroma_db"
CHUNK_SIZE = 500   # karakter per chunk
OVERLAP = 50       # karakter overlap antar chunk
COLLECTION_NAME = "ecommerce_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------- Helper functions -------------------------------------------------

def _read_documents() -> List[Tuple[str, str]]:
    """Baca semua file .txt dan .md di ``data/corpus/``.

    Returns
    -------
    List[Tuple[str, str]]
        Daftar ``(doc_id, raw_text)``.  ``doc_id`` = nama file tanpa ekstensi.
    """
    documents: List[Tuple[str, str]] = []
    for ext in ("*.txt", "*.md"):
        for path in sorted(CORPUS_DIR.rglob(ext)):
            with open(path, "r", encoding="utf-8") as f:
                documents.append((path.stem, f.read()))
    return documents


def _chunk_text(text: str) -> List[str]:
    """Potong *text* menjadi chunk berukuran ``CHUNK_SIZE`` dengan overlap."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - OVERLAP
    return chunks


# ---------- Fungsi utama -----------------------------------------------------

def build_vector_store() -> None:
    """Buat (atau refresh) koleksi ChromaDB persisten dari korpus dokumen.

    Fungsi ini idempotent — memanggil ulang akan menghapus koleksi lama
    dan membuatnya kembali dari korpus terkini.
    """
    from sentence_transformers import SentenceTransformer
    import chromadb

    # Pastikan direktori output ada
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Inisialisasi model embedding
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Inisialisasi ChromaDB client PERSISTEN (API baru)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Hapus koleksi lama jika ada, lalu buat ulang
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # eksplisit cosine metric
    )

    # Proses setiap dokumen
    documents = _read_documents()
    if not documents:
        print("[PERINGATAN] Tidak ada dokumen ditemukan di", CORPUS_DIR)
        print("   Silakan tambahkan file .txt/.md ke folder tersebut.")
        return

    total_chunks = 0
    for doc_id, raw in documents:
        chunks = _chunk_text(raw)
        embeddings = model.encode(chunks).tolist()

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "source_file": f"{doc_id}.txt",
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"[OK] Vector store berhasil dibangun: {total_chunks} chunk dari {len(documents)} dokumen.")
    print(f"     Disimpan di: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    build_vector_store()
