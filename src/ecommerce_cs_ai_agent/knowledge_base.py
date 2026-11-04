"""
knowledge_base.py
-----------------
Modul untuk membangun **persistent** vector store (ChromaDB) dari dokumen
SOP/FAQ bisnis ritel perlengkapan sekolah.

Alur kerja:
1. Membaca semua file .txt / .md di folder ``data/corpus/``.
2. Memotong (chunk) tiap dokumen menggunakan pemotong kata utuh (word-aware).
3. Meng-embed tiap chunk dengan model Sentence-Transformers
   ``paraphrase-multilingual-MiniLM-L12-v2``.
4. Menyimpan chunk + embedding + metadata ke ChromaDB **persisten** di
   folder ``data/chroma_db/``.
"""

import pathlib
from typing import List, Tuple

# Impor pemotong teks berbasis kata utuh
try:
    from ecommerce_cs_ai_agent.chunking import chunk_text_word_aware
except ImportError:
    from chunking import chunk_text_word_aware

# ---------- Konfigurasi ------------------------------------------------------
DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data"
CORPUS_DIR = DATA_ROOT / "corpus"
CHROMA_DB_DIR = DATA_ROOT / "chroma_db"

UKURAN_CHUNK = 500   # Ganti dengan hasil optimal dari langkah 4
OVERLAP_CHUNK = 50   # Sekitar 10% dari ukuran chunk

COLLECTION_NAME = "ecommerce_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Kamus Kategori berdasarkan nama dokumen
KATEGORI = {
    "faq_umum": "Umum",
    "kebijakan_garansi": "Produk Bermasalah",
    "kebijakan_grosir": "Transaksi",
    "kebijakan_keluhan": "Layanan",
    "kebijakan_privasi": "Akun dan Data",
    "kebijakan_promo": "Pembayaran dan Promo",
    "panduan_lacak_pesanan": "Pengiriman",
    "panduan_pembayaran": "Pembayaran dan Promo",
    "sla_layanan": "Layanan",
    "sop_barang_kurang": "Produk Bermasalah",
    "sop_pengiriman": "Pengiriman",
    "sop_refund": "Retur dan Refund",
    "sop_retur_seni": "Retur dan Refund",
    "sop_tukar_ukuran": "Retur dan Refund",
}


# ---------- Helper functions -------------------------------------------------

def _read_documents() -> List[Tuple[str, str]]:
    """Baca semua file .txt dan .md di ``data/corpus/``."""
    documents: List[Tuple[str, str]] = []
    for ext in ("*.txt", "*.md"):
        for path in sorted(CORPUS_DIR.rglob(ext)):
            with open(path, "r", encoding="utf-8") as f:
                documents.append((path.stem, f.read()))
    return documents


# ---------- Fungsi utama -----------------------------------------------------

def build_vector_store() -> None:
    """Buat (atau refresh) koleksi ChromaDB persisten dari korpus dokumen."""
    from sentence_transformers import SentenceTransformer
    import chromadb

    # Pastikan direktori output ada
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Inisialisasi model embedding & client ChromaDB persisten
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Hapus koleksi lama jika ada, lalu buat ulang
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Proses setiap dokumen
    documents = _read_documents()
    if not documents:
        print("[PERINGATAN] Tidak ada dokumen ditemukan di", CORPUS_DIR)
        return

    total_chunks = 0
    for berkas_stem, raw in documents:
        # Menggunakan pemotong kata utuh
        chunks = chunk_text_word_aware(raw, UKURAN_CHUNK, OVERLAP_CHUNK)
        if not chunks:
            continue

        embeddings = model.encode(chunks).tolist()

        ids = [f"{berkas_stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": berkas_stem,
                "kategori": KATEGORI.get(berkas_stem, "Umum"),
                "chunk_index": i,
                "source_file": f"{berkas_stem}.txt",
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