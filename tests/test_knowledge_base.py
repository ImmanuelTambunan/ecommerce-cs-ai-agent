"""
test_knowledge_base.py
----------------------
Unit test untuk memvalidasi:
1. Keberadaan korpus dokumen dan database vektor persisten ChromaDB.
2. Fungsionalitas embedding dan temu balik semantik (Semantic Search).
3. Evaluasi Kueri Relevan (In-Domain), Sinonim, dan Ambigu.
4. Presisi Penolakan Kueri Out-of-Domain (OOD Filtering).
"""

import pytest
from ecommerce_cs_ai_agent.knowledge_base import CHROMA_DB_DIR, CORPUS_DIR, _read_documents
from ecommerce_cs_ai_agent.search_engine import search, embed_query, _load_chroma_collection


def test_corpus_dokumen_memadai():
    """Memastikan minimal 10-15 dokumen korpus tersedia di data/corpus."""
    docs = _read_documents()
    assert len(docs) >= 10, f"Jumlah dokumen korpus ({len(docs)}) kurang dari syarat minimal 10 dokumen."


def test_chromadb_persisten_terisi():
    """Memastikan database ChromaDB persisten sudah berisi embeddings chunk."""
    collection = _load_chroma_collection()
    count = collection.count()
    assert count > 0, "ChromaDB kosong, belum ada chunk terindeks."
    assert count >= 30, f"Jumlah chunk terindeks ({count}) terlalu sedikit untuk 10+ dokumen."


def test_embed_query_berhasil():
    """Memastikan fungsi embed_query menghasilkan vektor dense dengan dimensi 384 (MiniLM-L12)."""
    vec = embed_query("tes embedding kueri")
    assert isinstance(vec, list)
    assert len(vec) == 384


def test_kueri_relevan_langsung():
    """Kueri relevan langsung harus diterima dan mengembalikan dokumen target dengan skor tinggi."""
    query = "Bagaimana prosedur dan syarat pengajuan refund dana?"
    results = search(query, top_k=3, similarity_threshold=0.45)
    assert len(results) > 0, "Kueri relevan seharusnya diterima oleh sistem."
    top_doc_ids = [r["metadata"]["doc_id"] for r in results]
    assert any(doc in top_doc_ids for doc in ["sop_refund", "kebijakan_keluhan", "faq_umum"])
    assert results[0]["similarity"] >= 0.50


def test_kueri_sinonim_percakapan():
    """Kueri sinonim/bahasa sehari-hari harus berhasil menarik dokumen target (SOP tukar ukuran)."""
    query = "Baju seragam yang kemarin sampai kekecilan, apakah bisa ditukar ukurannya?"
    results = search(query, top_k=3, similarity_threshold=0.45)
    assert len(results) > 0, "Kueri sinonim percakapan seharusnya berhasil ditarik."
    top_doc_ids = [r["metadata"]["doc_id"] for r in results]
    assert "sop_tukar_ukuran" in top_doc_ids
    assert results[0]["similarity"] >= 0.45


def test_kueri_out_of_domain_ditolak_kuliner():
    """Kueri tentang resep makanan di luar domain bisnis harus ditolak (OOD rejection)."""
    query = "Bagaimana resep dan takaran membuat kue bolu pandan yang empuk?"
    results = search(query, top_k=3, similarity_threshold=0.45)
    assert len(results) == 0, f"Kueri OOD kuliner harus ditolak, namun mengembalikan: {results}"


def test_kueri_out_of_domain_ditolak_tanaman():
    """Kueri tentang perawatan tanaman hias di luar domain bisnis harus ditolak (OOD rejection)."""
    query = "Bagaimana cara merawat tanaman hias monstera agar daunnya lebat?"
    results = search(query, top_k=3, similarity_threshold=0.45)
    assert len(results) == 0, f"Kueri OOD tanaman harus ditolak, namun mengembalikan: {results}"
