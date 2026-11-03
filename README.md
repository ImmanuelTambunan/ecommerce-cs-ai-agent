# ecommerce-cs-ai-agent

## Pengembangan Enterprise AI Copilot untuk Otomasi Customer Service Ritel Daring Perlengkapan Sekolah Berbasis Agentic AI dan Retrieval-Augmented Generation

Proyek mata kuliah Kecerdasan Buatan (10S3001), Program Studi Sarjana Sistem Informasi,
Institut Teknologi Del, Semester Gasal 2026/2027.

## Deskripsi Singkat
Sistem asisten cerdas (AI Copilot) yang dirancang untuk mengotomasi proses klasifikasi,
perutean, dan penanganan pesan keluhan pelanggan pada platform ritel daring perlengkapan
sekolah, dengan memanfaatkan pendekatan agentic AI dan retrieval-augmented generation (RAG).

## Anggota Tim (Kelompok 17)
1. Adithya Philip Jona Putra Silaban - 12S24029
2. Mutiara Y.H. Sianturi - 12S24045
3. Immanuel Alexander Tambunan - 12S24034

## Status Proyek
- [x] **Milestone 1** — Problem Framing, Spesifikasi PEAS, & Baseline Search (UCS)
- [x] **Milestone 2** — CSP Solver untuk Perutean Pesan Pelanggan Otomatis
- [x] **Milestone 3** — Pembangunan Basis Pengetahuan Korpus & Vector Semantic Search

---

## Fitur & Arsitektur Implementasi

Proyek ini membangun AI Copilot untuk otomatisasi customer service pada platform ritel
daring perlengkapan sekolah dengan fondasi AI berjenjang:

1. **Uniform Cost Search (UCS)** (`search.py`):
   Mencari jalur penanganan tiket dengan biaya waktu minimum dari status awal hingga selesai.
2. **Constraint Satisfaction Problem (CSP)** (`solver.py`):
   Merutekan pesan keluhan pelanggan secara otomatis ke channel L1 (balasan otomatis RAG),
   L2 (eksekusi transaksi), atau L3 (keputusan kebijakan otomatis), berdasarkan kategori,
   prioritas, dan skor keyakinan intent menggunakan AC-3, backtracking, dan heuristik MRV.
3. **Basis Pengetahuan Korpus & Vector Semantic Search** (`knowledge_base.py`, `search_engine.py`, `benchmark.py`):
   - **Enterprise Corpus**: 14 dokumen SOP dan regulasi bisnis spesifik ritel daring perlengkapan sekolah di `data/corpus/`.
   - **Chunking Terstruktur**: Word-aware recursive chunking (500 karakter) dengan overlap (50 karakter).
   - **Dense Vector Embedding**: Model multilingual `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensi).
   - **Persistent Vector Database**: Disimpan secara persisten pada disk menggunakan **ChromaDB** di `data/chroma_db/`.
   - **Semantic Retrieval & OOD Filtering**: Pencarian semantik berbasis Cosine Similarity dilengkapi threshold filtering (0.45) untuk menolak kueri di luar domain bisnis (*Out-of-Domain*).

---

## Struktur Proyek

```text
ecommerce-cs-ai-agent/
├── data/
│   ├── corpus/                     # 14 dokumen SOP & regulasi bisnis (format .txt)
│   └── chroma_db/                  # Database vektor persisten ChromaDB
├── docs/
│   ├── Grup17_Tugas01.pdf
│   ├── Grup17_Tugas02.pdf
│   └── Grup17_Tugas03.pdf          # Laporan teknis Milestone 3
├── src/
│   └── ecommerce_cs_ai_agent/
│       ├── __init__.py
│       ├── search.py               # Milestone 1: UCS
│       ├── solver.py               # Milestone 2: CSP
│       ├── knowledge_base.py       # Milestone 3: Ingestion & indexing ChromaDB
│       ├── search_engine.py        # Milestone 3: Semantic search & OOD filter
│       └── benchmark.py            # Milestone 3: Skrip evaluasi benchmark 5 kueri
├── tests/
│   ├── test_solver.py              # Unit test CSP (5 pengujian)
│   └── test_knowledge_base.py      # Unit test Knowledge Base & Retrieval (7 pengujian)
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Cara Instalasi

Pastikan menggunakan Python >= 3.11 dan paket terkelola dengan `uv`:

```bash
uv sync
```

---

## Panduan Menjalankan Sistem

### 1. Membangun Basis Pengetahuan (Vector Indexing)
Membaca seluruh dokumen pada `data/corpus/`, melakukan chunking, embedding, dan menyimpannya secara persisten ke ChromaDB:

```bash
uv run src/ecommerce_cs_ai_agent/knowledge_base.py
```

*Output yang diharapkan:*
```text
[OK] Vector store berhasil dibangun: 77 chunk dari 14 dokumen.
     Disimpan di: ...\data\chroma_db
```

### 2. Melakukan Pencarian Semantik (Semantic Search)
Mencari solusi SOP berdasarkan pertanyaan/keluhan pelanggan:

```bash
uv run src/ecommerce_cs_ai_agent/search_engine.py "baju seragam kemarin kekecilan, bisa ditukar ga?"
```

*Output yang diharapkan:*
```text
[SEARCH] Query: "baju seragam kemarin kekecilan, bisa ditukar ga?"

Ditemukan 3 hasil:
  1. [Score 0.5517] sop_tukar_ukuran
     SYARAT PENUKARAN UKURAN - Pengajuan penukaran ukuran harus dilakukan maksimal 14 hari setelah barang diterima...
```

*Contoh kueri di luar domain (Out-of-Domain) yang ditolak otomatis:*
```bash
uv run src/ecommerce_cs_ai_agent/search_engine.py "bagaimana resep membuat kue bolu pandan?"
```

*Output yang diharapkan:*
```text
[X] Tidak ada hasil relevan (semua di bawah similarity threshold).
    Query ini kemungkinan out-of-domain (OOD).
```

### 3. Menjalankan Tolok Ukur (Benchmark) Evaluasi Temu Balik
Menjalankan pengujian 5 kueri bisnis (kueri relevan, kueri sinonim, kueri ambigu, dan 2 kueri out-of-domain):

```bash
uv run src/ecommerce_cs_ai_agent/benchmark.py
```

### 4. Menjalankan Baseline UCS (Milestone 1)
```bash
uv run src/ecommerce_cs_ai_agent/search.py
```

### 5. Menjalankan CSP Solver (Milestone 2)
```bash
uv run src/ecommerce_cs_ai_agent/solver.py
```

---

## Menjalankan Seluruh Pengujian Otomatis

Menjalankan rangkaian unit test otomatis menggunakan `pytest`:

```bash
uv run pytest
```

*Hasil yang diharapkan: **12 passed** (5 solver tests + 7 knowledge base tests).*
