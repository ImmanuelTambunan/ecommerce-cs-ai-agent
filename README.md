# ecommerce-cs-ai-agent

## Pengembangan Enterprise AI Copilot untuk Otomasi Customer Service Ritel Daring Perlengkapan Sekolah Berbasis Agentic AI dan Retrieval-Augmented Generation

Proyek mata kuliah Kecerdasan Buatan (10S3001), Program Studi Sarjana Sistem Informasi,
Institut Teknologi Del, Semester Gasal 2026/2027.

---

## Deskripsi Singkat
Sistem asisten cerdas (AI Copilot) yang dirancang untuk mengotomasi proses klasifikasi,
perutean, dan penanganan pesan keluhan pelanggan pada platform ritel daring perlengkapan
sekolah, dengan memanfaatkan pendekatan *agentic AI* dan *retrieval-augmented generation* (RAG).

## Anggota Tim (Kelompok 17)
1. **Adithya Philip Jona Putra Silaban** — 12S24029
2. **Mutiara Y.H. Sianturi** — 12S24045
3. **Immanuel Alexander Tambunan** — 12S24034

## Status Proyek
- [x] **Milestone 1** — Problem Framing, Spesifikasi PEAS, & Baseline Search (UCS)
- [x] **Milestone 2** — CSP Solver untuk Perutean Pesan Pelanggan Otomatis
- [x] **Milestone 3** — Pembangunan Basis Pengetahuan Korpus & Vector Semantic Search

---

## Fitur & Arsitektur Implementasi

Proyek ini membangun AI Copilot untuk otomatisasi customer service pada platform ritel
daring perlengkapan sekolah fiktif **BeliCepat** dengan fondasi kecerdasan buatan berjenjang:

1. **Uniform Cost Search (UCS)** (`search.py`):
   Mencari jalur penanganan tiket dengan estimasi biaya waktu minimum dari status awal hingga tiket selesai ditangani.
2. **Constraint Satisfaction Problem (CSP)** (`solver.py`):
   Merutekan pesan keluhan pelanggan secara otomatis ke channel L1 (balasan otomatis RAG),
   L2 (eksekusi transaksi), atau L3 (keputusan kebijakan otomatis), berdasarkan kategori keluhan,
   tingkat prioritas, dan skor keyakinan intent menggunakan algoritma AC-3, backtracking search, dan heuristik MRV (*Minimum Remaining Values*).
3. **Basis Pengetahuan Korpus & Vector Semantic Search** (`knowledge_base.py`, `search_engine.py`, `benchmark.py`):
   - **Enterprise Knowledge Corpus**: 14 dokumen teks SOP dan manual regulasi bisnis perlengkapan sekolah di `data/corpus/`.
   - **Structured Word-Aware Chunking**: Memotong dokumen dengan ukuran 500 karakter dan overlap 50 karakter agar makna antarkalimat tidak terputus.
   - **Dense Vector Embedding**: Model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` yang menghasilkan representasi vektor padat 384 dimensi, mendukung bahasa Indonesia dan istilah bilingual.
   - **Persistent Vector Store**: Menggunakan **ChromaDB** dengan penyimpanan persisten pada disk (`data/chroma_db/`) dan metrik jarak *Cosine Similarity*.
   - **Semantic Retrieval & OOD Rejection**: Temu balik semantik dilengkapi mekanisme filtering dengan threshold (`0.45`) untuk secara presisi menolak kueri di luar domain bisnis (*Out-of-Domain*).

---

## Struktur Repositori

```text
ecommerce-cs-ai-agent/
├── data/
│   ├── corpus/                     # 14 dokumen teks SOP & kebijakan bisnis
│   │   ├── faq_umum.txt
│   │   ├── kebijakan_garansi.txt
│   │   ├── kebijakan_grosir.txt
│   │   ├── kebijakan_keluhan.txt
│   │   ├── kebijakan_privasi.txt
│   │   ├── kebijakan_promo.txt
│   │   ├── panduan_lacak_pesanan.txt
│   │   ├── panduan_pembayaran.txt
│   │   ├── sla_layanan.txt
│   │   ├── sop_barang_kurang.txt
│   │   ├── sop_pengiriman.txt
│   │   ├── sop_refund.txt
│   │   ├── sop_retur_seni.txt
│   │   └── sop_tukar_ukuran.txt
│   └── chroma_db/                  # Database vektor persisten ChromaDB (di-generate otomatis)
├── docs/
│   ├── Grup17_Tugas01.pdf          # Laporan teknis Milestone 1
│   ├── Grup17_Tugas02.pdf          # Laporan teknis Milestone 2
│   └── Grup17_Tugas03.pdf          # Laporan teknis Milestone 3
├── src/
│   └── ecommerce_cs_ai_agent/
│       ├── __init__.py
│       ├── search.py               # Milestone 1: Modul UCS
│       ├── solver.py               # Milestone 2: Solver CSP perutean tiket
│       ├── knowledge_base.py       # Milestone 3: Ingestion, chunking, & persistensi ChromaDB
│       ├── search_engine.py        # Milestone 3: Temu balik semantik & OOD filter
│       └── benchmark.py            # Milestone 3: Tolok ukur evaluasi 5 kueri bisnis
├── tests/
│   ├── test_solver.py              # Unit test CSP (5 test cases)
│   └── test_knowledge_base.py      # Unit test basis pengetahuan & retrieval (7 test cases)
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Panduan Instalasi & Persiapan Lingkungan

Proyek ini menggunakan manajer paket modern **`uv`** dan mendukung Python versi $\ge$ 3.11.

### 1. Kloning Repositori
```bash
git clone https://github.com/mutiarasianturi/ecommerce-cs-ai-agent.git
cd ecommerce-cs-ai-agent
```

### 2. Sinkronisasi Lingkungan Virtual & Dependensi
Jalankan perintah berikut untuk menginstal seluruh dependensi (termasuk `chromadb`, `sentence-transformers`, dan `pytest`):
```bash
uv sync
```

---

## Panduan Pengompilasian & Eksekusi Modul

### 1. Kompilasi & Pembangunan Basis Pengetahuan Vektor (Knowledge Base Ingestion)
Perintah ini membaca ke-14 dokumen pada folder `data/corpus/`, mengekstraksi dense embedding dari setiap potongan teks, dan menyusun indeks basis data vektor ChromaDB secara persisten ke folder `data/chroma_db/`.

**Perintah Kompilasi:**
```bash
uv run src/ecommerce_cs_ai_agent/knowledge_base.py
```

**Output yang Dihasilkan:**
```text
[OK] Vector store berhasil dibangun: 77 chunk dari 14 dokumen.
     Disimpan di: C:\Dokumen\GitHub\github\ecommerce-cs-ai-agent\data\chroma_db
```

---

### 2. Pengujian Temu Balik Semantik (Semantic Search Engine)

Modul `search_engine.py` menerima kueri bebas dalam tanda kutip ganda dan mencari dokumen SOP yang paling relevan berdasarkan kemiripan makna kosinus (*Cosine Similarity*).

#### Contoh Input 1: Kueri Bahasa Percakapan / Sinonim (In-Domain)
Pelanggan menggunakan bahasa sehari-hari (*"kekecilan"*, *"bisa ditukar ukurannya"*), bukan judul formal SOP:

```bash
uv run src/ecommerce_cs_ai_agent/search_engine.py "baju seragam yang kemarin sampai kekecilan, bisa ditukar ukurannya?"
```

**Output yang Dihasilkan:**
```text
[SEARCH] Query: "baju seragam yang kemarin sampai kekecilan, bisa ditukar ukurannya?"

Ditemukan 4 hasil:

  1. [Score 0.5436] sop_tukar_ukuran
     mesan 1 ukuran lebih besar agar
  bisa digunakan lebih lama.

6. CATATAN KHUSUS
- Seragam dengan bordir nama/logo sekolah custom tetap dapat ditukar u...

  2. [Score 0.5285] faq_umum
     kan riwayat pesanan.

2. PERTANYAAN TENTANG PRODUK

T: Apakah seragam bisa dipesan dengan logo/bordir sekolah?
J: Ya! Kami menyediakan layanan bordir ...

  3. [Score 0.4879] faq_umum
     k seragam dan sepatu dilengkapi tabel ukuran (size chart)
   dan panduan cara mengukur. Anda juga bisa menggunakan fitur "Virtual Fitting"
   di aplik...

  4. [Score 0.4843] sop_tukar_ukuran
     acakan.

4. BIAYA PENUKARAN
- Penukaran ukuran pertama: GRATIS (ongkos kirim ditanggung BeliCepat).
- Penukaran ukuran kedua dan seterusnya: ongkos ki...
```

---

#### Contoh Input 2: Kueri Prosedural / Relevan Langsung (In-Domain)
Pelanggan menanyakan prosedur pengembalian dana:

```bash
uv run src/ecommerce_cs_ai_agent/search_engine.py "bagaimana cara mengajukan refund dana?"
```

**Output yang Dihasilkan:**
```text
[SEARCH] Query: "bagaimana cara mengajukan refund dana?"

Ditemukan 5 hasil:

  1. [Score 0.6483] faq_umum
     k detail.

T: Siapa yang menanggung ongkir retur?
J: Untuk retur akibat kesalahan kami (barang cacat, salah kirim), ongkir
   ditanggung BeliCepat. Un...

  2. [Score 0.6237] sop_refund
     ARAT PENGAJUAN REFUND
- Pelanggan dapat mengajukan refund maksimal 7 hari kalender setelah barang diterima.
- Barang harus dalam kondisi belum digunak...

  3. [Score 0.5992] kebijakan_garansi
     eda), pelanggan berhak mendapat penggantian atau refund penuh.
- Klaim harus diajukan maksimal 3 hari setelah barang diterima.
...
```

---

#### Contoh Input 3: Kueri Ambigu Multi-Topik (In-Domain)
Pelanggan menyampaikan keluhan gabungan antara barang cacat dan keterlambatan pengiriman:

```bash
uv run src/ecommerce_cs_ai_agent/search_engine.py "Pesanan saya bermasalah, barang cacat dan pengiriman terlambat, bagaimana solusinya?"
```

**Output yang Dihasilkan:**
```text
[SEARCH] Query: "Pesanan saya bermasalah, barang cacat dan pengiriman terlambat, bagaimana solusinya?"

Ditemukan 3 hasil:

  1. [Score 0.5849] panduan_lacak_pesanan
     3 hari": hubungi CS untuk pengecekan ke pihak kurir.
- "Paket terkirim tapi belum diterima": laporkan ke CS dalam 1x24 jam
  untuk investigasi.
...

  2. [Score 0.5471] kebijakan_keluhan
     1x24 jam.
- SEDANG (Prioritas 3): keterlambatan pengiriman, ukuran tidak sesuai,
  voucher bermasalah. SLA: 2x24 jam.
- RENDAH (Prioritas 4): pertanyaan umum,...

  3. [Score 0.5212] sla_layanan
     SLA PENYELESAIAN MASALAH
- Pertanyaan umum (info produk, cara order, status pesanan): 1x24 jam.
- Keluhan pengiriman (keterlambatan, paket hilang): 2x24 ja...
```

---

#### Contoh Input 4: Kueri Luar Domain Kuliner (Out-of-Domain / DITOLAK)
Pertanyaan di luar konteks toko ritel sekolah akan disaring oleh *OOD threshold filter* (`similarity < 0.45`):

```bash
uv run src/ecommerce_cs_ai_agent/search_engine.py "Bagaimana resep dan takaran membuat kue bolu pandan yang empuk?"
```

**Output yang Dihasilkan:**
```text
[SEARCH] Query: "Bagaimana resep dan takaran membuat kue bolu pandan yang empuk?"

[X] Tidak ada hasil relevan (semua di bawah similarity threshold).
    Query ini kemungkinan out-of-domain (OOD).
```

---

#### Contoh Input 5: Kueri Luar Domain Flora / Tanaman (Out-of-Domain / DITOLAK)

```bash
uv run src/ecommerce_cs_ai_agent/search_engine.py "Bagaimana cara merawat tanaman hias monstera agar daunnya lebat dan subur?"
```

**Output yang Dihasilkan:**
```text
[SEARCH] Query: "Bagaimana cara merawat tanaman hias monstera agar daunnya lebat dan subur?"

[X] Tidak ada hasil relevan (semua di bawah similarity threshold).
    Query ini kemungkinan out-of-domain (OOD).
```

---

### 3. Eksekusi Skrip Tolok Ukur (Benchmark Temu Balik Semantik)
Menjalankan pengujian 5 variasi kueri uji bisnis sekaligus untuk mengevaluasi skor kemiripan kosinus dan presisi penolakan kueri luar domain:

**Perintah Eksekusi:**
```bash
uv run src/ecommerce_cs_ai_agent/benchmark.py
```

**Tabel Hasil Evaluasi Benchmark:**
```text
================================================================================
BENCHMARK TEMU BALIK SEMANTIK & OUT-OF-DOMAIN (OOD) FILTERING
Model Embedding   : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Metrik Similarity : Cosine Similarity (1.0 - Cosine Distance)
Ambang Batas (OOD): 0.45
================================================================================

================================================================================
TABEL RINGKASAN BENCHMARK EVALUASI
================================================================================
ID           | Cosine Sim | Status         | Top Doc               | Evaluasi  
--------------------------------------------------------------------------------
Q1_RELEVAN   | 0.6435     | DITERIMA       | kebijakan_keluhan     | PASS      
Q2_SINONIM   | 0.5517     | DITERIMA       | sop_tukar_ukuran      | PASS      
Q3_AMBIGU    | 0.5849     | DITERIMA       | panduan_lacak_pesanan | PASS      
Q4_OOD_1     | 0.2735     | DITOLAK (OOD)  | sop_retur_seni        | PASS      
Q5_OOD_2     | 0.1629     | DITOLAK (OOD)  | sop_retur_seni        | PASS      
================================================================================
```

---

### 4. Eksekusi Baseline Milestone Sebelumnya

#### Menjalankan Baseline Uniform Cost Search (Milestone 1)
```bash
uv run src/ecommerce_cs_ai_agent/search.py
```
*Output:*
```text
Jalur penanganan tiket optimal: Diterima -> Level_1 -> Selesai
Total estimasi waktu penanganan: 15 menit
```

#### Menjalankan CSP Message Routing Solver (Milestone 2)
```bash
uv run src/ecommerce_cs_ai_agent/solver.py
```
*Output:*
```text
=== HASIL PERUTEAN PESAN PELANGGAN (CSP) ===
MSG002 -> L2_Transaction_Agent
MSG003 -> L1_RAG_AutoReply
MSG001 -> L1_RAG_AutoReply
MSG004 -> L1_RAG_AutoReply
```

---

## Pengujian Otomatis (Automated Testing)

Seluruh komponen sistem diuji secara komprehensif menggunakan kerangka kerja `pytest`:

**Perintah Menjalankan Test:**
```bash
uv run pytest
```

**Ringkasan Hasil Uji (12 Tests Passed):**
```text
tests\test_knowledge_base.py .......                                     [ 58%]
tests\test_solver.py .....                                               [100%]

============================= 12 passed in 13.67s =============================
```

Rincian cakupan pengujian:
- **`test_solver.py` (5 pengujian)**: Validasi solusi konsistensi CSP, aturan node consistency, batasan kapasitas agen, penanganan pesan berprioritas tinggi, dan pencegahan penugasan L3 pada intent confidence rendah.
- **`test_knowledge_base.py` (7 pengujian)**: Validasi kuantitas korpus ($\ge$ 10 dokumen), integritas indeks ChromaDB persisten, dimensi dense embedding (384), keberhasilan temu balik kueri relevan, keakuratan temu balik kueri sinonim, serta presisi penolakan untuk 2 skenario kueri *out-of-domain*.
