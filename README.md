# ecommerce-cs-ai-agent

## Pengembangan Enterprise AI Copilot untuk Otomasi Customer Service Ritel Daring Perlengkapan Sekolah Berbasis Agentic AI dan Retrieval-Augmented Generation

Proyek mata kuliah Kecerdasan Buatan (10S3001), Program Studi Sarjana Sistem Informasi,
Institut Teknologi Del, Semester Gasal 2026/2027.

## Deskripsi Singkat
Sistem asisten cerdas (AI Copilot) yang dirancang untuk mengotomasi proses klasifikasi,
perutean, dan penanganan pesan keluhan pelanggan pada platform ritel daring perlengkapan
sekolah, dengan memanfaatkan pendekatan agentic AI dan retrieval-augmented generation (RAG).

## Anggota Tim
1. Adithya Philip Jona Putra Silaban - 12S24029
2. Mutiara Y.H. Sianturi - 12S24045
3. Immanuel Alexander Tambunan - 12S24034

## Status Proyek
✅ Milestone 1 — Problem Framing, Spesifikasi PEAS, & Baseline Search (UCS)
🚧 Milestone 2 — CSP Solver untuk Perutean Pesan Pelanggan Otomatis

## Fitur & Baseline Implementasi
Proyek ini membangun AI Copilot untuk otomatisasi customer service pada platform ritel
daring perlengkapan sekolah dengan pendekatan agentic AI dan retrieval-augmented
generation (RAG). Sistem dirancang untuk menerima pesan keluhan pelanggan, mengklasifikasikan
jenis permasalahannya, dan merutekannya secara otomatis ke channel tindakan yang tepat
tanpa keterlibatan supervisor manusia.

Pada tahap yang saat ini tersedia, proyek ini menggunakan dua pendekatan utama:
- Uniform Cost Search (UCS) untuk mencari jalur penanganan pesan dengan biaya waktu
  minimum dari status awal hingga selesai.
- Constraint Satisfaction Problem (CSP) untuk merutekan pesan keluhan pelanggan secara
  otomatis ke channel L1 (balasan otomatis berbasis RAG), L2 (eksekusi transaksi), atau
  L3 (keputusan kebijakan otomatis), berdasarkan kategori keluhan, tingkat prioritas,
  dan skor keyakinan intent.

## Baseline yang Tersedia
- `src/ecommerce_cs_ai_agent/search.py` — simulasi graf penanganan pesan dan pencarian
  jalur optimal menggunakan UCS.
- `src/ecommerce_cs_ai_agent/solver.py` — implementasi solver CSP untuk perutean pesan
  pelanggan ke channel otomatisasi dengan AC-3, backtracking, dan heuristik MRV.
- `tests/test_solver.py` — pengujian otomatis untuk validasi solusi perutean pesan.

## Struktur Proyek
ecommerce-cs-ai-agent/
├── docs/
│ ├── Grup17_Tugas01.pdf
│ └── Grup17_Tugas02.pdf
├── src/
│ └── ecommerce_cs_ai_agent/
│ ├── search.py
│ └── solver.py
├── tests/
│ └── test_solver.py
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock


## Cara Instalasi
```bash
uv sync
```

## Menjalankan Baseline UCS
uv run src/ecommerce_cs_ai_agent/search.py

Output yang diharapkan:

Jalur penanganan tiket optimal: Diterima -> Level_1 -> Selesai
Total estimasi waktu penanganan: 15 menit

## Menjalankan CSP Solver
uv run src/ecommerce_cs_ai_agent/solver.py

Output yang diharapkan:

=== HASIL PERUTEAN PESAN PELANGGAN (CSP) ===
MSG002 -> L2_Transaction_Agent
MSG003 -> L1_RAG_AutoReply
MSG001 -> L1_RAG_AutoReply
MSG004 -> L1_RAG_AutoReply

Urutan tiket dapat berbeda karena pemilihan variabel menggunakan heuristik MRV dan prioritas.

## Menjalankan Pengujian
uv run pytest

Hasil yang diharapkan setelah struktur folder tests/ sudah benar: 5 passed
