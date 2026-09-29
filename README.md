# ecommerce-cs-ai-agent
## Pengembangan Enterprise AI Copilot untuk Otomasi Customer Service E-Commerce Berbasis Agentic AI dan Retrieval-Augmented Generation

Proyek mata kuliah Kecerdasan Buatan (10S3001), Program Studi Sarjana Sistem Informasi,
Institut Teknologi Del, Semester Gasal 2026/2027.

## Deskripsi Singkat
Sistem asisten cerdas (AI Copilot) yang dirancang untuk mengotomasi proses klasifikasi,
perutean, dan penanganan tiket layanan pelanggan pada platform e-commerce, dengan
memanfaatkan pendekatan agentic AI dan retrieval-augmented generation (RAG).

## Anggota Tim
1. Adithya Philip Jona Putra Silaban - 12S24029
2. Mutiara Y.H. Sianturi - 12S24045
3. Immanuel Alexander Tambunan - 12S24034 


## Status Proyek
🚧 Milestone 1 — Problem Framing, Spesifikasi PEAS, & Baseline Search (UCS)

## Fitur & Baseline Implementasi
Proyek ini bertujuan untuk membangun AI Copilot untuk otomatisasi customer service pada platform e-commerce dengan pendekatan agentic AI dan retrieval-augmented generation (RAG). Secara umum, sistem ini dirancang untuk mengelola alur penanganan tiket pelanggan, mengklasifikasikan permasalahan, serta mengarahkan kasus ke unit yang tepat secara efisien.

Pada tahap baseline yang saat ini tersedia, proyek ini menggunakan dua pendekatan utama:
- Uniform Cost Search (UCS) untuk mencari jalur penanganan tiket dengan biaya minimum dari status awal hingga selesai.
- Constraint Satisfaction Problem (CSP) untuk menyelesaikan penjadwalan shift customer service dengan batasan-batasan konsistensi yang relevan.

Kedua komponen ini menjadi fondasi untuk pengembangan sistem yang lebih luas ke arah AI Copilot berbasis RAG dan workflow agentic pada layanan pelanggan e-commerce.

## Baseline yang Tersedia
- `src/ecommerce_cs_ai_agent/search.py` — simulasi graf penanganan tiket dan pencarian jalur optimal menggunakan UCS.
- `src/ecommerce_cs_ai_agent/solver.py` — implementasi solver CSP untuk penjadwalan shift CS dengan AC-3, backtracking, dan heuristik MRV.
- `test/test_solver.py` — pengujian otomatis untuk validasi solusi jadwal shift.

## Struktur Proyek
ecommerce-cs-ai-agent/
├── docs/                   # Dokumen problem framing, PEAS, dan laporan
│   ├── Grup17_Tugas01.pdf
│   └── Grup17_Tugas2.pdf
├── src/                    # Kode sumber program
│   └── ecommerce_cs_ai_agent/
├── test/                   # Skrip pengujian otomatis (pytest)
├── .gitignore              # File yang diabaikan Git
├── LICENSE                 # Lisensi proyek
├── README.md               # Dokumentasi proyek
├── pyproject.toml          # Konfigurasi proyek dan dependensi
└── uv.lock                 # Lock file dependensi

## Cara Instalasi
```bash
uv sync
```

## Cara Menjalankan
```bash
uv run src/ecommerce_cs_ai_agent/search.py
```
