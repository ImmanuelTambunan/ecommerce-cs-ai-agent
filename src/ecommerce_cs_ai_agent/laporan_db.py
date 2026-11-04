"""
laporan_db.py - Mengambil angka-angka untuk laporan dari database yang sudah jadi.

Mencetak: jumlah kata tiap dokumen korpus, statistik chunk, field metadata, dan
jumlah chunk per dokumen. Semua dalam bentuk tabel yang bisa langsung disalin ke
laporan Bagian 2.1, 3.3, dan 4.3/4.4. Tidak memakai model, jadi cepat.

Jalankan dari folder utama repo:
    uv run src/ecommerce_cs_ai_agent/laporan_db.py
"""

from collections import Counter
from pathlib import Path

import chromadb

REPO = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO / "data" / "corpus"
DB_PATH = REPO / "data" / "chroma_db"
COLLECTION_NAME = None  # None = pakai koleksi pertama

KUNCI_NAMA = ("nama_dokumen", "source", "sumber", "dokumen", "doc", "filename")


def nama_dokumen(meta):
    for kunci in KUNCI_NAMA:
        if kunci in meta:
            return str(meta[kunci]).replace(".txt", "")
    return "(tidak diketahui)"


def main():
    # ---- 1. Korpus: jumlah kata per dokumen (untuk tabel Bagian 2.1) ----
    print("| No | Dokumen | Jumlah Kata |")
    print("|---|---|---|")
    total_kata = 0
    berkas_korpus = sorted(CORPUS_DIR.glob("*.txt"))
    for i, berkas in enumerate(berkas_korpus, 1):
        jumlah = len(berkas.read_text(encoding="utf-8").split())
        total_kata += jumlah
        print(f"| {i} | {berkas.stem} | {jumlah} |")
    print(f"\nTotal: {len(berkas_korpus)} dokumen, {total_kata} kata.\n")

    # ---- 2. Database: statistik chunk (untuk Bagian 3.3) ----
    klien = chromadb.PersistentClient(path=str(DB_PATH))
    semua = klien.list_collections()
    if not semua:
        raise SystemExit("Database kosong. Jalankan knowledge_base.py dulu.")
    pertama = semua[0]
    nama = COLLECTION_NAME or (pertama if isinstance(pertama, str) else pertama.name)
    koleksi = klien.get_collection(nama)

    data = koleksi.get(include=["documents", "metadatas"])
    panjang_kar = [len(d) for d in data["documents"]]
    panjang_kata = [len(d.split()) for d in data["documents"]]
    print(f"Koleksi: {nama}")
    print(f"Total chunk            : {len(data['documents'])}")
    print(f"Rata-rata karakter     : {sum(panjang_kar) / len(panjang_kar):.0f}")
    print(f"Rata-rata kata         : {sum(panjang_kata) / len(panjang_kata):.1f}")
    print(f"Chunk terpendek/terpanjang (karakter): {min(panjang_kar)} / {max(panjang_kar)}")

    # ---- 3. Metadata (untuk Bagian 4.3) ----
    kunci = Counter()
    for m in data["metadatas"]:
        kunci.update(m.keys())
    print("\nField metadata (jumlah chunk yang memilikinya):")
    for k, n in kunci.items():
        print(f"  - {k}: {n}/{len(data['metadatas'])}")

    # ---- 4. Chunk per dokumen ----
    per_dok = Counter(nama_dokumen(m) for m in data["metadatas"])
    print("\nChunk per dokumen:")
    for dok, n in sorted(per_dok.items()):
        print(f"  - {dok}: {n}")

    # ---- 5. Deteksi kata terpotong di awal chunk (bukti perbaikan chunking) ----
    print("\nContoh awal 5 chunk pertama (periksa apakah ada kata yang terpotong):")
    for d in data["documents"][:5]:
        print(f"  \"{d[:50]}...\"")


if __name__ == "__main__":
    main()