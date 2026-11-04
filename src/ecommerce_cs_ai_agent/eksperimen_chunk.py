"""
eksperimen_chunk.py - Membandingkan beberapa ukuran chunk untuk dasar justifikasi.

Tujuan: rubrik meminta ukuran chunk dan overlap "dijustifikasi". Skrip ini
membangun database sementara (di memori, TIDAK menyentuh data/chroma_db) untuk
tiap ukuran chunk, lalu mengukur seberapa sering dokumen yang benar muncul di
urutan teratas. Hasilnya dipakai sebagai alasan di laporan Bagian 3.2.

Jalankan dari folder utama repo:
    uv run src/ecommerce_cs_ai_agent/eksperimen_chunk.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from ecommerce_cs_ai_agent.chunking import chunk_text_word_aware
except ImportError:  # dijalankan langsung sebagai skrip
    from chunking import chunk_text_word_aware

REPO = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO / "data" / "corpus"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# (ukuran chunk dalam karakter, overlap dalam karakter). Overlap dibuat ~10%.
KONFIGURASI = [(300, 30), (500, 50), (800, 80)]

# Kueri uji dan dokumen yang SEHARUSNYA muncul. Nama dokumen = nama file tanpa .txt.
# PENTING: ini tebakan awal. Baca isi dokumen, lalu sesuaikan daftar ini.
UJI = [
    ("Seragam yang saya beli kekecilan, bisa ditukar ukurannya?", {"sop_tukar_ukuran"}),
    ("Bagaimana cara mengajukan pengembalian dana?", {"sop_refund"}),
    ("Paket sudah seminggu belum sampai", {"sop_pengiriman", "panduan_lacak_pesanan"}),
    ("Jumlah barang yang diterima kurang dari pesanan", {"sop_barang_kurang"}),
    ("Mau beli pulpen 500 buah untuk dijual lagi", {"kebijakan_grosir"}),
    ("Harga di keranjang beda dengan halaman produk, kode promo tidak berlaku", {"kebijakan_promo"}),
    ("Resleting tas sekolah rusak, masih bergaransi?", {"kebijakan_garansi"}),
    ("Data pribadi saya dipakai untuk apa saja?", {"kebijakan_privasi"}),
    ("Metode pembayaran apa saja yang tersedia?", {"panduan_pembayaran"}),
    ("Berapa lama komplain saya diproses?", {"sla_layanan", "kebijakan_keluhan"}),
]


def bangun_koleksi(client, model, ukuran, overlap):
    """Bangun satu koleksi sementara untuk satu konfigurasi chunk."""
    nama = f"uji_{ukuran}_{overlap}"
    try:
        client.delete_collection(nama)
    except Exception:
        pass  # belum ada, tidak masalah
    koleksi = client.create_collection(nama, metadata={"hnsw:space": "cosine"})

    teks, meta, ids = [], [], []
    for berkas in sorted(CORPUS_DIR.glob("*.txt")):
        isi = berkas.read_text(encoding="utf-8")
        for i, potongan in enumerate(chunk_text_word_aware(isi, ukuran, overlap)):
            teks.append(potongan)
            meta.append({"nama_dokumen": berkas.stem, "chunk_index": i})
            ids.append(f"{berkas.stem}_{i}")

    vektor = model.encode(teks, normalize_embeddings=True).tolist()
    koleksi.add(ids=ids, documents=teks, metadatas=meta, embeddings=vektor)
    return koleksi, teks


def main():
    if not any(CORPUS_DIR.glob("*.txt")):
        raise SystemExit(f"Tidak ada berkas .txt di {CORPUS_DIR}")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model: {MODEL_NAME}")
    print(f"Batas panjang input model (max_seq_length): {model.max_seq_length} token\n")

    client = chromadb.EphemeralClient()
    vektor_uji = model.encode([q for q, _ in UJI], normalize_embeddings=True).tolist()

    print(f"{'Ukuran':>7} {'Overlap':>8} {'#Chunk':>7} {'Rata2 Kar':>10} "
          f"{'Hit@1':>7} {'Hit@3':>7} {'Rata2 Skor Top-1':>17}")
    print("-" * 70)

    for ukuran, overlap in KONFIGURASI:
        koleksi, teks = bangun_koleksi(client, model, ukuran, overlap)
        hit1 = hit3 = 0
        total_skor = 0.0
        for (kueri, benar), vek in zip(UJI, vektor_uji):
            hasil = koleksi.query(query_embeddings=[vek], n_results=3,
                                  include=["metadatas", "distances"])
            nama_top = [m["nama_dokumen"] for m in hasil["metadatas"][0]]
            skor_top = 1.0 - hasil["distances"][0][0]  # jarak cosine -> kemiripan
            hit1 += nama_top[0] in benar
            hit3 += any(n in benar for n in nama_top)
            total_skor += skor_top
        n = len(UJI)
        rata_kar = sum(len(t) for t in teks) / len(teks)
        print(f"{ukuran:>7} {overlap:>8} {len(teks):>7} {rata_kar:>10.0f} "
              f"{hit1 / n:>7.0%} {hit3 / n:>7.0%} {total_skor / n:>17.4f}")

    print("\nCara membaca: pilih ukuran dengan Hit@1 dan Hit@3 tertinggi. "
          "Jika seri, pilih yang\npanjangnya masih muat di batas input model. "
          "Tulis tabel ini di laporan Bagian 3.2.")


if __name__ == "__main__":
    main()