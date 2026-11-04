"""
benchmark_lanjutan.py - Benchmark yang lebih lengkap + penentuan threshold dari data.

Melengkapi benchmark.py milik Milestone 3. Menambah kueri bertipe: typo, pendek,
dan out-of-domain yang dekat dengan topik toko. Lalu mencoba beberapa nilai
threshold untuk melihat mana yang paling memisahkan kueri relevan dari yang
harus ditolak. Hasilnya dipakai untuk laporan Bagian 5.2 dan 5.3.

Syarat: database sudah dibangun (jalankan knowledge_base.py lebih dulu).
Jalankan dari folder utama repo:
    uv run src/ecommerce_cs_ai_agent/benchmark_lanjutan.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "chroma_db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = None   # isi dengan nama koleksi dari knowledge_base.py; None = pakai koleksi pertama
THRESHOLD_SAAT_INI = 0.45

# (id, jenis, kueri, dokumen yang diharapkan; None bila harus ditolak)
# Daftar dokumen yang diharapkan adalah tebakan awal: cek dan sesuaikan dengan isi korpus.
KUERI = [
    ("Q01", "relevan",    "Bagaimana cara mengajukan refund?", {"sop_refund"}),
    ("Q02", "sinonim",    "Baju seragam kekecilan, bisa ditukar?", {"sop_tukar_ukuran"}),
    ("Q03", "sinonim",    "Paket belum nyampe padahal sudah seminggu", {"sop_pengiriman", "panduan_lacak_pesanan"}),
    ("Q04", "sinonim",    "Mau beli pulpen 500 buah buat dijual lagi", {"kebijakan_grosir"}),
    ("Q05", "ambigu",     "Pesanan saya bermasalah, bagaimana solusinya?",
        {"kebijakan_keluhan", "sop_barang_kurang", "sop_pengiriman", "sop_refund", "faq_umum"}),
    ("Q06", "typo",       "brg sy rusak pas smpai, bs dikembalikan ga?",
        {"kebijakan_garansi", "kebijakan_keluhan", "sop_refund"}),
    ("Q07", "typo",       "sragam kekcilan bsa tuker ukuran?", {"sop_tukar_ukuran"}),
    ("Q08", "pendek",     "refund", {"sop_refund"}),
    ("Q09", "pendek",     "ongkir", {"faq_umum", "sop_pengiriman", "sop_refund"}),
    ("Q10", "ood_jauh",   "Bagaimana cara membuat kue bolu pandan yang empuk?", None),
    ("Q11", "ood_jauh",   "Siapa presiden pertama Indonesia?", None),
    ("Q12", "ood_dekat",  "Rekomendasi laptop gaming terbaik untuk kuliah", None),
    ("Q13", "ood_dekat",  "Berapa harga emas hari ini?", None),
    ("Q14", "ood_dekat",  "Bagaimana cara mengurus beasiswa kuliah?", None),
]

KUNCI_NAMA = ("nama_dokumen", "source", "sumber", "dokumen", "doc", "filename")


def nama_dokumen(meta):
    """Ambil nama dokumen dari metadata, apa pun nama field yang dipakai."""
    for kunci in KUNCI_NAMA:
        if kunci in meta:
            return str(meta[kunci]).replace(".txt", "")
    return str(meta)


def buka_koleksi():
    klien = chromadb.PersistentClient(path=str(DB_PATH))
    if COLLECTION_NAME:
        return klien.get_collection(COLLECTION_NAME)
    semua = klien.list_collections()
    if not semua:
        raise SystemExit("Database kosong. Jalankan knowledge_base.py dulu.")
    pertama = semua[0]
    return klien.get_collection(pertama if isinstance(pertama, str) else pertama.name)


def main():
    koleksi = buka_koleksi()
    print(f"Koleksi: {koleksi.name} | jumlah chunk: {koleksi.count()}")
    model = SentenceTransformer(MODEL_NAME)
    vektor = model.encode([k[2] for k in KUERI], normalize_embeddings=True).tolist()

    baris = []
    print(f"\n{'ID':<4} {'Jenis':<10} {'Skor':>6} {'Status':<8} {'Dok. Teratas':<24} {'Sesuai':<6}")
    print("-" * 66)
    for (qid, jenis, kueri, benar), vek in zip(KUERI, vektor):
        hasil = koleksi.query(query_embeddings=[vek], n_results=3, include=["metadatas", "distances"])
        teratas = nama_dokumen(hasil["metadatas"][0][0])
        skor = 1.0 - hasil["distances"][0][0]
        diterima = skor >= THRESHOLD_SAAT_INI
        if benar is None:                    # harus ditolak
            sesuai = not diterima
        else:                                # harus diterima dan dokumennya tepat
            sesuai = diterima and teratas in benar
        baris.append((qid, jenis, skor, benar is None))
        print(f"{qid:<4} {jenis:<10} {skor:>6.4f} {'DITERIMA' if diterima else 'DITOLAK':<8} "
              f"{teratas:<24} {'YA' if sesuai else 'TIDAK':<6}")

    # ---- Menentukan threshold dari data ----
    # Kueri typo/pendek wajar bernilai lebih rendah, jadi dipisah sebagai "kasus sulit".
    skor_relevan = [s for _, j, s, ood in baris if not ood and j not in ("typo", "pendek")]
    skor_sulit = [s for _, j, s, ood in baris if not ood and j in ("typo", "pendek")]
    skor_ood = [s for _, _, s, ood in baris if ood]
    print(f"\nSkor kueri relevan/sinonim/ambigu : {min(skor_relevan):.4f} - {max(skor_relevan):.4f}")
    print(f"Skor kasus sulit (typo, pendek)    : {min(skor_sulit):.4f} - {max(skor_sulit):.4f}")
    print(f"Skor kueri OOD                     : {min(skor_ood):.4f} - {max(skor_ood):.4f}")
    celah = min(skor_relevan) - max(skor_ood)
    if celah > 0:
        print(f"Celah relevan vs OOD = {celah:.4f}. Titik tengah {max(skor_ood) + celah / 2:.4f} "
              f"adalah kandidat threshold.")
    else:
        print("PERINGATAN: skor relevan dan OOD saling tumpang tindih. Tidak ada threshold yang "
              "memisahkan\nkeduanya dengan sempurna. Bahas ini sebagai keterbatasan di laporan.")
    if min(skor_sulit) < max(skor_ood):
        print("Catatan: sebagian kasus sulit (typo/pendek) skornya di bawah kueri OOD tertinggi. "
              "Ini bahan analisis:\nthreshold yang terlalu tinggi akan menolak pesan pelanggan yang valid.")

    print(f"\n{'Threshold':>10} {'Relevan diterima':>18} {'Sulit diterima':>16} {'OOD ditolak':>13}")
    print("-" * 61)
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        rel = sum(x >= t for x in skor_relevan)
        sul = sum(x >= t for x in skor_sulit)
        ood = sum(x < t for x in skor_ood)
        print(f"{t:>10.2f} {rel:>11}/{len(skor_relevan):<6} {sul:>9}/{len(skor_sulit):<6} {ood:>7}/{len(skor_ood)}")

    # ---- Cek metadata untuk hybrid search ----
    contoh = koleksi.get(limit=1, include=["metadatas"])["metadatas"][0]
    print(f"\nField metadata yang tersimpan: {sorted(contoh.keys())}")
    if "kategori" in contoh:
        kat = contoh["kategori"]
        h = koleksi.query(query_embeddings=[vektor[0]], n_results=2, where={"kategori": kat},
                          include=["metadatas", "distances"])
        print(f"Hybrid search (filter kategori='{kat}') berhasil: {len(h['ids'][0])} hasil.")
    else:
        print("Field 'kategori' belum ada. Tambahkan di knowledge_base.py agar hybrid search "
              "bisa didemokan.")


if __name__ == "__main__":
    main()