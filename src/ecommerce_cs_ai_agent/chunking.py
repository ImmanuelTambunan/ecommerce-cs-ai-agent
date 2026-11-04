"""
chunking.py - Pemotongan teks yang menghormati batas kata (word-aware).

Masalah di versi lama: teks dipotong tiap 500 karakter sehingga kata bisa
terbelah, misalnya "pesan" menjadi "mesan" atau "SYARAT" menjadi "ARAT".
Modul ini memotong per KATA, jadi tidak ada kata yang terputus, dan
potongan berikutnya mengulang beberapa kata terakhir (overlap).
"""

from typing import List


def chunk_text_word_aware(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Potong `text` menjadi potongan berukuran maksimal `chunk_size` karakter.

    Parameter:
        chunk_size: batas panjang satu potongan (dalam karakter).
        overlap   : panjang kira-kira (dalam karakter) bagian akhir potongan
                    yang diulang di awal potongan berikutnya.

    Mengembalikan daftar potongan. Teks kosong menghasilkan daftar kosong.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih besar dari 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap harus >= 0 dan lebih kecil dari chunk_size")

    # split() membuang spasi/baris baru berlebih sekaligus, sehingga
    # pembersihan spasi ganda terjadi di sini.
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        # 1) Tambahkan kata satu per satu sampai batas karakter tercapai.
        end = start
        length = 0
        while end < len(words):
            tambahan = len(words[end]) + (1 if end > start else 0)  # +1 untuk spasi
            if length + tambahan > chunk_size and end > start:
                break
            length += tambahan
            end += 1

        chunks.append(" ".join(words[start:end]))

        if end >= len(words):  # semua kata sudah masuk
            break

        # 2) Hitung titik mulai potongan berikutnya: mundur dari `end`
        #    selama total panjang kata yang diulang belum melebihi `overlap`.
        mundur = end
        diulang = 0
        while mundur > start + 1 and diulang + len(words[mundur - 1]) + 1 <= overlap:
            mundur -= 1
            diulang += len(words[mundur]) + 1
        start = mundur  # selalu > start lama, jadi tidak ada loop tak berujung

    return chunks