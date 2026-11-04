"""Tes untuk chunking.py. Murni Python, tidak butuh model embedding, jadi cepat."""

import sys
from pathlib import Path

import pytest

from ecommerce_cs_ai_agent.chunking import chunk_text_word_aware

CONTOH = (
    "SYARAT PENGAJUAN REFUND. Pelanggan dapat mengajukan refund maksimal 7 hari "
    "kalender setelah barang diterima. Barang harus dalam kondisi belum digunakan "
    "dan kemasan asli masih lengkap. Pesan 1 ukuran lebih besar agar bisa dipakai "
    "lebih lama. Seragam dengan bordir nama sekolah tetap dapat ditukar ukurannya."
)


def test_tidak_ada_kata_yang_terpotong():
    kata_asli = set(CONTOH.split())
    for potongan in chunk_text_word_aware(CONTOH, chunk_size=80, overlap=15):
        assert set(potongan.split()) <= kata_asli


def test_panjang_tidak_melebihi_batas():
    for potongan in chunk_text_word_aware(CONTOH, chunk_size=80, overlap=15):
        assert len(potongan) <= 80


def test_ada_overlap_antar_potongan():
    hasil = chunk_text_word_aware(CONTOH, chunk_size=80, overlap=20)
    assert len(hasil) > 1
    for sebelum, sesudah in zip(hasil, hasil[1:]):
        assert sebelum.split()[-1] in sesudah.split()  # kata terakhir diulang


def test_semua_kata_tercakup():
    gabungan = " ".join(chunk_text_word_aware(CONTOH, chunk_size=80, overlap=15)).split()
    assert set(CONTOH.split()) <= set(gabungan)


def test_teks_kosong_menghasilkan_daftar_kosong():
    assert chunk_text_word_aware("   \n  ") == []


def test_parameter_tidak_valid_ditolak():
    with pytest.raises(ValueError):
        chunk_text_word_aware("halo", chunk_size=50, overlap=50)


def test_kata_lebih_panjang_dari_chunk_tidak_membuat_loop_tak_berujung():
    hasil = chunk_text_word_aware("a" * 300 + " pendek", chunk_size=100, overlap=10)
    assert len(hasil) >= 1