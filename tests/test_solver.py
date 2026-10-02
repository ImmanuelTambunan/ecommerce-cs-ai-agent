"""
Pengujian Otomatis Modul CSP Message Routing AI Copilot
Mata Kuliah: Kecerdasan Buatan (10S3001) - Milestone 2 (W04)
Kelompok: 17
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ecommerce_cs_ai_agent"))

from solver import Ticket, TicketRoutingCSP


def test_solusi_normal_ditemukan():
    messages = [
        Ticket(id="MSG001", category="Info_Stok", priority=3, confidence=0.95),
        Ticket(id="MSG002", category="Cacat_Produk", priority=1, confidence=0.88),
        Ticket(id="MSG003", category="Retur_Ukuran", priority=2, confidence=0.80),
    ]
    csp = TicketRoutingCSP(messages, min_l2_l3_capacity=1, max_l2_l3_capacity=50)
    solution = csp.solve()
    assert solution is not None
    assert len(solution) == len(messages)


def test_pesan_keyakinan_rendah_tidak_boleh_l3():
    messages = [Ticket(id="MSG010", category="Lacak_Paket", priority=2, confidence=0.35)]
    csp = TicketRoutingCSP(messages, min_l2_l3_capacity=0, max_l2_l3_capacity=50)
    solution = csp.solve()
    assert solution is not None
    assert solution["MSG010"] != "L3_Auto_Policy_Resolution"


def test_pesan_cacat_produk_tidak_boleh_l1():
    messages = [Ticket(id="MSG020", category="Cacat_Produk", priority=1, confidence=0.90)]
    csp = TicketRoutingCSP(messages, min_l2_l3_capacity=1, max_l2_l3_capacity=50)
    solution = csp.solve()
    assert solution is not None
    assert solution["MSG020"] != "L1_RAG_AutoReply"


def test_kombinasi_cacat_produk_dan_keyakinan_rendah_tetap_ada_solusi():
    """Kasus edge case penting: dulu menghasilkan domain kosong (bug),
    sekarang harus tetap menghasilkan solusi (masuk L2)."""
    messages = [Ticket(id="MSG030", category="Cacat_Produk", priority=1, confidence=0.35)]
    csp = TicketRoutingCSP(messages, min_l2_l3_capacity=1, max_l2_l3_capacity=50)
    solution = csp.solve()
    assert solution is not None
    assert solution["MSG030"] == "L2_Transaction_Agent"


def test_kasus_ekstrem_kapasitas_tidak_cukup():
    messages = [
        Ticket(id=f"MSG{i}", category="Cacat_Produk", priority=1, confidence=0.90)
        for i in range(8)
    ]
    csp = TicketRoutingCSP(messages, min_l2_l3_capacity=1, max_l2_l3_capacity=3)
    solution = csp.solve()
    assert solution is None