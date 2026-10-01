import pytest
from ecommerce_cs_ai_agent.solver import Ticket, TicketRoutingCSP

def test_csp_routing_basic():
    """Pengujian alur perutean dasar pada kondisi normal."""
    tickets = [
        Ticket(id="T1", category="Info_Stok", priority=3, confidence=0.85),
        Ticket(id="T2", category="Retur_Ukuran", priority=2, confidence=0.90)
    ]
    solver = TicketRoutingCSP(tickets, max_transaction_capacity=5)
    sol = solver.solve()
    assert sol is not None
    assert len(sol) == 2

def test_c_kategori_cacat_produk_constraint():
    """
    Batasan C_kategori: Tiket Cacat_Produk wajib ke L2 atau L3,
    tidak boleh masuk ke kanal L1_RAG_AutoReply.
    """
    tickets = [
        Ticket(id="T1", category="Cacat_Produk", priority=1, confidence=0.95)
    ]
    solver = TicketRoutingCSP(tickets, max_transaction_capacity=5)
    sol = solver.solve()
    assert sol is not None
    assert sol["T1"] in ["L2_Transaction_Agent", "L3_Auto_Policy_Resolution"]
    assert sol["T1"] != "L1_RAG_AutoReply"

def test_c_ambang_low_confidence_constraint():
    """
    Batasan C_ambang: Tiket dengan confidence < 0.60 (typo tinggi)
    wajib diarahkan ke L1_RAG_AutoReply.
    """
    tickets = [
        Ticket(id="T1", category="Retur_Ukuran", priority=2, confidence=0.48)
    ]
    solver = TicketRoutingCSP(tickets, max_transaction_capacity=5)
    sol = solver.solve()
    assert sol is not None
    assert sol["T1"] == "L1_RAG_AutoReply"

def test_conflict_unsatisfiable_scenario():
    """
    Kasus Ekstrem Unsatisfiable:
    Tiket Cacat_Produk (wajib L2/L3) tetapi memiliki confidence < 0.60 (wajib L1).
    Solver harus mengembalikan None karena batasan saling bertentangan.
    """
    tickets = [
        Ticket(id="T1", category="Cacat_Produk", priority=1, confidence=0.45)
    ]
    solver = TicketRoutingCSP(tickets, max_transaction_capacity=5)
    sol = solver.solve()
    assert sol is None

def test_c_kapasitas_exceeded():
    """
    Batasan C_kapasitas: Ketika kuota L2/L3 penuh, tiket transaksi yang
    memerlukan L2/L3 tidak dapat dialokasikan (Unsatisfiable).
    """
    tickets = [
        Ticket(id="T1", category="Cacat_Produk", priority=1, confidence=0.95),
        Ticket(id="T2", category="Cacat_Produk", priority=1, confidence=0.90),
    ]
    # Kapasitas eksekusi disetel hanya 1 tiket
    solver = TicketRoutingCSP(tickets, max_transaction_capacity=1)
    sol = solver.solve()
    assert sol is None