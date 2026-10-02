"""
Modul Pemecahan Batasan Keputusan Bisnis (CSP Message Routing AI Copilot)
Mata Kuliah: Kecerdasan Buatan (10S3001) - Milestone 2 (W04)
Kelompok: 17
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import copy

DOMAIN_ACTIONS = [
    "L1_RAG_AutoReply",
    "L2_Transaction_Agent",
    "L3_Auto_Policy_Resolution"
]


@dataclass
class Ticket:
    id: str
    category: str
    priority: int
    confidence: float


class TicketRoutingCSP:
    def __init__(self, tickets: List[Ticket], min_l2_l3_capacity: int = 1, max_l2_l3_capacity: int = 50):
        self.tickets: Dict[str, Ticket] = {t.id: t for t in tickets}
        self.variables: List[str] = [t.id for t in tickets]
        self.min_capacity = min_l2_l3_capacity
        self.max_capacity = max_l2_l3_capacity
        self.domains: Dict[str, List[str]] = {t.id: list(DOMAIN_ACTIONS) for t in tickets}

    def node_consistency(self, domains: Dict[str, List[str]]) -> bool:
        """C_kategori dan C_ambang. Kombinasi Cacat_Produk + confidence
        rendah tetap punya solusi (diarahkan ke L2), tidak lagi menghasilkan
        domain kosong seperti pada versi awal."""
        for t_id, ticket in self.tickets.items():
            valid_actions = list(domains[t_id])
            if ticket.category == "Cacat_Produk" and "L1_RAG_AutoReply" in valid_actions:
                valid_actions.remove("L1_RAG_AutoReply")
            if ticket.confidence < 0.60 and "L3_Auto_Policy_Resolution" in valid_actions:
                valid_actions.remove("L3_Auto_Policy_Resolution")
            domains[t_id] = valid_actions
            if len(domains[t_id]) == 0:
                return False
        return True

    def select_unassigned_variable_mrv(self, assignment: Dict[str, str], domains: Dict[str, List[str]]) -> str:
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: (len(domains[v]), self.tickets[v].priority))

    def order_domain_values(self, var: str, domains: Dict[str, List[str]]) -> List[str]:
        ticket = self.tickets[var]
        available = list(domains[var])
        if ticket.priority == 1:
            order = ["L3_Auto_Policy_Resolution", "L2_Transaction_Agent", "L1_RAG_AutoReply"]
            available.sort(key=lambda x: order.index(x) if x in order else 99)
        elif ticket.priority == 3:
            order = ["L1_RAG_AutoReply", "L2_Transaction_Agent", "L3_Auto_Policy_Resolution"]
            available.sort(key=lambda x: order.index(x) if x in order else 99)
        return available

    def is_assignment_consistent(self, assignment: Dict[str, str]) -> bool:
        count_l2_l3 = sum(1 for v in assignment.values() if v != "L1_RAG_AutoReply")
        if count_l2_l3 > self.max_capacity:
            return False
        if len(assignment) == len(self.variables) and count_l2_l3 < self.min_capacity:
            return False
        return True

    def backtrack(self, assignment: Dict[str, str], domains: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
        if len(assignment) == len(self.variables):
            return dict(assignment) if self.is_assignment_consistent(assignment) else None

        # Forward-checking: pangkas dini jika sisa variabel yang WAJIB L2/L3
        # saja sudah melebihi kapasitas tersisa.
        unassigned = [v for v in self.variables if v not in assignment]
        forced_remaining = sum(1 for v in unassigned if "L1_RAG_AutoReply" not in domains[v])
        current_l2l3 = sum(1 for v in assignment.values() if v != "L1_RAG_AutoReply")
        if current_l2l3 + forced_remaining > self.max_capacity:
            return None

        var = self.select_unassigned_variable_mrv(assignment, domains)
        for value in self.order_domain_values(var, domains):
            assignment[var] = value
            if self.is_assignment_consistent(assignment):
                result = self.backtrack(assignment, domains)
                if result is not None:
                    return result
            del assignment[var]
        return None

    def solve(self) -> Optional[Dict[str, str]]:
        working_domains = copy.deepcopy(self.domains)
        if not self.node_consistency(working_domains):
            return None
        return self.backtrack({}, working_domains)


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(10000)  # wajib untuk skala besar (ribuan pesan)

    sample_messages = [
        Ticket(id="MSG001", category="Info_Stok", priority=3, confidence=0.95),
        Ticket(id="MSG002", category="Cacat_Produk", priority=1, confidence=0.40),
        Ticket(id="MSG003", category="Retur_Ukuran", priority=2, confidence=0.80),
        Ticket(id="MSG004", category="Lacak_Paket", priority=3, confidence=0.97),
    ]
    csp = TicketRoutingCSP(sample_messages, min_l2_l3_capacity=1, max_l2_l3_capacity=50)
    solution = csp.solve()

    print("=== HASIL PERUTEAN PESAN PELANGGAN (CSP) ===")
    if solution:
        for msg_id, channel in solution.items():
            print(f"{msg_id} -> {channel}")
    else:
        print("Tidak ditemukan solusi yang memenuhi seluruh batasan (Unsatisfiable).")