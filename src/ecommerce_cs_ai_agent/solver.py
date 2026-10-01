"""
Modul Pemecahan Batasan Keputusan Bisnis (CSP Ticket Routing AI Copilot)
Mata Kuliah: Kecerdasan Buatan (10S3001) - Milestone 2 (W04)
Kelompok: 17
"""

from typing import Dict, List, Tuple, Optional
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
    category: str      # Info_Stok, Retur_Ukuran, Cacat_Produk, Lacak_Paket
    priority: int      # 1=Tinggi/Klaim, 2=Transaksi, 3=Info
    confidence: float  # Skala 0.0 - 1.0


class TicketRoutingCSP:
    def __init__(
        self, 
        tickets: List[Ticket], 
        min_l2_l3_capacity: int = 1, 
        max_l2_l3_capacity: int = 50
    ):
        self.tickets: Dict[str, Ticket] = {t.id: t for t in tickets}
        self.variables: List[str] = [t.id for t in tickets]
        self.min_capacity = min_l2_l3_capacity
        self.max_capacity = max_l2_l3_capacity
        
        self.domains: Dict[str, List[str]] = {
            t.id: list(DOMAIN_ACTIONS) for t in tickets
        }

    def node_consistency(self, domains: Dict[str, List[str]]) -> bool:
        """Memangkas domain awal berdasarkan C_kategori dan C_ambang."""
        for t_id, ticket in self.tickets.items():
            valid_actions = []
            for action in domains[t_id]:
                # C_ambang: Keyakinan rendah (< 0.60) wajib L1
                if ticket.confidence < 0.60 and action != "L1_RAG_AutoReply":
                    continue
                # C_kategori: Cacat_Produk wajib L2 atau L3
                if ticket.category == "Cacat_Produk" and action == "L1_RAG_AutoReply":
                    continue
                valid_actions.append(action)

            domains[t_id] = valid_actions
            if len(domains[t_id]) == 0:
                return False
        return True

    def ac3(self, domains: Dict[str, List[str]]) -> bool:
        """
        Algoritma Arc Consistency 3 (AC-3) untuk memvalidasi konsistensi busur
        antar-tiket terhadap keterbatasan slot kuota bersama.
        """
        # Inisialisasi antrean semua busur terarah (Xi, Xj)
        queue: List[Tuple[str, str]] = [
            (xi, xj) for xi in self.variables for xj in self.variables if xi != xj
        ]

        while queue:
            xi, xj = queue.pop(0)
            if self._revise(domains, xi, xj):
                if len(domains[xi]) == 0:
                    return False
                # Tambahkan kembali busur tetangga (Xk, Xi)
                for xk in self.variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(self, domains: Dict[str, List[str]], xi: str, xj: str) -> bool:
        """Merevisi domain Xi jika tidak ada pasangan nilai legal di Xj."""
        revised = False
        for x_val in list(domains[xi]):
            # Periksa apakah ada nilai y di domain Xj yang konsisten
            has_support = any(
                self._is_pair_consistent(xi, x_val, xj, y_val) 
                for y_val in domains[xj]
            )
            if not has_support:
                domains[xi].remove(x_val)
                revised = True
        return revised

    def _is_pair_consistent(self, xi: str, x_val: str, xj: str, y_val: str) -> bool:
        """Memvalidasi batasan biner lokal antar-tiket."""
        # Jika kapasitas server dibatasi 1 slot, dua tiket tidak boleh sama-sama mengambil L3
        if self.max_capacity <= 1:
            if x_val == "L3_Auto_Policy_Resolution" and y_val == "L3_Auto_Policy_Resolution":
                return False
        return True

    def is_assignment_consistent(self, assignment: Dict[str, str]) -> bool:
        """Pemeriksaan batas kapasitas sistem (C_kapasitas)."""
        assigned_l2_l3 = sum(
            1 for val in assignment.values() 
            if val in ("L2_Transaction_Agent", "L3_Auto_Policy_Resolution")
        )

        # Cek batas atas
        if assigned_l2_l3 > self.max_capacity:
            return False

        # Jika semua variabel sudah terisi, cek jaminan batas minimum
        if len(assignment) == len(self.variables):
            if assigned_l2_l3 < self.min_capacity:
                return False

        return True

    def select_unassigned_variable_mrv(
        self, assignment: Dict[str, str], domains: Dict[str, List[str]]
    ) -> str:
        """Heuristik MRV dengan tie-breaker prioritas tiket."""
        unassigned = [v for v in self.variables if v not in assignment]
        return min(
            unassigned, 
            key=lambda v: (len(domains[v]), self.tickets[v].priority)
        )

    def order_domain_values(self, var: str, domains: Dict[str, List[str]]) -> List[str]:
        """Pengurutan nilai berdasarkan C_prioritas."""
        ticket = self.tickets[var]
        available = list(domains[var])
        if ticket.priority == 1:
            order = ["L3_Auto_Policy_Resolution", "L2_Transaction_Agent", "L1_RAG_AutoReply"]
            available.sort(key=lambda x: order.index(x) if x in order else 99)
        elif ticket.priority == 3:
            order = ["L1_RAG_AutoReply", "L2_Transaction_Agent", "L3_Auto_Policy_Resolution"]
            available.sort(key=lambda x: order.index(x) if x in order else 99)
        return available

    def backtrack(
        self, assignment: Dict[str, str], domains: Dict[str, List[str]]
    ) -> Optional[Dict[str, str]]:
        if len(assignment) == len(self.variables):
            return assignment if self.is_assignment_consistent(assignment) else None

        var = self.select_unassigned_variable_mrv(assignment, domains)

        for value in self.order_domain_values(var, domains):
            assignment[var] = value
            if self.is_assignment_consistent(assignment):
                local_domains = copy.deepcopy(domains)
                local_domains[var] = [value]
                
                # Propagasi AC-3 selama pencarian
                if self.ac3(local_domains):
                    result = self.backtrack(assignment, local_domains)
                    if result is not None:
                        return result

            del assignment[var]

        return None

    def solve(self) -> Optional[Dict[str, str]]:
        working_domains = copy.deepcopy(self.domains)
        if not self.node_consistency(working_domains):
            return None
        if not self.ac3(working_domains):
            return None
        return self.backtrack({}, working_domains)