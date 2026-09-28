"""
Modul Pemecahan Batasan Keputusan Bisnis (CSP Shift Scheduling CS BeliCepat.id)
Mata Kuliah: Kecerdasan Buatan (10S3001) - Milestone 2 (W04)
Grup: 17
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import copy

class CSShiftSolver:
    """
    Constraint Satisfaction Problem (CSP) Solver menggunakan AC-3 Arc Consistency
    dan Backtracking Search dengan MRV (Minimum Remaining Values) Heuristic.
    """

    def __init__(self, staff_data: List[Dict[str, Any]], days: int = 7):
        """
        Inisialisasi solver.
        
        staff_data: List dictionary memuat 'id', 'nama', dan 'skills' (e.g. ['Order', 'Refund'])
        days: Jumlah hari penjadwalan (default: 7 hari)
        """
        self.staff_data = staff_data
        self.days = days
        self.shifts = ['Pagi', 'Siang', 'Malam', 'Off']
        
        # Inisialisasi Variabel: Tuple (staff_id, day_index)
        self.variables: List[Tuple[str, int]] = [
            (s['id'], d) for s in self.staff_data for d in range(1, days + 1)
        ]
        
        # Inisialisasi Domain awal
        self.domains: Dict[Tuple[str, int], List[str]] = {
            v: list(self.shifts) for v in self.variables
        }

    def is_consistent(self, var: Tuple[str, int], value: str, assignment: Dict[Tuple[str, int], str]) -> bool:
        """
        Memeriksa apakah penugasan (var = value) konsisten dengan assignment saat ini.
        """
        staff_id, day = var
        
        # Hard Constraint 2: Istirahat pasca Shift Malam (Hari berikutnya wajib Off)
        if day > 1:
            prev_var = (staff_id, day - 1)
            if assignment.get(prev_var) == 'Malam' and value != 'Off':
                return False
                
        if day < self.days:
            next_var = (staff_id, day + 1)
            if assignment.get(next_var) is not None and value == 'Malam' and assignment[next_var] != 'Off':
                return False

        # Hard Constraint 3: Maksimal 2 Shift Malam berturut-turut
        if value == 'Malam':
            if day >= 3:
                if assignment.get((staff_id, day - 1)) == 'Malam' and assignment.get((staff_id, day - 2)) == 'Malam':
                    return False

        return True

    def ac3(self, domains: Dict[Tuple[str, int], List[str]]) -> bool:
        """
        Algoritma Arc Consistency 3 (AC-3) untuk mengeliminasi nilai domain yang tidak konsisten.
        """
        queue: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
        
        # Tambahkan semua pasang variabel bertetangga (inter-day constraint) ke antrean
        for s in self.staff_data:
            s_id = s['id']
            for d in range(1, self.days):
                queue.append(((s_id, d), (s_id, d + 1)))
                queue.append(((s_id, d + 1), (s_id, d)))

        while queue:
            (xi, xj) = queue.pop(0)
            if self._revise(domains, xi, xj):
                if len(domains[xi]) == 0:
                    return False  # Domain kosong, kegagalan propagasi
                # Tambahkan kembali tetangga ke antrean
                s_id, d = xi
                neighbors = [(s_id, d - 1), (s_id, d + 1)]
                for nbr in neighbors:
                    if nbr in domains and nbr != xj:
                        queue.append((nbr, xi))
        return True

    def _revise(self, domains: Dict[Tuple[str, int], List[str]], xi: Tuple[str, int], xj: Tuple[str, int]) -> bool:
        """
        Merevisi domain xi berdasarkan batasan terhadap xj.
        """
        revised = False
        s_id1, d1 = xi
        s_id2, d2 = xj

        # Aturan spesifik antara hari d1 dan d2 (apabila d2 = d1 + 1)
        if d2 == d1 + 1:
            for x in list(domains[xi]):
                # Jika xi = Malam, maka xj harus Off
                if x == 'Malam':
                    if not any(y == 'Off' for y in domains[xj]):
                        domains[xi].remove(x)
                        revised = True
        return revised

    def select_unassigned_variable_mrv(
        self, assignment: Dict[Tuple[str, int], str], domains: Dict[Tuple[str, int], List[str]]
    ) -> Tuple[str, int]:
        """
        Heuristik Minimum Remaining Values (MRV): Memilih variabel dengan sisa domain terkecil.
        """
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: len(domains[v]))

    def backtrack(
        self, assignment: Dict[Tuple[str, int], str], domains: Dict[Tuple[str, int], List[str]]
    ) -> Optional[Dict[Tuple[str, int], str]]:
        """
        Algoritma Backtracking Search dengan Propagasi AC-3 dan MRV.
        """
        if len(assignment) == len(self.variables):
            return assignment  # Seluruh variabel telah terisi (Solusi Ditemukan)

        var = self.select_unassigned_variable_mrv(assignment, domains)
        
        for value in domains[var]:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                
                # Buat salinan domain untuk backtracking
                local_domains = copy.deepcopy(domains)
                local_domains[var] = [value]
                
                # Jalankan AC-3 pasca penugasan nilai
                if self.ac3(local_domains):
                    result = self.backtrack(assignment, local_domains)
                    if result is not None:
                        return result
                        
                del assignment[var]  # Backtrack

        return None

    def solve(self) -> Optional[Dict[Tuple[str, int], str]]:
        """
        Fungsi utama untuk menjalankan solver CSP.
        """
        initial_domains = copy.deepcopy(self.domains)
        if not self.ac3(initial_domains):
            return None  # Terbukti tidak memiliki solusi dari awal
            
        return self.backtrack({}, initial_domains)


if __name__ == "__main__":
    # Data Simulasi Staf CS BeliCepat.id
    sample_staff = [
        {'id': 'S01', 'nama': 'Adithya', 'skills': ['Order', 'Refund']},
        {'id': 'S02', 'nama': 'Immanuel', 'skills': ['Teknis', 'Order']},
        {'id': 'S03', 'nama': 'Mutiara', 'skills': ['Refund', 'Teknis']},
        {'id': 'S04', 'nama': 'Budi', 'skills': ['Order']},
    ]

    solver = CSShiftSolver(staff_data=sample_staff, days=5)
    solution = solver.solve()

    if solution:
        print("=== JADWAL SHIFT CS BELICEPAT.ID (SOLUSI CSP VALID) ===")
        for day in range(1, 6):
            print(f"\nHari {day}:")
            for staff in sample_staff:
                shift = solution.get((staff['id'], day))
                print(f"  - {staff['nama']} ({staff['id']}): Shift {shift}")
    else:
        print("Tidak ditemukan solusi yang memenuhi seluruh batasan (Unsatisfiable).")