import pytest
from ecommerce_cs_ai_agent.solver import CSShiftSolver

def test_csp_solver_basic():
    staff_data = [
        {'id': 'S01', 'nama': 'Adithya', 'skills': ['Order']},
        {'id': 'S02', 'nama': 'Immanuel', 'skills': ['Teknis']},
        {'id': 'S03', 'nama': 'Mutiara', 'skills': ['Refund']}
    ]

    solver = CSShiftSolver(staff_data, days=3)
    sol = solver.solve()
    assert sol is not None
    assert len(sol) == len(staff_data) * 3

def test_night_shift_rest_constraint():
    staff_data = [{'id': 'S01', 'nama': 'Adithya', 'skills': ['Order']}]
    solver = CSShiftSolver(staff_data, days=2)
    sol = solver.solve()

    if sol and sol.get(('S01', 1)) == 'Malam':
        assert sol.get(('S01', 2)) == 'Off'