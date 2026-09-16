"""
Baseline Search - Uniform Cost Search (UCS)
Proyek: Enterprise AI Copilot untuk Otomasi Customer Service E-Commerce
Kasus: Pencarian jalur penanganan tiket dengan biaya (waktu) minimum
"""

import heapq

# Representasi graf: {status: [(status_tujuan, biaya), ...]}
graph = {
    "Diterima": [("Level_1", 5)],
    "Level_1": [("Selesai", 10), ("Level_2", 15)],
    "Level_2": [("Selesai", 12), ("Supervisor", 20)],
    "Supervisor": [("Selesai", 8)],
    "Selesai": []
}


def uniform_cost_search(graph, start, goal):
    """
    Mencari jalur dengan biaya (cost) minimum dari start ke goal
    menggunakan Uniform Cost Search dengan priority queue (heapq).
    """
    frontier = [(0, start, [start])]  # (biaya_kumulatif, node_saat_ini, jalur)
    visited = set()

    while frontier:
        cost, current, path = heapq.heappop(frontier)

        if current == goal:
            return path, cost

        if current in visited:
            continue
        visited.add(current)

        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                heapq.heappush(frontier, (cost + weight, neighbor, path + [neighbor]))

    return None, float("inf")


if __name__ == "__main__":
    path, total_cost = uniform_cost_search(graph, "Diterima", "Selesai")
    print(f"Jalur penanganan tiket optimal: {' -> '.join(path)}")
    print(f"Total estimasi waktu penanganan: {total_cost} menit")