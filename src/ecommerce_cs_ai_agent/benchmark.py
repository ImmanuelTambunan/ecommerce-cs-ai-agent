"""
benchmark.py
------------
Skrip tolok ukur (benchmark) temu balik semantik untuk mengevaluasi:
1. Kueri relevan langsung (Direct In-Domain Query)
2. Kueri sinonim / bahasa percakapan (Synonym / Colloquial Query)
3. Kueri ambigu (Ambiguous Multi-topic Query)
4. Kueri out-of-domain 1 (OOD Query: Resep Masakan)
5. Kueri out-of-domain 2 (OOD Query: Otomotif)

Menghitung skor Cosine Similarity, mengevaluasi presisi penolakan OOD,
serta mencetak tabel komparasi untuk bahan Laporan Teknis Milestone 3.
"""

from typing import List, Dict, Any
from ecommerce_cs_ai_agent.search_engine import _load_chroma_collection, embed_query


BENCHMARK_QUERIES = [
    {
        "id": "Q1_RELEVAN",
        "jenis": "Kueri Relevan Langsung",
        "query": "Bagaimana prosedur dan syarat pengajuan refund dana?",
        "expected_docs": ["sop_refund", "faq_umum", "kebijakan_keluhan"],
        "is_ood": False,
    },
    {
        "id": "Q2_SINONIM",
        "jenis": "Kueri Sinonim / Percakapan",
        "query": "Baju seragam yang kemarin sampai kekecilan, apakah bisa ditukar ukurannya?",
        "expected_docs": ["sop_tukar_ukuran"],
        "is_ood": False,
    },
    {
        "id": "Q3_AMBIGU",
        "jenis": "Kueri Ambigu (Multi-topik: Garansi/Pengiriman/Keluhan)",
        "query": "Pesanan saya bermasalah, barang cacat dan pengiriman terlambat, bagaimana solusinya?",
        "expected_docs": ["kebijakan_garansi", "kebijakan_keluhan", "sop_pengiriman", "sop_barang_kurang"],
        "is_ood": False,
    },
    {
        "id": "Q4_OOD_1",
        "jenis": "Kueri Out-of-Domain (Kuliner)",
        "query": "Bagaimana resep dan takaran membuat kue bolu pandan yang empuk?",
        "expected_docs": [],
        "is_ood": True,
    },
    {
        "id": "Q5_OOD_2",
        "jenis": "Kueri Out-of-Domain (Tanaman/Flora)",
        "query": "Bagaimana cara merawat tanaman hias monstera agar daunnya lebat dan subur?",
        "expected_docs": [],
        "is_ood": True,
    },
]


def run_benchmark(similarity_threshold: float = 0.45) -> List[Dict[str, Any]]:
    """Menjalankan benchmark 5 kueri bisnis dan menganalisis performa retrieval."""
    collection = _load_chroma_collection()
    results_summary = []

    print("=" * 80)
    print("BENCHMARK TEMU BALIK SEMANTIK & OUT-OF-DOMAIN (OOD) FILTERING")
    print(f"Model Embedding   : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print(f"Metrik Similarity : Cosine Similarity (1.0 - Cosine Distance)")
    print(f"Ambang Batas (OOD): {similarity_threshold:.2f}")
    print("=" * 80)

    for item in BENCHMARK_QUERIES:
        q_text = item["query"]
        q_vec = embed_query(q_text)

        raw = collection.query(
            query_embeddings=[q_vec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        top_docs = [m["doc_id"] for m in raw["metadatas"][0]] if raw["metadatas"] else []
        top_doc_id = top_docs[0] if top_docs else "None"
        raw_dist = raw["distances"][0][0] if raw["distances"][0] else 1.0
        cosine_sim = round(1.0 - raw_dist, 4)

        # Evaluasi threshold
        is_accepted = cosine_sim >= similarity_threshold
        if item["is_ood"]:
            passed = not is_accepted  # Harus ditolak
        else:
            passed = is_accepted and any(d in item["expected_docs"] for d in top_docs)

        status_str = "DITERIMA" if is_accepted else "DITOLAK (OOD)"
        eval_str = "BERHASIL" if passed else "GAGAL"

        res_entry = {
            "id": item["id"],
            "jenis": item["jenis"],
            "query": q_text,
            "top_doc": top_doc_id,
            "top_docs": top_docs,
            "cosine_sim": cosine_sim,
            "threshold": similarity_threshold,
            "status": status_str,
            "passed": passed,
        }
        results_summary.append(res_entry)

        print(f"\n[{item['id']}] {item['jenis']}")
        print(f"  Kueri          : \"{q_text}\"")
        print(f"  Top Match Dok  : {top_doc_id} (Top 3: {', '.join(top_docs)})")
        print(f"  Cosine Sim     : {cosine_sim:.4f}")
        print(f"  Keputusan      : {status_str} (Threshold = {similarity_threshold})")
        print(f"  Evaluasi       : [{eval_str}]")

    print("\n" + "=" * 80)
    print("TABEL RINGKASAN BENCHMARK EVALUASI")
    print("=" * 80)
    print(f"{'ID':<12} | {'Cosine Sim':<10} | {'Status':<14} | {'Top Doc':<20} | {'Evaluasi':<10}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['id']:<12} | {r['cosine_sim']:<10.4f} | {r['status']:<14} | {r['top_doc']:<20} | {'PASS' if r['passed'] else 'FAIL':<10}")
    print("=" * 80)

    return results_summary


if __name__ == "__main__":
    run_benchmark(similarity_threshold=0.45)
