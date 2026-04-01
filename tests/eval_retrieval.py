"""Retrieval evaluation for nl_query_patients.

Metrics computed per query (top-K results, default K=10):
  - Precision@K  — fraction of top-K that are relevant
  - Recall@K     — fraction of relevant set found in top-K
  - AP@K         — average precision (area under precision-recall curve)
  - NDCG@K       — normalised discounted cumulative gain (binary relevance)

Aggregate:
  - MAP@K        — mean average precision across queries
  - Mean Recall@K
  - Mean NDCG@K

Usage:
    python -m tests.eval_retrieval [--mode auto|vector|hybrid] [--k 10] [--query-file tests/eval_queries.json]

Only queries with at least one `relevant_ids` entry are evaluated; others are skipped.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ddm.query_engine import query_patients


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top = ranked_ids[:k]
    return sum(1 for r in top if r in relevant) / k if k else 0.0


def _recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def _ap_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Average precision at K."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(len(relevant), k)


def _dcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def _ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    dcg = _dcg_at_k(ranked_ids, relevant, k)
    # Ideal DCG: all relevant docs at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate(query_file: str, search_mode: str, k: int) -> None:
    with open(query_file) as f:
        data = json.load(f)

    queries = data["queries"]
    evaluable = [q for q in queries if q.get("relevant_ids")]
    skipped = [q["id"] for q in queries if not q.get("relevant_ids")]

    if skipped:
        print(f"Skipping {len(skipped)} unlabeled queries: {skipped}\n")

    if not evaluable:
        print("No labeled queries found. Populate relevant_ids in eval_queries.json first.")
        return

    results = []
    for entry in evaluable:
        qid = entry["id"]
        question = entry["question"]
        relevant = set(entry["relevant_ids"])

        print(f"[{qid}] {question!r} ({len(relevant)} relevant) … ", end="", flush=True)
        try:
            result = await query_patients(question, search_mode=search_mode)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        ranked_ids = [p["id"] for p in result.patients]

        p_k = _precision_at_k(ranked_ids, relevant, k)
        r_k = _recall_at_k(ranked_ids, relevant, k)
        ap_k = _ap_at_k(ranked_ids, relevant, k)
        ndcg_k = _ndcg_at_k(ranked_ids, relevant, k)

        print(
            f"mode={result.mode}  n={result.count}  "
            f"P@{k}={p_k:.3f}  R@{k}={r_k:.3f}  "
            f"AP@{k}={ap_k:.3f}  NDCG@{k}={ndcg_k:.3f}"
        )
        results.append({
            "id": qid,
            "question": question,
            "mode": result.mode,
            "n_retrieved": result.count,
            f"P@{k}": p_k,
            f"R@{k}": r_k,
            f"AP@{k}": ap_k,
            f"NDCG@{k}": ndcg_k,
        })

    if not results:
        print("No results to aggregate.")
        return

    map_k = sum(r[f"AP@{k}"] for r in results) / len(results)
    mean_r_k = sum(r[f"R@{k}"] for r in results) / len(results)
    mean_ndcg_k = sum(r[f"NDCG@{k}"] for r in results) / len(results)

    print(f"\n{'─' * 60}")
    print(f"Evaluated {len(results)} queries  |  search_mode={search_mode}  |  K={k}")
    print(f"  MAP@{k}         = {map_k:.4f}")
    print(f"  Mean Recall@{k} = {mean_r_k:.4f}")
    print(f"  Mean NDCG@{k}   = {mean_ndcg_k:.4f}")
    print(f"{'─' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate nl_query_patients retrieval quality")
    parser.add_argument("--mode", choices=["auto", "vector", "hybrid"], default="auto")
    parser.add_argument("--k", type=int, default=10, help="Cut-off rank (default 10)")
    parser.add_argument(
        "--query-file",
        default=str(Path(__file__).parent / "eval_queries.json"),
        help="Path to eval_queries.json",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(args.query_file, args.mode, args.k))


if __name__ == "__main__":
    main()
