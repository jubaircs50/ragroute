"""Recall@k, Precision@k, MRR@k and NDCG@k."""
from __future__ import annotations
import math

def retrieval_metrics(ranked_doc_ids: list[str], relevance: dict[str, float], k: int = 10) -> dict[str, float | None]:
    if k < 1:
        raise ValueError("k must be at least 1.")
    rel = {str(d): float(s) for d, s in relevance.items() if float(s) > 0}
    if not rel:
        return {"recall": None, "precision": None, "mrr": None, "ndcg": None}
    ranked = [str(x) for x in ranked_doc_ids[:k]]
    hits = {d for d in ranked if d in rel}
    recall, precision = len(hits) / len(rel), len(hits) / k
    mrr = next((1.0 / rank for rank, d in enumerate(ranked, 1) if d in rel), 0.0)
    gain = lambda score: 2.0 ** score - 1.0
    dcg = sum(gain(rel.get(d, 0.0)) / math.log2(rank + 1) for rank, d in enumerate(ranked, 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(gain(s) / math.log2(rank + 1) for rank, s in enumerate(ideal, 1))
    return {"recall": recall, "precision": precision, "mrr": mrr, "ndcg": dcg / idcg if idcg else 0.0}
