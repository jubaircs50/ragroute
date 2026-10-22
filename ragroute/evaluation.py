"""Compare retrieval modes over judged queries."""
from __future__ import annotations
import statistics
from typing import Any
from .metrics import retrieval_metrics
from .retrieval import RetrievalEngine
MODES = ["bm25", "dense", "hybrid", "adaptive"]
def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.mean(clean) if clean else None

def evaluate_retrievers(engine: RetrievalEngine, queries: list[dict[str, Any]], modes: list[str] | None = None, top_k: int = 10, rerank: bool = False) -> list[dict[str, Any]]:
    rows = []
    for mode in (modes or MODES):
        metrics = {key: [] for key in ("recall", "precision", "mrr", "ndcg")}; latencies = []
        for item in queries:
            result = engine.search(str(item.get("text", item.get("query", ""))), mode=mode, top_k=top_k, rerank=rerank)
            latencies.append(result["latency_ms"])
            values = retrieval_metrics([d["doc_id"] for d in result["results"]], item.get("relevance", {}), top_k)
            for key in metrics: metrics[key].append(values[key])
        row = {"mode":mode,"queries":len(queries),"top_k":top_k,"mean_latency_ms":_mean(latencies)}
        row.update({f"mean_{key}":_mean(vals) for key, vals in metrics.items()}); rows.append(row)
    return rows
