"""BM25, dense, hybrid RRF and heuristic adaptive retrieval."""
from __future__ import annotations
import re, time
from typing import Any
import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:  # Lightweight fallback for tests before installing requirements.
    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: list[list[str]]) -> None:
            self.corpus = corpus
            self.doc_freq: dict[str, int] = {}
            for tokens in corpus:
                for token in set(tokens):
                    self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
            self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            import math
            n, k1, b = len(self.corpus), 1.5, 0.75
            scores = []
            for tokens in self.corpus:
                tf: dict[str, int] = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1
                score = 0.0
                for term in query_tokens:
                    freq = tf.get(term, 0)
                    if not freq:
                        continue
                    df = self.doc_freq.get(term, 0)
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    denom = freq + k1 * (1 - b + b * len(tokens) / max(self.avgdl, 1e-9))
                    score += idf * freq * (k1 + 1) / denom
                scores.append(score)
            return scores
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())

class RetrievalEngine:
    def __init__(self, documents: list[dict[str, str]], embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        if not documents: raise ValueError("At least one document is required.")
        self.documents, seen = [], set()
        for i, raw in enumerate(documents):
            doc_id = str(raw.get("doc_id", raw.get("_id", f"doc-{i}")))
            if doc_id in seen: raise ValueError(f"Duplicate document ID: {doc_id}")
            seen.add(doc_id)
            self.documents.append({"doc_id":doc_id,"title":str(raw.get("title", "") or ""),"text":str(raw.get("text", "") or "")})
        self._by_id = {d["doc_id"]: d for d in self.documents}
        self._bm25 = BM25Okapi([tokenize(self._text(d)) for d in self.documents])
        self.embedding_model_name, self.reranker_model_name = embedding_model_name, reranker_model_name
        self._dense_model = self._reranker = self._embeddings = None

    @staticmethod
    def _text(doc: dict[str, str]) -> str: return f'{doc.get("title", "")} {doc.get("text", "")}'.strip()
    @staticmethod
    def choose_adaptive_mode(query: str) -> str:
        # Simple baseline heuristic, not a learned routing policy.
        return "bm25" if len(tokenize(query)) <= 3 or any(c.isdigit() for c in query) else "hybrid"
    def _ensure_dense(self) -> None:
        if self._dense_model is None:
            try: from sentence_transformers import SentenceTransformer
            except ImportError as exc: raise RuntimeError("Dense retrieval requires sentence-transformers.") from exc
            self._dense_model = SentenceTransformer(self.embedding_model_name)
        if self._embeddings is None:
            self._embeddings = np.asarray(self._dense_model.encode([self._text(d) for d in self.documents], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    def _result(self, doc: dict[str, str], score: float, rank: int) -> dict[str, Any]: return {**doc, "score":score, "rank":rank}
    def _bm25_search(self, query: str, k: int) -> list[dict[str, Any]]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        ids = np.argsort(-scores)[:k]
        return [self._result(self.documents[int(i)], float(scores[int(i)]), rank) for rank, i in enumerate(ids, 1)]
    def _dense_search(self, query: str, k: int) -> list[dict[str, Any]]:
        self._ensure_dense()
        qvec = np.asarray(self._dense_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)
        scores = self._embeddings @ qvec
        ids = np.argsort(-scores)[:k]
        return [self._result(self.documents[int(i)], float(scores[int(i)]), rank) for rank, i in enumerate(ids, 1)]
    def _rrf(self, rankings: list[list[dict[str, Any]]], k: int, rrf_k: int = 60) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for item in ranking: scores[item["doc_id"]] = scores.get(item["doc_id"], 0.0) + 1.0 / (rrf_k + item["rank"])
        ids = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self._result(self._by_id[d], scores[d], rank) for rank, d in enumerate(ids, 1)]
    def _rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._reranker is None:
            try: from sentence_transformers import CrossEncoder
            except ImportError as exc: raise RuntimeError("Reranking requires sentence-transformers.") from exc
            self._reranker = CrossEncoder(self.reranker_model_name)
        scores = np.asarray(self._reranker.predict([(query, self._text(d)) for d in results])).reshape(-1)
        ordered = [dict(results[int(i)]) for i in np.argsort(-scores)]
        for rank, item in enumerate(ordered, 1): item["score"], item["rank"] = float(scores[np.argsort(-scores)[rank-1]]), rank
        return ordered
    def search(self, query: str, mode: str = "adaptive", top_k: int = 5, rerank: bool = False) -> dict[str, Any]:
        if not query.strip(): raise ValueError("Query cannot be empty.")
        if top_k < 1: raise ValueError("top_k must be at least 1.")
        if mode not in {"bm25", "dense", "hybrid", "adaptive"}: raise ValueError(f"Unsupported mode: {mode}")
        start = time.perf_counter(); effective = self.choose_adaptive_mode(query) if mode == "adaptive" else mode
        k = min(max(top_k * 3, top_k), len(self.documents))
        if effective == "bm25": results = self._bm25_search(query, k if rerank else top_k)
        elif effective == "dense": results = self._dense_search(query, k if rerank else top_k)
        else: results = self._rrf([self._bm25_search(query, k), self._dense_search(query, k)], k if rerank else top_k)
        if rerank: results = self._rerank(query, results)[:top_k]
        for rank, item in enumerate(results[:top_k], 1): item["rank"] = rank
        return {"query":query,"requested_mode":mode,"effective_mode":effective,"reranked":rerank,"latency_ms":round((time.perf_counter()-start)*1000, 3),"results":results[:top_k]}
