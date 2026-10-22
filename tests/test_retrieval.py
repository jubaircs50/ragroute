import pytest
from ragroute.retrieval import RetrievalEngine
DOCS = [
 {"doc_id":"d1","title":"BM25","text":"BM25 ranks documents by lexical term matching."},
 {"doc_id":"d2","title":"Dense retrieval","text":"Dense retrieval uses vector embeddings."},
 {"doc_id":"d3","title":"Hybrid retrieval","text":"Hybrid retrieval combines lexical and semantic search."},
]
def test_bm25_rank():
    r = RetrievalEngine(DOCS).search("BM25 lexical term matching", "bm25", 2)
    assert r["effective_mode"] == "bm25" and len(r["results"]) == 2 and r["results"][0]["doc_id"] == "d1"
def test_adaptive_short_query():
    assert RetrievalEngine(DOCS).search("BM25", "adaptive", 2)["effective_mode"] == "bm25"
def test_invalid_inputs():
    e = RetrievalEngine(DOCS)
    with pytest.raises(ValueError): e.search("query", "unknown")
    with pytest.raises(ValueError): e.search("  ")
def test_duplicate_ids():
    with pytest.raises(ValueError): RetrievalEngine([{"doc_id":"x","text":"a"},{"doc_id":"x","text":"b"}])
