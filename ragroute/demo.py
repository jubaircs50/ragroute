"""Small smoke-test examples, not a research benchmark."""
DEMO_DOCUMENTS = [
 {"doc_id":"demo-1","title":"BM25","text":"BM25 is a lexical information retrieval algorithm that ranks documents using query-term matching and term-frequency statistics."},
 {"doc_id":"demo-2","title":"Dense retrieval","text":"Dense retrieval encodes queries and documents into vectors and ranks documents using embedding similarity."},
 {"doc_id":"demo-3","title":"Hybrid retrieval","text":"Hybrid retrieval combines lexical and dense retrieval for exact term matching and semantic similarity."},
 {"doc_id":"demo-4","title":"Cross-encoder reranking","text":"A cross-encoder jointly scores a query and candidate document pair. It can improve ranking quality but costs more latency."},
]
DEMO_QUERIES = [
 {"query_id":"demo-q1","text":"How does BM25 rank documents?","relevance":{"demo-1":1.0}},
 {"query_id":"demo-q2","text":"What does dense retrieval use?","relevance":{"demo-2":1.0}},
 {"query_id":"demo-q3","text":"How can lexical and semantic search be combined?","relevance":{"demo-3":1.0}},
]
