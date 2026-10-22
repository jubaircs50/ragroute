import pytest
from ragroute.data import attach_qrels_tsv, load_corpus_jsonl, load_queries_jsonl, read_jsonl

def test_load_corpus():
    d = load_corpus_jsonl('{"_id":"d1","title":"Test","text":"Passage"}\n')
    assert d[0]["doc_id"] == "d1" and d[0]["title"] == "Test"

def test_duplicate_doc_ids():
    with pytest.raises(ValueError): load_corpus_jsonl('{"_id":"d1","text":"a"}\n{"_id":"d1","text":"b"}')

def test_attach_qrels():
    qs = load_queries_jsonl('{"_id":"q1","text":"RAG?"}\n{"_id":"q2","text":"BM25?"}')
    out = attach_qrels_tsv(qs, 'query-id\tcorpus-id\tscore\nq1\td2\t1\nq1\td3\t1\nq2\td1\t2\n')
    assert out[0]["relevance"] == {"d2":1.0,"d3":1.0}

def test_bad_jsonl_line_number():
    with pytest.raises(ValueError, match="line 2"): read_jsonl('{"ok":true}\nbad-json')
