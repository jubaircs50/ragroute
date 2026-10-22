import pytest
from ragroute.metrics import retrieval_metrics

def test_recall_precision_mrr_ndcg():
    m = retrieval_metrics(["d1","d2","d3"], {"d2":1,"d3":1}, k=2)
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["mrr"] == 0.5
    assert 0 <= m["ndcg"] <= 1

def test_no_judgments():
    assert all(v is None for v in retrieval_metrics(["d1"], {}, 5).values())

def test_perfect_ranking():
    assert retrieval_metrics(["d1","d2"], {"d1":1,"d2":1}, 2) == {"recall":1.0,"precision":1.0,"mrr":1.0,"ndcg":1.0}

def test_invalid_k():
    with pytest.raises(ValueError): retrieval_metrics([], {}, 0)
