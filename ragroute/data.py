"""Load JSONL documents/queries and BEIR-style qrels TSV."""
from __future__ import annotations
import csv, io, json
from typing import Any


def read_jsonl(content: str) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object.")
        records.append(record)
    return records


def load_corpus_jsonl(content: str) -> list[dict[str, str]]:
    docs, seen = [], set()
    for i, item in enumerate(read_jsonl(content), 1):
        raw_id = item.get("_id", item.get("doc_id"))
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(f"Corpus record {i} has no _id.")
        doc_id = str(raw_id)
        if doc_id in seen:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        seen.add(doc_id)
        title, text = str(item.get("title", "") or ""), str(item.get("text", "") or "")
        if not (title.strip() or text.strip()):
            raise ValueError(f"Document {doc_id} has neither title nor text.")
        docs.append({"doc_id": doc_id, "title": title, "text": text})
    return docs


def load_queries_jsonl(content: str) -> list[dict[str, Any]]:
    queries, seen = [], set()
    for i, item in enumerate(read_jsonl(content), 1):
        raw_id = item.get("_id", item.get("query_id"))
        query_text = item.get("text", item.get("query", ""))
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(f"Query record {i} has no _id.")
        qid = str(raw_id)
        if qid in seen:
            raise ValueError(f"Duplicate query ID: {qid}")
        if not str(query_text).strip():
            raise ValueError(f"Query {qid} has empty text.")
        seen.add(qid)
        queries.append({"query_id": qid, "text": str(query_text), "relevance": {}})
    return queries


def attach_qrels_tsv(queries: list[dict[str, Any]], content: str) -> list[dict[str, Any]]:
    by_id = {str(q["query_id"]): {**q, "relevance": dict(q.get("relevance", {}))} for q in queries}
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    if not reader.fieldnames:
        raise ValueError("qrels TSV is empty.")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    qcol, dcol, scol = cols.get("query-id") or cols.get("query_id"), cols.get("corpus-id") or cols.get("corpus_id"), cols.get("score")
    if not (qcol and dcol and scol):
        raise ValueError("qrels header must contain query-id, corpus-id, and score.")
    for line, row in enumerate(reader, 2):
        qid, did = str(row.get(qcol, "")).strip(), str(row.get(dcol, "")).strip()
        try:
            score = float(row.get(scol, ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid qrels score on line {line}.") from exc
        if not qid or not did:
            raise ValueError(f"Missing query/document ID on qrels line {line}.")
        if qid in by_id:
            by_id[qid]["relevance"][did] = score
    return [by_id[str(q["query_id"])] for q in queries]
