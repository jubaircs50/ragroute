# RAGRoute

Runnable MVP for adaptive retrieval and evidence-grounded question answering.

## Features
- BM25 lexical retrieval
- Dense retrieval with Sentence Transformers
- Hybrid retrieval using Reciprocal Rank Fusion
- Transparent heuristic adaptive routing (not a learned policy)
- Optional cross-encoder reranking
- OpenAI-compatible grounded answer generation with structural citation-ID checks
- Recall@k, Precision@k, MRR@k, NDCG@k
- Streamlit dashboard, FastAPI endpoint, BEIR-format data loaders and tests

## Windows setup
Python 3.11+ recommended. From this folder in VS Code Terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
```

If PowerShell activation is blocked, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\python.exe -m pytest -q`.

## Run dashboard
```powershell
streamlit run app.py
```
The built-in documents and queries are smoke tests only. Dense/hybrid modes download an embedding model on first use and may need substantial memory for large corpora.

## Run API
```powershell
uvicorn api:app --reload
```
API docs: http://127.0.0.1:8000/docs. Default API corpus is the built-in demo. To use a BEIR corpus file, set `RAGROUTE_CORPUS_PATH` in `.env` to the full path of `corpus.jsonl`.

## BEIR file format
- `corpus.jsonl`: each line has `_id`, optional `title`, and `text`
- `queries.jsonl`: each line has `_id` and `text`
- `qrels/test.tsv`: tab-separated `query-id`, `corpus-id`, `score`

Upload the three files in the dashboard. Avoid tuning on the same test queries used for final reporting.

## Generation
Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in `.env`. The base URL/model must match your provider. No key is bundled in this project.

## Limitations
Adaptive routing is a transparent heuristic; it is not learned or novel by itself. Citation IDs are validated structurally, not for semantic entailment. Benchmark claims require fixed protocols, repeated runs, uncertainty estimates and failure analysis.
