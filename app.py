"""Streamlit dashboard for RAGRoute."""
from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from ragroute.data import attach_qrels_tsv, load_corpus_jsonl, load_queries_jsonl
from ragroute.demo import DEMO_DOCUMENTS, DEMO_QUERIES
from ragroute.evaluation import MODES, evaluate_retrievers
from ragroute.generation import generate_grounded_answer
from ragroute.retrieval import RetrievalEngine
load_dotenv()
st.set_page_config(page_title="RAGRoute", page_icon="🧭", layout="wide")
st.title("🧭 RAGRoute")
st.caption("Adaptive retrieval and evidence-grounded QA — runnable MVP")
if "documents" not in st.session_state: st.session_state.documents = DEMO_DOCUMENTS
if "queries" not in st.session_state: st.session_state.queries = DEMO_QUERIES
if "engine" not in st.session_state: st.session_state.engine = None
with st.sidebar:
    st.header("Corpus")
    source = st.radio("Data source", ["Built-in smoke test", "Upload BEIR-format files"])
    if source == "Upload BEIR-format files":
        cf = st.file_uploader("corpus.jsonl", type=["jsonl"])
        qf = st.file_uploader("queries.jsonl", type=["jsonl"])
        rf = st.file_uploader("qrels/test.tsv", type=["tsv", "txt"])
        if st.button("Load benchmark", type="primary"):
            if not (cf and qf and rf): st.error("Upload all three files.")
            else:
                try:
                    docs = load_corpus_jsonl(cf.getvalue().decode("utf-8-sig"))
                    qs = attach_qrels_tsv(load_queries_jsonl(qf.getvalue().decode("utf-8-sig")), rf.getvalue().decode("utf-8-sig"))
                    st.session_state.documents, st.session_state.queries, st.session_state.engine = docs, qs, None
                    st.success(f"Loaded {len(docs)} documents and {len(qs)} queries.")
                except Exception as exc: st.error(str(exc))
    st.write(f"Documents: **{len(st.session_state.documents)}** · Queries: **{len(st.session_state.queries)}**")
docs = st.session_state.documents
if st.session_state.engine is None:
    try: st.session_state.engine = RetrievalEngine(docs)
    except Exception as exc: st.error(str(exc)); st.stop()
engine: RetrievalEngine = st.session_state.engine
search_tab, eval_tab, answer_tab, data_tab = st.tabs(["Search", "Evaluation", "Grounded answer", "Data"])
with search_tab:
    query = st.text_input("Query", "How does hybrid retrieval combine search?")
    c1,c2,c3 = st.columns(3)
    mode = c1.selectbox("Mode", MODES, index=3)
    k = c2.slider("Top K", 1, min(20,len(docs)), min(5,len(docs)))
    rerank = c3.checkbox("Cross-encoder rerank (model download)")
    if st.button("Search", type="primary"):
        try:
            result = engine.search(query, mode, k, rerank)
            st.caption(f"Effective mode: {result['effective_mode']} · {result['latency_ms']} ms")
            for item in result["results"]:
                with st.container(border=True):
                    st.markdown(f"**{item['rank']}. {item['title'] or item['doc_id']}**")
                    st.caption(f"ID: `{item['doc_id']}` · score: {item['score']:.4f}")
                    st.write(item["text"])
        except Exception as exc: st.error(f"Search failed: {exc}")
with eval_tab:
    st.info("Built-in examples are smoke tests, not a research benchmark. Dense retrieval downloads a model on first use.")
    ek = st.slider("Evaluation K", 1, min(10,len(docs)), min(5,len(docs)), key="evalk")
    modes = st.multiselect("Modes to compare", MODES, default=["bm25"])
    if st.button("Run evaluation", type="primary"):
        try:
            rows = evaluate_retrievers(engine, st.session_state.queries, modes, ek)
            df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "ragroute-results.csv", "text/csv")
        except Exception as exc: st.error(f"Evaluation failed: {exc}")
with answer_tab:
    question = st.text_input("Question", "What is hybrid retrieval?")
    gm = st.selectbox("Retrieval mode", MODES, index=3, key="generationmode")
    gk = st.slider("Evidence passages", 1, min(10,len(docs)), min(4,len(docs)), key="gk")
    if st.button("Retrieve and generate", type="primary"):
        try:
            found = engine.search(question, gm, gk)["results"]
            st.markdown("### Retrieved evidence")
            for d in found: st.markdown(f"**[{d['doc_id']}] {d['title']}**\n\n{d['text']}")
            output = generate_grounded_answer(base_url=os.getenv("OPENAI_BASE_URL", ""), api_key=os.getenv("OPENAI_API_KEY", ""), model=os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"), query=question, retrieved_documents=found)
            st.markdown("### Answer"); st.write(output["answer"])
            st.caption(f"Valid citation IDs: {output['citations']} · Invalid IDs: {output['invalid_citations']}")
        except Exception as exc: st.error(f"Generation failed: {exc}")
with data_tab:
    st.subheader("Documents"); st.dataframe(pd.DataFrame(docs), use_container_width=True)
    st.subheader("Queries"); st.dataframe(pd.DataFrame([{"query_id":q["query_id"],"text":q["text"],"judged_documents":len(q.get("relevance",{}))} for q in st.session_state.queries]), use_container_width=True)
