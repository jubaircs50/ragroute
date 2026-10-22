"""Evidence-grounded generation with structural citation-ID validation."""
from __future__ import annotations
import re, time
from typing import Any
from openai import OpenAI
CITATION_PATTERN = re.compile(r"\[\[([^\[\]]+)\]\]")
def generate_grounded_answer(*, base_url: str, api_key: str, model: str, query: str, retrieved_documents: list[dict[str, Any]], input_price_per_million: float | None = None, output_price_per_million: float | None = None) -> dict[str, Any]:
    if not query.strip(): raise ValueError("Query cannot be empty.")
    if not retrieved_documents: return {"answer":"I could not find retrieved evidence to answer this question.","citations":[],"invalid_citations":[],"latency_ms":0.0,"estimated_cost":None,"usage":{}}
    if not api_key.strip(): raise ValueError("Set OPENAI_API_KEY in .env before generating answers.")
    valid_ids, blocks = set(), []
    for i, doc in enumerate(retrieved_documents, 1):
        did = str(doc.get("doc_id", f"doc-{i}")); valid_ids.add(did)
        blocks.append(f"DOCUMENT ID: {did}\nTITLE: {doc.get('title','')}\nPASSAGE: {doc.get('text','')}")
    client = OpenAI(api_key=api_key, base_url=base_url or None); start = time.perf_counter()
    response = client.chat.completions.create(model=model, temperature=0.0, messages=[
        {"role":"system","content":"Answer only from supplied evidence. If insufficient, say so. Cite factual claims with exact IDs in [[doc_id]] format. Never invent IDs."},
        {"role":"user","content":f"QUESTION:\n{query}\n\nEVIDENCE:\n" + "\n\n---\n\n".join(blocks)}])
    elapsed = (time.perf_counter()-start)*1000; answer = response.choices[0].message.content or ""
    found = list(dict.fromkeys(CITATION_PATTERN.findall(answer))); usage_obj = response.usage
    usage = {"prompt_tokens":getattr(usage_obj,"prompt_tokens",None),"completion_tokens":getattr(usage_obj,"completion_tokens",None),"total_tokens":getattr(usage_obj,"total_tokens",None)}
    cost = None
    if input_price_per_million is not None and output_price_per_million is not None and usage["prompt_tokens"] is not None and usage["completion_tokens"] is not None:
        cost = (usage["prompt_tokens"]*input_price_per_million + usage["completion_tokens"]*output_price_per_million)/1_000_000
    return {"answer":answer,"citations":[x for x in found if x in valid_ids],"invalid_citations":[x for x in found if x not in valid_ids],"latency_ms":round(elapsed,3),"estimated_cost":cost,"usage":usage,"note":"Citation IDs are checked structurally; semantic support is not verified."}
