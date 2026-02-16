import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import local

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:
    CrossEncoder = None
    SentenceTransformer = None


CITATION_RE = re.compile(r"\[(\d+):([^\]]+)\]")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def norm_scores(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn + 1e-12)


@dataclass
class RetrievalArtifacts:
    bm25: BM25Okapi
    bm25_tokens: list[list[str]]
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: any
    dense_model: any
    dense_matrix: np.ndarray
    rerank_model: any


def build_artifacts(corpus: list[dict], dense_model_name: str, rerank_model_name: str) -> RetrievalArtifacts:
    if BM25Okapi is None:
        raise RuntimeError("Missing dependency rank_bm25. Please run: pip install -r requirements.txt")
    texts = [normalize_text(c["text"]) for c in corpus]
    bm25_tokens = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(bm25_tokens)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    if SentenceTransformer is not None:
        dense_model = SentenceTransformer(dense_model_name)
        dense_matrix = dense_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        dense_model = None
        # Fallback: use TF-IDF latent-like dense representation
        dense_matrix = tfidf_matrix.toarray()
        row_norm = np.linalg.norm(dense_matrix, axis=1, keepdims=True) + 1e-12
        dense_matrix = dense_matrix / row_norm

    if CrossEncoder is not None:
        rerank_model = CrossEncoder(rerank_model_name)
    else:
        rerank_model = None

    return RetrievalArtifacts(
        bm25=bm25,
        bm25_tokens=bm25_tokens,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        dense_model=dense_model,
        dense_matrix=dense_matrix,
        rerank_model=rerank_model,
    )


def retrieve_sparse(query: str, artifacts: RetrievalArtifacts, top_k: int) -> tuple[list[int], np.ndarray]:
    q_tokens = tokenize(query)
    scores = np.array(artifacts.bm25.get_scores(q_tokens), dtype=float)
    idx = np.argsort(-scores)[:top_k].tolist()
    return idx, scores[idx]


def retrieve_dense(query: str, artifacts: RetrievalArtifacts, top_k: int) -> tuple[list[int], np.ndarray]:
    if artifacts.dense_model is not None:
        q_vec = artifacts.dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    else:
        q_vec = artifacts.tfidf_vectorizer.transform([query]).toarray()[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    scores = artifacts.dense_matrix @ q_vec
    idx = np.argsort(-scores)[:top_k].tolist()
    return idx, scores[idx]


def retrieve_hybrid(query: str, artifacts: RetrievalArtifacts, top_k: int, alpha: float) -> tuple[list[int], np.ndarray]:
    q_tokens = tokenize(query)
    sparse_scores = np.array(artifacts.bm25.get_scores(q_tokens), dtype=float)
    if artifacts.dense_model is not None:
        q_vec = artifacts.dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    else:
        q_vec = artifacts.tfidf_vectorizer.transform([query]).toarray()[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    dense_scores = artifacts.dense_matrix @ q_vec
    fusion = alpha * norm_scores(sparse_scores) + (1.0 - alpha) * norm_scores(dense_scores)
    idx = np.argsort(-fusion)[:top_k].tolist()
    return idx, fusion[idx]


def rerank(query: str, corpus: list[dict], indices: list[int], artifacts: RetrievalArtifacts) -> tuple[list[int], list[float]]:
    pairs = [(query, corpus[i]["text"]) for i in indices]
    if artifacts.rerank_model is not None:
        scores = artifacts.rerank_model.predict(pairs).tolist()
    else:
        # fallback by tfidf cosine
        q = artifacts.tfidf_vectorizer.transform([query])
        d = artifacts.tfidf_matrix[indices]
        scores = (d @ q.T).toarray().reshape(-1).tolist()
    order = np.argsort(-np.array(scores))
    reranked = [indices[i] for i in order]
    rerank_scores = [float(scores[i]) for i in order]
    return reranked, rerank_scores


def build_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c['doc_id']}:{c['chunk_id']}] {normalize_text(c['text'])}")
    return "\n".join(lines)


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a biomedical QA assistant.\n"
        "Answer the question using ONLY the provided evidence chunks.\n"
        "Every key claim must include citations in the form [doc_id:chunk_id].\n"
        "If evidence is insufficient, say so explicitly and still cite relevant chunks.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n\n"
        "Now provide a concise answer with citations."
    )


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int, timeout: float, retries: int) -> str:
    last_err = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Be precise and cite evidence chunks."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                top_p=1.0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except OpenAIError as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"Generation failed: {last_err}")


def extract_citations(answer: str) -> list[dict]:
    hits = []
    for m in CITATION_RE.finditer(str(answer or "")):
        hits.append({"doc_id": m.group(1), "chunk_id": m.group(2)})
    return hits


def run_strategy(
    strategy: str,
    qa_rows: list[dict],
    corpus: list[dict],
    artifacts: RetrievalArtifacts,
    top_k: int,
    top_n: int,
    alpha: float,
    api_key: str,
    base_url: str,
    llm_model: str,
    skip_generation: bool,
    max_gen_tokens: int,
    req_timeout: float,
    req_retries: int,
    workers: int,
) -> list[dict]:
    thread_local = local()

    def get_client() -> OpenAI | None:
        if skip_generation:
            return None
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=base_url or None)
            thread_local.client = c
        return c

    def process_one(q: dict) -> dict:
        question = q["question"]
        t0 = time.perf_counter()
        if strategy == "sparse_only":
            idx, scores = retrieve_sparse(question, artifacts, top_k)
        elif strategy == "dense_only":
            idx, scores = retrieve_dense(question, artifacts, top_k)
        elif strategy == "hybrid":
            idx, scores = retrieve_hybrid(question, artifacts, top_k, alpha=alpha)
        elif strategy == "hybrid_rerank":
            base_idx, _ = retrieve_hybrid(question, artifacts, top_k, alpha=alpha)
            idx, scores = rerank(question, corpus, base_idx, artifacts)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        top_chunks = [corpus[i] for i in idx[:top_n]]
        context = build_context(top_chunks)

        answer = ""
        gen_ms = 0.0
        if not skip_generation:
            client = get_client()
            p = build_prompt(question, context)
            tg = time.perf_counter()
            answer = call_llm(client, llm_model, p, max_gen_tokens, req_timeout, req_retries)
            gen_ms = (time.perf_counter() - tg) * 1000.0

        citations = extract_citations(answer)
        return {
            "question_id": q["question_id"],
            "question": question,
            "strategy": strategy,
            "retrieved_chunks": [
                {
                    "doc_id": corpus[i]["doc_id"],
                    "chunk_id": corpus[i]["chunk_id"],
                    "score": float(scores[j]) if j < len(scores) else 0.0,
                    "text": corpus[i]["text"],
                }
                for j, i in enumerate(idx)
            ],
            "context": context,
            "answer": answer,
            "citations": citations,
            "latency_ms": float(retrieval_ms + gen_ms),
            "retrieval_latency_ms": float(retrieval_ms),
            "generation_latency_ms": float(gen_ms),
            "gold_docs": q.get("gold_docs", []),
            "ideal_answer": q.get("ideal_answer", ""),
        }

    if workers <= 1:
        records = []
        for q in tqdm(qa_rows, desc=f"exp1 {strategy}"):
            records.append(process_one(q))
        return records

    records = [None] * len(qa_rows)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, q): i for i, q in enumerate(qa_rows)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"exp1 {strategy}"):
            idx = futures[fut]
            records[idx] = fut.result()
    return records


def write_jsonl(path: str, rows: list[dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Experiment 1 retrieval and optional QA generation.")
    parser.add_argument("--corpus", default="corpus/bioasq_chunks.jsonl")
    parser.add_argument("--qa", default="data/bioasq/qa.jsonl")
    parser.add_argument("--out_dir", default="runs/exp1")
    parser.add_argument("--strategies", default="sparse_only,dense_only,hybrid,hybrid_rerank")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rerank_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--llm_model", default=os.environ.get("OPENAI_MODEL", "").strip())
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "").strip())
    parser.add_argument("--max_gen_tokens", type=int, default=260)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--request_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4, help="Question-level parallel workers per strategy.")
    args = parser.parse_args()

    corpus = read_jsonl(args.corpus)
    qa_rows = read_jsonl(args.qa)
    if args.max_questions > 0:
        qa_rows = qa_rows[: args.max_questions]

    artifacts = build_artifacts(corpus, args.dense_model, args.rerank_model)

    api_key = ""
    if not args.skip_generation:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for generation. Use --skip_generation to run retrieval only.")
        if not args.llm_model:
            raise RuntimeError("Missing --llm_model (or OPENAI_MODEL env) for generation.")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        rows = run_strategy(
            strategy=s,
            qa_rows=qa_rows,
            corpus=corpus,
            artifacts=artifacts,
            top_k=args.top_k,
            top_n=args.top_n,
            alpha=args.alpha,
            api_key=api_key,
            base_url=args.base_url,
            llm_model=args.llm_model,
            skip_generation=args.skip_generation,
            max_gen_tokens=args.max_gen_tokens,
            req_timeout=args.request_timeout,
            req_retries=args.request_retries,
            workers=args.workers,
        )
        out_path = os.path.join(args.out_dir, s, "predictions.jsonl")
        write_jsonl(out_path, rows)
        print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
