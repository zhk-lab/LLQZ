import argparse
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from tqdm import tqdm


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
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def extract_citations(answer: str) -> list[dict]:
    return [{"doc_id": m.group(1), "chunk_id": m.group(2)} for m in CITATION_RE.finditer(str(answer or ""))]


def call_chat(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    last_err = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except OpenAIError as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"LLM call failed: {last_err}")


def parse_json_block(text: str) -> dict:
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        # best effort extract first {...}
        l = text.find("{")
        r = text.rfind("}")
        if l >= 0 and r > l:
            return json.loads(text[l : r + 1])
        raise


def build_context_from_retrieved(retrieved_chunks: list[dict], top_n: int) -> str:
    lines = []
    for c in (retrieved_chunks or [])[:top_n]:
        lines.append(f"[{c.get('doc_id')}:{c.get('chunk_id')}] {normalize_text(c.get('text', ''))}")
    return "\n".join(lines)


def build_naive_prompt(question: str, context: str) -> str:
    return (
        "You are a biomedical QA assistant.\n"
        "Answer using ONLY the evidence below. Every key claim must include citation [doc_id:chunk_id].\n"
        "If evidence is insufficient, clearly say so and avoid unsupported claims.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n\n"
        "Return a concise final answer with citations."
    )


def build_candidate_prompt(question: str, context: str) -> str:
    return (
        "Generate one candidate biomedical answer from the evidence.\n"
        "Return strict JSON with keys: answer, base_confidence.\n"
        "- answer: concise answer with citations [doc_id:chunk_id]\n"
        "- base_confidence: float in [0,1], model confidence in factual correctness.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n"
    )


def build_critic_prompt(question: str, context: str, answer: str) -> str:
    return (
        "You are a strict critic in Self-RAG.\n"
        "Evaluate answer quality by three dimensions and return strict JSON:\n"
        "{\n"
        '  "rel_label": "Rel|Irrel", "rel_prob": float,\n'
        '  "sup_label": "Fully|Partial|No", "sup_prob": float,\n'
        '  "use_score": int, "use_prob": float,\n'
        '  "notes": "short rationale"\n'
        "}\n"
        "Definition:\n"
        "- rel: evidence relevance to question.\n"
        "- sup: whether answer is supported by evidence.\n"
        "- use: usefulness score 1..5.\n"
        "Probabilities must be in [0,1].\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n\n"
        f"Candidate Answer:\n{answer}\n"
    )


def clamp01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return max(0.0, min(1.0, v))


def clamp15(x: Any, default: int = 3) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        return default
    return max(1, min(5, v))


def score_candidate(base_prob: float, critic: dict, theta: float, w_rel: float, w_sup: float, w_use: float) -> float:
    rel = clamp01(critic.get("rel_prob", 0.5))
    sup = clamp01(critic.get("sup_prob", 0.5))
    use = clamp01(critic.get("use_prob", critic.get("use_score", 3) / 5.0))
    return float(base_prob + theta * (w_rel * rel + w_sup * sup + w_use * use))


def run_selfrag_variant(
    records: list[dict],
    client: OpenAI,
    model: str,
    top_n: int,
    num_candidates: int,
    theta: float,
    w_rel: float,
    w_sup: float,
    w_use: float,
    no_reflection_weight: bool,
    max_tokens_answer: int,
    timeout: float,
    retries: int,
) -> list[dict]:
    out = []
    for r in tqdm(records, desc="exp3 self_rag"):
        qid = r.get("question_id")
        question = r.get("question", "")
        context = build_context_from_retrieved(r.get("retrieved_chunks", []), top_n=top_n)
        cand_list = []
        for _ in range(num_candidates):
            raw = call_chat(
                client,
                model,
                "You produce strict JSON.",
                build_candidate_prompt(question, context),
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens_answer,
                timeout=timeout,
                retries=retries,
            )
            obj = parse_json_block(raw)
            answer = normalize_text(obj.get("answer", ""))
            base_prob = clamp01(obj.get("base_confidence", 0.5))
            critic_raw = call_chat(
                client,
                model,
                "You produce strict JSON only.",
                build_critic_prompt(question, context, answer),
                temperature=0.0,
                top_p=1.0,
                max_tokens=220,
                timeout=timeout,
                retries=retries,
            )
            critic = parse_json_block(critic_raw)

            score = score_candidate(
                base_prob=base_prob,
                critic=critic,
                theta=theta,
                w_rel=0.0 if no_reflection_weight else w_rel,
                w_sup=0.0 if no_reflection_weight else w_sup,
                w_use=0.0 if no_reflection_weight else w_use,
            )
            cand_list.append(
                {
                    "answer": answer,
                    "base_prob": base_prob,
                    "critic": {
                        "rel_label": str(critic.get("rel_label", "Irrel")),
                        "rel_prob": clamp01(critic.get("rel_prob", 0.5)),
                        "sup_label": str(critic.get("sup_label", "Partial")),
                        "sup_prob": clamp01(critic.get("sup_prob", 0.5)),
                        "use_score": clamp15(critic.get("use_score", 3)),
                        "use_prob": clamp01(critic.get("use_prob", 0.6)),
                        "notes": str(critic.get("notes", ""))[:300],
                    },
                    "score": score,
                }
            )

        best = sorted(cand_list, key=lambda x: -x["score"])[0]
        out.append(
            {
                "question_id": qid,
                "question": question,
                "retrieved_chunks": (r.get("retrieved_chunks", [])[:top_n]),
                "context": context,
                "answer": best["answer"],
                "citations": extract_citations(best["answer"]),
                "candidates": cand_list,
                "selected_score": best["score"],
                "gold_docs": r.get("gold_docs", []),
                "ideal_answer": r.get("ideal_answer", ""),
            }
        )
    return out


def run_naive(records: list[dict], client: OpenAI, model: str, top_n: int, timeout: float, retries: int) -> list[dict]:
    out = []
    for r in tqdm(records, desc="exp3 naive"):
        question = r.get("question", "")
        context = build_context_from_retrieved(r.get("retrieved_chunks", []), top_n=top_n)
        ans = call_chat(
            client,
            model,
            "You are a biomedical QA assistant.",
            build_naive_prompt(question, context),
            temperature=0.2,
            top_p=1.0,
            max_tokens=260,
            timeout=timeout,
            retries=retries,
        )
        out.append(
            {
                "question_id": r.get("question_id"),
                "question": question,
                "retrieved_chunks": (r.get("retrieved_chunks", [])[:top_n]),
                "context": context,
                "answer": ans,
                "citations": extract_citations(ans),
                "gold_docs": r.get("gold_docs", []),
                "ideal_answer": r.get("ideal_answer", ""),
            }
        )
    return out


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Experiment 3 Naive RAG vs Self-RAG variants.")
    parser.add_argument("--retrieval_source", default="runs/exp1/hybrid_rerank/predictions.jsonl")
    parser.add_argument("--out_root", default="runs/exp3")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "").strip())
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "").strip())
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--w_rel", type=float, default=1.0)
    parser.add_argument("--w_sup", type=float, default=2.0)
    parser.add_argument("--w_use", type=float, default=1.0)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--max_tokens_answer", type=int, default=240)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--request_retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    if not args.model:
        raise RuntimeError("Missing --model (or OPENAI_MODEL).")

    src = read_jsonl(args.retrieval_source)
    if args.max_questions > 0:
        src = src[: args.max_questions]

    client = OpenAI(api_key=api_key, base_url=args.base_url or None)

    naive = run_naive(src, client, args.model, args.top_n, args.request_timeout, args.request_retries)
    wo = run_selfrag_variant(
        src,
        client,
        args.model,
        top_n=args.top_n,
        num_candidates=args.num_candidates,
        theta=args.theta,
        w_rel=args.w_rel,
        w_sup=args.w_sup,
        w_use=args.w_use,
        no_reflection_weight=True,
        max_tokens_answer=args.max_tokens_answer,
        timeout=args.request_timeout,
        retries=args.request_retries,
    )
    full = run_selfrag_variant(
        src,
        client,
        args.model,
        top_n=args.top_n,
        num_candidates=args.num_candidates,
        theta=args.theta,
        w_rel=args.w_rel,
        w_sup=args.w_sup,
        w_use=args.w_use,
        no_reflection_weight=False,
        max_tokens_answer=args.max_tokens_answer,
        timeout=args.request_timeout,
        retries=args.request_retries,
    )

    p1 = os.path.join(args.out_root, "naive_rag", "predictions.jsonl")
    p2 = os.path.join(args.out_root, "self_rag_wo_weight", "predictions.jsonl")
    p3 = os.path.join(args.out_root, "self_rag_full", "predictions.jsonl")
    write_jsonl(p1, naive)
    write_jsonl(p2, wo)
    write_jsonl(p3, full)
    print(json.dumps({"status": "ok", "naive": p1, "wo_weight": p2, "full": p3}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
