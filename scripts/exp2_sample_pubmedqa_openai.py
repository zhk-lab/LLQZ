import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError
from tqdm import tqdm


FINAL_RE = re.compile(r"(?im)^\s*Final\s*:\s*(yes|no|maybe)\s*$")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _truncate_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[TRUNCATED]..."


def _truncate_head_tail(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    half = max_chars // 2
    head = s[: max(0, half - 12)]
    tail = s[-max(0, half - 12) :]
    return head + "\n...[TRUNCATED]...\n" + tail


def _build_prompt(question: str, contexts: list[str], concise_mode: bool, append_no_think: bool) -> str:
    ctx = "\n\n".join([c.strip() for c in contexts if c and c.strip()])
    answer_rule = (
        "Answer: <3-5 concise sentences rationale grounded in context>\n"
        if concise_mode
        else "Answer: <2-6 sentences rationale grounded in context>\n"
    )
    no_think_hint = "\n/no_think" if append_no_think else ""
    return (
        "You are answering a biomedical yes/no/maybe question.\n"
        "Use ONLY the provided context to decide.\n"
        "Label policy:\n"
        "- Choose yes when evidence clearly supports the question.\n"
        "- Choose no when evidence clearly refutes the question.\n"
        "- Choose maybe ONLY when evidence is genuinely insufficient or conflicting after careful reading.\n"
        "- Do NOT default to maybe just because there is uncertainty.\n\n"
        "Output format (STRICT):\n"
        f"{answer_rule}"
        "Final: yes|no|maybe\n\n"
        f"Question: {question.strip()}\n\n"
        f"Context:\n{ctx}\n"
        f"{no_think_hint}"
    )


def _parse_answer_and_final(text: str) -> tuple[str, str]:
    """
    Returns (rationale_text, final_label). If parsing fails, final_label="maybe".
    """
    raw = (text or "").strip()
    # final label: take the last matching "Final: ..."
    matches = list(FINAL_RE.finditer(raw))
    final = matches[-1].group(1).lower() if matches else "maybe"

    # rationale: everything before the last "Final:" line if present; else whole text
    if matches:
        rationale = raw[: matches[-1].start()].strip()
    else:
        rationale = raw

    # best-effort: strip leading "Answer:" label if present
    rationale = re.sub(r"(?is)^\s*Answer\s*:\s*", "", rationale).strip()
    return rationale, final


def _load_existing_samples(path: str) -> dict[str, set[int]]:
    existing: dict[str, set[int]] = {}
    if not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("question_id", "")).strip()
            sid = row.get("sample_id")
            if not qid or not isinstance(sid, int):
                continue
            if qid not in existing:
                existing[qid] = set()
            existing[qid].add(sid)
    return existing


def _sample_one_question(
    qid: str,
    obj: dict,
    args: argparse.Namespace,
    run_id: str,
    api_key: str,
    existing_sids: set[int],
) -> tuple[list[dict], int]:
    client = OpenAI(api_key=api_key, base_url=args.base_url or None)

    question = (obj.get("QUESTION") or "").strip()
    contexts = obj.get("CONTEXTS") or []
    gold_label = (obj.get("final_decision") or "").strip().lower() or None

    raw_ctx = "\n\n".join(contexts)
    if args.context_trim_mode == "head_tail":
        ctx = _truncate_head_tail(raw_ctx, args.max_context_chars)
    else:
        ctx = _truncate_text(raw_ctx, args.max_context_chars)
    prompt = _build_prompt(question, [ctx], concise_mode=args.concise_mode, append_no_think=args.append_no_think)

    records = []
    skipped = 0
    for sample_id in range(args.m):
        if sample_id in existing_sids:
            skipped += 1
            continue

        last_err = None
        resp = None
        for attempt in range(args.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Follow the user's requested output format exactly.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout=args.request_timeout,
                )
                break
            except OpenAIError as e:
                last_err = e
                if attempt < args.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        if resp is None:
            raise RuntimeError(
                f"Failed to sample question_id={qid}, sample_id={sample_id}: {last_err}"
            )

        content = (resp.choices[0].message.content or "").strip()
        rationale_text, final_label = _parse_answer_and_final(content)
        records.append(
            {
                "run_id": run_id,
                "question_id": str(qid),
                "sample_id": sample_id,
                "question": question,
                "gold_label": gold_label,
                "rationale_text": rationale_text,
                "final_label": final_label,
                "raw_response": content,
            }
        )
    return records, skipped


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Sample PubMedQA M times for semantic entropy.")
    ap.add_argument("--input", required=True, help="Path to PubMedQA json (dict keyed by id).")
    ap.add_argument("--out", required=True, help="Output jsonl path (runs/exp2/samples.jsonl).")
    ap.add_argument("--model", required=True, help="Model name for OpenAI-compatible chat API.")
    ap.add_argument("--m", type=int, default=10, help="Samples per question.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=140)
    ap.add_argument("--max_context_chars", type=int, default=12000, help="Truncate concatenated contexts.")
    ap.add_argument(
        "--context_trim_mode",
        choices=["head", "head_tail"],
        default="head_tail",
        help="How to trim long contexts before prompting.",
    )
    ap.add_argument("--max_items", type=int, default=0, help="Debug: limit number of questions (0=all).")
    ap.add_argument("--request_timeout", type=float, default=120.0)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1, help="Question-level parallel workers.")
    ap.add_argument(
        "--concise_mode",
        action="store_true",
        help="Request 3-5 concise sentence rationales for balanced readability and speed.",
    )
    ap.add_argument(
        "--append_no_think",
        action="store_true",
        help="Append '/no_think' hint for models that support disabling long reasoning mode.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append mode: skip already generated (question_id, sample_id) records in output file.",
    )
    ap.add_argument(
        "--base_url",
        default=os.environ.get("OPENAI_BASE_URL", "").strip(),
        help="Optional OpenAI-compatible base URL (or set OPENAI_BASE_URL).",
    )
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())
    if args.max_items and args.max_items > 0:
        items = items[: args.max_items]

    _ensure_parent_dir(args.out)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing = _load_existing_samples(args.out) if args.resume else {}
    mode = "a" if args.resume else "w"

    with open(args.out, mode, encoding="utf-8") as w:
        total_skipped = 0
        if args.workers <= 1:
            for qid, obj in tqdm(items, desc="PubMedQA questions"):
                records, skipped = _sample_one_question(
                    qid=str(qid),
                    obj=obj,
                    args=args,
                    run_id=run_id,
                    api_key=api_key,
                    existing_sids=existing.get(str(qid), set()),
                )
                total_skipped += skipped
                for record in records:
                    w.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(
                        _sample_one_question,
                        str(qid),
                        obj,
                        args,
                        run_id,
                        api_key,
                        existing.get(str(qid), set()),
                    )
                    for qid, obj in items
                ]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="PubMedQA questions"):
                    records, skipped = fut.result()
                    total_skipped += skipped
                    for record in records:
                        w.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.resume:
        print(
            json.dumps(
                {"status": "ok", "resume": True, "skipped_existing_samples": total_skipped},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

