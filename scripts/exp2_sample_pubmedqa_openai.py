import argparse
import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
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


def _build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join([c.strip() for c in contexts if c and c.strip()])
    return (
        "You are answering a biomedical yes/no/maybe question.\n"
        "Use ONLY the provided context to decide. If the context is insufficient or ambiguous, answer maybe.\n\n"
        "Output format (STRICT):\n"
        "Answer: <2-6 sentences rationale grounded in context>\n"
        "Final: yes|no|maybe\n\n"
        f"Question: {question.strip()}\n\n"
        f"Context:\n{ctx}\n"
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


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Sample PubMedQA M times for semantic entropy.")
    ap.add_argument("--input", required=True, help="Path to PubMedQA json (dict keyed by id).")
    ap.add_argument("--out", required=True, help="Output jsonl path (runs/exp2/samples.jsonl).")
    ap.add_argument("--model", required=True, help="Model name for OpenAI-compatible chat API.")
    ap.add_argument("--m", type=int, default=10, help="Samples per question.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_context_chars", type=int, default=12000, help="Truncate concatenated contexts.")
    ap.add_argument("--max_items", type=int, default=0, help="Debug: limit number of questions (0=all).")
    ap.add_argument(
        "--base_url",
        default=os.environ.get("OPENAI_BASE_URL", "").strip(),
        help="Optional OpenAI-compatible base URL (or set OPENAI_BASE_URL).",
    )
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key, base_url=args.base_url or None)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())
    if args.max_items and args.max_items > 0:
        items = items[: args.max_items]

    _ensure_parent_dir(args.out)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(args.out, "w", encoding="utf-8") as w:
        for qid, obj in tqdm(items, desc="PubMedQA questions"):
            question = (obj.get("QUESTION") or "").strip()
            contexts = obj.get("CONTEXTS") or []
            gold_label = (obj.get("final_decision") or "").strip().lower() or None

            ctx = _truncate_text("\n\n".join(contexts), args.max_context_chars)
            prompt = _build_prompt(question, [ctx])

            for sample_id in range(args.m):
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
                )

                content = (resp.choices[0].message.content or "").strip()
                rationale_text, final_label = _parse_answer_and_final(content)

                record = {
                    "run_id": run_id,
                    "question_id": str(qid),
                    "sample_id": sample_id,
                    "question": question,
                    "gold_label": gold_label,
                    "rationale_text": rationale_text,
                    "final_label": final_label,
                    "raw_response": content,
                }
                w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

