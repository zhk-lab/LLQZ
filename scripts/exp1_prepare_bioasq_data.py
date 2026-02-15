import argparse
import json
import os
import re
from collections import defaultdict


PMID_RE = re.compile(r"/pubmed/(\d+)")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: list[dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as w:
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_pmid(url: str) -> str:
    if not url:
        return ""
    m = PMID_RE.search(url)
    return m.group(1) if m else ""


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BioASQ corpus and QA files for Experiment 1.")
    parser.add_argument("--input", default="data/BioASQ/trainining14b.json")
    parser.add_argument("--corpus_out", default="corpus/bioasq_chunks.jsonl")
    parser.add_argument("--qa_out", default="data/bioasq/qa.jsonl")
    parser.add_argument("--max_questions", type=int, default=0, help="Debug cap. 0 means all.")
    args = parser.parse_args()

    raw = read_json(args.input)
    questions = raw.get("questions", [])
    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    doc_texts: dict[str, set[str]] = defaultdict(set)
    qa_rows = []

    for q in questions:
        qid = str(q.get("id", "")).strip()
        body = normalize_text(q.get("body", ""))
        qtype = str(q.get("type", "")).strip()
        ideal_answers = q.get("ideal_answer", [])
        if isinstance(ideal_answers, list):
            ideal_answer = normalize_text(" ".join(str(x) for x in ideal_answers))
        else:
            ideal_answer = normalize_text(str(ideal_answers))

        doc_urls = q.get("documents", []) or []
        gold_docs = []
        for u in doc_urls:
            pmid = extract_pmid(str(u))
            if pmid:
                gold_docs.append(pmid)
        gold_docs = sorted(set(gold_docs))

        snippets = q.get("snippets", []) or []
        for sn in snippets:
            txt = normalize_text(sn.get("text", ""))
            if not txt:
                continue
            pmid = extract_pmid(str(sn.get("document", "")))
            if not pmid:
                continue
            doc_texts[pmid].add(txt)

        qa_rows.append(
            {
                "question_id": qid,
                "question": body,
                "type": qtype,
                "gold_docs": gold_docs,
                "ideal_answer": ideal_answer,
            }
        )

    corpus_rows = []
    for doc_id, texts in doc_texts.items():
        ordered = sorted(texts)
        for i, txt in enumerate(ordered):
            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{i}",
                    "text": txt,
                    "meta": {"source": "bioasq_snippet"},
                }
            )

    write_jsonl(args.corpus_out, corpus_rows)
    write_jsonl(args.qa_out, qa_rows)

    print(
        json.dumps(
            {
                "status": "ok",
                "num_questions": len(qa_rows),
                "num_chunks": len(corpus_rows),
                "corpus_out": args.corpus_out,
                "qa_out": args.qa_out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
