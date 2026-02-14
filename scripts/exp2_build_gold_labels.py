import argparse
import json
import os

VALID_LABELS = {"yes", "no", "maybe"}


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PubMedQA gold label jsonl from ori_pqal.json."
    )
    parser.add_argument(
        "--input",
        default="data/pubmedqa/ori_pqal.json",
        help="Path to PubMedQA source json.",
    )
    parser.add_argument(
        "--out",
        default="data/pubmedqa/gold_labels.jsonl",
        help="Output path for question_id + gold_label jsonl.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    ensure_parent_dir(args.out)
    written = 0
    skipped = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for qid, obj in data.items():
            gold = str(obj.get("final_decision", "")).strip().lower()
            if gold not in VALID_LABELS:
                skipped += 1
                continue

            record = {"question_id": str(qid), "gold_label": gold}
            w.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "status": "ok",
                "written": written,
                "skipped_invalid_label": skipped,
                "output": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
