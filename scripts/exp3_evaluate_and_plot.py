import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BAR_COLORS = ["#C9D4E5", "#95ACC9", "#4A6E9E"]
LINE_COLOR = "#2F4E79"


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


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def token_f1(pred: str, gold: str) -> float:
    p = re.findall(r"[a-z0-9]+", normalize_text(pred))
    g = re.findall(r"[a-z0-9]+", normalize_text(gold))
    if not p or not g:
        return 0.0
    cp, cg = defaultdict(int), defaultdict(int)
    for t in p:
        cp[t] += 1
    for t in g:
        cg[t] += 1
    common = sum(min(cp[t], cg[t]) for t in cp)
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall + 1e-12)


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Georgia", "serif"]
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["figure.dpi"] = 150


def eval_one(rows: list[dict], correctness_threshold: float) -> tuple[dict, dict]:
    f1s, cit_prec, abstain, correct = [], [], [], []
    by_q = {}
    for r in rows:
        ans = r.get("answer", "")
        ideal = r.get("ideal_answer", "")
        f1 = token_f1(ans, ideal) if ideal else 0.0
        f1s.append(f1)
        is_correct = int(f1 >= correctness_threshold)
        correct.append(is_correct)
        a = normalize_text(ans)
        abstain.append(int(("insufficient evidence" in a) or ("cannot determine" in a) or ("not enough evidence" in a)))

        gold_docs = set(str(x) for x in (r.get("gold_docs") or []))
        cits = r.get("citations") or []
        if cits:
            good = sum(1 for c in cits if str(c.get("doc_id", "")) in gold_docs)
            cit_prec.append(good / max(1, len(cits)))
        by_q[str(r.get("question_id"))] = is_correct

    metrics = {
        "factscore_proxy_f1": float(np.mean(f1s)) if f1s else 0.0,
        "citation_precision": float(np.mean(cit_prec)) if cit_prec else 0.0,
        "abstention_rate": float(np.mean(abstain)) if abstain else 0.0,
        "correct_rate": float(np.mean(correct)) if correct else 0.0,
    }
    return metrics, by_q


def correction_stats(naive: dict, other: dict) -> dict:
    ids = sorted(set(naive.keys()) & set(other.keys()))
    if not ids:
        return {"fix_rate": 0.0, "introduce_rate": 0.0, "delta_correct_rate": 0.0}
    fix = intro = 0
    n0 = 0
    for qid in ids:
        a = naive[qid]
        b = other[qid]
        if a == 0:
            n0 += 1
            if b == 1:
                fix += 1
        if a == 1 and b == 0:
            intro += 1
    return {
        "fix_rate": fix / max(1, n0),
        "introduce_rate": intro / max(1, sum(naive[q] == 1 for q in ids)),
        "delta_correct_rate": float(np.mean([other[q] for q in ids]) - np.mean([naive[q] for q in ids])),
    }


def plot_metrics(summary: dict, out_path: str) -> None:
    ensure_parent(out_path)
    setup_plot_style()
    systems = ["naive_rag", "self_rag_wo_weight", "self_rag_full"]
    labels = ["Naive RAG", "Self-RAG w/o w", "Self-RAG full"]
    f1 = [summary[s]["metrics"]["factscore_proxy_f1"] for s in systems]
    cp = [summary[s]["metrics"]["citation_precision"] for s in systems]
    cr = [summary[s]["metrics"]["correct_rate"] for s in systems]
    x = np.arange(len(systems))
    w = 0.22
    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111)
    ax.bar(x - w, f1, width=w, color=BAR_COLORS[0], label="FactScore Proxy (F1)")
    ax.bar(x, cp, width=w, color=BAR_COLORS[1], label="Citation Precision")
    ax.bar(x + w, cr, width=w, color=BAR_COLORS[2], label="Correct Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 3: Quality Comparison of Naive vs Self-RAG", pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_correction(summary: dict, out_path: str) -> None:
    ensure_parent(out_path)
    setup_plot_style()
    labels = ["Self-RAG w/o w", "Self-RAG full"]
    fix = [summary["comparison"]["wo_weight_vs_naive"]["fix_rate"], summary["comparison"]["full_vs_naive"]["fix_rate"]]
    intro = [summary["comparison"]["wo_weight_vs_naive"]["introduce_rate"], summary["comparison"]["full_vs_naive"]["introduce_rate"]]
    delta = [
        summary["comparison"]["wo_weight_vs_naive"]["delta_correct_rate"],
        summary["comparison"]["full_vs_naive"]["delta_correct_rate"],
    ]
    x = np.arange(len(labels))
    fig = plt.figure(figsize=(7.8, 5.2))
    ax = fig.add_subplot(111)
    ax.bar(x, fix, color="#95ACC9", width=0.36, label="Fix Rate (Wrong→Correct)")
    ax.bar(x, intro, bottom=fix, color="#4A6E9E", width=0.36, label="Introduce Rate (Correct→Wrong)")
    ax2 = ax.twinx()
    ax2.plot(x, delta, color=LINE_COLOR, marker="o", linewidth=2.0, label="Δ Correct Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax2.set_ylim(min(-0.2, min(delta) - 0.02), max(0.2, max(delta) + 0.02))
    ax.set_ylabel("Rate")
    ax2.set_ylabel("Δ Correct Rate")
    ax.set_title("Experiment 3: Correction Gain and Side Effects", pad=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 3 outputs and create single-figure plots.")
    parser.add_argument("--naive", default="runs/exp3/naive_rag/predictions.jsonl")
    parser.add_argument("--wo_weight", default="runs/exp3/self_rag_wo_weight/predictions.jsonl")
    parser.add_argument("--full", default="runs/exp3/self_rag_full/predictions.jsonl")
    parser.add_argument("--correctness_threshold", type=float, default=0.20)
    parser.add_argument("--summary_out", default="runs/exp3/metrics_summary.json")
    parser.add_argument("--plot_metrics_out", default="plots/exp3_quality_metrics.png")
    parser.add_argument("--plot_correction_out", default="plots/exp3_correction_gain.png")
    args = parser.parse_args()

    naive_rows = read_jsonl(args.naive)
    wo_rows = read_jsonl(args.wo_weight)
    full_rows = read_jsonl(args.full)

    naive_m, naive_byq = eval_one(naive_rows, args.correctness_threshold)
    wo_m, wo_byq = eval_one(wo_rows, args.correctness_threshold)
    full_m, full_byq = eval_one(full_rows, args.correctness_threshold)

    summary = {
        "naive_rag": {"metrics": naive_m},
        "self_rag_wo_weight": {"metrics": wo_m},
        "self_rag_full": {"metrics": full_m},
        "comparison": {
            "wo_weight_vs_naive": correction_stats(naive_byq, wo_byq),
            "full_vs_naive": correction_stats(naive_byq, full_byq),
        },
    }

    ensure_parent(args.summary_out)
    with open(args.summary_out, "w", encoding="utf-8") as w:
        json.dump(summary, w, ensure_ascii=False, indent=2)
    plot_metrics(summary, args.plot_metrics_out)
    plot_correction(summary, args.plot_correction_out)

    print(
        json.dumps(
            {
                "status": "ok",
                "summary_out": args.summary_out,
                "plot_metrics_out": args.plot_metrics_out,
                "plot_correction_out": args.plot_correction_out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
