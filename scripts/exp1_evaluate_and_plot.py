import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BAR_COLORS = ["#C9D4E5", "#95ACC9", "#6F8FB4", "#4A6E9E"]
POINT_COLORS = ["#4A6E9E", "#95ACC9", "#2F4E79", "#7D93B5"]


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
    p_set = defaultdict(int)
    g_set = defaultdict(int)
    for t in p:
        p_set[t] += 1
    for t in g:
        g_set[t] += 1
    common = 0
    for t, c in p_set.items():
        common += min(c, g_set[t])
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall + 1e-12)


def mrr_at_k(retrieved_docs: list[str], gold_docs: set[str], k: int) -> float:
    for i, d in enumerate(retrieved_docs[:k], start=1):
        if d in gold_docs:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], gold_docs: set[str], k: int) -> float:
    gains = [1.0 if d in gold_docs else 0.0 for d in retrieved_docs[:k]]
    dcg = 0.0
    for i, g in enumerate(gains, start=1):
        dcg += g / np.log2(i + 1)
    ideal_gains = [1.0] * min(len(gold_docs), k)
    idcg = 0.0
    for i, g in enumerate(ideal_gains, start=1):
        idcg += g / np.log2(i + 1)
    return float(dcg / (idcg + 1e-12))


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Georgia", "serif"]
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["figure.dpi"] = 150


def plot_metrics_bar(metrics: dict, out_path: str) -> None:
    ensure_parent(out_path)
    setup_plot_style()
    strategies = list(metrics.keys())
    vals_recall = [metrics[s]["retrieval"]["recall_at_k"] for s in strategies]
    vals_mrr = [metrics[s]["retrieval"]["mrr_at_k"] for s in strategies]
    vals_ndcg = [metrics[s]["retrieval"]["ndcg_at_k"] for s in strategies]
    vals_hit = [metrics[s]["retrieval"]["hit_at_k"] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.18
    fig = plt.figure(figsize=(8.4, 5.4))
    ax = fig.add_subplot(111)
    ax.bar(x - 1.5 * width, vals_recall, width=width, color=BAR_COLORS[0], label="Recall@K")
    ax.bar(x - 0.5 * width, vals_mrr, width=width, color=BAR_COLORS[1], label="MRR@K")
    ax.bar(x + 0.5 * width, vals_ndcg, width=width, color=BAR_COLORS[2], label="nDCG@K")
    ax.bar(x + 1.5 * width, vals_hit, width=width, color=BAR_COLORS[3], label="Hit@K")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric Value")
    ax.set_title("Experiment 1: Retrieval Quality Across Strategies", pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_effect_cost(metrics: dict, out_path: str) -> None:
    ensure_parent(out_path)
    setup_plot_style()
    strategies = list(metrics.keys())
    x = [metrics[s]["efficiency"]["avg_latency_ms"] for s in strategies]
    y = [metrics[s]["retrieval"]["recall_at_k"] for s in strategies]

    fig = plt.figure(figsize=(7.8, 5.2))
    ax = fig.add_subplot(111)
    for i, s in enumerate(strategies):
        ax.scatter(x[i], y[i], s=90, color=POINT_COLORS[i % len(POINT_COLORS)], alpha=0.95)
        ax.text(x[i] * 1.01 + 1e-6, y[i] + 0.004, s, fontsize=10)
    ax.set_xlabel("Average End-to-End Latency (ms)")
    ax.set_ylabel("Recall@K")
    ax.set_title("Experiment 1: Effectiveness vs Cost", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_strategy(rows: list[dict], top_k: int) -> dict:
    recall_vals, hit_vals, mrr_vals, ndcg_vals = [], [], [], []
    latency_vals = []
    answer_f1_vals = []
    citation_precision_vals = []

    for r in rows:
        gold_docs = set(str(x) for x in (r.get("gold_docs") or []))
        retrieved = r.get("retrieved_chunks") or []
        retrieved_docs = [str(x.get("doc_id", "")) for x in retrieved]
        retrieved_topk = retrieved_docs[:top_k]
        hit = int(any(d in gold_docs for d in retrieved_topk)) if gold_docs else 0
        inter = len(set(retrieved_topk) & gold_docs) if gold_docs else 0
        recall = inter / len(gold_docs) if gold_docs else 0.0
        mrr = mrr_at_k(retrieved_topk, gold_docs, top_k) if gold_docs else 0.0
        ndcg = ndcg_at_k(retrieved_topk, gold_docs, top_k) if gold_docs else 0.0

        recall_vals.append(recall)
        hit_vals.append(hit)
        mrr_vals.append(mrr)
        ndcg_vals.append(ndcg)
        latency_vals.append(float(r.get("latency_ms", 0.0)))

        pred_answer = r.get("answer", "")
        ideal = r.get("ideal_answer", "")
        if pred_answer and ideal:
            answer_f1_vals.append(token_f1(pred_answer, ideal))

        cits = r.get("citations") or []
        if cits:
            c_doc = [str(c.get("doc_id", "")) for c in cits]
            good = sum(1 for d in c_doc if d in gold_docs)
            citation_precision_vals.append(good / max(1, len(c_doc)))

    return {
        "retrieval": {
            "recall_at_k": float(np.mean(recall_vals)) if recall_vals else 0.0,
            "hit_at_k": float(np.mean(hit_vals)) if hit_vals else 0.0,
            "mrr_at_k": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
            "ndcg_at_k": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        },
        "answer": {
            "token_f1": float(np.mean(answer_f1_vals)) if answer_f1_vals else 0.0,
            "citation_doc_precision": float(np.mean(citation_precision_vals)) if citation_precision_vals else 0.0,
        },
        "efficiency": {
            "avg_latency_ms": float(np.mean(latency_vals)) if latency_vals else 0.0,
            "p95_latency_ms": float(np.percentile(latency_vals, 95)) if latency_vals else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 1 results and generate publication-style plots.")
    parser.add_argument("--pred_root", default="runs/exp1")
    parser.add_argument("--strategies", default="sparse_only,dense_only,hybrid,hybrid_rerank")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--metrics_out", default="runs/exp1/metrics_summary.json")
    parser.add_argument("--plot_metrics_out", default="plots/exp1_retrieval_metrics.png")
    parser.add_argument("--plot_cost_out", default="plots/exp1_effect_vs_cost.png")
    args = parser.parse_args()

    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    summary = {}
    for s in strategies:
        path = os.path.join(args.pred_root, s, "predictions.jsonl")
        rows = read_jsonl(path)
        summary[s] = evaluate_strategy(rows, top_k=args.top_k)

    ensure_parent(args.metrics_out)
    with open(args.metrics_out, "w", encoding="utf-8") as w:
        json.dump(summary, w, ensure_ascii=False, indent=2)

    plot_metrics_bar(summary, args.plot_metrics_out)
    plot_effect_cost(summary, args.plot_cost_out)
    print(
        json.dumps(
            {
                "status": "ok",
                "metrics_out": args.metrics_out,
                "plot_metrics_out": args.plot_metrics_out,
                "plot_cost_out": args.plot_cost_out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
