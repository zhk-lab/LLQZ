import argparse
import json
import math
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


VALID_LABELS = {"yes", "no", "maybe"}
BAR_COLORS = ["#C9D4E5", "#95ACC9", "#4A6E9E"]
SCATTER_COLOR = "#4A6E9E"
LINE_COLOR = "#2F4E79"


def ensure_parent_dir(path: str) -> None:
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


def normalize_label(label: str) -> str:
    value = str(label or "").strip().lower()
    return value if value in VALID_LABELS else "maybe"


def cluster_by_similarity(
    texts: list[str], labels: list[str], threshold: float, require_same_label: bool
) -> tuple[list[int], list[list[int]]]:
    n = len(texts)
    if n == 0:
        return [], []
    if n == 1:
        return [0], [[0]]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x = vectorizer.fit_transform(texts)
    # x is l2-normalized by default, so dot product = cosine similarity
    sim = (x * x.T).toarray()

    # Build connected components under threshold constraint.
    # This is a practical approximation of semantic equivalence classes.
    visited = [False] * n
    cluster_ids = [-1] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        component = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in range(n):
                if visited[v]:
                    continue
                if sim[u][v] < threshold:
                    continue
                if require_same_label and labels[u] != labels[v]:
                    continue
                visited[v] = True
                stack.append(v)
        cid = len(clusters)
        for idx in component:
            cluster_ids[idx] = cid
        clusters.append(sorted(component))

    return cluster_ids, clusters


def shannon_entropy(probs: list[float]) -> float:
    value = 0.0
    for p in probs:
        if p > 0.0:
            value -= p * math.log(p)
    return value


def majority_vote(labels: list[str]) -> str:
    counts = Counter(labels)
    # Stable tie-break to keep result deterministic.
    order = {"yes": 0, "no": 1, "maybe": 2}
    return sorted(counts.items(), key=lambda x: (-x[1], order.get(x[0], 99)))[0][0]


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "DejaVu Serif",
        "Georgia",
        "serif",
    ]
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["figure.dpi"] = 150


def plot_entropy_scatter(rows: list[dict], out_path: str) -> None:
    ensure_parent_dir(out_path)
    setup_plot_style()

    x = np.array([r["semantic_entropy"] for r in rows], dtype=float)
    y = np.array([r["error"] for r in rows], dtype=float)

    # Add small y-jitter for readability when y in {0,1}
    rng = np.random.default_rng(7)
    y_jitter = y + rng.normal(0, 0.025, size=len(y))
    y_jitter = np.clip(y_jitter, -0.05, 1.05)

    fig = plt.figure(figsize=(7.6, 5.2))
    ax = fig.add_subplot(111)

    ax.scatter(
        x,
        y_jitter,
        s=22,
        alpha=0.6,
        c=SCATTER_COLOR,
        edgecolors="none",
        label="Questions",
    )

    if len(x) >= 2 and np.std(x) > 1e-9:
        # Linear trend line over binary target for visual guidance.
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        yy = slope * xx + intercept
        ax.plot(xx, yy, color="#1F3552", linewidth=2.2, label="Trend")

    ax.set_xlabel("Semantic Entropy SE(x)")
    ax.set_ylabel("Error (0=Correct, 1=Wrong)")
    ax.set_title("Experiment 2: Semantic Entropy vs Prediction Error", pad=12)
    ax.set_ylim(-0.08, 1.08)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_bucket_bar(rows: list[dict], out_path: str, n_bins: int) -> None:
    ensure_parent_dir(out_path)
    setup_plot_style()
    if not rows:
        return

    entropy = np.array([r["semantic_entropy"] for r in rows], dtype=float)
    labels = [r["pred_label"] for r in rows]
    errors = np.array([r["error"] for r in rows], dtype=float)

    # Quantile bins are more balanced for publication plots.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(entropy, quantiles)
    edges[0] = float(np.min(entropy))
    edges[-1] = float(np.max(entropy)) + 1e-9

    bin_indices = np.digitize(entropy, edges[1:-1], right=False)

    yes_share, no_share, maybe_share = [], [], []
    err_rate, bin_names = [], []

    for b in range(n_bins):
        mask = bin_indices == b
        idx = np.where(mask)[0]
        if len(idx) == 0:
            yes_share.append(0.0)
            no_share.append(0.0)
            maybe_share.append(0.0)
            err_rate.append(0.0)
            bin_names.append(f"B{b+1}")
            continue

        c = Counter(labels[i] for i in idx)
        total = float(len(idx))
        yes_share.append(c.get("yes", 0) / total)
        no_share.append(c.get("no", 0) / total)
        maybe_share.append(c.get("maybe", 0) / total)
        err_rate.append(float(np.mean(errors[idx])))
        bin_names.append(f"B{b+1}")

    y_pos = np.arange(n_bins)

    fig = plt.figure(figsize=(8.0, 5.8))
    ax = fig.add_subplot(111)
    ax.barh(y_pos, yes_share, color=BAR_COLORS[0], edgecolor="none", label="Pred: yes")
    ax.barh(
        y_pos,
        no_share,
        left=yes_share,
        color=BAR_COLORS[1],
        edgecolor="none",
        label="Pred: no",
    )
    left2 = np.array(yes_share) + np.array(no_share)
    ax.barh(
        y_pos,
        maybe_share,
        left=left2,
        color=BAR_COLORS[2],
        edgecolor="none",
        label="Pred: maybe",
    )

    # Overlay bucket error rate as a line on a twin x-axis.
    ax2 = ax.twiny()
    ax2.plot(err_rate, y_pos, color=LINE_COLOR, linewidth=2.1, marker="o", markersize=4)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Error Rate by Entropy Bucket")
    ax2.grid(False)

    ax.set_xlim(0, 1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bin_names)
    ax.set_xlabel("Prediction Label Share")
    ax.set_ylabel("Entropy Buckets (Low → High)")
    ax.set_title("Experiment 2: Entropy Buckets, Label Mix, and Error Rate", pad=12)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster sampled answers and compute semantic entropy.")
    parser.add_argument(
        "--samples",
        default="runs/exp2/samples.jsonl",
        help="Input samples jsonl generated by exp2_sample_pubmedqa_openai.py.",
    )
    parser.add_argument(
        "--clusters_out",
        default="runs/exp2/clusters.jsonl",
        help="Output path for semantic clusters.",
    )
    parser.add_argument(
        "--entropy_out",
        default="runs/exp2/semantic_entropy.jsonl",
        help="Output path for per-question entropy and error.",
    )
    parser.add_argument(
        "--correlation_out",
        default="runs/exp2/correlation.json",
        help="Output path for Pearson/Spearman metrics.",
    )
    parser.add_argument(
        "--scatter_plot_out",
        default="plots/exp2_entropy_scatter.png",
        help="Output path for entropy scatter chart.",
    )
    parser.add_argument(
        "--bucket_plot_out",
        default="plots/exp2_entropy_bucket_bar.png",
        help="Output path for entropy bucket chart.",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.62,
        help="Cosine threshold for semantic clustering.",
    )
    parser.add_argument(
        "--require_same_label",
        action="store_true",
        help="If set, only merge samples with identical final_label.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=6,
        help="Number of entropy buckets for visualization.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.samples)
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        qid = str(r.get("question_id", "")).strip()
        if not qid:
            continue
        by_qid[qid].append(r)

    ensure_parent_dir(args.clusters_out)
    ensure_parent_dir(args.entropy_out)
    ensure_parent_dir(args.correlation_out)

    entropy_rows = []
    with open(args.clusters_out, "w", encoding="utf-8") as wc, open(
        args.entropy_out, "w", encoding="utf-8"
    ) as we:
        for qid, items in tqdm(by_qid.items(), desc="Analyzing questions"):
            items = sorted(items, key=lambda x: int(x.get("sample_id", 0)))
            texts = [str(it.get("rationale_text", "")).strip() for it in items]
            labels = [normalize_label(it.get("final_label", "")) for it in items]
            gold = normalize_label(items[0].get("gold_label", ""))

            cluster_ids, clusters = cluster_by_similarity(
                texts=texts,
                labels=labels,
                threshold=args.sim_threshold,
                require_same_label=args.require_same_label,
            )

            # Uniform sample probability when token-level logprobs are unavailable.
            n = max(1, len(items))
            cluster_sizes = [len(c) for c in clusters]
            cluster_probs = [s / n for s in cluster_sizes]
            se = shannon_entropy(cluster_probs)

            pred_label = majority_vote(labels)
            error = int(pred_label != gold)

            cluster_payload = {
                "question_id": qid,
                "question": items[0].get("question", ""),
                "gold_label": gold,
                "pred_label_majority_vote": pred_label,
                "semantic_cluster_ids": cluster_ids,
                "semantic_clusters": [
                    {
                        "cluster_id": cid,
                        "sample_indices": member_indices,
                        "size": len(member_indices),
                        "prob": cluster_probs[cid],
                        "representative_label": labels[member_indices[0]] if member_indices else "maybe",
                        "representative_text": texts[member_indices[0]][:380] if member_indices else "",
                    }
                    for cid, member_indices in enumerate(clusters)
                ],
            }
            wc.write(json.dumps(cluster_payload, ensure_ascii=False) + "\n")

            entropy_payload = {
                "question_id": qid,
                "gold_label": gold,
                "pred_label": pred_label,
                "error": error,
                "num_samples": n,
                "num_clusters": len(clusters),
                "semantic_entropy": se,
            }
            we.write(json.dumps(entropy_payload, ensure_ascii=False) + "\n")
            entropy_rows.append(entropy_payload)

    if len(entropy_rows) >= 2:
        x = np.array([r["semantic_entropy"] for r in entropy_rows], dtype=float)
        y = np.array([r["error"] for r in entropy_rows], dtype=float)
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0

    corr_payload = {
        "num_questions": len(entropy_rows),
        "sim_threshold": args.sim_threshold,
        "require_same_label": args.require_same_label,
        "pearson": {"r": float(pearson_r), "p_value": float(pearson_p)},
        "spearman": {"rho": float(spearman_rho), "p_value": float(spearman_p)},
    }
    with open(args.correlation_out, "w", encoding="utf-8") as w:
        json.dump(corr_payload, w, ensure_ascii=False, indent=2)

    plot_entropy_scatter(entropy_rows, args.scatter_plot_out)
    plot_entropy_bucket_bar(entropy_rows, args.bucket_plot_out, args.n_bins)

    print(
        json.dumps(
            {
                "status": "ok",
                "num_questions": len(entropy_rows),
                "clusters_out": args.clusters_out,
                "entropy_out": args.entropy_out,
                "correlation_out": args.correlation_out,
                "scatter_plot_out": args.scatter_plot_out,
                "bucket_plot_out": args.bucket_plot_out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
