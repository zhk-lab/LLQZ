import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI, OpenAIError
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


VALID_LABELS = {"yes", "no", "maybe"}
BAR_COLORS = ["#C9D4E5", "#95ACC9", "#4A6E9E"]
SCATTER_COLOR = "#4A6E9E"
LINE_COLOR = "#2F4E79"
LABEL_TO_SCORE = {"no": -1.0, "maybe": 0.0, "yes": 1.0}


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


def safe_corr(x: np.ndarray, y: np.ndarray, mode: str) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    if mode == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return float(r), float(p)


def load_cache(path: str) -> dict[str, bool]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return {str(k): bool(v) for k, v in payload.items()}


def save_cache(path: str, cache: dict[str, bool]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as w:
        json.dump(cache, w, ensure_ascii=False)


def cache_key(premise: str, hypothesis: str) -> str:
    raw = f"{premise}\n<SEP>\n{hypothesis}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_yes_no(text: str) -> bool:
    t = str(text or "").strip().lower()
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return t.startswith("y")


def llm_entails(
    client: OpenAI,
    model: str,
    premise: str,
    hypothesis: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> bool:
    prompt = (
        "Task: textual entailment.\n"
        "Given Premise and Hypothesis, decide if Hypothesis is logically entailed by Premise.\n"
        "Answer with exactly one token: YES or NO.\n\n"
        f"Premise: {premise}\n\n"
        f"Hypothesis: {hypothesis}\n"
    )
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strict NLI judge."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=1.0,
                max_tokens=3,
                timeout=timeout,
            )
            content = (resp.choices[0].message.content or "").strip()
            return parse_yes_no(content)
        except OpenAIError as e:
            last_err = e
    raise RuntimeError(f"NLI request failed after retries: {last_err}")


def build_tfidf_similarity(texts: list[str]) -> np.ndarray:
    n = len(texts)
    if n <= 1:
        return np.ones((n, n), dtype=float)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x = vectorizer.fit_transform(texts)
    return (x * x.T).toarray()


def greedy_semantic_cluster(
    texts: list[str],
    labels: list[str],
    backend: str,
    require_same_label: bool,
    tfidf_threshold: float,
    client: OpenAI | None,
    nli_model: str | None,
    nli_temp: float,
    nli_timeout: float,
    nli_retries: int,
    cache: dict[str, bool],
    stats: dict[str, int],
    cache_lock=None,
) -> tuple[list[int], list[list[int]]]:
    n = len(texts)
    if n == 0:
        return [], []

    sim = build_tfidf_similarity(texts) if backend == "tfidf" else None
    clusters: list[list[int]] = []
    reps: list[int] = []
    cluster_ids = [-1] * n

    def equivalent(i: int, j: int) -> bool:
        if texts[i] == texts[j]:
            return True
        if require_same_label and labels[i] != labels[j]:
            return False
        if backend == "tfidf":
            return bool(sim[i][j] >= tfidf_threshold)

        assert client is not None and nli_model is not None
        k1 = cache_key(texts[i], texts[j])
        if cache_lock is not None:
            with cache_lock:
                has_k1 = k1 in cache
                if has_k1:
                    e_ij = cache[k1]
                    stats["cache_hit"] += 1
        else:
            has_k1 = k1 in cache
            if has_k1:
                e_ij = cache[k1]
                stats["cache_hit"] += 1

        if not has_k1:
            e_ij = llm_entails(client, nli_model, texts[i], texts[j], nli_temp, nli_timeout, nli_retries)
            if cache_lock is not None:
                with cache_lock:
                    if k1 not in cache:
                        cache[k1] = e_ij
                        stats["cache_miss"] += 1
                    else:
                        e_ij = cache[k1]
                        stats["cache_hit"] += 1
            else:
                cache[k1] = e_ij
                stats["cache_miss"] += 1

        if not e_ij:
            return False

        k2 = cache_key(texts[j], texts[i])
        if cache_lock is not None:
            with cache_lock:
                has_k2 = k2 in cache
                if has_k2:
                    e_ji = cache[k2]
                    stats["cache_hit"] += 1
        else:
            has_k2 = k2 in cache
            if has_k2:
                e_ji = cache[k2]
                stats["cache_hit"] += 1

        if not has_k2:
            e_ji = llm_entails(client, nli_model, texts[j], texts[i], nli_temp, nli_timeout, nli_retries)
            if cache_lock is not None:
                with cache_lock:
                    if k2 not in cache:
                        cache[k2] = e_ji
                        stats["cache_miss"] += 1
                    else:
                        e_ji = cache[k2]
                        stats["cache_hit"] += 1
            else:
                cache[k2] = e_ji
                stats["cache_miss"] += 1
        return bool(e_ij and e_ji)

    for i in range(n):
        assigned = False
        for cid, rep_idx in enumerate(reps):
            if equivalent(i, rep_idx):
                clusters[cid].append(i)
                cluster_ids[i] = cid
                assigned = True
                break
        if not assigned:
            cid = len(clusters)
            clusters.append([i])
            reps.append(i)
            cluster_ids[i] = cid

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


def maybe_debiased_vote(
    labels: list[str], maybe_margin_max: int, maybe_alt_min_count: int
) -> tuple[str, dict]:
    counts = Counter(labels)
    major = majority_vote(labels)
    debug = {"counts": dict(counts), "raw_majority": major, "switched": False}
    if major != "maybe":
        return major, debug
    yes_c = counts.get("yes", 0)
    no_c = counts.get("no", 0)
    maybe_c = counts.get("maybe", 0)
    alt = "yes" if yes_c >= no_c else "no"
    alt_c = max(yes_c, no_c)
    if alt_c >= maybe_alt_min_count and (maybe_c - alt_c) <= maybe_margin_max:
        debug["switched"] = True
        debug["to"] = alt
        return alt, debug
    return major, debug


def compute_errors(pred: str, gold: str) -> dict:
    pred = normalize_label(pred)
    gold = normalize_label(gold)
    binary = int(pred != gold)
    ordinal = abs(LABEL_TO_SCORE[pred] - LABEL_TO_SCORE[gold]) / 2.0
    severe = int((pred == "yes" and gold == "no") or (pred == "no" and gold == "yes"))
    maybe_overuse = int(pred == "maybe" and gold in {"yes", "no"})
    return {
        "error_binary": binary,
        "error_ordinal": float(ordinal),
        "error_severe": severe,
        "maybe_overuse": maybe_overuse,
    }


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
    y = np.array([r["error_for_plot"] for r in rows], dtype=float)

    # Add small y-jitter for readability.
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
    ax.set_ylabel("Error Score (0=Correct, 1=Wrong)")
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
    errors = np.array([r["error_for_plot"] for r in rows], dtype=float)

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
        "--semantic_backend",
        choices=["nli_llm", "tfidf"],
        default="nli_llm",
        help="Semantic equivalence backend. nli_llm follows paper-style bidirectional entailment.",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.62,
        help="Cosine threshold for tfidf backend.",
    )
    parser.add_argument(
        "--require_same_label",
        action="store_true",
        help="If set, only merge samples with identical final_label.",
    )
    parser.add_argument(
        "--prediction_mode",
        choices=["majority_vote", "maybe_debiased"],
        default="maybe_debiased",
        help="How to derive final prediction label from M samples.",
    )
    parser.add_argument(
        "--plot_error_metric",
        choices=["binary", "ordinal"],
        default="ordinal",
        help="Error metric visualized on plots and used for primary correlation.",
    )
    parser.add_argument(
        "--maybe_margin_max",
        type=int,
        default=1,
        help="For maybe_debiased mode: allow switch from maybe if margin is small.",
    )
    parser.add_argument(
        "--maybe_alt_min_count",
        type=int,
        default=3,
        help="For maybe_debiased mode: min vote count required for yes/no alternative.",
    )
    parser.add_argument(
        "--nli_model",
        default=os.environ.get("NLI_MODEL", "").strip(),
        help="NLI judge model for nli_llm backend. If empty, use OPENAI_MODEL env.",
    )
    parser.add_argument(
        "--nli_base_url",
        default=os.environ.get("OPENAI_BASE_URL", "").strip(),
        help="OpenAI-compatible base URL for NLI calls.",
    )
    parser.add_argument(
        "--nli_temp",
        type=float,
        default=0.0,
        help="Temperature for NLI judge calls.",
    )
    parser.add_argument("--nli_timeout", type=float, default=90.0)
    parser.add_argument("--nli_retries", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Question-level parallel workers for semantic clustering/NLI.",
    )
    parser.add_argument(
        "--nli_cache",
        default="runs/exp2/nli_cache.json",
        help="Cache file for entailment decisions to speed re-runs.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=6,
        help="Number of entropy buckets for visualization.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.samples)
    nli_client = None
    nli_model = None
    nli_cache = {}
    nli_stats = {"cache_hit": 0, "cache_miss": 0}
    cache_lock = Lock()
    if args.semantic_backend == "nli_llm":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for nli_llm backend.")
        nli_model = args.nli_model or os.environ.get("OPENAI_MODEL", "").strip()
        if not nli_model:
            raise RuntimeError("Missing --nli_model (or OPENAI_MODEL) for nli_llm backend.")
        nli_client = OpenAI(api_key=api_key, base_url=args.nli_base_url or None)
        nli_cache = load_cache(args.nli_cache)
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        qid = str(r.get("question_id", "")).strip()
        if not qid:
            continue
        by_qid[qid].append(r)

    ensure_parent_dir(args.clusters_out)
    ensure_parent_dir(args.entropy_out)
    ensure_parent_dir(args.correlation_out)

    def process_one(qid: str, items: list[dict]) -> tuple[str, dict, dict]:
        items = sorted(items, key=lambda x: int(x.get("sample_id", 0)))
        texts = [str(it.get("rationale_text", "")).strip() for it in items]
        labels = [normalize_label(it.get("final_label", "")) for it in items]
        gold = normalize_label(items[0].get("gold_label", ""))

        cluster_ids, clusters = greedy_semantic_cluster(
            texts=texts,
            labels=labels,
            backend=args.semantic_backend,
            require_same_label=args.require_same_label,
            tfidf_threshold=args.sim_threshold,
            client=nli_client,
            nli_model=nli_model,
            nli_temp=args.nli_temp,
            nli_timeout=args.nli_timeout,
            nli_retries=args.nli_retries,
            cache=nli_cache,
            stats=nli_stats,
            cache_lock=cache_lock,
        )

        n = max(1, len(items))
        cluster_sizes = [len(c) for c in clusters]
        cluster_probs = [s / n for s in cluster_sizes]
        se = shannon_entropy(cluster_probs)

        raw_majority = majority_vote(labels)
        if args.prediction_mode == "maybe_debiased":
            pred_label, vote_debug = maybe_debiased_vote(
                labels, maybe_margin_max=args.maybe_margin_max, maybe_alt_min_count=args.maybe_alt_min_count
            )
        else:
            pred_label = raw_majority
            vote_debug = {"raw_majority": raw_majority, "switched": False}
        err = compute_errors(pred_label, gold)

        cluster_payload = {
            "question_id": qid,
            "question": items[0].get("question", ""),
            "gold_label": gold,
            "pred_label_majority_vote": pred_label,
            "raw_majority_label": raw_majority,
            "vote_debug": vote_debug,
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
        entropy_payload = {
            "question_id": qid,
            "gold_label": gold,
            "pred_label_raw_majority": raw_majority,
            "pred_label": pred_label,
            "error_binary": err["error_binary"],
            "error_ordinal": err["error_ordinal"],
            "error_severe": err["error_severe"],
            "maybe_overuse": err["maybe_overuse"],
            "num_samples": n,
            "num_clusters": len(clusters),
            "semantic_entropy": se,
        }
        entropy_payload["error_for_plot"] = (
            err["error_ordinal"] if args.plot_error_metric == "ordinal" else err["error_binary"]
        )
        return qid, cluster_payload, entropy_payload

    q_items = list(by_qid.items())
    results: dict[str, tuple[dict, dict]] = {}
    if args.workers <= 1:
        for qid, items in tqdm(q_items, desc="Analyzing questions"):
            _, cluster_payload, entropy_payload = process_one(qid, items)
            results[qid] = (cluster_payload, entropy_payload)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, qid, items) for qid, items in q_items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Analyzing questions"):
                qid, cluster_payload, entropy_payload = fut.result()
                results[qid] = (cluster_payload, entropy_payload)

    entropy_rows = []
    with open(args.clusters_out, "w", encoding="utf-8") as wc, open(
        args.entropy_out, "w", encoding="utf-8"
    ) as we:
        for qid, _ in q_items:
            cluster_payload, entropy_payload = results[qid]
            wc.write(json.dumps(cluster_payload, ensure_ascii=False) + "\n")
            we.write(json.dumps(entropy_payload, ensure_ascii=False) + "\n")
            entropy_rows.append(entropy_payload)

    if args.semantic_backend == "nli_llm":
        save_cache(args.nli_cache, nli_cache)

    x = np.array([r["semantic_entropy"] for r in entropy_rows], dtype=float)
    y_bin = np.array([r["error_binary"] for r in entropy_rows], dtype=float)
    y_ord = np.array([r["error_ordinal"] for r in entropy_rows], dtype=float)
    y_plot = np.array([r["error_for_plot"] for r in entropy_rows], dtype=float)
    pearson_bin_r, pearson_bin_p = safe_corr(x, y_bin, "pearson")
    spearman_bin_r, spearman_bin_p = safe_corr(x, y_bin, "spearman")
    pearson_ord_r, pearson_ord_p = safe_corr(x, y_ord, "pearson")
    spearman_ord_r, spearman_ord_p = safe_corr(x, y_ord, "spearman")
    pearson_plot_r, pearson_plot_p = safe_corr(x, y_plot, "pearson")
    spearman_plot_r, spearman_plot_p = safe_corr(x, y_plot, "spearman")

    raw_counts = Counter(r["pred_label_raw_majority"] for r in entropy_rows)
    final_counts = Counter(r["pred_label"] for r in entropy_rows)
    n_q = max(1, len(entropy_rows))

    corr_payload = {
        "num_questions": len(entropy_rows),
        "semantic_backend": args.semantic_backend,
        "sim_threshold_tfidf": args.sim_threshold,
        "require_same_label": args.require_same_label,
        "prediction_mode": args.prediction_mode,
        "plot_error_metric": args.plot_error_metric,
        "pearson_plot": {"r": pearson_plot_r, "p_value": pearson_plot_p},
        "spearman_plot": {"rho": spearman_plot_r, "p_value": spearman_plot_p},
        "pearson_binary": {"r": pearson_bin_r, "p_value": pearson_bin_p},
        "spearman_binary": {"rho": spearman_bin_r, "p_value": spearman_bin_p},
        "pearson_ordinal": {"r": pearson_ord_r, "p_value": pearson_ord_p},
        "spearman_ordinal": {"rho": spearman_ord_r, "p_value": spearman_ord_p},
        "maybe_rate_raw_majority": raw_counts.get("maybe", 0) / n_q,
        "maybe_rate_final_pred": final_counts.get("maybe", 0) / n_q,
        "pred_distribution_raw_majority": dict(raw_counts),
        "pred_distribution_final": dict(final_counts),
        "mean_error_binary": float(np.mean(y_bin)) if len(y_bin) else 0.0,
        "mean_error_ordinal": float(np.mean(y_ord)) if len(y_ord) else 0.0,
        "maybe_overuse_rate": float(np.mean([r["maybe_overuse"] for r in entropy_rows])) if entropy_rows else 0.0,
        "nli_cache": {
            "path": args.nli_cache,
            "cache_hit": nli_stats["cache_hit"],
            "cache_miss": nli_stats["cache_miss"],
            "cache_size": len(nli_cache),
        },
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
