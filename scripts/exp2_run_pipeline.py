import argparse
import os
import subprocess
import sys


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def with_run_tag(path: str, run_tag: str) -> str:
    if not run_tag:
        return path
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}_{run_tag}{ext}"
    return f"{path}_{run_tag}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2 pipeline end-to-end.")
    parser.add_argument("--input", default="data/pubmedqa/ori_pqal.json")
    parser.add_argument("--gold_out", default="data/pubmedqa/gold_labels.jsonl")
    parser.add_argument("--samples_out", default="runs/exp2/samples.jsonl")
    parser.add_argument("--clusters_out", default="runs/exp2/clusters.jsonl")
    parser.add_argument("--entropy_out", default="runs/exp2/semantic_entropy.jsonl")
    parser.add_argument("--correlation_out", default="runs/exp2/correlation.json")
    parser.add_argument("--scatter_out", default="plots/exp2_entropy_scatter.png")
    parser.add_argument("--bucket_out", default="plots/exp2_entropy_bucket_bar.png")
    parser.add_argument(
        "--run_tag",
        default="",
        help="Optional output suffix (e.g., v2). Writes samples_v2/clusters_v2/... without overwriting existing results.",
    )
    parser.add_argument("--model", required=True, help="LLM model name for sampling stage.")
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=140)
    parser.add_argument("--max_context_chars", type=int, default=2500)
    parser.add_argument(
        "--context_trim_mode",
        choices=["head", "head_tail"],
        default="head_tail",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--concise_mode", action="store_true")
    parser.add_argument("--append_no_think", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--sim_threshold", type=float, default=0.62)
    parser.add_argument("--require_same_label", action="store_true")
    parser.add_argument(
        "--semantic_backend",
        choices=["nli_llm", "tfidf"],
        default="nli_llm",
    )
    parser.add_argument(
        "--prediction_mode",
        choices=["majority_vote", "maybe_debiased"],
        default="maybe_debiased",
    )
    parser.add_argument(
        "--plot_error_metric",
        choices=["binary", "ordinal"],
        default="ordinal",
    )
    parser.add_argument("--nli_model", default="")
    parser.add_argument("--nli_base_url", default="")
    parser.add_argument("--nli_timeout", type=float, default=90.0)
    parser.add_argument("--nli_retries", type=int, default=3)
    parser.add_argument("--nli_workers", type=int, default=4)
    parser.add_argument("--nli_cache", default="runs/exp2/nli_cache.json")
    parser.add_argument("--maybe_margin_max", type=int, default=1)
    parser.add_argument("--maybe_alt_min_count", type=int, default=3)
    args = parser.parse_args()

    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set it before running this pipeline.")

    samples_out = with_run_tag(args.samples_out, args.run_tag)
    clusters_out = with_run_tag(args.clusters_out, args.run_tag)
    entropy_out = with_run_tag(args.entropy_out, args.run_tag)
    correlation_out = with_run_tag(args.correlation_out, args.run_tag)
    scatter_out = with_run_tag(args.scatter_out, args.run_tag)
    bucket_out = with_run_tag(args.bucket_out, args.run_tag)

    run_step(
        [
            sys.executable,
            "scripts/exp2_build_gold_labels.py",
            "--input",
            args.input,
            "--out",
            args.gold_out,
        ],
        env,
    )

    sample_cmd = [
        sys.executable,
        "scripts/exp2_sample_pubmedqa_openai.py",
        "--input",
        args.input,
        "--out",
        samples_out,
        "--model",
        args.model,
        "--m",
        str(args.m),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_tokens",
        str(args.max_tokens),
        "--max_context_chars",
        str(args.max_context_chars),
        "--context_trim_mode",
        args.context_trim_mode,
        "--workers",
        str(args.workers),
        "--request_timeout",
        str(args.request_timeout),
        "--max_retries",
        str(args.max_retries),
    ]
    if args.concise_mode:
        sample_cmd.append("--concise_mode")
    if args.append_no_think:
        sample_cmd.append("--append_no_think")
    if args.resume:
        sample_cmd.append("--resume")
    if args.max_items > 0:
        sample_cmd.extend(["--max_items", str(args.max_items)])
    run_step(sample_cmd, env)

    analyze_cmd = [
        sys.executable,
        "scripts/exp2_analyze_semantic_entropy.py",
        "--samples",
        samples_out,
        "--clusters_out",
        clusters_out,
        "--entropy_out",
        entropy_out,
        "--correlation_out",
        correlation_out,
        "--scatter_plot_out",
        scatter_out,
        "--bucket_plot_out",
        bucket_out,
        "--sim_threshold",
        str(args.sim_threshold),
        "--semantic_backend",
        args.semantic_backend,
        "--prediction_mode",
        args.prediction_mode,
        "--plot_error_metric",
        args.plot_error_metric,
        "--nli_timeout",
        str(args.nli_timeout),
        "--nli_retries",
        str(args.nli_retries),
        "--workers",
        str(args.nli_workers),
        "--nli_cache",
        args.nli_cache,
        "--maybe_margin_max",
        str(args.maybe_margin_max),
        "--maybe_alt_min_count",
        str(args.maybe_alt_min_count),
    ]
    if args.nli_model:
        analyze_cmd.extend(["--nli_model", args.nli_model])
    if args.nli_base_url:
        analyze_cmd.extend(["--nli_base_url", args.nli_base_url])
    if args.require_same_label:
        analyze_cmd.append("--require_same_label")
    run_step(analyze_cmd, env)

    print("\nExperiment 2 pipeline finished successfully.")


if __name__ == "__main__":
    main()
