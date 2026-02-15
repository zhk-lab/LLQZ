import argparse
import os
import subprocess
import sys


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 3 pipeline end-to-end.")
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
    parser.add_argument("--correctness_threshold", type=float, default=0.20)
    parser.add_argument("--summary_out", default="runs/exp3/metrics_summary.json")
    parser.add_argument("--plot_metrics_out", default="plots/exp3_quality_metrics.png")
    parser.add_argument("--plot_correction_out", default="plots/exp3_correction_gain.png")
    args = parser.parse_args()

    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")
    if not args.model:
        raise RuntimeError("Missing --model (or OPENAI_MODEL env).")

    run_step(
        [
            sys.executable,
            "scripts/exp3_run_selfrag.py",
            "--retrieval_source",
            args.retrieval_source,
            "--out_root",
            args.out_root,
            "--model",
            args.model,
            "--base_url",
            args.base_url,
            "--top_n",
            str(args.top_n),
            "--num_candidates",
            str(args.num_candidates),
            "--theta",
            str(args.theta),
            "--w_rel",
            str(args.w_rel),
            "--w_sup",
            str(args.w_sup),
            "--w_use",
            str(args.w_use),
            "--max_questions",
            str(args.max_questions),
            "--max_tokens_answer",
            str(args.max_tokens_answer),
            "--request_timeout",
            str(args.request_timeout),
            "--request_retries",
            str(args.request_retries),
        ],
        env,
    )

    run_step(
        [
            sys.executable,
            "scripts/exp3_evaluate_and_plot.py",
            "--naive",
            os.path.join(args.out_root, "naive_rag", "predictions.jsonl"),
            "--wo_weight",
            os.path.join(args.out_root, "self_rag_wo_weight", "predictions.jsonl"),
            "--full",
            os.path.join(args.out_root, "self_rag_full", "predictions.jsonl"),
            "--correctness_threshold",
            str(args.correctness_threshold),
            "--summary_out",
            args.summary_out,
            "--plot_metrics_out",
            args.plot_metrics_out,
            "--plot_correction_out",
            args.plot_correction_out,
        ],
        env,
    )

    print("\nExperiment 3 pipeline finished successfully.")


if __name__ == "__main__":
    main()
