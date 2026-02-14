import argparse
import os
import subprocess
import sys


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


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
    parser.add_argument("--model", required=True, help="LLM model name for sampling stage.")
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--sim_threshold", type=float, default=0.62)
    parser.add_argument("--require_same_label", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set it before running this pipeline.")

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
        args.samples_out,
        "--model",
        args.model,
        "--m",
        str(args.m),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
    ]
    if args.max_items > 0:
        sample_cmd.extend(["--max_items", str(args.max_items)])
    run_step(sample_cmd, env)

    analyze_cmd = [
        sys.executable,
        "scripts/exp2_analyze_semantic_entropy.py",
        "--samples",
        args.samples_out,
        "--clusters_out",
        args.clusters_out,
        "--entropy_out",
        args.entropy_out,
        "--correlation_out",
        args.correlation_out,
        "--scatter_plot_out",
        args.scatter_out,
        "--bucket_plot_out",
        args.bucket_out,
        "--sim_threshold",
        str(args.sim_threshold),
    ]
    if args.require_same_label:
        analyze_cmd.append("--require_same_label")
    run_step(analyze_cmd, env)

    print("\nExperiment 2 pipeline finished successfully.")


if __name__ == "__main__":
    main()
