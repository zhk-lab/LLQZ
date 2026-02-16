import argparse
import os
import subprocess
import sys


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 pipeline end-to-end.")
    parser.add_argument("--input", default="data/BioASQ/trainining14b.json")
    parser.add_argument("--corpus_out", default="corpus/bioasq_chunks.jsonl")
    parser.add_argument("--qa_out", default="data/bioasq/qa.jsonl")
    parser.add_argument("--pred_root", default="runs/exp1")
    parser.add_argument("--strategies", default="sparse_only,dense_only,hybrid,hybrid_rerank")
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rerank_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--llm_model", default=os.environ.get("OPENAI_MODEL", "").strip())
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "").strip())
    parser.add_argument("--max_gen_tokens", type=int, default=260)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--request_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume_completed",
        action="store_true",
        help="Skip strategies whose predictions.jsonl already has target question count.",
    )
    parser.add_argument("--metrics_out", default="runs/exp1/metrics_summary.json")
    parser.add_argument("--plot_metrics_out", default="plots/exp1_retrieval_metrics.png")
    parser.add_argument("--plot_cost_out", default="plots/exp1_effect_vs_cost.png")
    args = parser.parse_args()

    env = os.environ.copy()
    if not args.skip_generation and not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required unless --skip_generation is set.")

    run_step(
        [
            sys.executable,
            "scripts/exp1_prepare_bioasq_data.py",
            "--input",
            args.input,
            "--corpus_out",
            args.corpus_out,
            "--qa_out",
            args.qa_out,
            "--max_questions",
            str(args.max_questions),
        ],
        env,
    )

    target_questions = count_jsonl_lines(args.qa_out)
    strategy_list = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if args.resume_completed:
        pending = []
        for s in strategy_list:
            pred_path = os.path.join(args.pred_root, s, "predictions.jsonl")
            done_n = count_jsonl_lines(pred_path)
            if done_n >= target_questions > 0:
                print(f"[exp1] skip completed strategy: {s} ({done_n}/{target_questions})")
            else:
                pending.append(s)
        strategy_list = pending

    run_cmd = [
        sys.executable,
        "scripts/exp1_run_retrieval_qa.py",
        "--corpus",
        args.corpus_out,
        "--qa",
        args.qa_out,
        "--out_dir",
        args.pred_root,
        "--strategies",
        ",".join(strategy_list),
        "--top_k",
        str(args.top_k),
        "--top_n",
        str(args.top_n),
        "--alpha",
        str(args.alpha),
        "--dense_model",
        args.dense_model,
        "--rerank_model",
        args.rerank_model,
        "--max_questions",
        str(args.max_questions),
        "--max_gen_tokens",
        str(args.max_gen_tokens),
        "--request_timeout",
        str(args.request_timeout),
        "--request_retries",
        str(args.request_retries),
        "--workers",
        str(args.workers),
    ]
    if strategy_list:
        if args.skip_generation:
            run_cmd.append("--skip_generation")
        else:
            run_cmd.extend(["--llm_model", args.llm_model, "--base_url", args.base_url])
        run_step(run_cmd, env)
    else:
        print("[exp1] all strategies already completed. Skip retrieval+generation stage.")

    eval_cmd = [
        sys.executable,
        "scripts/exp1_evaluate_and_plot.py",
        "--pred_root",
        args.pred_root,
        "--strategies",
        args.strategies,
        "--top_k",
        str(args.top_k),
        "--metrics_out",
        args.metrics_out,
        "--plot_metrics_out",
        args.plot_metrics_out,
        "--plot_cost_out",
        args.plot_cost_out,
    ]
    run_step(eval_cmd, env)

    print("\nExperiment 1 pipeline finished successfully.")


if __name__ == "__main__":
    main()
