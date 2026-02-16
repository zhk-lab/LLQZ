import argparse
import os
import subprocess
import sys


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 then Experiment 3 sequentially.")
    parser.add_argument("--input", default="data/BioASQ/trainining14b.json")
    parser.add_argument("--llm_model", default=os.environ.get("OPENAI_MODEL", "").strip())
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "").strip())
    parser.add_argument("--max_questions", type=int, default=0, help="Experiment 1 question cap. 0 means all.")
    parser.add_argument("--exp3_max_questions", type=int, default=200, help="Experiment 3 question cap.")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rerank_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--exp1_workers", type=int, default=4)
    parser.add_argument("--exp3_num_candidates", type=int, default=3)
    parser.add_argument("--exp3_workers", type=int, default=4)
    parser.add_argument("--exp3_theta", type=float, default=1.0)
    parser.add_argument("--exp3_w_rel", type=float, default=1.0)
    parser.add_argument("--exp3_w_sup", type=float, default=2.0)
    parser.add_argument("--exp3_w_use", type=float, default=1.0)
    parser.add_argument("--exp1_resume_completed", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")
    if not args.llm_model:
        raise RuntimeError("Missing --llm_model (or OPENAI_MODEL env).")

    exp1_cmd = [
        sys.executable,
        "scripts/exp1_run_pipeline.py",
        "--input",
        args.input,
        "--llm_model",
        args.llm_model,
        "--base_url",
        args.base_url,
        "--max_questions",
        str(args.max_questions),
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
        "--workers",
        str(args.exp1_workers),
    ]
    if args.exp1_resume_completed:
        exp1_cmd.append("--resume_completed")
    run_step(exp1_cmd, env)

    run_step(
        [
            sys.executable,
            "scripts/exp3_run_pipeline.py",
            "--retrieval_source",
            "runs/exp1/hybrid_rerank/predictions.jsonl",
            "--model",
            args.llm_model,
            "--base_url",
            args.base_url,
            "--top_n",
            str(args.top_n),
            "--num_candidates",
            str(args.exp3_num_candidates),
            "--workers",
            str(args.exp3_workers),
            "--theta",
            str(args.exp3_theta),
            "--w_rel",
            str(args.exp3_w_rel),
            "--w_sup",
            str(args.exp3_w_sup),
            "--w_use",
            str(args.exp3_w_use),
            "--max_questions",
            str(args.exp3_max_questions),
        ],
        env,
    )

    print("\nExperiment 1 and 3 chained pipeline finished successfully.")


if __name__ == "__main__":
    main()
