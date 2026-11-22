"""
Automated Benchmark Runner for the Best 3 Extra Small Models:
1. qwen2.5-coder:1.5b (1.54B params, ~1.3 GB VRAM)
2. llama3.2:3b         (3.21B params, ~2.2 GB VRAM)
3. codegemma:2b        (2.50B params, ~1.8 GB VRAM)

Features:
- 1-click execution for any or all 3 small models
- Automatic VRAM unloading (keep_alive=0) between models to prevent GPU memory pressure
- Smoke test mode (--smoke_test N) for rapid verification
- Full HumanEval (164 tasks) or MBPP evaluation
- Summary report generation with Pass@1, tokens, iterations, and latency

Usage:
  # Quick 5-task smoke test on Qwen2.5-Coder 1.5B:
  python -m my_pipeline.run_small_models --model qwen2.5-coder:1.5b --smoke_test 5

  # Run all 3 small models sequentially on HumanEval:
  python -m my_pipeline.run_small_models --all --config +module1

  # Run Llama-3.2 3B full benchmark:
  python -m my_pipeline.run_small_models --model llama3.2:3b --config +module1
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path

from shared.logger import ResultLogger
from shared.dataset_loader import load_dataset
from shared.models import query_llm, OLLAMA_URL
from my_pipeline.pipeline import run_task, parse_config

# The 3 Best Extra Small Models (< 3.5B params)
EXTRA_SMALL_MODELS = [
    {
        "name": "qwen2.5-coder:1.5b",
        "description": "Alibaba Qwen2.5-Coder 1.5B (1.54B params, ~1.3 GB VRAM)",
        "ollama_tag": "qwen2.5-coder:1.5b",
    },
    {
        "name": "llama3.2:3b",
        "description": "Meta Llama-3.2 3B Instruct (3.21B params, ~2.2 GB VRAM)",
        "ollama_tag": "llama3.2:3b",
    },
    {
        "name": "codegemma:2b",
        "description": "Google CodeGemma 2B Instruct (2.50B params, ~1.8 GB VRAM)",
        "ollama_tag": "codegemma:2b",
    },
    {
        "name": "starcoder:1b",
        "description": "BigCode StarCoder 1B (1.14B params, ~0.9 GB VRAM)",
        "ollama_tag": "starcoder:1b",
    },
]

_BUNDLE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = {
    "humaneval": str(_BUNDLE_ROOT / "shared" / "datasets" / "HumanEval.jsonl"),
    "mbpp": str(_BUNDLE_ROOT / "shared" / "datasets" / "mbpp.jsonl"),
}


def unload_model_from_vram(model_name: str):
    """Tell Ollama to immediately unload the model from GPU VRAM."""
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
        time.sleep(2)
    except Exception:
        pass


def prewarm_model(model_name: str) -> bool:
    """Send a tiny test prompt to ensure model is pulled and resident in VRAM."""
    print(f"--> Pre-warming {model_name} in GPU VRAM...")
    try:
        res = query_llm("def ping(): return True", model=model_name, max_tokens=16)
        print(f"    [OK] Model warm! Latency: {res['latency_sec']}s, Generated {res['completion_tokens']} tokens.")
        return True
    except Exception as e:
        print(f"    [!] Error warming {model_name}: {e}")
        print(f"    Make sure you pulled it first with: ollama pull {model_name}")
        return False


def run_benchmark_for_model(
    model_name: str,
    dataset_name: str = "humaneval",
    config_str: str = "+module1",
    smoke_test: int = None,
    resume: bool = True,
):
    print("\n" + "=" * 70)
    print(f" RUNNING BENCHMARK: {model_name} on {dataset_name.upper()} (Config: {config_str})")
    print("=" * 70)

    # 1. Warm model
    if not prewarm_model(model_name):
        print(f"Skipping {model_name} due to warm-up failure.")
        return None

    # 2. Load dataset
    dataset_path = DEFAULT_DATASETS[dataset_name]
    tasks = load_dataset(dataset_name, dataset_path)
    if smoke_test:
        tasks = tasks[:smoke_test]
        print(f"Smoke test mode: Running first {len(tasks)} tasks.")
    else:
        print(f"Loaded {len(tasks)} tasks from {dataset_path}.")

    # 3. Setup log path
    safe_model = model_name.replace(":", "-")
    config_slug = config_str.lstrip("+")
    os.makedirs("results", exist_ok=True)
    log_path = f"results/mypipeline_{dataset_name}_{config_slug}_{safe_model}.jsonl"
    logger = ResultLogger(log_path)

    already_done = logger.already_done() if resume else set()
    remaining = [t for t in tasks if t["task_id"] not in already_done]
    print(f"Total tasks: {len(tasks)} | Already completed: {len(already_done)} | Remaining to execute: {len(remaining)}")

    # 4. Parse config modules
    modules = parse_config(config_str)

    # 5. Execute benchmark loop
    passed_count = 0
    total_tokens = 0
    total_iters = 0
    total_time = 0

    start_all = time.time()
    for idx, task in enumerate(tasks, start=1):
        task_id = task["task_id"]
        if task_id in already_done:
            continue

        prompt = task["prompt"]
        entry_point = task["entry_point"]
        tests = task["tests"]

        print(f"[{idx}/{len(tasks)}] Task {task_id} ...", end=" ", flush=True)
        task_start = time.time()

        try:
            row = run_task(
                task,
                model=model_name,
                config=config_str,
                dataset=dataset_name,
                oracle_model=model_name,
                judge_model=model_name,
            )
            elapsed = time.time() - task_start
            status = "PASS" if row.get("final_passed", False) else "FAIL"
            iters = row.get("iterations", 1)
            toks = row.get("total_tokens", 0)

            print(f"{status} | Iters: {iters} | Tokens: {toks} | Time: {elapsed:.2f}s")
            logger.log(**row)

        except Exception as e:
            elapsed = time.time() - task_start
            print(f"ERROR ({e}) | Time: {elapsed:.2f}s")
            fallback_row = {
                "task_id": task_id,
                "framework": "mypipeline",
                "model": model_name,
                "dataset": dataset_name,
                "config": config_str,
                "final_passed": False,
                "total_tokens": 0,
                "iterations": 1,
                "total_time_sec": round(elapsed, 2),
                "error_type": type(e).__name__,
            }
            logger.log(**fallback_row)

    # 6. Compute statistics from log file
    total_done = 0
    passed_done = 0
    tokens_list = []
    iters_list = []

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    total_done += 1
                    if rec.get("final_passed", False):
                        passed_done += 1
                    tokens_list.append(rec.get("total_tokens", 0))
                    iters_list.append(rec.get("iterations", 1))

    pass_rate = (passed_done / total_done * 100) if total_done > 0 else 0.0
    avg_tokens = (sum(tokens_list) / len(tokens_list)) if tokens_list else 0.0
    avg_iters = (sum(iters_list) / len(iters_list)) if iters_list else 0.0
    total_wall_time = time.time() - start_all

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "config": config_str,
        "total_tasks": total_done,
        "passed": passed_done,
        "pass_rate_pct": round(pass_rate, 2),
        "avg_tokens": round(avg_tokens, 1),
        "avg_iterations": round(avg_iters, 2),
        "total_time_sec": round(total_wall_time, 2),
        "log_path": log_path,
    }

    print("\n" + "-" * 50)
    print(f"SUMMARY FOR {model_name}:")
    print(f"  Pass@1:          {summary['pass_rate_pct']}% ({passed_done}/{total_done})")
    print(f"  Avg Tokens:      {summary['avg_tokens']}")
    print(f"  Avg Iterations:  {summary['avg_iterations']}")
    print(f"  Results saved to: {log_path}")
    print("-" * 50)

    # 7. Unload from VRAM so next model has clean GPU memory
    unload_model_from_vram(model_name)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on the best extra small models (<3.5B)")
    parser.add_argument(
        "--model",
        choices=["qwen2.5-coder:1.5b", "llama3.2:3b", "codegemma:2b", "starcoder:1b", "starcoder2:3b"],
        default=None,
        help="Specify which small model to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all 3 small models sequentially")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--config", default="+module1", help="Pipeline configuration (e.g. baseline, +module1, +module1+module2+module3)")
    parser.add_argument("--smoke_test", type=int, default=None, help="Run only first N tasks for quick testing (e.g. --smoke_test 5)")
    parser.add_argument("--no_resume", action="store_true", help="Do not skip already completed tasks")

    args = parser.parse_args()

    models_to_run = []
    if args.all:
        models_to_run = [m["name"] for m in EXTRA_SMALL_MODELS]
    elif args.model:
        models_to_run = [args.model]
    else:
        print("Please specify --model <name> or --all to run all 3 small models.")
        print("\nAvailable Small Models:")
        for m in EXTRA_SMALL_MODELS:
            print(f"  - {m['name']:<22} : {m['description']}")
        print("\nExample:")
        print("  python -m my_pipeline.run_small_models --model qwen2.5-coder:1.5b --smoke_test 5")
        sys.exit(1)

    results_table = []
    for model in models_to_run:
        summary = run_benchmark_for_model(
            model_name=model,
            dataset_name=args.dataset,
            config_str=args.config,
            smoke_test=args.smoke_test,
            resume=not args.no_resume,
        )
        if summary:
            results_table.append(summary)

    if results_table:
        print("\n" + "=" * 80)
        print(f" FINAL BENCHMARK SUMMARY TABLE ({args.dataset.upper()} - {args.config})")
        print("=" * 80)
        print(f"{'Model':<22} | {'Pass@1 (%)':<12} | {'Avg Tokens':<12} | {'Avg Iters':<10} | {'Solved / Total':<14}")
        print("-" * 80)
        for r in results_table:
            print(f"{r['model']:<22} | {str(r['pass_rate_pct']) + '%':<12} | {r['avg_tokens']:<12} | {r['avg_iterations']:<10} | {str(r['passed']) + '/' + str(r['total_tasks']):<14}")
        print("=" * 80)


if __name__ == "__main__":
    main()
