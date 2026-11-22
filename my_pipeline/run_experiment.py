"""
Colab driver for Phases C/D/E. Runs the pipeline over HumanEval or MBPP for
one model and one config, logging one row per task to
results/mypipeline_<dataset>_<config_slug>_<model>.jsonl.

--config takes "baseline" or any "+"-joined module combo -- pipeline.py's
parse_config() is the source of truth for what's valid, so this file
doesn't keep its own separate list to fall out of sync.

Examples:
    python -m my_pipeline.run_experiment --dataset_name humaneval --model qwen2.5-coder:3b --config baseline
    python -m my_pipeline.run_experiment --dataset_name mbpp --model qwen2.5-coder:3b --config +module1
    python -m my_pipeline.run_experiment --dataset_name humaneval --model deepseek-coder:6.7b --config +module1+module2
    python -m my_pipeline.run_experiment --dataset_name humaneval --model deepseek-coder:6.7b --config +module1+module2+module3

    # quick smoke test before committing to a full run
    python -m my_pipeline.run_experiment --dataset_name mbpp --model qwen2.5-coder:3b --config baseline --sample_size 5

--resume skips task_ids already present in the log file, so a Colab session
timeout doesn't cost you progress.
"""
import argparse
import random
import sys

from shared.logger import ResultLogger
from shared.dataset_loader import load_dataset
from shared.trace_logger import TraceLogger
from my_pipeline.pipeline import run_task, parse_config

from pathlib import Path

_BUNDLE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = {
    "humaneval": str(_BUNDLE_ROOT / "shared" / "datasets" / "HumanEval.jsonl"),
    "mbpp": str(_BUNDLE_ROOT / "shared" / "datasets" / "mbpp.jsonl"),
}


def _config_arg(value: str) -> str:
    """argparse type= hook -- reuses pipeline.py's own validation so a typo
    like '+module5' fails at startup instead of an hour into a Colab run."""
    parse_config(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=_config_arg, required=True,
                         help='"baseline" or a "+"-joined module combo, e.g. "+module1+module2+module3"')
    parser.add_argument("--oracle_model", default=None,
                         help="Model for module2's test-writer. Defaults to --model.")
    parser.add_argument("--judge_model", default=None,
                         help="Model for module3's LLM judge. Defaults to --model.")
    parser.add_argument("--dataset_path", default=None,
                         help="Override the default path for --dataset_name")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--strip_tests", action="store_true",
                         help="Strip test cases from the debugging loop for Phase D zero-test evaluation.")
    parser.add_argument("--task_list", default=None,
                         help="Path to JSON file containing specific task IDs to run (e.g. evidence_package/selected_100_task_ids.json)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun_failed", action="store_true",
                         help="Re-runs previously failed tasks while skipping tasks that already passed.")
    parser.add_argument("--trace", action="store_true",
                         help="Log every regeneration step (plan + code) for diagnosis. "
                              "Use on a small --sample_size, not full runs.")
    args = parser.parse_args()

    dataset_path = args.dataset_path or DEFAULT_PATHS[args.dataset_name]
    tasks = load_dataset(args.dataset_name, dataset_path)

    if args.task_list:
        import json
        with open(args.task_list) as f:
            selected_ids = set(json.load(f))
        tasks = [t for t in tasks if t["task_id"] in selected_ids]
        print(f"Filtered to {len(tasks)} tasks from {args.task_list}")

    if args.sample_size:
        random.seed(args.seed)
        tasks = random.sample(tasks, min(args.sample_size, len(tasks)))

    safe_model_name = args.model.replace(":", "-")
    config_slug = args.config.lstrip("+")
    if args.strip_tests:
        log_path = f"results/mypipeline_zerotest_{config_slug}_{safe_model_name}.jsonl"
    else:
        log_path = f"results/mypipeline_{args.dataset_name}_{config_slug}_{safe_model_name}.jsonl"
    logger = ResultLogger(log_path)

    if args.rerun_failed:
        already_done = logger.passed_tasks()
    elif args.resume:
        already_done = logger.already_done()
    else:
        already_done = set()
    remaining = [t for t in tasks if t["task_id"] not in already_done]

    trace = None
    if args.trace:
        trace_path = f"results/traces/{args.dataset_name}_{config_slug}_{safe_model_name}.jsonl"
        trace = TraceLogger(trace_path)
        print(f"Tracing enabled -> {trace_path}")

    print(f"[{args.dataset_name} / {args.config} / {args.model}{' [ZERO-TEST]' if args.strip_tests else ''}] "
          f"{len(remaining)} of {len(tasks)} tasks to run ({len(already_done)} already logged)")

    for i, task in enumerate(remaining, 1):
        row = run_task(
            task, args.model, args.config, dataset=args.dataset_name, trace=trace,
            oracle_model=args.oracle_model, judge_model=args.judge_model,
            strip_tests=args.strip_tests,
        )
        logger.log(**row)
        status = "PASS" if row["final_passed"] else "FAIL"
        print(f"  [{i}/{len(remaining)}] {task['task_id']}: {status} "
              f"({row['iterations']} iters, {row['total_tokens']} tokens)")

    print(f"Done -> {log_path}")


if __name__ == "__main__":
    sys.exit(main())
