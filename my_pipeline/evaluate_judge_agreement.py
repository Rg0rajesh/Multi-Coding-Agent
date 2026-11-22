"""
Phase E deliverable -- how much should Module 3's LLM-judge fallback
actually be trusted? Generates code for a sample of tasks, then checks it
two independent ways: real execution (ground truth) and the judge's
verdict on the exact same code. Where they disagree is the whole point of
this report.

This never touches hybrid_verify()'s own oracle-selection logic -- it
forces the judge path directly (test_code=None) on code whose real
correctness is already known from execution, which is the only way to
measure judge accuracy without needing the pipeline to have actually
failed to produce a test oracle first.

Usage:
    python -m my_pipeline.evaluate_judge_agreement \
        --model deepseek-coder:6.7b --judge_model codellama:7b-python --sample_size 40

Output: results/judge_agreement_analysis.csv (one row per task) plus a
printed summary -- agreement rate, false-positive rate, false-negative
rate. Both go straight into evidence_package/ per Phase G.
"""
import argparse
import csv
import random
from pathlib import Path

from shared.dataset_loader import load_dataset
from shared.models import LLMError
from my_pipeline.pipeline import generate_candidate
from my_pipeline.sandbox_exec import run_with_diagnosis
from my_pipeline.module3_hybrid_verification import hybrid_verify

DEFAULT_DATASET_PATH = "shared/datasets/HumanEval.jsonl"
REPORT_FIELDS = ["task_id", "ground_truth_passed", "judge_passed", "agree"]


def score_task(task: dict, model: str, judge_model: str) -> dict | None:
    """Returns one report row, or None if generation itself failed (that's
    an LLMError problem, not a judge-accuracy data point)."""
    try:
        code, _ = generate_candidate(task["prompt"], model)
    except LLMError as exc:
        print(f"  skip {task['task_id']}: generation failed ({exc})")
        return None

    ground_truth_passed, _, _ = run_with_diagnosis(code, task["test"], task["entry_point"])
    judge_result = hybrid_verify(code, None, task["prompt"], task["entry_point"], model=judge_model)
    judge_passed = judge_result["passed"]

    return {
        "task_id": task["task_id"],
        "ground_truth_passed": ground_truth_passed,
        "judge_passed": judge_passed,
        "agree": ground_truth_passed == judge_passed,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    actual_fails = sum(1 for r in rows if not r["ground_truth_passed"])
    actual_passes = n - actual_fails
    false_positives = sum(1 for r in rows if r["judge_passed"] and not r["ground_truth_passed"])
    false_negatives = sum(1 for r in rows if not r["judge_passed"] and r["ground_truth_passed"])

    return {
        "n": n,
        "agreement_rate": round(sum(r["agree"] for r in rows) / n, 4) if n else 0.0,
        # rate is "of the tasks that actually failed, how many did the judge wrongly pass"
        "false_positive_rate": round(false_positives / actual_fails, 4) if actual_fails else None,
        "false_negative_rate": round(false_negatives / actual_passes, 4) if actual_passes else None,
    }


def write_report(rows: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(stats: dict, output_path: str) -> None:
    print(f"\n{stats['n']} tasks scored -> {output_path}")
    print(f"  agreement rate:      {stats['agreement_rate']:.2%}")
    if stats["false_positive_rate"] is None:
        print("  false positive rate: n/a (no real failures in this sample)")
    else:
        print(f"  false positive rate: {stats['false_positive_rate']:.2%}")
    if stats["false_negative_rate"] is None:
        print("  false negative rate: n/a (no real passes in this sample)")
    else:
        print(f"  false negative rate: {stats['false_negative_rate']:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", default="deepseek-coder:6.7b",
                         help="Model that generates the candidate code being judged.")
    parser.add_argument("--judge_model", default="codellama:7b-python")
    parser.add_argument("--sample_size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/judge_agreement_analysis.csv")
    args = parser.parse_args()

    tasks = load_dataset("humaneval", args.dataset_path)
    random.seed(args.seed)
    sample = random.sample(tasks, min(args.sample_size, len(tasks)))

    print(f"Scoring {len(sample)} tasks: generator={args.model}, judge={args.judge_model}")

    rows = []
    for i, task in enumerate(sample, 1):
        row = score_task(task, args.model, args.judge_model)
        if row is None:
            continue
        rows.append(row)
        match = "OK" if row["agree"] else "MISMATCH"
        print(f"  [{i}/{len(sample)}] {task['task_id']}: "
              f"truth={row['ground_truth_passed']} judge={row['judge_passed']} {match}")

    if not rows:
        print("No tasks scored -- nothing to report.")
        return

    write_report(rows, args.output)
    print_summary(summarize(rows), args.output)


if __name__ == "__main__":
    main()
