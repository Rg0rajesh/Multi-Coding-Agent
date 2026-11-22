"""
Paired Task-Level Comparison Tool (Phase F deliverable).
Identifies tasks newly fixed (recovered) vs newly broken (regressed)
between any two runs (e.g. baseline vs +module1+module2+module3).
"""
import argparse
import csv
import json
from pathlib import Path


def load_run_dict(path: Path) -> dict:
    task_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                task_map[row["task_id"]] = row
    return task_map


def paired_compare(baseline_path: Path, ablated_path: Path, output_csv: Path | None = None):
    base = load_run_dict(baseline_path)
    ablate = load_run_dict(ablated_path)

    common_tasks = sorted(set(base.keys()) & set(ablate.keys()))
    newly_fixed = []
    newly_broken = []
    both_pass = []
    both_fail = []

    for tid in common_tasks:
        b_p = base[tid].get("final_passed", False)
        a_p = ablate[tid].get("final_passed", False)

        if not b_p and a_p:
            newly_fixed.append((tid, base[tid].get("error_type"), ablate[tid].get("iterations"), ablate[tid].get("total_tokens")))
        elif b_p and not a_p:
            newly_broken.append((tid, ablate[tid].get("error_type"), ablate[tid].get("iterations"), ablate[tid].get("total_tokens")))
        elif b_p and a_p:
            both_pass.append(tid)
        else:
            both_fail.append(tid)

    print(f"=== PAIRED COMPARISON: {baseline_path.name} vs {ablated_path.name} ===")
    print(f"Total Shared Tasks : {len(common_tasks)}")
    print(f"Passed by Both     : {len(both_pass)}")
    print(f"Failed by Both     : {len(both_fail)}")
    print(f"Newly Fixed (+)    : {len(newly_fixed)}")
    print(f"Newly Broken (-)   : {len(newly_broken)}")
    net = len(newly_fixed) - len(newly_broken)
    print(f"Net Gain           : {net:+d} tasks")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "task_id", "error_type", "iterations", "tokens"])
            for t in newly_fixed:
                writer.writerow(["newly_fixed", t[0], t[1], t[2], t[3]])
            for t in newly_broken:
                writer.writerow(["newly_broken", t[0], t[1], t[2], t[3]])
        print(f"Saved paired comparison details to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Paired Task-Level Comparison")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ablated", required=True)
    parser.add_argument("--output_csv", default="evidence_package/paired_comparisons.csv")
    args = parser.parse_args()

    paired_compare(Path(args.baseline), Path(args.ablated), Path(args.output_csv))


if __name__ == "__main__":
    main()
