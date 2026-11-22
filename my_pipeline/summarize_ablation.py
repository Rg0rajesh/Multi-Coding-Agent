"""
Turns the raw per-task JSONL logs into the Phase C deliverable: one row per
model comparing "baseline" vs "+module1" on pass@1, avg iterations,
avg tokens, avg time.

Usage:
    python -m my_pipeline.summarize_ablation --results_dir results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "pass_at_1": 0.0, "avg_iterations": 0.0,
                "avg_tokens": 0.0, "avg_time_sec": 0.0}
    return {
        "n": n,
        "pass_at_1": round(sum(r["final_passed"] for r in rows) / n, 4),
        "avg_iterations": round(sum(r["iterations"] for r in rows) / n, 2),
        "avg_tokens": round(sum(r["total_tokens"] for r in rows) / n, 1),
        "avg_time_sec": round(sum(r["total_time_sec"] for r in rows) / n, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    by_model = defaultdict(dict)

    for path in results_dir.glob("mypipeline_*.jsonl"):
        # filenames look like mypipeline_<dataset>_<config>_<model>.jsonl
        stem = path.stem.replace("mypipeline_", "")
        dataset, config, model = stem.split("_", 2)
        key = f"{dataset}/{model}"
        by_model[key][config] = summarize(load_rows(path))

    header = f"{'dataset/model':<36} {'config':<10} {'n':>4} {'pass@1':>8} {'avg_iters':>10} {'avg_tokens':>11} {'avg_time':>9}"
    print(header)
    print("-" * len(header))
    for key, configs in sorted(by_model.items()):
        for config, stats in sorted(configs.items()):
            print(f"{key:<36} {config:<10} {stats['n']:>4} {stats['pass_at_1']:>8.2%} "
                  f"{stats['avg_iterations']:>10.2f} {stats['avg_tokens']:>11.1f} "
                  f"{stats['avg_time_sec']:>9.2f}")


if __name__ == "__main__":
    main()
