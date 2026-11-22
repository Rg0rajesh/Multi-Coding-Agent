"""
Turns the raw per-task JSONL logs into actual report files -- a per-model
CSV and one combined CSV -- instead of a table that only lives in notebook
output and disappears when the session ends.

Usage:
    python -m my_pipeline.generate_reports

Output:
    results/reports/<model>_summary.csv       -- one file per model
    results/reports/full_ablation_summary.csv -- everything in one table

Compares against results/published_baseline_citations.csv if it exists --
that file holds published pass@1 numbers for both base papers (AdaCoder's
own two-phase framework, and MapCoder as reproduced in AdaCoder's Table
II/VII -- MapCoder's own paper never evaluated these open-weight models,
so its numbers here are always attributed to AdaCoder's reproduction, not
MapCoder's original claims). Missing citations are left blank, not guessed.

Note: model/dataset/config are read from each JSONL row's own fields, not
parsed out of the filename. Filenames can drift (e.g. ":" or "." getting
swapped for "-"/"_" by whatever tool wrote them); the JSONL schema fields
are the source of truth per the locked logging schema.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPORT_FIELDS = [
    "model", "dataset", "config", "n", "pass_at_1",
    "avg_tokens", "avg_iterations", "avg_time_sec",
    "adacoder_pass_at_1", "beats_adacoder",
    "mapcoder_pass_at_1", "beats_mapcoder",
]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping a malformed line in {path.name}")
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "pass_at_1": 0.0, "avg_tokens": 0.0,
                "avg_iterations": 0.0, "avg_time_sec": 0.0}
    return {
        "n": n,
        "pass_at_1": round(sum(r["final_passed"] for r in rows) / n, 4),
        "avg_tokens": round(sum(r["total_tokens"] for r in rows) / n, 1),
        "avg_iterations": round(sum(r["iterations"] for r in rows) / n, 2),
        "avg_time_sec": round(sum(r["total_time_sec"] for r in rows) / n, 2),
    }


def collect_results(results_dir: Path) -> list[dict]:
    """
    Groups every task row across all result files by its own (model, dataset,
    config) fields -- not by filename -- so naming drift in the files
    themselves can't silently misfile results under the wrong model.
    """
    grouped = defaultdict(list)
    for path in results_dir.glob("mypipeline_*.jsonl"):
        for row in load_rows(path):
            key = (row.get("model"), row.get("dataset"), row.get("config"))
            if None in key:
                print(f"Warning: a row in {path.name} is missing model/dataset/config, skipping it")
                continue
            grouped[key].append(row)

    return [
        {"model": model, "dataset": dataset, "config": config, **summarize(rows)}
        for (model, dataset, config), rows in grouped.items()
    ]


def load_citations(path: Path) -> dict:
    """
    Expected columns: model, dataset, framework, pass_at_1.
    framework is "AdaCoder" or "MapCoder" (the latter as reproduced in
    AdaCoder's own Table II/VII, since MapCoder's paper never evaluated
    these models itself).
    Returns {(model, dataset, framework): pass_at_1}, or {} if missing/unreadable.
    """
    if not path.exists():
        return {}

    citations = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        required = {"model", "dataset", "framework", "pass_at_1"}
        if not required.issubset(reader.fieldnames or []):
            print(f"Warning: {path} is missing columns {required - set(reader.fieldnames or [])}, "
                  f"skipping citation comparison.")
            return {}
        for row in reader:
            try:
                citations[(row["model"], row["dataset"], row["framework"])] = float(row["pass_at_1"])
            except (ValueError, KeyError):
                continue
    return citations


def attach_citations(rows: list[dict], citations: dict) -> None:
    for row in rows:
        for framework, prefix in (("AdaCoder", "adacoder"), ("MapCoder", "mapcoder")):
            cited = citations.get((row["model"], row["dataset"], framework))
            row[f"{prefix}_pass_at_1"] = cited if cited is not None else ""
            row[f"beats_{prefix}"] = (row["pass_at_1"] > cited) if cited is not None else ""


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REPORT_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="results/reports")
    parser.add_argument("--citations_file", default="results/published_baseline_citations.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    rows = collect_results(results_dir)
    if not rows:
        print(f"No result files found in {results_dir} (expected mypipeline_*.jsonl)")
        return

    citations = load_citations(Path(args.citations_file))
    attach_citations(rows, citations)

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    for model, model_rows in by_model.items():
        safe_name = model.replace(":", "-").replace(".", "_")
        write_csv(model_rows, output_dir / f"{safe_name}_summary.csv")

    write_csv(rows, output_dir / "full_ablation_summary.csv")

    print(f"Wrote {len(by_model)} per-model CSVs and 1 combined CSV to {output_dir}/")
    if not citations:
        print("(No published_baseline_citations.csv found -- citation comparison columns left blank.)")


if __name__ == "__main__":
    main()
