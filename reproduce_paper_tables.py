#!/usr/bin/env python3
"""
Multi-Coding Agent: Paper Table Reproduction Tool
=================================================
Generates formatted tables (Markdown, Plain Text, LaTeX) directly from raw JSONL
experimental execution traces.

Usage:
  python reproduce_paper_tables.py
  python reproduce_paper_tables.py --format latex
"""

import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

MODELS = [
    ("qwen2.5-coder-0.5b", "Qwen2.5-Coder 0.5B"),
    ("qwen2.5-coder-1.5b", "Qwen2.5-Coder 1.5B"),
    ("deepseek-coder-1.3b", "DeepSeek-Coder 1.3B"),
    ("llama3.2-3b", "Llama 3.2 3B"),
    ("qwen2.5-coder-3b", "Qwen2.5-Coder 3B"),
    ("deepseek-coder-6.7b", "DeepSeek-Coder 6.7B"),
]


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_table_1():
    print("TABLE I: Full-Pipeline Performance on HumanEval (164 tasks) and MBPP (974 tasks)")
    print("-" * 105)
    print(f"{'Model':<22} | {'HumanEval Pass@1':<18} {'HE Tokens':<11} {'HE Latency':<12} | {'MBPP Pass@1':<15} {'MBPP Tokens':<13} {'MBPP Latency':<13}")
    print("-" * 105)

    for slug, name in MODELS:
        # HumanEval
        p_he = RESULTS_DIR / "humaneval" / f"mypipeline_humaneval_module1+module2+module3_{slug}.jsonl"
        if not p_he.exists(): p_he = RESULTS_DIR / f"mypipeline_humaneval_module1+module2+module3_{slug}.jsonl"
        rec_he = load_jsonl(p_he)
        solved_he = sum(1 for r in rec_he if r.get("final_passed"))
        pass_he = (solved_he / len(rec_he)) * 100
        tok_he = sum(r.get("total_tokens", 0) for r in rec_he) / len(rec_he)
        lat_he = sum(r.get("total_time_sec", 0) for r in rec_he) / len(rec_he)

        # MBPP
        p_mbpp = RESULTS_DIR / "mbpp" / f"mypipeline_mbpp_module1+module2+module3_{slug}.jsonl"
        if not p_mbpp.exists(): p_mbpp = RESULTS_DIR / f"mypipeline_mbpp_module1+module2+module3_{slug}.jsonl"
        rec_mbpp = load_jsonl(p_mbpp)
        solved_mbpp = sum(1 for r in rec_mbpp if r.get("final_passed"))
        pass_mbpp = (solved_mbpp / len(rec_mbpp)) * 100
        tok_mbpp = sum(r.get("total_tokens", 0) for r in rec_mbpp) / len(rec_mbpp)
        lat_mbpp = sum(r.get("total_time_sec", 0) for r in rec_mbpp) / len(rec_mbpp)

        print(f"{name:<22} | {pass_he:6.2f}% ({solved_he:>3}/164)   {tok_he:>6.0f}      {lat_he:>5.2f}s       | {pass_mbpp:6.2f}% ({solved_mbpp:>3}/974)  {tok_mbpp:>7.0f}       {lat_mbpp:>5.2f}s")
    print("-" * 105)


def print_iteration_tables():
    print("\nTABLE IX: Iteration Convergence (Tasks First Solved per Iteration)")
    print("-" * 85)
    print(f"{'Dataset':<11} {'Model':<22} {'It 1':>6} {'It 2':>6} {'It 3':>6} {'It 4':>6} {'It 5':>6} {'It 6':>6} {'Total':>7}")
    print("-" * 85)

    for ds, max_n in [("HumanEval", 164), ("MBPP", 974)]:
        for slug, name in MODELS:
            sub = "humaneval" if ds == "HumanEval" else "mbpp"
            fn = f"mypipeline_{sub}_module1+module2+module3_{slug}.jsonl"
            p = RESULTS_DIR / sub / fn
            if not p.exists(): p = RESULTS_DIR / fn
            rec = load_jsonl(p)
            cnt = Counter(r.get("iterations", 1) for r in rec if r.get("final_passed"))
            tot = sum(cnt.values())
            print(f"{ds:<11} {name:<22} {cnt[1]:6d} {cnt[2]:6d} {cnt[3]:6d} {cnt[4]:6d} {cnt[5]:6d} {cnt[6]:6d} {tot:7d}")
        print("-" * 85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["text", "latex"], default="text")
    args = parser.parse_args()

    print_table_1()
    print_iteration_tables()


if __name__ == "__main__":
    main()

