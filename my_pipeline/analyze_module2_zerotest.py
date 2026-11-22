"""
Phase D Analysis -- Module 2 Zero-Test Evaluation.
Computes Table 2 deliverables:
- Pass@1 with exact counts (e.g. 62/100)
- 95% Wilson score confidence interval
- McNemar test (chi-square or exact binomial)
- Valid oracles generated (%)
- Flipped task IDs (b and c)
"""
import argparse
import json
import math
from pathlib import Path
# # # # # # # # # from scipy import stats if False else None  # standard library fallback


import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std_dev = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    return max(0.0, lower), min(1.0, upper)


def mcnemar_test(b: int, c: int) -> dict:
    """Computes McNemar test with continuity correction and exact binomial test."""
    total = b + c
    if total == 0:
        return {"chi2": 0.0, "p_value": 1.0, "method": "none"}
    
    # Exact binomial if discordant pairs < 25
    if total < 25:
        k = min(b, c)
        p_val = 0.0
        for i in range(k + 1):
            p_val += math.comb(total, i) * (0.5 ** total)
        p_val = min(1.0, 2 * p_val)
        return {"chi2": (abs(b - c) - 1)**2 / total, "p_value": p_val, "method": "exact_binomial"}
    else:
        chi2 = (abs(b - c) - 1)**2 / total
        p_val = math.erfc(math.sqrt(chi2) / math.sqrt(2))
        return {"chi2": chi2, "p_value": p_val, "method": "chi_square_continuity"}


def resolve_log_path(path_str: str, default_model: str = None) -> str:
    """Intelligently resolves zero-test log file paths handling ':', '-', and alias variations."""
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_file():
        return str(p)

    variations = [
        path_str.replace(":", "-"),
        path_str.replace("-", ":"),
        path_str.replace("_module1_", "_m1_"),
        path_str.replace("_m1_", "_module1_"),
        path_str.replace("_module1+module2_", "_m1+m2_"),
        path_str.replace("_m1+m2_", "_module1+module2_"),
        path_str.replace(":", "-").replace("_module1_", "_m1_"),
        path_str.replace(":", "-").replace("_m1_", "_module1_"),
        path_str.replace(":", "-").replace("_module1+module2_", "_m1+m2_"),
        path_str.replace(":", "-").replace("_m1+m2_", "_module1+module2_"),
    ]

    for v in variations:
        if Path(v).is_file():
            return str(Path(v))

    # Also search inside results/ directory
    results_dir = Path("results")
    if results_dir.exists():
        fname = p.name
        alt_names = [
            fname,
            fname.replace(":", "-"),
            fname.replace("-", ":"),
            fname.replace("_module1_", "_m1_"),
            fname.replace("_m1_", "_module1_"),
            fname.replace("_module1+module2_", "_m1+m2_"),
            fname.replace("_m1+m2_", "_module1+module2_"),
            fname.replace(":", "-").replace("_module1_", "_m1_"),
            fname.replace(":", "-").replace("_m1_", "_module1_"),
            fname.replace(":", "-").replace("_module1+module2_", "_m1+m2_"),
            fname.replace(":", "-").replace("_m1+m2_", "_module1+module2_"),
        ]
        for name in alt_names:
            candidate = results_dir / name
            if candidate.is_file():
                return str(candidate)

    return ""


def analyze_zerotest(blind_path: str, m2_path: str, output_csv: str = None) -> str:
    resolved_blind = resolve_log_path(blind_path)
    resolved_m2 = resolve_log_path(m2_path)

    missing = []
    if not resolved_blind:
        missing.append(f"Blind log file (+m1): '{blind_path}'")
    if not resolved_m2:
        missing.append(f"Module 2 log file (+m1+m2): '{m2_path}'")

    if missing:
        print("\n" + "!" * 75)
        print("[ERROR] Required Zero-Test results file(s) not found:")
        for m in missing:
            print(f"  - {m}")
        print("\nAvailable JSONL files in results/ directory:")
        results_dir = Path("results")
        if results_dir.exists():
            files = sorted(results_dir.glob("*.jsonl"))
            if files:
                for f in files:
                    print(f"  * {f}")
            else:
                print("  (results/ folder is empty)")
        else:
            print("  (results/ directory does not exist yet)")

        print("\n[TROUBLESHOOTING & HOW TO RUN]:")
        print("Before running the analysis script, you must execute the zero-test experiment commands:")
        print("  1. !python -m my_pipeline.run_experiment --model <MODEL> --config +module1 --strip_tests --task_list evidence_package/selected_100_task_ids.json --resume")
        print("  2. !python -m my_pipeline.run_experiment --model <MODEL> --config +module1+module2 --strip_tests --task_list evidence_package/selected_100_task_ids.json --resume")
        print("!" * 75 + "\n")
        raise FileNotFoundError(f"Missing zero-test log file(s): {', '.join(missing)}")

    def load_run(path):
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    data[r["task_id"]] = r
        return data

    print(f"Loading blind log: {resolved_blind}")
    blind_data = load_run(resolved_blind)
    print(f"Loading m2 log:    {resolved_m2}")
    m2_data = load_run(resolved_m2)

    common_ids = sorted(set(blind_data.keys()) & set(m2_data.keys()))
    n = len(common_ids)
    if n == 0:
        msg = f"No overlapping task IDs found between {resolved_blind} and {resolved_m2}."
        print(msg)
        return msg

    blind_passed = {t for t in common_ids if blind_data[t]["final_passed"]}
    m2_passed = {t for t in common_ids if m2_data[t]["final_passed"]}

    k_blind = len(blind_passed)
    k_m2 = len(m2_passed)

    ci_blind = wilson_score_interval(k_blind, n)
    ci_m2 = wilson_score_interval(k_m2, n)

    # Discordant pairs:
    b_tasks = sorted(blind_passed - m2_passed)
    c_tasks = sorted(m2_passed - blind_passed)
    b = len(b_tasks)
    c = len(c_tasks)

    mcn = mcnemar_test(b, c)

    valid_oracles = sum(1 for t in common_ids if "NoOracleError" not in str(m2_data[t].get("error_type") or ""))
    oracle_pct = (valid_oracles / n) * 100.0

    report = []
    report.append("\n" + "=" * 75)
    report.append("# Phase D Deliverable: Table 2 (Zero-Test Evaluation)")
    report.append(f"Analyzed {n} paired tasks.")
    report.append("=" * 75)
    report.append("| Configuration | Pass@1 (n=" + str(n) + ") | 95% CI (Wilson) | Valid Oracles Generated |")
    report.append("|---|---|---|---|")
    report.append(f"| +m1 (blind) | {k_blind}/{n} ({k_blind/n*100:.1f}%) | [{ci_blind[0]*100:.1f}%, {ci_blind[1]*100:.1f}%] | — |")
    report.append(f"| +m1+m2 | {k_m2}/{n} ({k_m2/n*100:.1f}%) | [{ci_m2[0]*100:.1f}%, {ci_m2[1]*100:.1f}%] | {oracle_pct:.1f}% ({valid_oracles}/{n}) |")
    report.append("")
    report.append("### Statistical Significance (McNemar Test):")
    report.append(f"- Improvements (blind failed -> m2 passed): {c} tasks {c_tasks}")
    report.append(f"- Regressions (blind passed -> m2 failed): {b} tasks {b_tasks}")
    report.append(f"- Test Method: {mcn['method']}")
    report.append(f"- McNemar chi2 = {mcn['chi2']:.4f}, p-value = {mcn['p_value']:.4e}")
    sig_str = "Statistically Significant (p < 0.05)" if mcn['p_value'] < 0.05 else "Not Statistically Significant (p >= 0.05)"
    report.append(f"- Verdict: **{sig_str}**")
    report.append("=" * 75)

    full_text = "\n".join(report)
    print(full_text)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write("config,passed_count,total_count,pass_rate,ci_lower,ci_upper,valid_oracles_pct\n")
            f.write(f"+m1,{k_blind},{n},{k_blind/n*100:.2f},{ci_blind[0]*100:.2f},{ci_blind[1]*100:.2f},N/A\n")
            f.write(f"+m1+m2,{k_m2},{n},{k_m2/n*100:.2f},{ci_m2[0]*100:.2f},{ci_m2[1]*100:.2f},{oracle_pct:.2f}\n")
        print(f"\nSaved CSV summary to {output_csv}")

    return full_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Phase D Zero-Test results.")
    parser.add_argument("--model", default=None, help="Model name (e.g. qwen2.5-coder:1.5b) to auto-locate logs")
    parser.add_argument("--blind_log", default=None, help="Path to +m1 (blind) JSONL log")
    parser.add_argument("--m2_log", default=None, help="Path to +m1+m2 JSONL log")
    parser.add_argument("--output_csv", default=None, help="Output CSV path")
    args = parser.parse_args()

    blind_log = args.blind_log
    m2_log = args.m2_log
    output_csv = args.output_csv

    if args.model:
        safe_m = args.model.replace(":", "-")
        if not blind_log:
            blind_log = f"results/mypipeline_zerotest_module1_{safe_m}.jsonl"
        if not m2_log:
            m2_log = f"results/mypipeline_zerotest_module1+module2_{safe_m}.jsonl"
        if not output_csv:
            output_csv = f"evidence_package/table2_zerotest_{safe_m}.csv"

    if not blind_log or not m2_log:
        parser.error("Either provide --model OR both --blind_log and --m2_log")

    if not output_csv:
        output_csv = "evidence_package/table2_zerotest_summary.csv"

    analyze_zerotest(blind_log, m2_log, output_csv)
