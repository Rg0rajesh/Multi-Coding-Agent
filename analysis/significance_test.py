"""
Statistical Significance Tool (McNemar's Test) - Phase F deliverable.
Evaluates whether the difference in pass/fail rates between baseline and an ablated configuration
is statistically significant (p < 0.05).
"""
import argparse
import json
import math
from pathlib import Path


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """
    McNemar's test with continuity correction:
    chi2 = (|b - c| - 1)^2 / (b + c)
    b: passed in baseline, failed in treatment
    c: failed in baseline, passed in treatment
    """
    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0

    chi2 = (abs(b - c) - 1) ** 2 / total_discordant

    # Approximate p-value from chi-square distribution with 1 degree of freedom
    # using complementary error function
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p_value


def run_significance(baseline_path: Path, ablated_path: Path, output_file: Path | None = None):
    with open(baseline_path, "r", encoding="utf-8") as f:
        base_rows = {json.loads(l)["task_id"]: json.loads(l) for l in f if l.strip()}
    with open(ablated_path, "r", encoding="utf-8") as f:
        ablate_rows = {json.loads(l)["task_id"]: json.loads(l) for l in f if l.strip()}

    common = sorted(set(base_rows.keys()) & set(ablate_rows.keys()))

    a = 0  # both pass
    b = 0  # base pass, ablate fail
    c = 0  # base fail, ablate pass
    d = 0  # both fail

    for tid in common:
        bp = base_rows[tid].get("final_passed", False)
        ap = ablate_rows[tid].get("final_passed", False)

        if bp and ap:
            a += 1
        elif bp and not ap:
            b += 1
        elif not bp and ap:
            c += 1
        else:
            d += 1

    chi2, p_value = mcnemar_test(b, c)
    significant = p_value < 0.05

    report = (
        f"=== STATISTICAL SIGNIFICANCE (McNemar's Test) ===\n"
        f"Baseline: {baseline_path.name}\n"
        f"Ablation: {ablated_path.name}\n"
        f"Total Tasks: {len(common)}\n"
        f"Contingency Table:\n"
        f"  Both Passed (a): {a}\n"
        f"  Baseline only (b): {b}\n"
        f"  Ablation only (c): {c}\n"
        f"  Both Failed (d): {d}\n"
        f"McNemar Chi-Square: {chi2:.4f}\n"
        f"p-value: {p_value:.5f}\n"
        f"Statistically Significant (p < 0.05): {significant}\n"
    )
    print(report)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved significance report to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Test")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ablated", required=True)
    parser.add_argument("--output_file", default="evidence_package/significance_tests.txt")
    args = parser.parse_args()

    run_significance(Path(args.baseline), Path(args.ablated), Path(args.output_file))


if __name__ == "__main__":
    main()
