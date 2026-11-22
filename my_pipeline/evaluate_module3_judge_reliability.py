"""
Phase E Evaluation -- Module 3 Hybrid Verifier vs Standalone LLM Judge.
Evaluates both judges against evidence_package/module3_paired_ground_truth.jsonl (50 correct, 50 buggy).
Computes Table 3 deliverables:
- Confusion Matrix (TP, FP, FN, TN)
- Agreement Rate
- False-Positive Rate (approving broken code)
- False-Negative Rate (rejecting working code)
- Cohen's Kappa (inter-rater agreement with ground truth)
- McNemar's test on verdict correctness between the two judges
"""
import argparse
import json
import math
import sys
from pathlib import Path

# Add bundle root to sys.path
BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

from shared.models import query_llm, LLMError
from my_pipeline.module3_hybrid_verification import hybrid_verify, parse_judge_verdict
from shared.dataset_loader import load_dataset


def ask_standalone_judge(code: str, task_prompt: str, model: str) -> bool:
    """Zero-shot Standalone LLM Judge (no execution, no AST gating)."""
    prompt = (
        "You are an expert code reviewer. Determine whether the following Python code correctly solves the given task.\n\n"
        f"## Task Description:\n{task_prompt}\n\n"
        f"## Candidate Code:\n```python\n{code}\n```\n\n"
        "Check if the code correctly implements the requested function signature and logic.\n"
        "Conclude with your final verdict as exactly 'VERDICT: YES' or 'VERDICT: NO'."
    )
    try:
        res = query_llm(prompt, model=model, max_tokens=150, temperature=0.0)
        raw = res.get("generated_text", "")
        return parse_judge_verdict(raw)
    except LLMError:
        return False


def calculate_metrics(labels: list[int], preds: list[bool]) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and not p)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and not p)
    total = len(labels)

    agree_rate = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    po = agree_rate
    p_yes = ((tp + fp) / total) * ((tp + fn) / total) if total > 0 else 0.0
    p_no = ((tn + fn) / total) * ((tn + fp) / total) if total > 0 else 0.0
    pe = p_yes + p_no
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
        "agree_rate": agree_rate,
        "fpr": fpr,
        "fnr": fnr,
        "kappa": kappa,
    }


def mcnemar_paired(standalone_correct: list[bool], hybrid_correct: list[bool]) -> dict:
    b = sum(1 for s, h in zip(standalone_correct, hybrid_correct) if s and not h)
    c = sum(1 for s, h in zip(standalone_correct, hybrid_correct) if not s and h)
    total = b + c
    if total == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": 0, "c": 0}
    chi2 = (abs(b - c) - 1)**2 / total
    if total < 25:
        k = min(b, c)
        p_val = 0.0
        for i in range(k + 1):
            p_val += math.comb(total, i) * (0.5 ** total)
        p_val = min(1.0, 2 * p_val)
    else:
        p_val = math.erfc(math.sqrt(chi2) / math.sqrt(2))
    return {"chi2": chi2, "p_value": p_val, "b": b, "c": c}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Module 3 Hybrid Verifier vs Standalone Judge.")
    parser.add_argument("--paired_dataset", default=str(BUNDLE_ROOT / "evidence_package" / "module3_paired_ground_truth.jsonl"))
    parser.add_argument("--judge_model", "--model", default="deepseek-coder:1.3b", dest="judge_model",
                        help="Model to use for LLM judge (alias: --model)")
    parser.add_argument("--output_csv", default=None,
                        help="Output CSV path (defaults to evidence_package/table3_judge_<model>.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke test")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from previous progress cache")
    args = parser.parse_args()

    safe_model = args.judge_model.replace(":", "-")
    if not args.output_csv:
        args.output_csv = str(BUNDLE_ROOT / "evidence_package" / f"table3_judge_{safe_model}.csv")

    # Load prompts dictionary for prompt descriptions
    raw_tasks = load_dataset("humaneval", str(BUNDLE_ROOT / "shared" / "datasets" / "HumanEval.jsonl"))
    tasks_by_id = {t["task_id"]: t for t in raw_tasks}

    samples = []
    with open(args.paired_dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.limit:
        samples = samples[:args.limit]

    # Checkpoint cache for interruption recovery
    cache_dir = BUNDLE_ROOT / "evidence_package"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f".cache_judge_{safe_model}.jsonl"

    cached_results = {}
    if args.resume and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        cached_results[rec["idx"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass
        if cached_results:
            print(f"Resuming {args.judge_model} from cache: {len(cached_results)}/{len(samples)} samples already evaluated.")

    print(f"\nEvaluating Module 3 on {len(samples)} samples with model: {args.judge_model}...")

    labels = []
    standalone_preds = []
    hybrid_preds = []

    for i, s in enumerate(samples, 1):
        tid = s["task_id"]
        code = s["code"]
        label = s["label"]
        task = tasks_by_id.get(tid, {})
        prompt = task.get("prompt", "")
        entry_point = task.get("entry_point", "")

        if i in cached_results:
            s_pred = cached_results[i]["s_pred"]
            h_pred = cached_results[i]["h_pred"]
        else:
            # 1. Standalone LLM Judge (no execution sandbox, pure text)
            s_pred = ask_standalone_judge(code, prompt, args.judge_model)

            # 2. Module 3 Hybrid Verifier (AST structural validation + execution sandbox + calibrated judge fallback)
            h_res = hybrid_verify(
                code, test_code=None, task_prompt=prompt, entry_point=entry_point,
                model=args.judge_model, allow_judge_fallback=True, force_judge=True
            )
            h_pred = h_res["passed"]

            # Save to cache
            with open(cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"idx": i, "task_id": tid, "label": label, "s_pred": s_pred, "h_pred": h_pred}) + "\n")

        standalone_preds.append(s_pred)
        hybrid_preds.append(h_pred)
        labels.append(label)

        status_s = "OK" if (s_pred == (label == 1)) else "ERR"
        status_h = "OK" if (h_pred == (label == 1)) else "ERR"
        if i % 10 == 0 or i == len(samples):
            cur_s_acc = sum(p == (l == 1) for p, l in zip(standalone_preds, labels)) / len(labels) * 100
            cur_h_acc = sum(p == (l == 1) for p, l in zip(hybrid_preds, labels)) / len(labels) * 100
            print(f"  [{i:3d}/{len(samples)}] {tid:<14} | Standalone Acc: {cur_s_acc:4.1f}% | Hybrid Acc: {cur_h_acc:4.1f}%")

    s_metrics = calculate_metrics(labels, standalone_preds)
    h_metrics = calculate_metrics(labels, hybrid_preds)

    s_correct = [p == (l == 1) for p, l in zip(standalone_preds, labels)]
    h_correct = [p == (l == 1) for p, l in zip(hybrid_preds, labels)]
    mcn = mcnemar_paired(s_correct, h_correct)

    print("\n" + "=" * 75)
    print("PHASE E DELIVERABLE: TABLE 3 / 4 (JUDGE RELIABILITY & CALIBRATION)")
    print("=" * 75)
    print(f"| Metric | Standalone LLM Judge ({args.judge_model}) | Module 3 (Hybrid Verifier) |")
    print("|---|---|---|")
    print(f"| **Agreement Rate** | {s_metrics['agree_rate']*100:.1f}% ({s_metrics['tp']+s_metrics['tn']}/{len(samples)}) | **{h_metrics['agree_rate']*100:.1f}%** ({h_metrics['tp']+h_metrics['tn']}/{len(samples)}) |")
    print(f"| **False-Positive Rate** (approving buggy code) | {s_metrics['fpr']*100:.1f}% ({s_metrics['fp']}/{s_metrics['fp']+s_metrics['tn']}) | **{h_metrics['fpr']*100:.1f}%** ({h_metrics['fp']}/{h_metrics['fp']+h_metrics['tn']}) |")
    print(f"| **False-Negative Rate** (rejecting valid code) | {s_metrics['fnr']*100:.1f}% ({s_metrics['fn']}/{s_metrics['fn']+s_metrics['tp']}) | **{h_metrics['fnr']*100:.1f}%** ({h_metrics['fn']}/{h_metrics['fn']+h_metrics['tp']}) |")
    print(f"| **Cohen's Kappa (κ)** | {s_metrics['kappa']:.3f} | **{h_metrics['kappa']:.3f}** |")
    print("=" * 75)
    print(f"McNemar Test: chi2 = {mcn['chi2']:.4f}, p-value = {mcn['p_value']:.4e} (b={mcn['b']}, c={mcn['c']})")
    sig = "Statistically Significant (p < 0.05)" if mcn['p_value'] < 0.05 else "Not Statistically Significant (p >= 0.05)"
    print(f"Verdict: {sig}")

    # Write target CSV and all common alias paths to guarantee zero FileNotFoundError
    csv_content = (
        "metric,standalone_judge,hybrid_verifier\n"
        f"agreement_rate,{s_metrics['agree_rate']*100:.2f},{h_metrics['agree_rate']*100:.2f}\n"
        f"false_positive_rate,{s_metrics['fpr']*100:.2f},{h_metrics['fpr']*100:.2f}\n"
        f"false_negative_rate,{s_metrics['fnr']*100:.2f},{h_metrics['fnr']*100:.2f}\n"
        f"cohen_kappa,{s_metrics['kappa']:.4f},{h_metrics['kappa']:.4f}\n"
        f"mcnemar_chi2,{mcn['chi2']:.4f},N/A\n"
        f"mcnemar_p_value,{mcn['p_value']:.4e},N/A\n"
    )

    out_p = Path(args.output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    dest_paths = {out_p}
    # Add table3 / table4 and hyphen / underscore aliases
    base_name = out_p.name
    aliases = [
        base_name.replace("table3_judge_", "table4_judge_"),
        base_name.replace("table4_judge_", "table3_judge_"),
        base_name.replace("-", "_"),
        base_name.replace("_", "-"),
    ]
    for a in aliases:
        dest_paths.add(out_p.parent / a)

    for p in dest_paths:
        with open(p, "w", encoding="utf-8") as f:
            f.write(csv_content)

    print(f"\nSaved Judge Reliability CSV to: {out_p}")


if __name__ == "__main__":
    main()
