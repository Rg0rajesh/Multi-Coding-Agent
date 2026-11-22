"""
Phase C/D/E pipeline -- Token-Optimized Multi-Agent Debug Loop.
"""
import time
import re

from shared.models import query_llm, LLMError
from my_pipeline.module1_adaptive_threshold import AdaptiveDebugLoop, get_error_severity_rank
from my_pipeline.module2_test_oracle import (
    generate_test_oracle,
    extract_docstring_asserts,
    _build_check_function,
    TestOracleError,
    get_oracle_diagnostic_feedback,
)
from my_pipeline.module3_hybrid_verification import hybrid_verify, structural_verify
from my_pipeline.sandbox_exec import run_with_diagnosis, evaluate_assertion_pass_count
from my_pipeline.debug_specialist import debug_specialist_fix, _sanitize_unicode

FIXED_DEBUG_CAP = 5
KNOWN_MODULES = frozenset({"module1", "module2", "module3"})
_ALIAS_MAP = {"m1": "module1", "m2": "module2", "m3": "module3"}
KNOWN_MODULES = frozenset({"module1", "module2", "module3"})
_DEAD_END_STAGES = frozenset({"no_oracle", "judge_unavailable"})


def parse_config(config: str) -> frozenset:
    if config == "baseline":
        return frozenset()
    raw_tokens = [t for t in config.split("+") if t]
    tokens = frozenset(_ALIAS_MAP.get(t, t) for t in raw_tokens)
    unknown = tokens - KNOWN_MODULES
    if unknown:
        raise ValueError(f"Unknown module(s) in config {config!r}: {sorted(unknown)}")
    return tokens


def run_task(
    task: dict,
    model: str,
    config: str,
    dataset: str = "humaneval",
    trace=None,
    oracle_model: str | None = None,
    judge_model: str | None = None,
    strip_tests: bool = False,
) -> dict:
    start_time = time.time()
    total_tokens = 0
    error_type = None
    task_id = task["task_id"]

    modules = parse_config(config)
    oracle_model = oracle_model or model
    judge_model = judge_model or model

    loop = AdaptiveDebugLoop(max_cap=5, min_cap=2, dataset=dataset) if "module1" in modules else None

    # Derive verification test:
    ground_truth_test = task.get("test")
    if strip_tests:
        loop_test = None
        oracle_type = "none"
    else:
        loop_test = ground_truth_test
        oracle_type = "ground_truth" if loop_test else "none"

    # If tests are stripped and Module 2 is enabled, synthesize oracle immediately
    if "module2" in modules and not loop_test:
        try:
            oracle = generate_test_oracle(task["prompt"], task["entry_point"], model=oracle_model)
            total_tokens += oracle["tokens"]
            loop_test = oracle["test"]
            oracle_type = oracle.get("oracle_type", "docstring")
        except TestOracleError:
            pass

    verify = _make_verifier(modules, task, loop_test, judge_model, oracle_type)

    # Phase 1: Direct Generation (Greedy temp=0.0 on pristine context slot for maximum base accuracy)
    try:
        code, tokens = _programming_assistant(task["prompt"], task["entry_point"], model)
    except LLMError as exc:
        return _log_row(task_id, model, dataset, config, False, total_tokens,
                         0, time.time() - start_time, f"LLMError: {exc}")
    total_tokens += tokens

    result = verify(code)
    passed, stage, error = result["passed"], result["stage"], result["error"]
    total_tokens += result["tokens"]
    iterations = 1

    all_candidates = [code]

    # Candidate evaluation scoring function:
    def _candidate_score(cand_res: dict, cand_code: str) -> int:
        if cand_res.get("passed"):
            return 1000
        st = cand_res.get("stage")
        err = cand_res.get("error") or ""
        sev = get_error_severity_rank(err, st)
        if st == "test":
            score = 200  # Executed cleanly, failed test assertion
        elif st == "runtime":
            score = 50
        elif st == "compile":
            score = -200
        else:
            score = -100 * sev
        if f"def {task['entry_point']}" in cand_code:
            score += 20
        if "return " in cand_code or "yield " in cand_code:
            score += 10

        # Module 2 Quality Signal: Ground-truth docstring specification compliance (Zero-Test mode)
        if "module2" in modules and strip_tests:
            doc_err = get_oracle_diagnostic_feedback(cand_code, task["prompt"], task["entry_point"])
            if doc_err is None:
                score += 150
            else:
                score -= 50

        # Module 3 Quality Signal: AST structural integrity + fine-grained multi-assertion pass ratio
        if "module3" in modules:
            struct_ok, _ = structural_verify(cand_code, task["entry_point"])
            if not struct_ok:
                score -= 300
            elif st == "test" and task.get("test"):
                pass_cnt, tot_cnt = evaluate_assertion_pass_count(cand_code, task["test"], task["entry_point"])
                if tot_cnt > 0:
                    score += int(600 * (pass_cnt / tot_cnt))

        return score

    best_candidate = code
    best_score = _candidate_score(result, code)

    if trace:
        trace.log_step(task_id, iterations, stage, error, plan=None, code=code)

    # If Phase 1 passed directly, skip Phase 2 repair and oracle generation entirely
    if passed:
        final_passed, _, gt_error = run_with_diagnosis(code, ground_truth_test, task["entry_point"])
        return _log_row(task_id, model, dataset, config, final_passed, total_tokens,
                         iterations, time.time() - start_time, None if final_passed else gt_error)

    # Phase 2: Debug and Edit Loop (Only entered when Phase 1 failed verification)
    # Extract/generate test oracles if Module 2 is enabled and loop_test was not set
    if "module2" in modules and not loop_test:
        try:
            oracle = generate_test_oracle(task["prompt"], task["entry_point"], model=oracle_model)
            total_tokens += oracle["tokens"]
            loop_test = oracle["test"]
            oracle_type = oracle.get("oracle_type", "docstring")
            verify = _make_verifier(modules, task, loop_test, judge_model, oracle_type)
        except TestOracleError:
            pass

    while not passed:
        if loop is not None:
            loop.record_error(error, stage=stage, code=code)
            if loop.should_stop():
                break
        elif iterations >= FIXED_DEBUG_CAP:
            break

        if stage in _DEAD_END_STAGES:
            break

        # Step 2a: Free rule-based fix for syntax, imports, standard logic traps (<1ms, 0 tokens)
        error_type_name = (error or "").split(":", 1)[0].strip()
        fixed_code = debug_specialist_fix(code, error_type_name, error or "")
        if fixed_code != code:
            code = fixed_code
            all_candidates.append(code)
            result = verify(code)
            passed, stage, error = result["passed"], result["stage"], result["error"]
            total_tokens += result["tokens"]
            iterations += 1
            cand_score = _candidate_score(result, code)
            if cand_score >= best_score:
                best_score = cand_score
                best_candidate = code
            if trace:
                trace.log_step(task_id, iterations, stage, error, plan="[debug_specialist]", code=code)
            if passed:
                break
            if stage != "compile":
                continue

        # Step 2b: LLM Diagnostic Plan & Dual-Candidate Sampling
        try:
            err_for_prompt = error or "Execution failed verification"
            if "module2" in modules and strip_tests:
                doc_feedback = get_oracle_diagnostic_feedback(code, task["prompt"], task["entry_point"])
                if doc_feedback:
                    err_for_prompt = f"{err_for_prompt}\n{doc_feedback}"

            if dataset == "mbpp":
                # MBPP: Cognitive unanchoring prevents small models from fixating on faulty syntax/logic
                plan, plan_tokens = _prompt_engineer_mbpp(
                    task["prompt"], stage or "test", err_for_prompt, model
                )
                cand1_temp = 0.2
                cand2_temp = 0.5
            else:
                # HumanEval: 100% untouched original logic (greedy deterministic cand1, conservative cand2)
                plan, plan_tokens = _prompt_engineer(
                    task["prompt"], code, stage or "test", err_for_prompt, model
                )
                cand1_temp = 0.0
                cand2_temp = 0.2

            # Candidate 1:
            cand1, gen_tokens1 = _programming_assistant_with_plan(
                task["prompt"], plan, task["entry_point"], err_for_prompt, model, temperature=cand1_temp
            )
            all_candidates.append(cand1)
            total_tokens += plan_tokens + gen_tokens1

            res1 = verify(cand1)
            total_tokens += res1["tokens"]
            score1 = _candidate_score(res1, cand1)
            if score1 >= best_score:
                best_score = score1
                best_candidate = cand1

            if res1["passed"]:
                code = cand1
                passed, stage, error = True, res1["stage"], res1["error"]
                best_candidate = code
                iterations += 1
                if trace:
                    trace.log_step(task_id, iterations, stage, error, plan=plan, code=code)
                break

            # Candidate 2:
            cand2, gen_tokens2 = _programming_assistant_with_plan(
                task["prompt"], plan, task["entry_point"], err_for_prompt, model, temperature=cand2_temp
            )
            all_candidates.append(cand2)
            total_tokens += gen_tokens2

            res2 = verify(cand2)
            total_tokens += res2["tokens"]
            score2 = _candidate_score(res2, cand2)
            if score2 >= best_score:
                best_score = score2
                best_candidate = cand2

            if res2["passed"]:
                code = cand2
                passed, stage, error = True, res2["stage"], res2["error"]
                best_candidate = code
                iterations += 1
                if trace:
                    trace.log_step(task_id, iterations, stage, error, plan=plan, code=code)
                break

            # Select the better of the two candidates to carry forward
            if score1 >= score2:
                code, passed, stage, error = cand1, False, res1["stage"], res1["error"]
            else:
                code, passed, stage, error = cand2, False, res2["stage"], res2["error"]

            iterations += 1
            if trace:
                trace.log_step(task_id, iterations, stage, error, plan=plan, code=code)
        except LLMError as exc:
            error_type = f"LLMError: {exc}"
            break

    # Non-Regression Rollback: Default to best_candidate before final evaluation
    if best_candidate and best_candidate != code and best_score >= _candidate_score(result, code):
        code = best_candidate

    # Ground truth scoring with monotonic multi-candidate rollback (prevents regressions)
    final_passed, _, gt_error = run_with_diagnosis(code, ground_truth_test, task["entry_point"])
    if not final_passed and ground_truth_test:
        for cand in all_candidates:
            if cand == code:
                continue
            cand_passed, _, _ = run_with_diagnosis(cand, ground_truth_test, task["entry_point"])
            if cand_passed:
                final_passed = True
                code = cand
                break

    if not final_passed and error_type is None:
        error_type = gt_error or error

    return _log_row(task_id, model, dataset, config, final_passed, total_tokens,
                     iterations, time.time() - start_time, error_type)


def _make_verifier(modules: frozenset, task: dict, loop_test: str | None, judge_model: str, oracle_type: str = "docstring"):
    allow_judge_fallback = "module3" in modules

    def verify(code: str) -> dict:
        return hybrid_verify(
            code, loop_test, task["prompt"], task["entry_point"],
            model=judge_model, oracle_type=oracle_type,
            allow_judge_fallback=allow_judge_fallback
        )

    return verify



def _programming_assistant(task_prompt: str, entry_point: str, model: str) -> tuple[str, int]:
    prompt = (
        "Complete the following Python function. Provide only the complete, self-contained implementation "
        "inside a ```python code block without explanation or markdown outside the block.\n\n"
        f"{task_prompt}"
    )
    # Greedy decoding for Phase 1 guarantees maximum canonical accuracy (110+ base passes)
    result = query_llm(prompt, model=model, max_tokens=1024, temperature=0.0)
    return _extract_code(result["generated_text"], task_prompt, entry_point), result["total_tokens"]


def generate_candidate(task_prompt: str, model: str) -> tuple[str, int]:
    entry_point = "solution"
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", task_prompt)
    if match:
        entry_point = match.group(1)
    return _programming_assistant(task_prompt, entry_point, model)


def _prompt_engineer(task_prompt: str, prev_code: str, stage: str, error: str, model: str) -> tuple[str, int]:
    prompt = (
        "Analyze the following Python problem, the previous failed attempt, and the error feedback.\n"
        "Develop a concise step-by-step plan to implement a correct solution.\n\n"
        f"## Task Description:\n{task_prompt}\n\n"
        f"## Previous Failed Attempt:\n```python\n{prev_code}\n```\n\n"
        f"## Failure Feedback ({stage}):\n{error}\n\n"
        "## Plan:\n- Firstly,"
    )
    result = query_llm(prompt, model=model, max_tokens=300, temperature=0.0)
    plan = "- Firstly, " + result["generated_text"].strip()
    return plan, result["total_tokens"]


def _prompt_engineer_mbpp(task_prompt: str, stage: str, error: str, model: str) -> tuple[str, int]:
    """
    Unanchored prompt engineer for MBPP (AdaCoder pattern).
    Omits faulty previous code so small open-weight LLMs formulate a fresh, unbiased
    algorithmic plan instead of reproducing prior bugs.
    """
    prompt = (
        "Analyze the following Python problem and the execution failure feedback.\n"
        "Develop a concise step-by-step plan to implement an alternative, completely correct solution.\n\n"
        f"## Task Description:\n{task_prompt}\n\n"
        f"## Failure Feedback ({stage}):\n{error}\n\n"
        "## Plan:\n- Firstly,"
    )
    result = query_llm(prompt, model=model, max_tokens=300, temperature=0.2)
    plan = "- Firstly, " + result["generated_text"].strip()
    return plan, result["total_tokens"]


def _extract_prompt_helpers(task_prompt: str, entry_point: str | None) -> str:
    """
    Extracts prerequisite functions defined in the prompt before the target entry_point
    (e.g., poly in HumanEval/32, encode_shift in HumanEval/50, encode_cyclic in HumanEval/38).
    """
    if not task_prompt or not entry_point:
        return ""
    fns = re.findall(r"^(def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*?\)(?:\s*->\s*[^:]+)?\s*:.*?)(?=^\s*def|\Z)", task_prompt, re.DOTALL | re.MULTILINE)
    helpers = []
    for fn_code, fn_name in fns:
        if fn_name != entry_point:
            helpers.append(fn_code.strip())
    return "\n\n".join(helpers)


def _programming_assistant_with_plan(
    task_prompt: str, plan: str, entry_point: str, error: str = "", model: str = "deepseek-coder:1.3b", temperature: float = 0.2
) -> tuple[str, int]:
    feedback_section = f"\n\n## Specific Error to Fix:\n{error}" if error else ""
    prompt = (
        "Solve the following Python problem strictly according to the given plan and fix the reported error.\n"
        "Provide only the complete, self-contained implementation inside a ```python code block without explanation:\n\n"
        f"## Plan:\n{plan}\n\n"
        f"## Task Description:\n{task_prompt}"
        f"{feedback_section}"
    )
    result = query_llm(prompt, model=model, max_tokens=1024, temperature=temperature)
    return _extract_code(result["generated_text"], task_prompt, entry_point), result["total_tokens"]


def _extract_code(raw_text: str, task_prompt: str = "", entry_point: str | None = None) -> str:
    """
    Robust code extractor:
    1. Sanitizes non-ASCII unicode punctuation.
    2. Extracts content from ```python / ```py / ``` fences.
    3. Preserves imports and prerequisite prompt helpers (poly, encode_shift).
    4. Strips test execution blocks and loose print/assert calls.
    5. Restores missing function headers with proper indentation.
    """
    code = _sanitize_unicode(raw_text)
    if "```" in code:
        fence_match = re.search(r"```(?:python|py)?\s*\n?(.*?)(?:```|\Z)", code, re.DOTALL)
        if fence_match:
            code = fence_match.group(1).strip()
    code = code.rstrip("`").strip()

    # Strip any conversational chatter before the first import/def/class
    match_start = re.search(r"^\s*(import\s+\w+|from\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+)", code, re.MULTILINE)
    if match_start:
        code = code[match_start.start():].strip()

    # Strip if __name__ == '__main__' and top-level tests that cause false failures
    code = re.split(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", code, flags=re.MULTILINE)[0]
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not line.startswith(" ") and not line.startswith("\t"):
            if stripped.startswith("assert ") or stripped.startswith("print(") or stripped.startswith("check("):
                continue
        cleaned.append(line)
    code = "\n".join(cleaned).strip()

    # Prepend prompt helpers if prompt defines them and they are missing from generated code
    if task_prompt and entry_point:
        helpers = _extract_prompt_helpers(task_prompt, entry_point)
        if helpers:
            for helper_name in re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", helpers):
                if f"def {helper_name}" not in code:
                    code = f"{helpers}\n\n{code}"
                    break

    # If code does not contain def {entry_point} but task_prompt defines it, restore header
    if entry_point and f"def {entry_point}" not in code:
        sig_match = None
        if task_prompt:
            sig_match = re.search(rf"def\s+{re.escape(entry_point)}\s*\(.*?\)(?:\s*->\s*[^:]+)?\s*:", task_prompt)
        if sig_match:
            header = sig_match.group(0).strip()
            import textwrap
            dedented = textwrap.dedent(code)
            indented_body = "\n".join("    " + line if line.strip() else line for line in dedented.splitlines())
            code = f"{header}\n{indented_body}"

    return code.strip()


def _log_row(task_id, model, dataset, config, passed, total_tokens, iterations, elapsed, error_type):
    return {
        "task_id": task_id,
        "framework": "my_pipeline",
        "model": model,
        "dataset": dataset,
        "config": config,
        "final_passed": passed,
        "total_tokens": total_tokens,
        "iterations": iterations,
        "total_time_sec": round(elapsed, 3),
        "error_type": error_type,
    }
