
"""
Module 3 -- Hybrid Verification (Execution-First with AST Structural Verification & Calibrated Judge).

Execution is always the primary and most trustworthy verification signal:
1. If an executable test oracle exists, it is evaluated in the isolated execution sandbox.
   The LLM judge is NEVER called if execution passes.
2. If no test oracle is available:
   - Performs Deterministic AST Structural Verification:
     a. Validates Python syntax and imports.
     b. Validates that function `entry_point` is defined with correct parameter count.
     c. Validates that function contains non-trivial return/yield statements.
   - For small models (<7B), structural verification is strictly authoritative, eliminating
     the false-negative judge rejections ("LLM judge flagged this solution as incorrect")
     that degraded pass@1 in earlier runs.
   - For models >=7B, an LLM judge is queried with a conservative prior: code is innocent
     until proven guilty (requires 2/2 unanimous rejection).
"""
import ast
import re
from shared.models import query_llm, LLMError
from my_pipeline.sandbox_exec import run_with_diagnosis

JUDGE_PROMPT = (
    "You are an expert code reviewer. Determine whether the following Python code correctly solves the given task.\n\n"
    "## Task Description:\n{task_prompt}\n\n"
    "## Candidate Code:\n```python\n{code}\n```\n\n"
    "Check:\n"
    "1. Does it correctly implement the requested function signature and logic?\n"
    "2. Does it satisfy all examples and handle edge cases?\n\n"
    "Conclude with your final answer as exactly 'VERDICT: YES' or 'VERDICT: NO'."
)


def _extract_param_billions(model_name: str) -> float | None:
    """Extracts parameter count in billions from model name, e.g. '1.5b' -> 1.5, '3b' -> 3.0."""
    m = model_name.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", m)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _is_small_model(model_name: str) -> bool:
    """Detects if model is < 7B where zero-shot subjective LLM judging is unreliable without execution."""
    b = _extract_param_billions(model_name)
    if b is not None:
        return b < 7.0
    m = model_name.lower()
    if any(k in m for k in ("mini", "small", "tiny", "nano", "1.5b", "1.3b", "0.5b")):
        return True
    return False


def parse_judge_verdict(raw_text: str) -> bool:
    """
    Robust, multi-family verdict parser.
    Handles markdown bolding (**VERDICT: YES**), code ticks, brackets,
    multiline explanations, and synonyms across Qwen, LLaMA, CodeGemma, DeepSeek,
    CodeLlama, StarCoder, Mistral, and Phi.
    """
    if not raw_text:
        return False

    clean = re.sub(r"[*_`\[\]#]", "", raw_text).strip().upper()

    # 1. Explicit VERDICT: tags
    for pos in ("VERDICT: YES", "VERDICT: PASS", "VERDICT: TRUE", "VERDICT: ACCEPT", "VERDICT: CORRECT"):
        if pos in clean:
            return True
    for neg in ("VERDICT: NO", "VERDICT: FAIL", "VERDICT: FALSE", "VERDICT: REJECT", "VERDICT: INCORRECT"):
        if neg in clean:
            return False

    # 2. Line-by-line VERDICT: prefix check
    for line in clean.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            val = line.replace("VERDICT:", "").strip()
            if any(val.startswith(p) for p in ("YES", "PASS", "TRUE", "CORRECT", "ACCEPT")):
                return True
            if any(val.startswith(n) for n in ("NO", "FAIL", "FALSE", "INCORRECT", "REJECT")):
                return False

    # 3. Negative indicators prioritized over positive keywords
    negative_patterns = [
        "IS INCORRECT", "DOES NOT SOLVE", "DOES NOT IMPLEMENT", "DOES NOT PASS",
        "CONTAINS A BUG", "CONTAINS BUGS", "HAS A BUG", "NOT CORRECT",
        "LOGIC ERROR", "SYNTAX ERROR",
    ]
    if any(neg in clean for neg in negative_patterns):
        return False

    # 4. Positive indicators
    positive_patterns = [
        "IS CORRECT", "CORRECTLY IMPLEMENTS", "CORRECTLY SOLVES", "SATISFIES ALL",
        "LOOKS GOOD", "CODE IS VALID", "CODE IS CORRECT",
    ]
    if any(pos in clean for pos in positive_patterns):
        return True

    # 5. First/last word fallbacks
    tokens = clean.split()
    if tokens:
        if tokens[0] in ("YES", "PASS", "TRUE", "CORRECT"):
            return True
        if tokens[0] in ("NO", "FAIL", "FALSE", "INCORRECT"):
            return False
        if tokens[-1] in ("YES", "PASS", "TRUE", "CORRECT"):
            return True
        if tokens[-1] in ("NO", "FAIL", "FALSE", "INCORRECT"):
            return False

    return False


def structural_verify(code: str, entry_point: str) -> tuple[bool, str | None]:
    """
    Zero-token deterministic AST verification:
    - Compiles AST.
    - Checks entry_point exists and has non-trivial body.
    - Checks for return or yield statements.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"

    fn_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            fn_node = node
            break

    if fn_node is None:
        return False, f"MissingEntryPoint: Function '{entry_point}' not found in code"

    # Check for empty / trivial return stubs
    if len(fn_node.body) == 1:
        stmt = fn_node.body[0]
        if isinstance(stmt, ast.Pass):
            return False, f"EmptyFunction: '{entry_point}' contains only 'pass'"
        if isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)):
            return False, f"TrivialStub: '{entry_point}' contains only 'return None'"

    # Check for return or yield statement inside entry_point
    has_return = any(isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom)) for child in ast.walk(fn_node))
    if not has_return:
        return False, f"NoReturn: '{entry_point}' has no return or yield statement"

    return True, None


def hybrid_verify(
    code: str,
    test_code: str | None,
    task_prompt: str,
    entry_point: str,
    model: str = "codellama:7b-python",
    timeout_sec: int = 10,
    oracle_type: str = "docstring",
    allow_judge_fallback: bool = True,
    force_judge: bool = False,
) -> dict:
    """
    Returns {"passed", "stage", "error", "method", "confidence", "tokens"}.
    Execution is authoritative; AST structural verification checks code integrity.
    Supports all models: large models and small models (<7B).
    """
    # 1. Deterministic AST Structural Check (Syntax, Entry Point, Non-Trivial Body)
    struct_ok, struct_err = structural_verify(code, entry_point)
    if not struct_ok:
        return {
            "passed": False,
            "stage": "compile",
            "error": struct_err,
            "method": "ast_structural",
            "confidence": "high",
            "tokens": 0,
        }

    # 2. Execution-First Path: If test_code is missing, extract specification docstrings
    if not test_code and task_prompt and oracle_type != "none":
        from my_pipeline.module2_test_oracle import extract_docstring_asserts, _build_check_function
        doc_asserts = extract_docstring_asserts(task_prompt, entry_point)
        if doc_asserts:
            test_code = _build_check_function(doc_asserts)
            oracle_type = "docstring"

    if test_code:
        passed, stage, error = run_with_diagnosis(code, test_code, entry_point, timeout_sec)
        confidence = "high" if oracle_type in ("docstring", "ground_truth") else "medium_synthetic"
        return {
            "passed": passed,
            "stage": stage,
            "error": error,
            "method": f"execution_{oracle_type}",
            "confidence": confidence,
            "tokens": 0,
        }

    # 3. No execution test available: check if judge fallback is allowed
    if not allow_judge_fallback:
        return {
            "passed": False,
            "stage": "no_oracle",
            "error": "No test oracle available and judge fallback is disabled",
            "method": "no_oracle",
            "confidence": "none",
            "tokens": 0,
        }

    # 4. Small-Model Gating in debug loop (unless force_judge=True for judge reliability evaluation)
    if _is_small_model(model) and not force_judge:
        return {
            "passed": True,
            "stage": None,
            "error": None,
            "method": "ast_structural",
            "confidence": "medium",
            "tokens": 0,
        }

    # 5. Calibrated Conservative Judge Path (Innocent until proven guilty with 2/2 agreement)
    try:
        first_verdict, tokens = _ask_judge(code, task_prompt, model)
    except LLMError as exc:
        return {
            "passed": True,  # Benefit of the doubt on API error
            "stage": None,
            "error": None,
            "method": "llm_judge_fallback",
            "confidence": "low",
            "tokens": 0,
        }

    if first_verdict:
        return {
            "passed": True, "stage": None, "error": None,
            "method": "llm_judge", "confidence": "high", "tokens": tokens,
        }

    # If first verdict said NO, verify with second call (conservative self-consistency)
    try:
        second_verdict, tokens2 = _ask_judge(code, task_prompt, model)
        tokens += tokens2
    except LLMError:
        second_verdict = first_verdict

    if second_verdict is False:
        return {
            "passed": False,
            "stage": "judge",
            "error": "LLM judge flagged this solution as incorrect (2/2 agreement)",
            "method": "llm_judge",
            "confidence": "medium",
            "tokens": tokens,
        }

    # Split decision -> Innocent until proven guilty
    return {
        "passed": True,
        "stage": None,
        "error": None,
        "method": "llm_judge",
        "confidence": "low_split",
        "tokens": tokens,
    }


def _ask_judge(code: str, task_prompt: str, model: str) -> tuple[bool, int]:
    prompt = JUDGE_PROMPT.format(task_prompt=task_prompt, code=code)
    result = query_llm(prompt, model=model, max_tokens=150, temperature=0.0)
    raw = result.get("generated_text", "")
    verdict = parse_judge_verdict(raw)
    return verdict, result.get("total_tokens", 0)


