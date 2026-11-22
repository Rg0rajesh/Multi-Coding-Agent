"""
Runs generated code in a separate process with a hard timeout. Uses
multiprocessing rather than signal.SIGALRM -- SIGALRM is Unix-only and
breaks on Windows.

Reports which STAGE a failure happened at, not just pass/fail:
  "compile" -- code itself doesn't even define/run (syntax error, missing
               import, truncated function, etc.)
  "test"    -- code runs fine but fails the actual test assertions (a
               genuine logic error)

This split matters because AdaCoder's own Code Evaluator routes these two
failure types to different agents: compile-stage failures go to the
rule-based Debug Specialist, test-stage failures go to the LLM-based
Prompt Engineer. Lumping them together (as an earlier version of this file
did) makes that routing impossible.
"""
import multiprocessing as mp


def _worker(code: str, test_code: str, entry_point: str, queue: mp.Queue):
    """
    Executes code and test assertions in an isolated namespace.
    Accurately classifies failure stages:
      - 'compile': syntax errors, indentation errors, missing module imports
      - 'runtime': exceptions during execution (IndexError, TypeError, ZeroDivisionError, etc.)
      - 'test': assertion failures (logic error / incorrect return value)
    """
    namespace = {
        "__builtins__": __builtins__,
    }
    # Pre-populate typing annotations and standard builtins to prevent artificial NameError
    exec("from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable, Iterable, Sequence, Mapping", namespace)
    
    # Stage 1: Compile and define user code
    try:
        exec(code, namespace)
    except Exception as exc:
        queue.put(("fail", "compile", type(exc).__name__, str(exc)))
        return

    # Stage 2: Validate entry point existence in candidate code
    if entry_point not in namespace:
        queue.put(("fail", "compile", "NameError", f"function '{entry_point}' was not defined in generated code"))
        return

    # Stage 3: Wrap candidate function to capture call inputs and outputs for rich diagnostic feedback
    original_fn = namespace[entry_point]
    last_call = {"args": None, "kwargs": None, "result": None}

    def instrumented_candidate(*args, **kwargs):
        last_call["args"] = args
        last_call["kwargs"] = kwargs
        res = original_fn(*args, **kwargs)
        last_call["result"] = res
        return res

    # Stage 4: Compile test code
    try:
        exec(test_code, namespace)
    except Exception as exc:
        queue.put(("fail", "compile", type(exc).__name__, f"Test setup error: {exc}"))
        return

    # Stage 5: Execute tests against candidate function
    try:
        namespace["check"](instrumented_candidate)
    except AssertionError as exc:
        msg = str(exc).strip()
        args_str = ", ".join(repr(a) for a in last_call["args"]) if last_call["args"] is not None else ""
        res_str = repr(last_call["result"]) if last_call["result"] is not None else ""
        if args_str and res_str:
            diag_msg = f"Logic Error: For input ({args_str}), returned {res_str}"
            if msg:
                diag_msg += f" (assertion: {msg})"
        elif msg:
            diag_msg = f"Logic Error: Test assertion failed ({msg})"
        else:
            diag_msg = "Logic Error: Test assertion failed (function returned incorrect output)"
        queue.put(("fail", "test", "AssertionError", diag_msg))
        return
    except NameError as exc:
        # A missing import that wasn't raised during definition but triggered on call
        queue.put(("fail", "compile", "NameError", str(exc)))
        return
    except Exception as exc:
        # Genuine runtime execution error (IndexError, TypeError, ValueError, ZeroDivisionError, etc.)
        args_str = ", ".join(repr(a) for a in last_call["args"]) if last_call["args"] is not None else ""
        if args_str:
            err_msg = f"{str(exc)} (occurred for input ({args_str}))"
        else:
            err_msg = str(exc)
        queue.put(("fail", "runtime", type(exc).__name__, err_msg))
        return

    queue.put(("pass", None, None, None))


def run_with_diagnosis(code: str, test_code: str, entry_point: str, timeout_sec: int = 10):
    """
    Returns (passed: bool, stage: str | None, error: str | None).
    stage is 'compile', 'runtime', or 'test' when passed is False, None when passed.
    error is 'ErrorType: message' for logging/display, None when passed.
    """
    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(code, test_code, entry_point, queue))
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "runtime", "TimeoutError: execution exceeded time limit (possible infinite loop)"

    if queue.empty():
        return False, "runtime", "CrashError: worker process exited without a result"

    status, stage, error_type, error_message = queue.get()
    if status == "pass":
        return True, None, None
    return False, stage, f"{error_type}: {error_message}"


def run_with_timeout(code: str, test_code: str, entry_point: str, timeout_sec: int = 10):
    """Back-compat wrapper for callers that only care about pass/fail + error."""
    passed, _stage, error = run_with_diagnosis(code, test_code, entry_point, timeout_sec)
    return passed, error


def _multi_assert_worker(code: str, test_code: str, entry_point: str, queue: mp.Queue):
    namespace = {"__builtins__": __builtins__}
    exec("from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable, Iterable, Sequence, Mapping", namespace)
    try:
        exec(code, namespace)
    except Exception:
        queue.put((0, 1))
        return

    if entry_point not in namespace:
        queue.put((0, 1))
        return

    fn = namespace[entry_point]

    assert_lines = [l.strip() for l in test_code.splitlines() if l.strip().startswith("assert ")]
    if not assert_lines:
        queue.put((1, 1))
        return

    passed = 0
    for a in assert_lines:
        single_test = f"def check(candidate):\n    {a}\n"
        try:
            test_ns = {"__builtins__": __builtins__}
            exec("from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable, Iterable, Sequence, Mapping", test_ns)
            exec(single_test, test_ns)
            test_ns["check"](fn)
            passed += 1
        except Exception:
            pass

    queue.put((passed, len(assert_lines)))


def evaluate_assertion_pass_count(code: str, test_code: str, entry_point: str, timeout_sec: int = 5) -> tuple[int, int]:
    """
    Evaluates individual assertions in test_code, returning (passed_count, total_count).
    Enables Module 3 to perform fine-grained candidate reranking.
    """
    queue = mp.Queue()
    proc = mp.Process(target=_multi_assert_worker, args=(code, test_code, entry_point, queue))
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return 0, 1
    if queue.empty():
        return 0, 1
    return queue.get()

