"""
Module 1 -- Token-Efficient Adaptive Debugging Loop with Progress-Aware Hierarchy.

Caps debugging iterations intelligently to prevent runaway token costs:
1. Progress-Aware Error Hierarchy:
   - Rank 0: NoError / Passed (Optimal)
   - Rank 1: AssertionError / LogicError (Function executes cleanly, refining logic)
   - Rank 2: RuntimeError / TypeError / ValueError / IndexError / ZeroDivisionError
   - Rank 3: SyntaxError / IndentationError / NameError (Fatal / Compilation block)
2. Halts early on dead-end stagnation (identical code or repeating fatal errors).
3. Halts on cyclic oscillation (A -> B -> A).
4. Dynamically expands budget up to max_cap when candidate code explores distinct logic on Rank 1 (solvable logic problems).
"""
import re

DEFAULT_MAX_CAP = 5
DEFAULT_MIN_CAP = 3


def get_error_severity_rank(error_str: str | None, stage: str | None = None) -> int:
    """
    Returns integer rank (0=best, 3=worst).
    """
    if not error_str:
        return 0
    err_lower = error_str.lower()
    if any(k in err_lower for k in ("syntaxerror", "indentationerror", "taberror", "nameerror", "timeouterror")):
        return 3
    if any(k in err_lower for k in ("zerodivisionerror", "typeerror", "valueerror", "indexerror", "keyerror", "attributeerror")):
        return 2
    if "assertionerror" in err_lower or "logic error" in err_lower or stage == "test":
        return 1
    return 2


def normalize_error_signature(error_str: str | None, stage: str | None = None) -> str:
    if not error_str:
        return f"{stage or 'unknown'}::NoError"
    clean = error_str.split("::", 1)[0].strip()
    clean = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", clean)
    clean = re.sub(r"line \d+", "line N", clean)
    prefix = f"{stage}::" if stage else ""
    return f"{prefix}{clean}"


def adaptive_threshold(
    error_history: list[str],
    stage_history: list[str] | None = None,
    code_history: list[str] | None = None,
    max_cap: int = DEFAULT_MAX_CAP,
    min_cap: int = DEFAULT_MIN_CAP,
    dataset: str = "humaneval",
) -> int:
    n = len(error_history)
    if n <= 1:
        return min_cap

    stages = stage_history or [""] * n
    normalized = [normalize_error_signature(e, s) for e, s in zip(error_history, stages)]
    ranks = [get_error_severity_rank(e, s) for e, s in zip(error_history, stages)]

    # 1. Exact duplicate code and error twice in a row -> True stagnation (STOP at 2 iters)
    if code_history and len(code_history) >= 2 and code_history[-1] == code_history[-2] and normalized[-1] == normalized[-2]:
        return n

    # 2. Repeated fatal syntax/compilation error twice in a row -> Stop early
    if ranks[-1] == 3 and len(ranks) >= 2 and ranks[-2] == 3:
        return n

    # 3. Oscillation: cycling back and forth (A -> B -> A) -> STOP
    if n >= 3 and normalized[-1] == normalized[-3]:
        if code_history and len(code_history) >= 3 and code_history[-1] == code_history[-3]:
            return n

    if n < min_cap:
        return min_cap

    # For MBPP: Solvable logic errors (Rank 1 AssertionError) explore up to max_cap
    if dataset == "mbpp" and ranks[-1] == 1:
        return max_cap

    # 4. Progress up the error severity hierarchy (e.g. Rank 3 -> Rank 2 or Rank 2 -> Rank 1)
    if n >= 2 and ranks[-1] < ranks[-2]:
        # Error severity improved! Grant +2 iterations
        return min(max_cap + 1, n + 2)

    # 5. Forward progress across stages (e.g. compile -> runtime/test)
    if n >= 2 and stages[-1] != stages[-2] and stages[-1] in ("test", "runtime"):
        return min(max_cap + 1, n + 2)

    # 6. Active implementation exploration on non-fatal errors (Rank 1 logic or Rank 2 runtime):
    # If code is distinct across iterations, the model is actively trying alternative implementations.
    # Allow full max_cap iterations so we don't prematurely abort viable solutions.
    if ranks[-1] in (1, 2) and code_history and len(code_history) >= 2 and code_history[-1] != code_history[-2]:
        return max_cap

    return min(n + 1, max_cap)


class AdaptiveDebugLoop:
    def __init__(self, max_cap: int = DEFAULT_MAX_CAP, min_cap: int = DEFAULT_MIN_CAP, dataset: str = "humaneval"):
        self.max_cap = max_cap
        self.min_cap = min_cap
        self.dataset = dataset
        self.error_history: list[str] = []
        self.stage_history: list[str] = []
        self.code_history: list[str] = []

    def record_error(self, error_signature: str | None, stage: str | None = None, code: str | None = None):
        self.error_history.append(error_signature or "")
        self.stage_history.append(stage or "")
        if code is not None:
            self.code_history.append(code.strip())

    def should_stop(self) -> bool:
        if not self.error_history:
            return False
        limit = adaptive_threshold(
            self.error_history,
            self.stage_history,
            self.code_history,
            self.max_cap,
            self.min_cap,
            dataset=self.dataset
        )
        return len(self.error_history) >= limit

    @property
    def iterations_used(self) -> int:
        return len(self.error_history)

