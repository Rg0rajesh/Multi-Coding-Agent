"""
Loads HumanEval and MBPP into the same normalized shape so pipeline.py
doesn't need to know which dataset it's running on:

    {"task_id": str, "prompt": str, "test": str, "entry_point": str}

HumanEval already ships in that shape. MBPP doesn't -- it gives a plain-text
description ("text") and a list of assert statements ("test_list") with no
declared entry point, so this module has to derive both the prompt and the
`check(candidate)` test wrapper.
"""
import json
import re
from pathlib import Path

_ASSERT_CALL_RE = re.compile(r"assert\s+(?:(?:int|str|float|bool|len|sorted|list|tuple|set)\s*\(\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def load_humaneval(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mbpp(path: str) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(_normalize_mbpp_task(json.loads(line)))
    return tasks


def _extract_signature(raw: dict, entry_point: str) -> str:
    """
    Extracts canonical function signature from reference code (e.g. def find_Product(arr, n):),
    standardizing MBPP tasks to match the HumanEval specification format (EvalPlus pattern).
    """
    code = raw.get("code", "")
    sig_match = re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\((.*?)\)\s*:", code, re.MULTILINE)
    if sig_match:
        return f"def {entry_point}({sig_match.group(1).strip()}):"
    return f"def {entry_point}(*args):"


def _normalize_mbpp_task(raw: dict) -> dict:
    entry_point = _find_entry_point(raw)
    signature = _extract_signature(raw, entry_point)
    test_examples = "\n".join(raw.get("test_list", [])[:2])
    prompt = (
        f"{raw['text']}\n\n"
        f"Write a function with the signature `{signature}`.\n"
        f"Examples:\n{test_examples}\n"
    )
    return {
        "task_id": f"MBPP/{raw['task_id']}",
        "prompt": prompt,
        "test": _build_check_function(raw, entry_point),
        "entry_point": entry_point,
    }


def _find_entry_point(raw: dict) -> str:
    """
    MBPP doesn't declare an entry point. The first assert in `test_list` is
    the reliable source -- it's the function actually being graded. The
    `code` field's first `def` is NOT safe to use as the primary source: a
    handful of MBPP tasks define a helper function before the real one, so
    picking the first `def` silently grades the wrong function. `code` is
    only a fallback for the rare case test_list itself doesn't parse.
    """
    for assertion in raw.get("test_list", []):
        match = _ASSERT_CALL_RE.search(assertion)
        if match:
            return match.group(1)

    match = _DEF_RE.search(raw.get("code", ""))
    if match:
        return match.group(1)

    raise ValueError(f"Couldn't determine entry_point for MBPP task {raw['task_id']}")


def _build_check_function(raw: dict, entry_point: str) -> str:
    """
    Rewrites MBPP's asserts (which call the function by its real name) into
    a HumanEval-style `check(candidate)`, so sandbox_exec.py can treat both
    datasets identically.
    """
    call_pattern = re.compile(rf"\b{re.escape(entry_point)}\s*\(")
    lines = []
    if raw.get("test_setup_code"):
        lines.append(raw["test_setup_code"])
    for assertion in raw["test_list"]:
        lines.append(call_pattern.sub("candidate(", assertion))

    body = "\n    ".join(lines)
    return f"def check(candidate):\n    {body}\n"


LOADERS = {
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
}


def load_dataset(name: str, path: str) -> list[dict]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}', expected one of {list(LOADERS)}")
    return LOADERS[name](path)
