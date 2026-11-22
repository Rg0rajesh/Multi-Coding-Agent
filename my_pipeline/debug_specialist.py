"""
Rule-based Debug Specialist -- fixes superficial errors deterministically without LLM calls:
1. Missing imports / type hints (NameError).
2. Inconsistent indentation / mixed tabs and spaces (IndentationError, TabError).
3. Truncated functions / unterminated blocks (SyntaxError).
4. Missing colons on compound statements (SyntaxError).

Deterministic fixes execute in <1ms with 0 tokens.
"""
import re

_MODULE_NAMES = {
    "math", "re", "itertools", "functools", "collections",
    "string", "sys", "os", "random", "datetime", "json", "copy",
    "heapq", "bisect", "typing", "statistics",
}

_MEMBER_TO_MODULE = {
    # typing
    "List": "typing", "Dict": "typing", "Tuple": "typing", "Optional": "typing",
    "Union": "typing", "Any": "typing", "Set": "typing", "Callable": "typing",
    "Iterable": "typing", "Iterator": "typing", "Sequence": "typing", "Mapping": "typing",
    "NamedTuple": "typing",
    # math
    "sqrt": "math", "floor": "math", "ceil": "math", "pi": "math", "factorial": "math",
    "gcd": "math", "lcm": "math", "log": "math", "log2": "math", "log10": "math",
    "pow": "math", "comb": "math", "perm": "math", "inf": "math", "nan": "math",
    "prod": "math", "fabs": "math", "isclose": "math", "sin": "math", "cos": "math", "tan": "math",
    # collections
    "Counter": "collections", "defaultdict": "collections",
    "deque": "collections", "OrderedDict": "collections", "namedtuple": "collections",
    "ChainMap": "collections",
    # itertools
    "permutations": "itertools", "combinations": "itertools",
    "combinations_with_replacement": "itertools", "accumulate": "itertools",
    "product": "itertools", "chain": "itertools", "groupby": "itertools", "islice": "itertools",
    "zip_longest": "itertools",
    # functools
    "reduce": "functools", "lru_cache": "functools", "partial": "functools", "cmp_to_key": "functools",
    # heapq & bisect
    "heappush": "heapq", "heappop": "heapq", "heapify": "heapq",
    "nlargest": "heapq", "nsmallest": "heapq", "heappushpop": "heapq",
    "bisect_left": "bisect", "bisect_right": "bisect", "insort": "bisect", "bisect": "bisect",
    # string
    "ascii_letters": "string", "ascii_lowercase": "string", "ascii_uppercase": "string",
    "digits": "string", "punctuation": "string", "hexdigits": "string", "whitespace": "string",
    # statistics
    "mean": "statistics", "median": "statistics", "mode": "statistics", "stdev": "statistics",
}

_NAME_ERROR_RE = re.compile(r"name '(\w+)' is not defined")


def debug_specialist_fix(code: str, error_type: str, error_message: str) -> str:
    """Applies whichever rule-based fix matches the error type; returns original if no fix."""
    code = _sanitize_unicode(code)
    code = _fix_loose_statements(code)

    if error_type == "NameError":
        fixed = _fix_missing_import(code, error_message)
        if fixed != code:
            return fixed

    if error_type in ("IndentationError", "TabError"):
        return _fix_indentation(code)

    if error_type == "SyntaxError":
        code = _sanitize_unicode(code)
        code = _fix_missing_colon(code)
        fixed = _fix_truncated_function(code, error_message)
        if fixed != code:
            return fixed
        return _fix_indentation(code)

    return code


def _sanitize_unicode(code: str) -> str:
    """Normalizes non-ASCII punctuation that causes SyntaxErrors in generated code."""
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "-",   # em-dash
        "\u2015": "-",   # horizontal bar
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u00ab": '"',   # left guillemet
        "\u00bb": '"',   # right guillemet
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
        "\u2026": "...", # horizontal ellipsis
    }
    for char, rep in replacements.items():
        if char in code:
            code = code.replace(char, rep)
    return code


def _fix_loose_statements(code: str) -> str:
    """Strips top-level asserts, print calls, and __main__ driver blocks."""
    code = re.split(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", code, flags=re.MULTILINE)[0]
    lines = code.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not line.startswith(" ") and not line.startswith("\t"):
            if stripped.startswith("assert ") or stripped.startswith("print(") or stripped.startswith("check("):
                continue
        filtered.append(line)
    return "\n".join(filtered)


def _fix_missing_import(code: str, error_message: str) -> str:
    match = _NAME_ERROR_RE.search(error_message)
    if not match:
        return code

    name = match.group(1)
    if name in _MODULE_NAMES:
        import_line = f"import {name}\n"
    elif name in _MEMBER_TO_MODULE:
        import_line = f"from {_MEMBER_TO_MODULE[name]} import {name}\n"
    else:
        return code

    if import_line in code:
        return code
    return import_line + code


def _fix_indentation(code: str) -> str:
    """Normalizes tabs to 4 spaces and repairs unindented function bodies."""
    code = code.replace("\t", "    ")
    lines = code.splitlines()
    if not lines:
        return code

    # Check if lines following a top-level def/class are missing indentation
    fixed = []
    in_def = False
    indent_needed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed.append(line)
            continue
        if line.startswith("def ") or line.startswith("class "):
            in_def = True
            indent_needed = True
            fixed.append(line)
            continue
        if indent_needed:
            if not line.startswith(" ") and not line.startswith("\t"):
                # Unindented body line detected -- reindent all subsequent body lines
                fixed.append("    " + line)
                continue
            else:
                indent_needed = False
        fixed.append(line)
    return "\n".join(fixed)


def _fix_missing_colon(code: str) -> str:
    """Fixes missing colons at the end of def/if/for/while/elif/else statements."""
    fixed_lines = []
    compound_re = re.compile(r"^\s*(def\s+\w+\(.*?\)|if\s+.*|for\s+.*|while\s+.*|elif\s+.*|else|try|except.*|finally)\s*$")
    for line in code.splitlines():
        if compound_re.match(line) and not line.strip().endswith(":"):
            line = line + ":"
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _fix_truncated_function(code: str, error_message: str = "") -> str:
    """
    Strips incomplete trailing lines caused by model token limits.
    """
    eof_indicators = [
        "unexpected eof", "unexpected end of file", "expected an indented block",
        "was never closed", "unmatched"
    ]
    msg_lower = error_message.lower()
    is_truncation = any(ind in msg_lower for ind in eof_indicators) or not error_message

    if not is_truncation:
        return code

    lines = code.splitlines()
    for _ in range(min(len(lines), 15)):
        try:
            compile("\n".join(lines), "<candidate>", "exec")
            return "\n".join(lines)
        except SyntaxError:
            lines = lines[:-1]
    return code

