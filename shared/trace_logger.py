"""
Optional diagnostic trace -- separate from the locked results schema
(shared/logger.py), so it's safe to add fields here without breaking Hard
Rule 3. Logs what the Prompt Engineer actually writes and what code comes
back from each regeneration attempt, so "the plan isn't helping" can
actually be checked against real output instead of guessed at.

Off by default. Turn on with --trace on run_experiment.py for a small
sample run when you need to see what's happening step by step, not for
full benchmark runs -- it roughly doubles what gets written to disk per
task and isn't needed once things are working.
"""
import json
from pathlib import Path


class TraceLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(self, task_id: str, iteration: int, stage: str, error: str,
                 plan: str = None, code: str = None):
        row = {
            "task_id": task_id,
            "iteration": iteration,
            "stage": stage,
            "error": error,
            "plan": plan,
            "code": code,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
