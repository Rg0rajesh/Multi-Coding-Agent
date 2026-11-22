"""
Append-only JSONL logger. The schema below is locked (Master Project Prompt,
Section 4/Hard Rule 3) -- don't add, remove, or rename fields once a real run
has started, or every analysis script downstream breaks.
"""
import json
from pathlib import Path
from threading import Lock

SCHEMA_FIELDS = [
    "task_id", "framework", "model", "dataset", "config",
    "final_passed", "total_tokens", "iterations", "total_time_sec", "error_type",
]

_write_lock = Lock()


class ResultLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields):
        missing = set(SCHEMA_FIELDS) - set(fields)
        if missing:
            raise ValueError(f"Missing required log fields: {missing}")
        extra = set(fields) - set(SCHEMA_FIELDS)
        if extra:
            raise ValueError(f"Unexpected fields, schema is locked: {extra}")

        row = {k: fields[k] for k in SCHEMA_FIELDS}
        with _write_lock, open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def already_done(self) -> set:
        """Task IDs already in this log file -- used for --resume."""
        if not self.log_path.exists():
            return set()

        done = set()
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # skip a corrupted line rather than crash a resumed run
        return done

    def passed_tasks(self) -> set:
        """Task IDs that have final_passed == True -- used for --rerun_failed."""
        if not self.log_path.exists():
            return set()

        passed = set()
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("final_passed"):
                        passed.add(d["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return passed

