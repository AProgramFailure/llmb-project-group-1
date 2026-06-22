"""
Run logger — writes one JSONL file per run to logs/.
Each line is one LLM call with all inputs, outputs, and timing.
Load any log back into a DataFrame with RunLogger.load(path).
"""

import json
from datetime import datetime
from pathlib import Path


LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


class RunLogger:
    def __init__(self, name: str = "run", logs_dir: Path = LOGS_DIR):
        """
        Parameters
        ----------
        name     : short label for the run (e.g. "esg_llm", "cfb_agent")
        logs_dir : directory where log files are saved (default: logs/)
        """
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = logs_dir / f"{timestamp}_{name}.jsonl"
        self._file = open(self.path, "a", encoding="utf-8")
        print(f"Logging to {self.path}")

    def log(
        self,
        *,
        question_id,
        model: str,
        perturbation_type: str,
        perturbation_level: int,
        prompt: str,
        response: str,
        ground_truth: str = None,
        latency_ms: float = None,
        tokens_used: int = None,
        extra: dict = None,
    ) -> None:
        """Write one LLM call to the log file."""
        record = {
            "timestamp":          datetime.now().isoformat(),
            "question_id":        question_id,
            "model":              model,
            "perturbation_type":  perturbation_type,
            "perturbation_level": perturbation_level,
            "prompt":             prompt,
            "response":           response,
            "ground_truth":       ground_truth,
            "latency_ms":         latency_ms,
            "tokens_used":        tokens_used,
            **(extra or {}),
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def load(path: str | Path):
        """Load a JSONL log file back into a pandas DataFrame."""
        import pandas as pd
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    @staticmethod
    def list_runs(logs_dir: Path = LOGS_DIR):
        """Print all saved log files with their row counts."""
        logs_dir = Path(logs_dir)
        files = sorted(logs_dir.glob("*.jsonl"))
        if not files:
            print("No log files found.")
            return
        for f in files:
            n = sum(1 for line in open(f, encoding="utf-8") if line.strip())
            print(f"{f.name}  ({n} rows)")
