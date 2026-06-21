"""
Run this script to inspect what entities spaCy finds in each dataset
and how many swap candidates exist per type.

Usage:
    python inspect_entity_pools.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from lib.perturbation.text.entity_pertubators import build_entity_pools

DATA_DIR = Path("datasets")

DATASETS = {
    "ESGenius": (DATA_DIR / "ESGenius_w_ref_1136q.csv", "source_text"),
    "CFB":      (DATA_DIR / "Climate Finance Bench - Dataset.xlsx", "Document extracts"),
}


def inspect(name: str, path: Path, column: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  —  column: '{column}'")
    print(f"{'='*60}")

    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    pools = build_entity_pools(df[column])

    if not pools:
        print("  No entities detected.")
        return

    for label, values in sorted(pools.items(), key=lambda x: -len(x[1])):
        n = len(values)
        status = "OK" if n >= 3 else ("LOW" if n == 2 else "ONLY 1 — swaps skipped")
        sample = ", ".join(sorted(values)[:5])
        if len(values) > 5:
            sample += f", ... (+{n - 5} more)"
        print(f"  [{status:18s}]  {label:<14} {n:>4} values   e.g. {sample}")


if __name__ == "__main__":
    for name, (path, column) in DATASETS.items():
        inspect(name, path, column)
    print()
