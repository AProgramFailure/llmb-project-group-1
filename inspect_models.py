"""
Query the university server for available models and their metadata.
Run with: python inspect_models.py

Usage:
    python inspect_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.llm.llm_client import get_client, MODEL_1, MODEL_2, MODEL_3


def inspect_model(n: int, model_name: str) -> None:
    if not model_name:
        print(f"\n  MODEL_{n}: not configured")
        return
    print(f"\n{'='*60}")
    print(f"  MODEL_{n} = {model_name}")
    print(f"{'='*60}")
    try:
        client = get_client(n)
        models = client.models.list()
        match = next((m for m in models.data if m.id == model_name), None)
        if match:
            print(f"  id          : {match.id}")
            print(f"  created     : {match.created}")
            print(f"  owned_by    : {match.owned_by}")
            extra = getattr(match, 'model_extra', None) or {}
            for key, val in extra.items():
                print(f"  {key:<12}: {val}")
        else:
            print(f"  '{model_name}' not found in server model list.")
            print("  Available models on this server:")
            for m in models.data:
                print(f"    - {m.id}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    for n, name in [(1, MODEL_1), (2, MODEL_2), (3, MODEL_3)]:
        inspect_model(n, name)
    print()
