"""
Quick connectivity test for MODEL_2.
Run with: python test_model2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.llm.llm_client import get_client, MODEL_2, timed_completion

print(f"Testing MODEL_2 = {MODEL_2} ...")

try:
    response, latency_ms = timed_completion(
        messages=[{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
        model=MODEL_2,
        n=2,
    )
    print(f"Response  : {response.choices[0].message.content.strip()}")
    print(f"Latency   : {latency_ms:.0f} ms")
    print(f"Tokens    : {response.usage.total_tokens}")
    print("MODEL_2 is reachable.")
except Exception as e:
    print(f"MODEL_2 is NOT reachable: {e}")
