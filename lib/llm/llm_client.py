import os
import time
from dotenv import load_dotenv

load_dotenv()

MODEL_1 = os.getenv("LLM_MODEL_1")
MODEL_2 = os.getenv("LLM_MODEL_2")
MODEL_3 = os.getenv("LLM_MODEL_3")

# Default model
MODEL = MODEL_2

# Pricing for MODEL_2 only (per 1M tokens) — MODEL_1 pricing unknown, ask teacher
COST_INPUT_PER_M  = 0.15
COST_OUTPUT_PER_M = 1.15


def compute_cost(response) -> float:
    """Return cost in USD for a single API response based on token usage."""
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return (input_tokens * COST_INPUT_PER_M + output_tokens * COST_OUTPUT_PER_M) / 1_000_000

_clients = {}


def get_client(n: int = 1):
    """Return the OpenAI client for model n (1, 2, or 3)."""
    if n not in _clients:
        from openai import OpenAI
        _clients[n] = OpenAI(
            api_key=os.getenv(f"LLM_API_KEY_{n}"),
            base_url=os.getenv(f"LLM_BASE_URL_{n}"),
        )
    return _clients[n]


def timed_completion(messages: list[dict], model: str = None, n: int = 1, **kwargs) -> tuple:
    """
    Call chat.completions.create and return (response, latency_ms).
    Use this instead of get_client().chat.completions.create() so latency is captured automatically.

    Example:
        response, latency_ms = timed_completion(messages, model=MODEL_1)
    """
    client = get_client(n)
    model  = model or globals().get(f"MODEL_{n}") or MODEL
    t0 = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000
    return response, latency_ms
