import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

MODEL_1 = os.getenv("LLM_MODEL_1")  # Qwen2-72B
MODEL_2 = os.getenv("LLM_MODEL_2")  # MiniMax
MODEL_3 = os.getenv("LLM_MODEL_3")  # OpenAI (to be configured)

# Default model — only change MODEL here, MODEL_N is derived automatically
MODEL = MODEL_1
MODEL_N = {MODEL_1: 1, MODEL_2: 2, MODEL_3: 3}.get(MODEL, 1)

# Pricing per 1M tokens, keyed by model number.
# MODEL_1 (Qwen):   $0.15 input / $0.95 output
# MODEL_2 (MiniMax): $0.15 input / $1.15 output
# MODEL_3 (OpenAI): to be configured
COSTS = {
    1: {"input": 0.15,  "output": 0.95},   # Qwen2-72B (UTM2)
    2: {"input": 0.15,  "output": 1.15},   # MiniMax (UTM)
    3: {"input": None,  "output": None},   # OpenAI (to be configured)
}

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that some models include in their output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Per-model extra_body to disable thinking at the API level.
# MODEL_1 (Qwen via vLLM): chat_template_kwargs works.
# MODEL_2 (MiniMax/UTM):   server has thinking hardcoded — no parameter disables it.
#                           strip_thinking() handles removal on the output side.
# MODEL_3 (OpenAI):        no thinking mode to disable.
_THINKING_OFF_EXTRA_BODY = {
    1: {"chat_template_kwargs": {"enable_thinking": False}},
    2: None,
    3: None,
}

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


def compute_cost(response, n: int) -> float | None:
    """
    Return cost in USD for a single API response based on token usage.
    Returns None if pricing for model n is not yet configured.
    """
    pricing = COSTS.get(n, {})
    if pricing.get("input") is None or pricing.get("output") is None:
        return None
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def timed_completion(messages: list[dict], model: str = None, n: int = 1, **kwargs) -> tuple:
    """
    Call chat.completions.create and return (response, latency_ms).
    Thinking is disabled automatically per model. Pass extra_body explicitly to override.

    Example:
        response, latency_ms = timed_completion(messages, model=MODEL_1, n=1)
    """
    client = get_client(n)
    model  = model or globals().get(f"MODEL_{n}") or MODEL

    extra_body = _THINKING_OFF_EXTRA_BODY.get(n)
    if extra_body is not None:
        kwargs.setdefault("extra_body", extra_body)

    t0 = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000
    return response, latency_ms
