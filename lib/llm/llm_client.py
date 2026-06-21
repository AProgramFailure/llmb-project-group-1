import os
from dotenv import load_dotenv

load_dotenv()

MODEL_1 = os.getenv("LLM_MODEL_1")
MODEL_2 = os.getenv("LLM_MODEL_2")
MODEL_3 = os.getenv("LLM_MODEL_3")

# Default model
MODEL = MODEL_2

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
