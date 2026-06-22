# LLMs for Business Analytics — Group 1

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

> `en_core_web_sm` is the spaCy language model used for automatic named entity detection in the perturbation engine. It is not installed by `uv sync` and must be downloaded manually once per machine.

## Environment variables

Create a `.env` file at the root of the project with the following structure:

```env
IPKERNELAPP_LOG_LEVEL=ERROR
IPYTHONDIR=.ipython

# Model 1 — Qwen  (pricing: $0.15 input / $0.95 output per 1M tokens)
LLM_MODEL_1=UTM2
LLM_API_KEY_1=your_qwen_api_key
LLM_BASE_URL_1=http://grading-llm.eemcs.utwente.nl:8801/v1/

# Model 2 — MiniMax  (pricing: $0.15 input / $1.15 output per 1M tokens)
LLM_MODEL_2=UTM
LLM_API_KEY_2=your_minimax_api_key
LLM_BASE_URL_2=http://grading-llm.eemcs.utwente.nl:8801/v1/

# Model 3 — OpenAI (to be configured)
LLM_MODEL_3=
LLM_API_KEY_3=
LLM_BASE_URL_3=
```

The active default model is set in `lib/llm/llm_client.py` via the `MODEL` variable.
