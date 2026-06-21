# LLMs for Business Analytics — Group 1

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

> `en_core_web_sm` is the spaCy language model used for automatic named entity detection in the perturbation engine. It is not installed by `uv sync` and must be downloaded manually once per machine.
