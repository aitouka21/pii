#!/usr/bin/env bash

PII_MODEL_CACHE_DIR="$PWD/.model-cache" ~/.local/bin/uv run python - <<'PY'
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "openai/privacy-filter"
cache_dir = ".model-cache"

AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
AutoModelForTokenClassification.from_pretrained(model_id, cache_dir=cache_dir)
PY
