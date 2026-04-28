#!/usr/bin/env bash

set -eu

PII_MODEL_CACHE_DIR="$PWD/.model-cache" TRANSFORMERS_OFFLINE=1 ~/.local/bin/uv run uvicorn pii_service.api:app --host 127.0.0.1 --port 8000
