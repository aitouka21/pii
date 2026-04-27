# PII Service POC

Minimal offline FastAPI service for PII redaction using the Hugging Face `openai/privacy-filter` token-classification model.

## Scope

This is a POC for local testing. It includes one redaction endpoint and does not include auth, persistence, queues, UI, production deployment, or compliance guarantees.

## Setup

Use `uv` from `~/.local/bin/uv` so dependencies stay inside the project-local `.venv`.

```bash
~/.local/bin/uv venv .venv
~/.local/bin/uv sync --all-groups
```

## Model cache

The service uses `openai/privacy-filter` by default. To run fully offline, download the model artifacts into a cache directory in the isolated test environment before setting offline mode.

```bash
PII_MODEL_CACHE_DIR="$PWD/.model-cache" ~/.local/bin/uv run python - <<'PY'
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "openai/privacy-filter"
cache_dir = ".model-cache"

AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
AutoModelForTokenClassification.from_pretrained(model_id, cache_dir=cache_dir)
PY
```

Then run with offline mode:

```bash
PII_MODEL_CACHE_DIR="$PWD/.model-cache" TRANSFORMERS_OFFLINE=1 ~/.local/bin/uv run uvicorn pii_service.api:app --host 127.0.0.1 --port 8000
```

## API

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Redact text:

```bash
curl -X POST http://127.0.0.1:8000/redact \
  -H 'Content-Type: application/json' \
  -d '{"text":"My name is Alice Smith and my email is alice@example.com"}'
```

Example response:

```json
{
  "redacted_text": "My name is [PRIVATE_PERSON] and my email is [PRIVATE_EMAIL]",
  "spans": [
    {
      "category": "private_person",
      "text": "Alice Smith",
      "start": 11,
      "end": 22,
      "score": 0.99
    },
    {
      "category": "private_email",
      "text": "alice@example.com",
      "start": 39,
      "end": 56,
      "score": 0.99
    }
  ]
}
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PII_MODEL_ID` | `openai/privacy-filter` | Hugging Face model ID or local model path |
| `PII_DEVICE` | unset | Optional Transformers pipeline device |
| `PII_MODEL_CACHE_DIR` | unset | Cache directory for model artifacts |
| `TRANSFORMERS_OFFLINE` | `0` | Set to `1` for offline-only model loading |

## Tests

```bash
~/.local/bin/uv run pytest -q
```

Tests use fake model inference and do not download the model.

## Limitations

The model is a PII minimization aid, not an anonymization or compliance guarantee. It can miss PII, over-redact benign text, and may need evaluation or fine-tuning for domain-specific policies.
