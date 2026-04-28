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
| `PII_ENABLE_CHUNKING` | `1` | Set to `0` to bypass token-window chunking and use whole-text inference |
| `PII_CHUNK_MAX_TOKENS` | `256` | Maximum model-token window size for long `/redact` inputs |
| `PII_CHUNK_OVERLAP_TOKENS` | `32` | Number of tokens repeated between adjacent windows to reduce boundary misses |
| `PII_CHUNK_BATCH_SIZE` | `4` | Number of token windows classified per Transformers pipeline batch |

## Performance and concurrency

Long `/redact` requests are split into overlapping token windows before model inference when `PII_ENABLE_CHUNKING=1`. The service batches those windows through the same Transformers pipeline, rebases chunk-local offsets to original-text offsets, then performs BIOES decoding and masking globally. Set `PII_ENABLE_CHUNKING=0` to bypass chunking and force whole-text inference for maximum-context mode, A/B benchmarking, or operational rollback.

The default guardrail settings are `PII_CHUNK_MAX_TOKENS=256`, `PII_CHUNK_OVERLAP_TOKENS=32`, and `PII_CHUNK_BATCH_SIZE=4`. During default selection, a local repeated PII benchmark of 7,520 characters / 1,681 tokens reduced model latency from 19.752s to 6.967s while preserving identical redacted text and identical span category/text/start/end output. A later smoke run on this implementation measured 14.774s to 6.160s on a 6,840-character payload with the same output parity. Hardware and input shape affect results, so tune these values against representative traffic.

The default `run.sh` starts one Uvicorn worker. FastAPI can accept concurrent synchronous requests through its threadpool, but all requests share the same in-process model and CPU resources. Multiple workers may improve throughput if the machine has enough memory for one model instance per worker, but workers do not reduce latency for one long request.

## Tests

```bash
~/.local/bin/uv run pytest -q
```

Tests use fake model inference and do not download the model.

## Limitations

The model is a PII minimization aid, not an anonymization or compliance guarantee. It can miss PII, over-redact benign text, and may need evaluation or fine-tuning for domain-specific policies.
