import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip() or None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


def _bounded_overlap(overlap: int, max_tokens: int) -> int:
    if overlap >= max_tokens:
        return max_tokens // 4
    return overlap


@dataclass(frozen=True)
class Settings:
    model_id: str = field(
        default_factory=lambda: _env_str("PII_MODEL_ID", "openai/privacy-filter")
    )
    device: str | None = field(default_factory=lambda: _env_optional_str("PII_DEVICE"))
    model_cache_dir: str | None = field(
        default_factory=lambda: _env_optional_str("PII_MODEL_CACHE_DIR")
    )
    transformers_offline: bool = field(
        default_factory=lambda: _env_bool("TRANSFORMERS_OFFLINE")
    )
    enable_chunking: bool = field(
        default_factory=lambda: _env_bool("PII_ENABLE_CHUNKING", True)
    )
    chunk_max_tokens: int = field(
        default_factory=lambda: _env_int("PII_CHUNK_MAX_TOKENS", 256)
    )
    chunk_overlap_tokens: int = field(
        default_factory=lambda: _bounded_overlap(
            _env_int("PII_CHUNK_OVERLAP_TOKENS", 32, minimum=0),
            _env_int("PII_CHUNK_MAX_TOKENS", 256),
        )
    )
    chunk_batch_size: int = field(
        default_factory=lambda: _env_int("PII_CHUNK_BATCH_SIZE", 4)
    )
