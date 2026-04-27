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


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


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
