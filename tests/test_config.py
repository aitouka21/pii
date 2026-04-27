from pii_service.config import Settings


def test_settings_defaults_to_privacy_filter_model():
    settings = Settings()

    assert settings.model_id == "openai/privacy-filter"
    assert settings.device is None
    assert settings.model_cache_dir is None
    assert settings.transformers_offline is False


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("PII_MODEL_ID", "local/privacy-filter")
    monkeypatch.setenv("PII_DEVICE", "cpu")
    monkeypatch.setenv("PII_MODEL_CACHE_DIR", "/tmp/privacy-filter-cache")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    settings = Settings()

    assert settings.model_id == "local/privacy-filter"
    assert settings.device == "cpu"
    assert settings.model_cache_dir == "/tmp/privacy-filter-cache"
    assert settings.transformers_offline is True


def test_settings_treats_empty_model_id_as_default(monkeypatch):
    for value in ["", "   ", "\t"]:
        monkeypatch.setenv("PII_MODEL_ID", value)
        settings = Settings()
        assert settings.model_id == "openai/privacy-filter"


def test_settings_strips_model_id(monkeypatch):
    monkeypatch.setenv("PII_MODEL_ID", " local/privacy-filter ")

    settings = Settings()

    assert settings.model_id == "local/privacy-filter"


def test_settings_treats_empty_optional_env_values_as_unset(monkeypatch):
    monkeypatch.setenv("PII_DEVICE", "")
    monkeypatch.setenv("PII_MODEL_CACHE_DIR", "")

    settings = Settings()

    assert settings.device is None
    assert settings.model_cache_dir is None


def test_settings_treats_whitespace_optional_env_values_as_unset(monkeypatch):
    monkeypatch.setenv("PII_DEVICE", "  ")
    monkeypatch.setenv("PII_MODEL_CACHE_DIR", "\t")

    settings = Settings()

    assert settings.device is None
    assert settings.model_cache_dir is None


def test_settings_handles_falsy_offline_values(monkeypatch):
    for value in ["0", "false", "no", "off", ""]:
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", value)
        settings = Settings()
        assert settings.transformers_offline is False
