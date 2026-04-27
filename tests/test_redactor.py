from pii_service.config import Settings
from pii_service.redactor import DetectedSpan, PrivacyFilterRedactor, redact_text_with_spans


def test_redact_text_with_spans_replaces_offsets_from_end_to_start():
    text = "Alice emailed alice@example.com from 123 Main St."
    spans = [
        DetectedSpan(category="private_person", text="Alice", start=0, end=5, score=0.99),
        DetectedSpan(category="private_email", text="alice@example.com", start=14, end=31, score=0.98),
        DetectedSpan(category="private_address", text="123 Main St.", start=37, end=49, score=0.97),
    ]

    result = redact_text_with_spans(text, spans)

    assert result == "[PRIVATE_PERSON] emailed [PRIVATE_EMAIL] from [PRIVATE_ADDRESS]"


def test_redact_text_with_spans_ignores_spans_without_offsets():
    text = "Call Alice"
    spans = [DetectedSpan(category="private_person", text="Alice", start=None, end=None, score=0.9)]

    result = redact_text_with_spans(text, spans)

    assert result == "Call Alice"


def test_redact_text_with_spans_validates_against_original_text():
    text = "x y"
    spans = [
        DetectedSpan(
            category="very_long_category",
            text="y",
            start=2,
            end=3,
            score=0.99,
        ),
        DetectedSpan(category="secret", text="invalid", start=0, end=10, score=0.99),
    ]

    result = redact_text_with_spans(text, spans)

    assert result == "x [VERY_LONG_CATEGORY]"


def test_redact_text_with_spans_prefers_longest_overlapping_span():
    text = "John's email is john@example.com"
    spans = [
        DetectedSpan(category="private_person", text="John", start=0, end=4, score=0.99),
        DetectedSpan(
            category="private_email",
            text="john@example.com",
            start=16,
            end=32,
            score=0.98,
        ),
        DetectedSpan(category="private_person", text="john", start=16, end=20, score=0.99),
    ]

    result = redact_text_with_spans(text, spans)

    assert result == "[PRIVATE_PERSON]'s email is [PRIVATE_EMAIL]"


def test_redact_text_with_spans_prefers_highest_score_for_equal_length_overlap():
    text = "secret code"
    spans = [
        DetectedSpan(category="low_confidence", text="secret", start=0, end=6, score=0.8),
        DetectedSpan(category="high_confidence", text="secret", start=0, end=6, score=0.9),
    ]

    result = redact_text_with_spans(text, spans)

    assert result == "[HIGH_CONFIDENCE] code"


def test_redactor_load_passes_cache_and_offline_kwargs(monkeypatch):
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)

        def fake_classifier(text, aggregation_strategy):
            return []

        return fake_classifier

    monkeypatch.setattr("pii_service.redactor.pipeline", fake_pipeline)
    settings = Settings(
        model_id="local/privacy-filter",
        device="cpu",
        model_cache_dir="/home/malkuth/pii/.cache/privacy-filter",
        transformers_offline=True,
    )

    redactor = PrivacyFilterRedactor(settings=settings)
    redactor.load()

    assert captured["task"] == "token-classification"
    assert captured["model"] == "local/privacy-filter"
    assert captured["device"] == "cpu"
    assert captured["model_kwargs"] == {
        "cache_dir": "/home/malkuth/pii/.cache/privacy-filter",
        "local_files_only": True,
    }
    assert captured["tokenizer_kwargs"] == {
        "cache_dir": "/home/malkuth/pii/.cache/privacy-filter",
        "local_files_only": True,
    }


def test_redactor_uses_injected_classifier():
    def fake_classifier(text, aggregation_strategy):
        assert text == "Email alice@example.com"
        assert aggregation_strategy == "simple"
        return [
            {
                "entity_group": "private_email",
                "word": " alice@example.com",
                "start": 5,
                "end": 23,
                "score": 0.99,
            }
        ]

    redactor = PrivacyFilterRedactor(classifier=fake_classifier)

    result = redactor.redact("Email alice@example.com")

    assert result.redacted_text == "Email[PRIVATE_EMAIL]"
    assert len(result.spans) == 1
    assert result.spans[0].category == "private_email"
    assert result.spans[0].text == " alice@example.com"
    assert result.spans[0].start == 5
    assert result.spans[0].end == 23
    assert result.spans[0].score == 0.99
