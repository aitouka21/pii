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


def test_redactor_keeps_longest_decoded_span_when_spans_overlap():
    def fake_classifier(text, aggregation_strategy):
        assert text == "john@example.com"
        assert aggregation_strategy == "none"
        return [
            {
                "entity": "B-private_email",
                "word": "john",
                "start": 0,
                "end": 4,
                "score": 0.98,
            },
            {
                "entity": "I-private_email",
                "word": "@example",
                "start": 4,
                "end": 12,
                "score": 0.99,
            },
            {
                "entity": "E-private_email",
                "word": ".com",
                "start": 12,
                "end": 16,
                "score": 0.97,
            },
            {
                "entity": "S-private_person",
                "word": "john",
                "start": 0,
                "end": 4,
                "score": 0.999,
            },
        ]

    redactor = PrivacyFilterRedactor(classifier=fake_classifier)

    result = redactor.redact("john@example.com")

    assert result.redacted_text == "[PRIVATE_EMAIL]"
    assert result.spans == [
        DetectedSpan(
            category="private_email",
            text="john@example.com",
            start=0,
            end=16,
            score=0.99,
        )
    ]


def test_redactor_uses_overlap_policy_for_equal_length_decoded_spans():
    def fake_classifier(text, aggregation_strategy):
        assert text == "Alice"
        assert aggregation_strategy == "none"
        return [
            {
                "entity": "B-private_person",
                "word": "Alice",
                "start": 0,
                "end": 5,
                "score": 0.70,
            },
            {
                "entity": "S-private_email",
                "word": "Alice",
                "start": 0,
                "end": 5,
                "score": 0.95,
            },
        ]

    redactor = PrivacyFilterRedactor(classifier=fake_classifier)

    result = redactor.redact("Alice")

    assert result.redacted_text == "[PRIVATE_EMAIL]"
    assert result.spans == [
        DetectedSpan(
            category="private_email",
            text="Alice",
            start=0,
            end=5,
            score=0.95,
        )
    ]


def test_redact_text_with_spans_prefers_highest_score_for_equal_length_overlap():
    text = "secret code"
    spans = [
        DetectedSpan(category="low_confidence", text="secret", start=0, end=6, score=0.8),
        DetectedSpan(category="high_confidence", text="secret", start=0, end=6, score=0.9),
    ]

    result = redact_text_with_spans(text, spans)

    assert result == "[HIGH_CONFIDENCE] code"


def test_redactor_decodes_bioes_chunks_from_raw_model_tokens():
    def fake_classifier(text, aggregation_strategy):
        assert text == "My name is Alice Smith and my email is alice@example.com"
        assert aggregation_strategy == "none"
        return [
            {
                "entity": "B-private_person",
                "word": "ĠAlice",
                "start": 10,
                "end": 16,
                "score": 0.9999974966049194,
            },
            {
                "entity": "E-private_person",
                "word": "ĠSmith",
                "start": 16,
                "end": 22,
                "score": 0.9999980926513672,
            },
            {
                "entity": "B-private_email",
                "word": "Ġalice",
                "start": 38,
                "end": 44,
                "score": 0.9999756813049316,
            },
            {
                "entity": "I-private_email",
                "word": "@example",
                "start": 44,
                "end": 52,
                "score": 0.9999961853027344,
            },
            {
                "entity": "E-private_email",
                "word": ".com",
                "start": 52,
                "end": 56,
                "score": 0.9999011754989624,
            },
        ]

    redactor = PrivacyFilterRedactor(classifier=fake_classifier)

    result = redactor.redact("My name is Alice Smith and my email is alice@example.com")

    assert result.redacted_text == (
        "My name is [PRIVATE_PERSON] and my email is [PRIVATE_EMAIL]"
    )
    assert result.spans == [
        DetectedSpan(
            category="private_person",
            text="Alice Smith",
            start=11,
            end=22,
            score=0.9999980926513672,
        ),
        DetectedSpan(
            category="private_email",
            text="alice@example.com",
            start=39,
            end=56,
            score=0.9999961853027344,
        ),
    ]


def test_redactor_keeps_separate_bioes_entities_apart():
    def fake_classifier(text, aggregation_strategy):
        assert text == "Alice and Bob"
        assert aggregation_strategy == "none"
        return [
            {
                "entity": "S-private_person",
                "word": "Alice",
                "start": 0,
                "end": 5,
                "score": 0.99,
            },
            {
                "entity": "S-private_person",
                "word": "ĠBob",
                "start": 9,
                "end": 13,
                "score": 0.98,
            },
        ]

    redactor = PrivacyFilterRedactor(classifier=fake_classifier)

    result = redactor.redact("Alice and Bob")

    assert result.redacted_text == "[PRIVATE_PERSON] and [PRIVATE_PERSON]"
    assert result.spans == [
        DetectedSpan(
            category="private_person",
            text="Alice",
            start=0,
            end=5,
            score=0.99,
        ),
        DetectedSpan(
            category="private_person",
            text="Bob",
            start=10,
            end=13,
            score=0.98,
        ),
    ]


def test_redactor_load_does_not_duplicate_transformers_offline_kwargs(monkeypatch):
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
    }
    assert "tokenizer_kwargs" not in captured


def test_redactor_uses_injected_classifier():
    def fake_classifier(text, aggregation_strategy):
        assert text == "Email alice@example.com"
        assert aggregation_strategy == "none"
        return [
            {
                "entity": "S-private_email",
                "word": "Ġalice@example.com",
                "start": 5,
                "end": 23,
                "score": 0.99,
            }
        ]

    redactor = PrivacyFilterRedactor(classifier=fake_classifier)

    result = redactor.redact("Email alice@example.com")

    assert result.redacted_text == "Email [PRIVATE_EMAIL]"
    assert result.spans == [
        DetectedSpan(
            category="private_email",
            text="alice@example.com",
            start=6,
            end=23,
            score=0.99,
        )
    ]
