from pii_service.config import Settings
from pii_service.redactor import (
    DetectedSpan,
    PrivacyFilterRedactor,
    RedactionResult,
    TextChunk,
    anonymize_text_with_presidio_results,
    plan_token_chunks,
    presidio_analyze_results,
    presidio_entity_for_category,
    rebase_span,
    redact_text_with_spans,
)


class FakeTokenizer:
    def __call__(self, text, *, return_offsets_mapping, add_special_tokens):
        assert return_offsets_mapping is True
        assert add_special_tokens is False
        offsets = []
        index = 0
        for token in text.split(" "):
            start = index
            end = index + len(token)
            offsets.append((start, end))
            index = end + 1
        return {"offset_mapping": offsets}


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


def test_plan_token_chunks_returns_single_chunk_when_text_fits():
    chunks = plan_token_chunks("one two three", FakeTokenizer(), 10, 2)

    assert chunks == [TextChunk(start_char=0, end_char=13, text="one two three")]


def test_plan_token_chunks_uses_overlapping_token_windows():
    chunks = plan_token_chunks("one two three four five", FakeTokenizer(), 3, 1)

    assert chunks == [
        TextChunk(start_char=0, end_char=13, text="one two three"),
        TextChunk(start_char=8, end_char=23, text="three four five"),
    ]


def test_rebase_span_moves_chunk_offsets_to_original_text_offsets():
    span = DetectedSpan("S-private_email", "alice@example.com", 3, 20, 0.99)

    assert rebase_span(span, 50) == DetectedSpan(
        "S-private_email",
        "alice@example.com",
        53,
        70,
        0.99,
    )


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


def test_presidio_entity_for_category_maps_known_categories():
    assert presidio_entity_for_category("private_person") == "PERSON"
    assert presidio_entity_for_category("private_email") == "EMAIL_ADDRESS"


def test_presidio_entity_for_category_uses_uppercase_fallback():
    assert presidio_entity_for_category("private_account_number") == "ACCOUNT_NUMBER"
    assert presidio_entity_for_category("custom_secret") == "CUSTOM_SECRET"


def test_presidio_analyze_results_translates_and_filters_entities():
    text = "Alice emailed alice@example.com"
    spans = [
        DetectedSpan("private_person", "Alice", 0, 5, 0.99),
        DetectedSpan("private_email", "alice@example.com", 14, 31, 0.98),
    ]

    assert presidio_analyze_results(text, spans, entities=["EMAIL_ADDRESS"]) == [
        {
            "entity_type": "EMAIL_ADDRESS",
            "start": 14,
            "end": 31,
            "score": 0.98,
            "recognition_metadata": {"recognizer_name": "private_email"},
        }
    ]


def test_anonymize_text_with_presidio_results_returns_text_and_items():
    result = anonymize_text_with_presidio_results(
        "Alice emailed alice@example.com",
        [
            {"entity_type": "PERSON", "start": 0, "end": 5, "score": 0.99},
            {"entity_type": "EMAIL_ADDRESS", "start": 14, "end": 31, "score": 0.98},
        ],
    )

    assert result == {
        "text": "<PERSON> emailed <EMAIL_ADDRESS>",
        "items": [
            {
                "start": 0,
                "end": 8,
                "entity_type": "PERSON",
                "text": "<PERSON>",
                "operator": "replace",
            },
            {
                "start": 17,
                "end": 32,
                "entity_type": "EMAIL_ADDRESS",
                "text": "<EMAIL_ADDRESS>",
                "operator": "replace",
            },
        ],
    }


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


def test_redactor_uses_single_model_call_for_short_text():
    calls = []

    def fake_classifier(text, aggregation_strategy):
        calls.append(text)
        assert aggregation_strategy == "none"
        return []

    settings = Settings(chunk_max_tokens=256, chunk_overlap_tokens=32, chunk_batch_size=4)
    redactor = PrivacyFilterRedactor(settings=settings, classifier=fake_classifier)
    redactor._tokenizer = FakeTokenizer()

    result = redactor.redact("one two")

    assert calls == ["one two"]
    assert result == RedactionResult(redacted_text="one two", spans=[])


def test_redactor_bypasses_chunking_when_disabled():
    calls = []

    def fake_classifier(text, aggregation_strategy):
        calls.append(text)
        assert aggregation_strategy == "none"
        return []

    settings = Settings(
        enable_chunking=False,
        chunk_max_tokens=4,
        chunk_overlap_tokens=1,
        chunk_batch_size=4,
    )
    redactor = PrivacyFilterRedactor(settings=settings, classifier=fake_classifier)
    redactor._tokenizer = FakeTokenizer()

    result = redactor.redact("zero one two three four five")

    assert calls == ["zero one two three four five"]
    assert result == RedactionResult(
        redacted_text="zero one two three four five",
        spans=[],
    )


def test_redactor_falls_back_to_full_text_when_tokenizer_cannot_plan_chunks():
    class BrokenTokenizer:
        def __call__(self, text, *, return_offsets_mapping, add_special_tokens):
            raise NotImplementedError("offset mapping unsupported")

    calls = []

    def fake_classifier(text, aggregation_strategy):
        calls.append(text)
        assert aggregation_strategy == "none"
        return []

    settings = Settings(chunk_max_tokens=4, chunk_overlap_tokens=1, chunk_batch_size=4)
    redactor = PrivacyFilterRedactor(settings=settings, classifier=fake_classifier)
    redactor._tokenizer = BrokenTokenizer()

    result = redactor.redact("zero one two three four five")

    assert calls == ["zero one two three four five"]
    assert result == RedactionResult(
        redacted_text="zero one two three four five",
        spans=[],
    )


def test_redactor_falls_back_to_full_text_when_tokenizer_lacks_offsets():
    class MissingOffsetsTokenizer:
        def __call__(self, text, *, return_offsets_mapping, add_special_tokens):
            return {}

    calls = []

    def fake_classifier(text, aggregation_strategy):
        calls.append(text)
        assert aggregation_strategy == "none"
        return []

    settings = Settings(chunk_max_tokens=4, chunk_overlap_tokens=1, chunk_batch_size=4)
    redactor = PrivacyFilterRedactor(settings=settings, classifier=fake_classifier)
    redactor._tokenizer = MissingOffsetsTokenizer()

    result = redactor.redact("zero one two three four five")

    assert calls == ["zero one two three four five"]
    assert result == RedactionResult(
        redacted_text="zero one two three four five",
        spans=[],
    )


def test_redactor_uses_tokenizer_from_injected_classifier():
    calls = []

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def __call__(self, text, aggregation_strategy, batch_size=None):
            calls.append((text, batch_size))
            assert aggregation_strategy == "none"
            return [[] for _ in text]

    settings = Settings(chunk_max_tokens=4, chunk_overlap_tokens=1, chunk_batch_size=4)
    redactor = PrivacyFilterRedactor(settings=settings, classifier=FakePipeline())

    result = redactor.redact("zero one two three four five")

    assert calls == [
        (["zero one two three", "three four five"], 4),
    ]
    assert result == RedactionResult(
        redacted_text="zero one two three four five",
        spans=[],
    )


def test_redactor_batches_long_text_and_rebases_offsets():
    calls = []

    def fake_classifier(text, aggregation_strategy, batch_size=None):
        calls.append((text, batch_size))
        assert aggregation_strategy == "none"
        assert batch_size == 4
        results = []
        for chunk_text in text:
            if "alice@example.com" in chunk_text:
                start = chunk_text.index("alice@example.com")
                results.append(
                    [
                        {
                            "entity": "S-private_email",
                            "word": "alice@example.com",
                            "start": start,
                            "end": start + len("alice@example.com"),
                            "score": 0.99,
                        }
                    ]
                )
            else:
                results.append([])
        return results

    text = "zero one two alice@example.com three four five six"
    settings = Settings(chunk_max_tokens=4, chunk_overlap_tokens=1, chunk_batch_size=4)
    redactor = PrivacyFilterRedactor(settings=settings, classifier=fake_classifier)
    redactor._tokenizer = FakeTokenizer()

    result = redactor.redact(text)

    assert len(calls) == 1
    assert result.redacted_text == "zero one two [PRIVATE_EMAIL] three four five six"
    assert result.spans == [
        DetectedSpan("private_email", "alice@example.com", 13, 30, 0.99)
    ]


def test_redactor_deduplicates_entities_detected_in_overlapping_chunks():
    def fake_classifier(text, aggregation_strategy, batch_size=None):
        assert aggregation_strategy == "none"
        assert batch_size == 4
        results = []
        for chunk_text in text:
            if "alice@example.com" not in chunk_text:
                results.append([])
                continue
            start = chunk_text.index("alice@example.com")
            results.append(
                [
                    {
                        "entity": "S-private_email",
                        "word": "alice@example.com",
                        "start": start,
                        "end": start + len("alice@example.com"),
                        "score": 0.99,
                    }
                ]
            )
        return results

    text = "zero one two alice@example.com three four"
    settings = Settings(chunk_max_tokens=5, chunk_overlap_tokens=3, chunk_batch_size=4)
    redactor = PrivacyFilterRedactor(settings=settings, classifier=fake_classifier)
    redactor._tokenizer = FakeTokenizer()

    result = redactor.redact(text)

    assert result.redacted_text == "zero one two [PRIVATE_EMAIL] three four"
    assert result.spans == [
        DetectedSpan("private_email", "alice@example.com", 13, 30, 0.99)
    ]


def test_redactor_deduplicates_overlapping_multi_token_entities():
    def fake_classifier(text, aggregation_strategy, batch_size=None):
        assert aggregation_strategy == "none"
        assert batch_size == 4
        results = []
        for chunk_text in text:
            if "Alice Smith" not in chunk_text:
                results.append([])
                continue
            start = chunk_text.index("Alice")
            results.append(
                [
                    {
                        "entity": "B-private_person",
                        "word": "Alice",
                        "start": start,
                        "end": start + len("Alice"),
                        "score": 0.98,
                    },
                    {
                        "entity": "E-private_person",
                        "word": "Smith",
                        "start": start + len("Alice "),
                        "end": start + len("Alice Smith"),
                        "score": 0.99,
                    },
                ]
            )
        return results

    text = "zero one Alice Smith two three"
    settings = Settings(chunk_max_tokens=5, chunk_overlap_tokens=3, chunk_batch_size=4)
    redactor = PrivacyFilterRedactor(settings=settings, classifier=fake_classifier)
    redactor._tokenizer = FakeTokenizer()

    result = redactor.redact(text)

    assert result.redacted_text == "zero one [PRIVATE_PERSON] two three"
    assert result.spans == [
        DetectedSpan("private_person", "Alice Smith", 9, 20, 0.99)
    ]
