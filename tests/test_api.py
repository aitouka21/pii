from fastapi.testclient import TestClient

from pii_service import api as api_module
from pii_service.api import app, create_app, get_redactor
from pii_service.redactor import DetectedSpan, PrivacyFilterRedactor, RedactionResult


class FakeRedactor:
    def redact(self, text: str) -> RedactionResult:
        assert text == "Email alice@example.com"
        return RedactionResult(
            redacted_text="Email [PRIVATE_EMAIL]",
            spans=[
                DetectedSpan(
                    category="private_email",
                    text="alice@example.com",
                    start=6,
                    end=23,
                    score=0.99,
                )
            ],
        )


class PassthroughRedactor:
    def redact(self, text: str) -> RedactionResult:
        return RedactionResult(redacted_text=text, spans=[])


class ExplodingRedactor:
    def redact(self, text: str) -> RedactionResult:
        raise ValueError("secret internal failure")


class RuntimeErrorRedactor:
    def redact(self, text: str) -> RedactionResult:
        raise RuntimeError("Classifier not loaded")


class WhitespacePreservingRedactor:
    def redact(self, text: str) -> RedactionResult:
        assert text == "  Email alice@example.com  "
        return RedactionResult(redacted_text=text, spans=[])


def test_create_app_can_skip_startup_model_load(monkeypatch):
    create_app = getattr(api_module, "create_app", None)
    assert create_app is not None

    def fail_load(self):
        raise AssertionError("load should not be called")

    monkeypatch.setattr(PrivacyFilterRedactor, "load", fail_load)

    with TestClient(create_app(load_model_on_startup=False)) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_returns_ok():
    client = TestClient(create_app(load_model_on_startup=False))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_redact_returns_redacted_text_and_spans():
    app.dependency_overrides[get_redactor] = lambda: FakeRedactor()
    client = TestClient(app)

    try:
        response = client.post("/redact", json={"text": "Email alice@example.com"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "redacted_text": "Email [PRIVATE_EMAIL]",
        "spans": [
            {
                "category": "private_email",
                "text": "alice@example.com",
                "start": 6,
                "end": 23,
                "score": 0.99,
            }
        ],
    }


def test_redact_returns_normalized_span_shape():
    class NormalizedExampleRedactor:
        def redact(self, text: str) -> RedactionResult:
            assert text == "My name is Alice Smith and my email is alice@example.com"
            return RedactionResult(
                redacted_text=(
                    "My name is [PRIVATE_PERSON] and my email is [PRIVATE_EMAIL]"
                ),
                spans=[
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
                ],
            )

    app.dependency_overrides[get_redactor] = lambda: NormalizedExampleRedactor()
    client = TestClient(app)

    try:
        response = client.post(
            "/redact",
            json={"text": "My name is Alice Smith and my email is alice@example.com"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "redacted_text": "My name is [PRIVATE_PERSON] and my email is [PRIVATE_EMAIL]",
        "spans": [
            {
                "category": "private_person",
                "text": "Alice Smith",
                "start": 11,
                "end": 22,
                "score": 0.9999980926513672,
            },
            {
                "category": "private_email",
                "text": "alice@example.com",
                "start": 39,
                "end": 56,
                "score": 0.9999961853027344,
            },
        ],
    }


def test_redact_preserves_non_blank_leading_and_trailing_whitespace():
    app.dependency_overrides[get_redactor] = lambda: WhitespacePreservingRedactor()
    client = TestClient(app)

    try:
        response = client.post("/redact", json={"text": "  Email alice@example.com  "})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "redacted_text": "  Email alice@example.com  ",
        "spans": [],
    }


def test_redact_rejects_empty_text():
    client = TestClient(app)

    response = client.post("/redact", json={"text": ""})

    assert response.status_code == 422


def test_redact_rejects_whitespace_only_text():
    app.dependency_overrides[get_redactor] = lambda: PassthroughRedactor()
    client = TestClient(app)

    try:
        response = client.post("/redact", json={"text": "   \n\t"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_redact_unexpected_exception_returns_generic_500_detail():
    app.dependency_overrides[get_redactor] = lambda: ExplodingRedactor()
    client = TestClient(app)

    try:
        response = client.post("/redact", json={"text": "Email alice@example.com"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "Redaction failed"}


def test_redact_runtime_error_returns_503_with_message():
    app.dependency_overrides[get_redactor] = lambda: RuntimeErrorRedactor()
    client = TestClient(app)

    try:
        response = client.post("/redact", json={"text": "Email alice@example.com"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Classifier not loaded"}
