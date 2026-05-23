from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from pii_service.redactor import (
    DetectedSpan,
    PrivacyFilterRedactor,
    anonymize_text_with_presidio_results,
    presidio_analyze_results,
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class RedactRequest(TextRequest):
    pass


class AnalyzeRequest(TextRequest):
    language: str | None = None
    ad_hoc_recognizers: list[Any] | None = None
    entities: list[str] | None = None


class SpanResponse(BaseModel):
    category: str
    text: str
    start: int | None
    end: int | None
    score: float


class RedactResponse(BaseModel):
    redacted_text: str
    spans: list[SpanResponse]


class AnalyzeResultItem(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float
    recognition_metadata: dict[str, Any] | None = None


class AnonymizeRequest(TextRequest):
    analyzer_results: list[AnalyzeResultItem]


class AnonymizeItemResponse(BaseModel):
    start: int
    end: int
    entity_type: str
    text: str
    operator: str


class AnonymizeResponse(BaseModel):
    text: str
    items: list[AnonymizeItemResponse]


redactor = PrivacyFilterRedactor()


def get_redactor() -> PrivacyFilterRedactor:
    return redactor


def create_app(load_model_on_startup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if load_model_on_startup:
            redactor.load()
        yield

    app = FastAPI(title="PII Service POC", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/redact", response_model=RedactResponse)
    def redact(
        request: RedactRequest,
        service: Annotated[PrivacyFilterRedactor, Depends(get_redactor)],
    ) -> RedactResponse:
        try:
            result = service.redact(request.text)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Redaction failed") from exc

        return RedactResponse(
            redacted_text=result.redacted_text,
            spans=[_span_response(span) for span in result.spans],
        )

    @app.post("/analyze", response_model=list[AnalyzeResultItem])
    def analyze(
        request: AnalyzeRequest,
        service: Annotated[PrivacyFilterRedactor, Depends(get_redactor)],
    ) -> list[AnalyzeResultItem]:
        try:
            spans = service.detect(request.text)
            results = presidio_analyze_results(
                request.text,
                spans,
                entities=request.entities,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Analysis failed") from exc

        return [AnalyzeResultItem(**item) for item in results]

    @app.post("/anonymize", response_model=AnonymizeResponse)
    def anonymize(request: AnonymizeRequest) -> AnonymizeResponse:
        try:
            result = anonymize_text_with_presidio_results(
                request.text,
                [item.model_dump() for item in request.analyzer_results],
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Anonymization failed") from exc

        return AnonymizeResponse(
            text=result["text"],
            items=[AnonymizeItemResponse(**item) for item in result["items"]],
        )

    return app


app = create_app()


def _span_response(span: DetectedSpan) -> SpanResponse:
    return SpanResponse(
        category=span.category,
        text=span.text,
        start=span.start,
        end=span.end,
        score=span.score,
    )
