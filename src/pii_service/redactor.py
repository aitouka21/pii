from dataclasses import dataclass
from typing import Any, Protocol

from transformers import pipeline

from pii_service.config import Settings


class Classifier(Protocol):
    def __call__(
        self,
        text: str | list[str],
        *,
        aggregation_strategy: str,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]: ...


@dataclass(frozen=True)
class DetectedSpan:
    category: str
    text: str
    start: int | None
    end: int | None
    score: float


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    spans: list[DetectedSpan]


@dataclass(frozen=True)
class TextChunk:
    start_char: int
    end_char: int
    text: str


def marker_for_category(category: str) -> str:
    return f"[{category.upper()}]"


def _valid_spans(text: str, spans: list[DetectedSpan]) -> list[DetectedSpan]:
    text_length = len(text)
    return [
        span
        for span in spans
        if span.start is not None
        and span.end is not None
        and 0 <= span.start < span.end <= text_length
    ]


def plan_token_chunks(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be at least 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be less than max_tokens")

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = [
        (start, end)
        for start, end in encoded["offset_mapping"]
        if isinstance(start, int) and isinstance(end, int) and end > start
    ]
    if len(offsets) <= max_tokens:
        return [TextChunk(start_char=0, end_char=len(text), text=text)]

    chunks: list[TextChunk] = []
    step = max_tokens - overlap_tokens
    start_token = 0
    while start_token < len(offsets):
        end_token = min(start_token + max_tokens, len(offsets))
        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]
        chunks.append(
            TextChunk(
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
            )
        )
        if end_token >= len(offsets):
            break
        start_token += step

    return chunks


def rebase_span(span: DetectedSpan, offset: int) -> DetectedSpan:
    return DetectedSpan(
        category=span.category,
        text=span.text,
        start=span.start + offset if span.start is not None else None,
        end=span.end + offset if span.end is not None else None,
        score=span.score,
    )


def _spans_overlap(left: DetectedSpan, right: DetectedSpan) -> bool:
    if (
        left.start is None
        or left.end is None
        or right.start is None
        or right.end is None
    ):
        return False
    return left.start < right.end and left.end > right.start


def _select_non_overlapping_spans(
    text: str, spans: list[DetectedSpan]
) -> list[DetectedSpan]:
    selected: list[DetectedSpan] = []
    prioritized_spans = sorted(
        _valid_spans(text, spans),
        key=lambda span: (-(span.end - span.start), -span.score, span.start),
    )

    for span in prioritized_spans:
        if not any(_spans_overlap(span, selected_span) for selected_span in selected):
            selected.append(span)

    return selected


def _split_entity_label(label: str) -> tuple[str | None, str]:
    prefix, separator, category = label.partition("-")
    if separator and prefix in {"B", "I", "E", "S"} and category:
        return prefix, category
    return None, label


def _trim_span_to_non_whitespace(text: str, span: DetectedSpan) -> DetectedSpan | None:
    if span.start is None or span.end is None:
        return span

    start = span.start
    end = span.end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None

    return DetectedSpan(
        category=span.category,
        text=text[start:end],
        start=start,
        end=end,
        score=span.score,
    )


def decode_token_spans(text: str, spans: list[DetectedSpan]) -> list[DetectedSpan]:
    decoded: list[DetectedSpan] = []
    current_category: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    current_score = 0.0

    def flush_current() -> None:
        nonlocal current_category, current_start, current_end, current_score
        if current_category is None or current_start is None or current_end is None:
            current_category = None
            current_start = None
            current_end = None
            current_score = 0.0
            return

        trimmed = _trim_span_to_non_whitespace(
            text,
            DetectedSpan(
                category=current_category,
                text=text[current_start:current_end],
                start=current_start,
                end=current_end,
                score=current_score,
            ),
        )
        if trimmed is not None:
            decoded.append(trimmed)

        current_category = None
        current_start = None
        current_end = None
        current_score = 0.0

    for span in sorted(_valid_spans(text, spans), key=lambda item: item.start or 0):
        prefix, category = _split_entity_label(span.category)
        if span.start is None or span.end is None:
            continue

        if prefix in {None, "S"}:
            standalone = DetectedSpan(
                category=category,
                text=text[span.start : span.end],
                start=span.start,
                end=span.end,
                score=span.score,
            )
            if current_end is not None and span.start < current_end:
                trimmed = _trim_span_to_non_whitespace(text, standalone)
                if trimmed is not None:
                    decoded.append(trimmed)
                continue

            flush_current()
            trimmed = _trim_span_to_non_whitespace(text, standalone)
            if trimmed is not None:
                decoded.append(trimmed)
            continue

        if prefix == "B" or current_category != category:
            flush_current()
            current_category = category
            current_start = span.start
            current_end = span.end
            current_score = span.score
        elif prefix in {"I", "E"}:
            current_end = span.end
            current_score = max(current_score, span.score)

        if prefix == "E":
            flush_current()

    flush_current()
    selected = _select_non_overlapping_spans(text, decoded)
    return sorted(selected, key=lambda span: span.start or 0)


def redact_text_with_spans(text: str, spans: list[DetectedSpan]) -> str:
    redacted = text
    selected_spans = _select_non_overlapping_spans(text, spans)
    for span in sorted(selected_spans, key=lambda item: item.start or 0, reverse=True):
        start = span.start or 0
        end = span.end or 0
        redacted = redacted[:start] + marker_for_category(span.category) + redacted[end:]
    return redacted


class PrivacyFilterRedactor:
    def __init__(
        self, settings: Settings | None = None, classifier: Classifier | None = None
    ) -> None:
        self._settings = settings or Settings()
        self._classifier = classifier
        self._tokenizer: Any | None = getattr(classifier, "tokenizer", None)

    def load(self) -> None:
        if self._classifier is not None:
            return

        pipeline_kwargs: dict[str, Any] = {
            "task": "token-classification",
            "model": self._settings.model_id,
        }
        if self._settings.device is not None:
            pipeline_kwargs["device"] = self._settings.device
        model_kwargs: dict[str, Any] = {}
        if self._settings.model_cache_dir is not None:
            model_kwargs["cache_dir"] = self._settings.model_cache_dir
        if model_kwargs:
            pipeline_kwargs["model_kwargs"] = model_kwargs

        self._classifier = pipeline(**pipeline_kwargs)
        self._tokenizer = getattr(self._classifier, "tokenizer", None)

    def _classify(self, text: str) -> list[DetectedSpan]:
        if self._classifier is None:
            raise RuntimeError("Privacy filter classifier is not loaded")

        def classify_full_text() -> list[DetectedSpan]:
            raw_spans = self._classifier(text, aggregation_strategy="none")
            return [self._to_span(raw_span) for raw_span in raw_spans]

        if not self._settings.enable_chunking or self._tokenizer is None:
            return classify_full_text()

        try:
            chunks = plan_token_chunks(
                text,
                self._tokenizer,
                self._settings.chunk_max_tokens,
                self._settings.chunk_overlap_tokens,
            )
        except (KeyError, TypeError, ValueError, NotImplementedError):
            return classify_full_text()
        if len(chunks) == 1:
            return classify_full_text()

        raw_results = self._classifier(
            [chunk.text for chunk in chunks],
            aggregation_strategy="none",
            batch_size=self._settings.chunk_batch_size,
        )
        token_spans: list[DetectedSpan] = []
        for chunk, chunk_raw_spans in zip(chunks, raw_results, strict=True):
            token_spans.extend(
                rebase_span(self._to_span(raw_span), chunk.start_char)
                for raw_span in chunk_raw_spans
            )
        return token_spans

    def redact(self, text: str) -> RedactionResult:
        if self._classifier is None:
            self.load()

        token_spans = self._classify(text)
        spans = decode_token_spans(text, token_spans)
        return RedactionResult(
            redacted_text=redact_text_with_spans(text, spans),
            spans=spans,
        )

    @staticmethod
    def _to_span(raw_span: dict[str, Any]) -> DetectedSpan:
        category = str(raw_span.get("entity_group") or raw_span.get("entity") or "unknown")
        return DetectedSpan(
            category=category,
            text=str(raw_span.get("word") or ""),
            start=raw_span.get("start") if isinstance(raw_span.get("start"), int) else None,
            end=raw_span.get("end") if isinstance(raw_span.get("end"), int) else None,
            score=float(raw_span.get("score") or 0.0),
        )
