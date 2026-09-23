"""Versioned, non-destructive source-to-output video timing graph.

The graph is a timing contract, not a promise that every legacy effect can be
reconstructed. Unsupported operations may retain a rendered anchor explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


EDIT_GRAPH_SCHEMA_VERSION = 1
MAX_SPANS = 10_000
MAX_SOURCES = 1_000


class EditGraphError(ValueError):
    pass


def rational(value: object) -> Fraction:
    try:
        if isinstance(value, Fraction):
            result = value
        elif isinstance(value, float):
            result = Fraction(str(value))
        else:
            result = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise EditGraphError(f"Invalid rational time: {value!r}") from exc
    if result.denominator > 10**9 or abs(result.numerator) > 10**15:
        raise EditGraphError("Rational time exceeds supported precision or range.")
    return result


def rational_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class SourceNode:
    source_id: str
    media_path: str
    duration: Fraction


@dataclass(frozen=True)
class ClipSpan:
    source_id: str
    source_start: Fraction
    source_end: Fraction
    output_start: Fraction
    output_end: Fraction

    @property
    def duration(self) -> Fraction:
        return self.output_end - self.output_start

    def source_at(self, output_time: Fraction) -> Fraction:
        return self.source_start + (
            (output_time - self.output_start)
            * (self.source_end - self.source_start)
            / self.duration
        )


@dataclass(frozen=True)
class EditGraph:
    sources: dict[str, SourceNode]
    spans: tuple[ClipSpan, ...]
    fps: Fraction
    provenance: str = "source"
    schema_version: int = EDIT_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EDIT_GRAPH_SCHEMA_VERSION:
            raise EditGraphError("Unsupported edit graph schema version.")
        if self.provenance not in {"source", "rendered_anchor"}:
            raise EditGraphError("Invalid edit graph provenance.")
        if not 0 < self.fps <= 240:
            raise EditGraphError("Edit graph frame rate must be between 0 and 240 fps.")
        if not self.sources or len(self.sources) > MAX_SOURCES:
            raise EditGraphError("Edit graph has an invalid source count.")
        if not self.spans or len(self.spans) > MAX_SPANS:
            raise EditGraphError("Edit graph has an invalid span count.")
        for source_id, source in self.sources.items():
            if source_id != source.source_id or not source_id or not source.media_path:
                raise EditGraphError("Edit graph source identity is invalid.")
            if not Path(source.media_path).is_absolute() or source.duration <= 0:
                raise EditGraphError("Edit graph source path or duration is invalid.")
        cursor = Fraction(0)
        for span in self.spans:
            source = self.sources.get(span.source_id)
            if source is None:
                raise EditGraphError(f"Unknown edit graph source: {span.source_id}")
            if span.output_start != cursor or span.output_end <= span.output_start:
                raise EditGraphError("Edit graph spans must be positive and contiguous.")
            if not 0 <= span.source_start < span.source_end <= source.duration:
                raise EditGraphError("Edit graph source range is outside its media.")
            cursor = span.output_end

    @property
    def duration(self) -> Fraction:
        return self.spans[-1].output_end

    @classmethod
    def from_source(
        cls,
        media_path: str | Path,
        *,
        duration: object,
        fps: object,
        source_id: str = "src_0",
        provenance: str = "source",
    ) -> "EditGraph":
        source_duration = rational(duration)
        frame_rate = rational(fps).limit_denominator(100_000)
        source_path = str(Path(media_path).expanduser().resolve(strict=False))
        return cls(
            sources={source_id: SourceNode(source_id, source_path, source_duration)},
            spans=(ClipSpan(source_id, Fraction(0), source_duration, Fraction(0), source_duration),),
            fps=frame_rate,
            provenance=provenance,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EditGraph":
        if not isinstance(raw, Mapping) or raw.get("schema_version") != EDIT_GRAPH_SCHEMA_VERSION:
            raise EditGraphError("Unsupported edit graph schema version.")
        raw_sources = raw.get("sources")
        raw_spans = raw.get("spans")
        if not isinstance(raw_sources, list) or not isinstance(raw_spans, list):
            raise EditGraphError("Edit graph sources and spans must be lists.")
        if len(raw_sources) > MAX_SOURCES or len(raw_spans) > MAX_SPANS:
            raise EditGraphError("Edit graph exceeds size limits.")
        sources: dict[str, SourceNode] = {}
        for item in raw_sources:
            if not isinstance(item, Mapping):
                raise EditGraphError("Invalid edit graph source.")
            source_id = str(item.get("source_id") or "")
            if source_id in sources:
                raise EditGraphError("Duplicate edit graph source id.")
            sources[source_id] = SourceNode(
                source_id=source_id,
                media_path=str(item.get("media_path") or ""),
                duration=rational(item.get("duration")),
            )
        spans: list[ClipSpan] = []
        for item in raw_spans:
            if not isinstance(item, Mapping):
                raise EditGraphError("Invalid edit graph span.")
            spans.append(
                ClipSpan(
                    source_id=str(item.get("source_id") or ""),
                    source_start=rational(item.get("source_start")),
                    source_end=rational(item.get("source_end")),
                    output_start=rational(item.get("output_start")),
                    output_end=rational(item.get("output_end")),
                )
            )
        graph = cls(
            sources=sources,
            spans=tuple(spans),
            fps=rational(raw.get("fps")),
            provenance=str(raw.get("provenance") or "source"),
        )
        if rational(raw.get("duration")) != graph.duration:
            raise EditGraphError("Edit graph duration does not match its spans.")
        return graph

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "fps": rational_text(self.fps),
            "duration": rational_text(self.duration),
            "sources": [
                {
                    "source_id": source.source_id,
                    "media_path": source.media_path,
                    "duration": rational_text(source.duration),
                }
                for source in self.sources.values()
            ],
            "spans": [
                {
                    "source_id": span.source_id,
                    "source_start": rational_text(span.source_start),
                    "source_end": rational_text(span.source_end),
                    "output_start": rational_text(span.output_start),
                    "output_end": rational_text(span.output_end),
                }
                for span in self.spans
            ],
        }

    def map_output_time(self, time_value: object) -> tuple[str, Fraction]:
        output_time = rational(time_value)
        if not 0 <= output_time <= self.duration:
            raise EditGraphError("Output time is outside the edit graph.")
        for span in self.spans:
            if span.output_start <= output_time < span.output_end:
                return span.source_id, span.source_at(output_time)
        last = self.spans[-1]
        return last.source_id, last.source_end

    def trim(self, start: object, end: object | None = None) -> "EditGraph":
        start_time = rational(start)
        end_time = self.duration if end is None else rational(end)
        if not 0 <= start_time < end_time <= self.duration:
            raise EditGraphError("Trim range must be inside the current output duration.")
        return self._keep_ranges(((start_time, end_time),))

    def cut(self, start: object, end: object) -> "EditGraph":
        start_time = rational(start)
        end_time = rational(end)
        if not 0 <= start_time < end_time <= self.duration:
            raise EditGraphError("Cut range must be inside the current output duration.")
        ranges = tuple(
            (a, b)
            for a, b in ((Fraction(0), start_time), (end_time, self.duration))
            if b > a
        )
        if not ranges:
            raise EditGraphError("A cut cannot remove the entire graph.")
        return self._keep_ranges(ranges)

    def _keep_ranges(self, ranges: tuple[tuple[Fraction, Fraction], ...]) -> "EditGraph":
        output_cursor = Fraction(0)
        selected: list[ClipSpan] = []
        for range_start, range_end in ranges:
            for span in self.spans:
                start = max(range_start, span.output_start)
                end = min(range_end, span.output_end)
                if end <= start:
                    continue
                source_start = span.source_at(start)
                source_end = span.source_at(end)
                selected.append(
                    ClipSpan(
                        source_id=span.source_id,
                        source_start=source_start,
                        source_end=source_end,
                        output_start=output_cursor,
                        output_end=output_cursor + end - start,
                    )
                )
                output_cursor += end - start
        return EditGraph(
            sources=self.sources,
            spans=tuple(selected),
            fps=self.fps,
            provenance=self.provenance,
        )
