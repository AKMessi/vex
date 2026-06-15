from __future__ import annotations

import re
from collections import Counter
from typing import Any


CONTEXT_DEPENDENT_OPENERS = {
    "also",
    "and",
    "because",
    "but",
    "he",
    "it",
    "she",
    "so",
    "that",
    "then",
    "they",
    "those",
    "which",
}
TRAILING_DEPENDENCIES = {
    "a",
    "an",
    "and",
    "because",
    "but",
    "for",
    "of",
    "or",
    "so",
    "the",
    "to",
    "with",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "which",
    "with",
    "you",
    "your",
}
SOURCE_ROLES = {
    "hook",
    "context",
    "setup",
    "tension",
    "proof",
    "payoff",
    "quote",
    "support",
    "button",
    "primary",
    "part",
}


def build_semantic_units(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]] | None = None,
    *,
    max_unit_duration_sec: float = 18.0,
    max_merge_gap_sec: float = 1.25,
) -> list[dict[str, Any]]:
    """Build stable editorial units without imposing arbitrary word/time cuts."""
    normalized = _normalize_segments(segments)
    units: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(normalized):
        if pending:
            gap = max(0.0, float(segment["start"]) - float(pending[-1]["end"]))
            pending_duration = float(pending[-1]["end"]) - float(pending[0]["start"])
            if gap > max_merge_gap_sec or pending_duration >= max_unit_duration_sec:
                units.append(
                    _unit_from_segments(
                        pending,
                        words,
                        complete_end=_lexically_closed(
                            _clean_text(" ".join(str(item["text"]) for item in pending))
                        ),
                    )
                )
                pending = []
        pending.append(segment)
        text = _clean_text(" ".join(str(item["text"]) for item in pending))
        duration = float(pending[-1]["end"]) - float(pending[0]["start"])
        next_gap = (
            max(
                0.0,
                float(normalized[segment_index + 1]["start"]) - float(segment["end"]),
            )
            if segment_index + 1 < len(normalized)
            else 0.0
        )
        terminal_boundary = _has_terminal_end(text)
        pause_boundary = next_gap >= 0.45 and _lexically_closed(text)
        duration_boundary = duration >= max_unit_duration_sec
        if terminal_boundary or pause_boundary or duration_boundary:
            units.append(
                _unit_from_segments(
                    pending,
                    words,
                    complete_end=terminal_boundary or pause_boundary,
                )
            )
            pending = []
    if pending:
        pending_text = _clean_text(" ".join(str(item["text"]) for item in pending))
        units.append(
            _unit_from_segments(
                pending,
                words,
                complete_end=_lexically_closed(pending_text),
            )
        )

    for index, unit in enumerate(units, start=1):
        unit["unit_id"] = f"unit_{index:04d}"
        unit["index"] = index
    return units


def build_story_chapters(
    units: list[dict[str, Any]],
    *,
    max_duration_sec: float = 150.0,
    max_units: int = 32,
    overlap_units: int = 2,
) -> list[dict[str, Any]]:
    if not units:
        return []
    chapters: list[dict[str, Any]] = []
    start_index = 0
    while start_index < len(units):
        end_index = start_index
        chapter_start = float(units[start_index]["start"])
        while end_index < len(units):
            duration = float(units[end_index]["end"]) - chapter_start
            unit_count = end_index - start_index + 1
            if end_index > start_index and (duration > max_duration_sec or unit_count > max_units):
                break
            end_index += 1
        selected = units[start_index:end_index]
        chapter_index = len(chapters) + 1
        chapters.append(
            {
                "chapter_id": f"chapter_{chapter_index:03d}",
                "index": chapter_index,
                "start": round(float(selected[0]["start"]), 3),
                "end": round(float(selected[-1]["end"]), 3),
                "unit_ids": [str(unit["unit_id"]) for unit in selected],
                "units": selected,
                "keywords": _keywords(" ".join(str(unit["text"]) for unit in selected), 12),
            }
        )
        if end_index >= len(units):
            break
        start_index = max(start_index + 1, end_index - max(0, overlap_units))
    return chapters


def format_units_for_planner(units: list[dict[str, Any]]) -> str:
    return "\n".join(
        (
            f"{unit['unit_id']} | {float(unit['start']):.2f}-{float(unit['end']):.2f} | "
            f"start={'clean' if unit.get('complete_start') else 'dependent'} | "
            f"end={'clean' if unit.get('complete_end') else 'fragment'} | {unit['text']}"
        )
        for unit in units
    )


def compile_story_proposal(
    proposal: dict[str, Any],
    units: list[dict[str, Any]],
    *,
    candidate_id: str,
    min_duration_sec: float,
    max_duration_sec: float,
    max_source_ranges: int = 3,
) -> dict[str, Any]:
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    unit_position = {str(unit["unit_id"]): index for index, unit in enumerate(units)}
    raw_ranges = proposal.get("source_ranges")
    compiled_groups: list[dict[str, Any]] = []
    errors: list[str] = []

    if isinstance(raw_ranges, list) and raw_ranges:
        for raw_range in raw_ranges:
            if not isinstance(raw_range, dict):
                continue
            unit_ids = _string_list(raw_range.get("unit_ids"))
            if not unit_ids:
                continue
            compiled_groups.append(
                {
                    "unit_ids": unit_ids,
                    "role": _source_role(raw_range.get("role")),
                    "reason": _truncate(str(raw_range.get("reason") or ""), 180),
                }
            )
    else:
        unit_ids = _string_list(proposal.get("unit_ids"))
        unknown_ids = [unit_id for unit_id in unit_ids if unit_id not in unit_by_id]
        if unknown_ids:
            errors.append(f"unknown semantic units: {', '.join(unknown_ids[:4])}")
        contiguous_groups = _contiguous_groups(unit_ids, unit_position)
        if len(contiguous_groups) > 1:
            errors.append(
                "discontiguous unit_ids require explicit source_ranges with story roles"
            )
        compiled_groups = [
            {
                "unit_ids": group,
                "role": "primary" if len(contiguous_groups) == 1 else "part",
                "reason": "",
            }
            for group in contiguous_groups
        ]

    source_ranges: list[dict[str, Any]] = []
    selected_units: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_position = -1
    for group_index, group in enumerate(compiled_groups, start=1):
        unit_ids = list(group["unit_ids"])
        missing = [unit_id for unit_id in unit_ids if unit_id not in unit_by_id]
        if missing:
            errors.append(f"unknown semantic units: {', '.join(missing[:4])}")
            continue
        positions = [unit_position[unit_id] for unit_id in unit_ids]
        if positions != sorted(positions) or any(
            second != first + 1 for first, second in zip(positions, positions[1:])
        ):
            errors.append(f"source range {group_index} is not a contiguous transcript span")
            continue
        if positions[0] <= previous_position:
            errors.append("source ranges are not in chronological story order")
            continue
        previous_position = positions[-1]
        group_units = [unit_by_id[unit_id] for unit_id in unit_ids]
        for unit in group_units:
            unit_id = str(unit["unit_id"])
            if unit_id not in seen_ids:
                selected_units.append(unit)
                seen_ids.add(unit_id)
        source_ranges.append(
            {
                "index": group_index,
                "start": round(float(group_units[0]["start"]), 3),
                "end": round(float(group_units[-1]["end"]), 3),
                "duration": round(
                    float(group_units[-1]["end"]) - float(group_units[0]["start"]),
                    3,
                ),
                "role": str(group["role"]),
                "reason": str(group["reason"] or _role_reason(str(group["role"]))),
                "transition": "open" if group_index == 1 else "hard_cut",
                "unit_ids": unit_ids,
            }
        )

    if not selected_units:
        errors.append("story proposal contains no usable semantic units")
    if len(source_ranges) > max_source_ranges:
        errors.append(
            f"story proposal uses {len(source_ranges)} source ranges; max is {max_source_ranges}"
        )

    duration = round(
        sum(float(item["duration"]) for item in source_ranges),
        3,
    )
    if duration < min_duration_sec:
        errors.append(
            f"compiled story is {duration:.2f}s; minimum is {min_duration_sec:.2f}s"
        )
    if duration > max_duration_sec:
        errors.append(
            f"compiled story is {duration:.2f}s; maximum is {max_duration_sec:.2f}s"
        )

    transcript = _clean_text(" ".join(str(unit["text"]) for unit in selected_units))
    critic = evaluate_story_candidate(
        transcript,
        source_ranges,
        selected_units,
        planner_confidence=_score(proposal.get("confidence"), 50.0),
    )
    errors.extend(str(item) for item in critic["errors"])
    passed = not errors
    start = min((float(item["start"]) for item in source_ranges), default=0.0)
    end = max((float(item["end"]) for item in source_ranges), default=start)
    return {
        "passed": passed,
        "errors": _dedupe(errors),
        "candidate": {
            "candidate_id": candidate_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": duration,
            "composition_mode": "single_window" if len(source_ranges) <= 1 else "story_compilation",
            "source_ranges": source_ranges,
            "excerpt": transcript,
            "keywords": _string_list(proposal.get("keywords")) or _keywords(transcript, 8),
            "story_plan": {
                "version": "shorts-story-compiler-v1",
                "title": _truncate(str(proposal.get("title") or ""), 72),
                "hook": _truncate(str(proposal.get("hook") or ""), 120),
                "reason": _truncate(str(proposal.get("reason") or ""), 260),
                "unit_ids": [str(unit["unit_id"]) for unit in selected_units],
                "planner_confidence": _score(proposal.get("confidence"), 50.0),
                "critic": critic,
            },
        },
        "critic": critic,
    }


def evaluate_story_candidate(
    transcript: str,
    source_ranges: list[dict[str, Any]],
    units: list[dict[str, Any]],
    *,
    planner_confidence: float = 50.0,
) -> dict[str, Any]:
    tokens = _tokens(transcript)
    first_token = tokens[0] if tokens else ""
    last_token = tokens[-1] if tokens else ""
    errors: list[str] = []
    warnings: list[str] = []

    clean_start = bool(units and units[0].get("complete_start"))
    clean_end = bool(units and units[-1].get("complete_end"))
    if first_token in CONTEXT_DEPENDENT_OPENERS and not clean_start:
        errors.append("story opens with unresolved prior context")
    if last_token in TRAILING_DEPENDENCIES or not clean_end:
        errors.append("story ends on an incomplete thought")
    if len(source_ranges) > 1:
        for index in range(1, len(source_ranges)):
            previous_ids = _string_list(source_ranges[index - 1].get("unit_ids"))
            current_ids = _string_list(source_ranges[index].get("unit_ids"))
            previous = [unit for unit in units if str(unit.get("unit_id")) in previous_ids]
            current = [unit for unit in units if str(unit.get("unit_id")) in current_ids]
            if current and not bool(current[0].get("complete_start")):
                errors.append(f"stitch {index} enters a context-dependent utterance")
            if previous and current:
                overlap = _topic_overlap(
                    " ".join(str(unit["text"]) for unit in previous),
                    " ".join(str(unit["text"]) for unit in current),
                )
                if overlap < 0.04:
                    errors.append(f"stitch {index} lacks a shared causal topic")

    standalone = 88.0 if clean_start else 42.0
    closure = 88.0 if clean_end else 35.0
    continuity = max(1.0, 100.0 - max(len(source_ranges) - 1, 0) * 11.0 - len(errors) * 24.0)
    specificity = min(100.0, 35.0 + len(set(tokens)) * 0.65)
    score = max(
        1.0,
        min(
            100.0,
            standalone * 0.28
            + closure * 0.24
            + continuity * 0.24
            + specificity * 0.12
            + planner_confidence * 0.12,
        ),
    )
    if len(tokens) < 18:
        warnings.append("story has limited spoken context")
        score -= 8.0
    return {
        "passed": not errors and score >= 58.0,
        "score": round(max(1.0, score), 2),
        "standalone": round(standalone, 2),
        "closure": round(closure, 2),
        "continuity": round(continuity, 2),
        "specificity": round(specificity, 2),
        "errors": _dedupe(errors),
        "warnings": _dedupe(warnings),
        "version": "shorts-story-critic-v1",
    }


def _normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        try:
            start = max(0.0, float(segment.get("start", 0.0)))
            end = max(start, float(segment.get("end", start)))
        except (TypeError, ValueError):
            continue
        text = _clean_text(str(segment.get("text") or ""))
        if not text or end <= start:
            continue
        normalized.append(
            {
                "source_index": source_index,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )
    return sorted(normalized, key=lambda item: (float(item["start"]), float(item["end"])))


def _unit_from_segments(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]] | None,
    *,
    complete_end: bool,
) -> dict[str, Any]:
    text = _clean_text(" ".join(str(segment["text"]) for segment in segments))
    start = float(segments[0]["start"])
    end = float(segments[-1]["end"])
    word_bounds = _word_bounds(words or [], start, end)
    if word_bounds:
        start, end = word_bounds
    tokens = _tokens(text)
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "text": text,
        "source_segment_start": int(segments[0]["source_index"]),
        "source_segment_end": int(segments[-1]["source_index"]),
        "complete_start": bool(tokens and tokens[0] not in CONTEXT_DEPENDENT_OPENERS),
        "complete_end": complete_end,
        "keywords": _keywords(text, 8),
    }


def _word_bounds(
    words: list[dict[str, Any]],
    start: float,
    end: float,
) -> tuple[float, float] | None:
    inside: list[dict[str, Any]] = []
    for word in words:
        try:
            word_start = float(word.get("start"))
            word_end = float(word.get("end"))
        except (TypeError, ValueError):
            continue
        if word_end > start - 0.05 and word_start < end + 0.05:
            inside.append(word)
    if not inside:
        return None
    return max(0.0, float(inside[0]["start"])), max(
        float(inside[0]["start"]),
        float(inside[-1]["end"]),
    )


def _has_terminal_end(text: str) -> bool:
    tokens = _tokens(text)
    if not tokens:
        return False
    if tokens[-1] in TRAILING_DEPENDENCIES:
        return False
    return text.rstrip().endswith((".", "?", "!", ";", ":"))


def _lexically_closed(text: str) -> bool:
    tokens = _tokens(text)
    return bool(
        tokens
        and tokens[-1] not in TRAILING_DEPENDENCIES
        and not text.rstrip().endswith((",", "-", "—"))
    )


def _contiguous_groups(
    unit_ids: list[str],
    positions: dict[str, int],
) -> list[list[str]]:
    valid = sorted(
        {unit_id for unit_id in unit_ids if unit_id in positions},
        key=lambda unit_id: positions[unit_id],
    )
    groups: list[list[str]] = []
    for unit_id in valid:
        if not groups or positions[unit_id] != positions[groups[-1][-1]] + 1:
            groups.append([unit_id])
        else:
            groups[-1].append(unit_id)
    return groups


def _topic_overlap(first: str, second: str) -> float:
    first_tokens = {token for token in _tokens(first) if token not in STOPWORDS and len(token) >= 4}
    second_tokens = {token for token in _tokens(second) if token not in STOPWORDS and len(token) >= 4}
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / max(min(len(first_tokens), len(second_tokens)), 1)


def _keywords(text: str, limit: int) -> list[str]:
    counts = Counter(
        token
        for token in _tokens(text)
        if token not in STOPWORDS and len(token) >= 4
    )
    return [token for token, _count in counts.most_common(limit)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def _clean_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", collapsed)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _source_role(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "").lower()).strip("_")
    return normalized if normalized in SOURCE_ROLES else "part"


def _role_reason(role: str) -> str:
    return {
        "hook": "opens with the cold-viewer hook",
        "context": "provides only the context required for comprehension",
        "setup": "establishes the problem or claim",
        "tension": "introduces the contradiction or unresolved problem",
        "proof": "supports the claim with explanation or evidence",
        "payoff": "delivers the conclusion or practical takeaway",
        "button": "closes with a concise final line",
    }.get(role, "preserves a required story beat")


def _score(value: Any, default: float) -> float:
    try:
        return round(max(1.0, min(float(value), 100.0)), 2)
    except (TypeError, ValueError):
        return default


def _truncate(text: str, limit: int) -> str:
    clean = _clean_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result
