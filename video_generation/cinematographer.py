from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from video_generation.models import Beat, BeatGraph, ScriptPlan, VideoGenerationRequest
from vex_hyperframes import build_composition
from vex_hyperframes.compiler import compile_hyperframes_plan
from vex_hyperframes.qa import (
    analyze_hyperframes_quality,
    build_rendered_visual_fingerprint,
    extract_quality_frames,
)
from vex_hyperframes.variants import build_variants


CINEMATOGRAPHER_VERSION = "semantic-cinematographer-v1"


@dataclass
class CinematicBeatComposition:
    beat_id: str
    start: float
    duration: float
    compiler_passed: bool
    spec: dict[str, Any] = field(default_factory=dict)
    compiler_issues: list[str] = field(default_factory=list)
    template: str = ""
    scene_type: str = ""
    variant_id: str = ""
    variant_index: int = 0
    composition_id: str = ""
    composition_src: str = ""
    composition_html: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "start": self.start,
            "duration": self.duration,
            "compiler_passed": self.compiler_passed,
            "compiler_issues": list(self.compiler_issues),
            "template": self.template,
            "scene_type": self.scene_type,
            "variant_id": self.variant_id,
            "variant_index": self.variant_index,
            "composition_id": self.composition_id,
            "composition_src": self.composition_src,
            "spec": _spec_summary(self.spec),
            "metadata": _metadata_summary(self.metadata),
        }


@dataclass
class CinematicPlan:
    version: str
    beat_compositions: list[CinematicBeatComposition]
    warnings: list[str] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for item in self.beat_compositions if item.compiler_passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "accepted_count": self.accepted_count,
            "beat_count": len(self.beat_compositions),
            "warnings": list(self.warnings),
            "beat_compositions": [item.to_dict() for item in self.beat_compositions],
        }


def build_cinematic_plan(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
) -> CinematicPlan:
    compositions: list[CinematicBeatComposition] = []
    warnings: list[str] = []
    visual_world_history: list[dict[str, Any]] = []
    for beat in beat_graph.beats:
        compiled_item = _compile_beat(
            request=request,
            plan=plan,
            beat=beat,
            visual_world_history=visual_world_history,
        )
        compositions.append(compiled_item)
        fingerprint = (
            dict(compiled_item.metadata.get("visual_world_program") or {})
            .get("fingerprint")
        )
        if compiled_item.compiler_passed and isinstance(fingerprint, dict):
            visual_world_history.append(dict(fingerprint))
    if not any(item.compiler_passed for item in compositions):
        warnings.append("semantic_cinematographer_compiled_no_beats")
    return CinematicPlan(
        version=CINEMATOGRAPHER_VERSION,
        beat_compositions=compositions,
        warnings=warnings,
    )


def write_cinematic_compositions(
    cinematic_plan: CinematicPlan | None,
    *,
    compositions_dir: Path,
) -> dict[str, str]:
    if cinematic_plan is None:
        return {}
    compositions_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for item in cinematic_plan.beat_compositions:
        if not item.compiler_passed or not item.composition_html:
            continue
        filename = f"{_safe_id(item.beat_id)}_{_safe_id(item.composition_id)}.html"
        path = compositions_dir / filename
        path.write_text(_external_template_html(item), encoding="utf-8")
        item.composition_src = f"compositions/{filename}"
        mapping[item.beat_id] = item.composition_src
        (compositions_dir / f"{filename}.metadata.json").write_text(
            json.dumps(item.metadata, indent=2),
            encoding="utf-8",
        )
    return mapping


def inline_cinematic_composition(
    item: CinematicBeatComposition,
    *,
    start: float,
    duration: float,
    track_index: int,
) -> str:
    body_match = re.search(
        r"<body[^>]*>(?P<body>.*?)</body>",
        item.composition_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    head_styles = "\n".join(
        match.group(0)
        for match in re.finditer(
            r"<style\b[^>]*>.*?</style>",
            item.composition_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    body = body_match.group("body").strip() if body_match else item.composition_html
    scoped_id = f"{_safe_id(item.composition_id)}-{_safe_id(item.beat_id)}"
    content = f"{head_styles}\n{body}"
    content = _demote_internal_hyperframes_clips(
        content,
        composition_id=item.composition_id,
    )
    content = _strip_script_blocks(content)
    content = _scope_composition_document(
        content,
        composition_id=item.composition_id,
        scoped_id=scoped_id,
        start=start,
        duration=duration,
        track_index=track_index,
    )
    return content


def _demote_internal_hyperframes_clips(content: str, *, composition_id: str) -> str:
    """Keep only the inlined composition root as a parent HyperFrames clip."""

    def replace_tag(match: re.Match[str]) -> str:
        tag = match.group(0)
        if f'data-composition-id="{composition_id}"' in tag:
            return tag
        tag = re.sub(r'\sdata-start="[^"]*"', "", tag)
        tag = re.sub(r'\sdata-duration="[^"]*"', "", tag)
        tag = re.sub(r'\sdata-track-index="[^"]*"', "", tag)

        def replace_class(class_match: re.Match[str]) -> str:
            classes = [
                part
                for part in class_match.group(1).split()
                if part != "clip"
            ]
            if not classes:
                return ""
            return f'class="{" ".join(classes)}"'

        return re.sub(r'class="([^"]*)"', replace_class, tag, count=1)

    return re.sub(r"<[a-zA-Z][^>]*>", replace_tag, content)


def _strip_script_blocks(content: str) -> str:
    return re.sub(
        r"<script\b[^>]*>.*?</script>\s*",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _external_template_html(item: CinematicBeatComposition) -> str:
    body_match = re.search(
        r"<body[^>]*>(?P<body>.*?)</body>",
        item.composition_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    head_styles = "\n".join(
        match.group(0)
        for match in re.finditer(
            r"<style\b[^>]*>.*?</style>",
            item.composition_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    body = body_match.group("body").strip() if body_match else item.composition_html
    template_id = f"{_safe_id(item.composition_id)}-template"
    scoped_id = f"{_safe_id(item.composition_id)}-{_safe_id(item.beat_id)}-template-root"
    content = _scope_external_template_document(
        f"{head_styles}\n{body}",
        composition_id=item.composition_id,
        scoped_id=scoped_id,
    )
    content = _move_template_styles_inside_root(content, scoped_id=scoped_id)
    return (
        f'<template id="{template_id}">\n'
        f"{content}\n"
        "</template>\n"
    )


def _scope_external_template_document(
    content: str,
    *,
    composition_id: str,
    scoped_id: str,
) -> str:
    content = _scope_common_document_selectors(
        content,
        scoped_id=scoped_id,
    )

    def replace_root(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        attrs = re.sub(r'\bid="root"', f'id="{scoped_id}"', attrs, count=1)
        if f'id="{scoped_id}"' not in attrs:
            attrs += f' id="{scoped_id}"'
        attrs = re.sub(r'\sdata-start="[^"]*"', "", attrs)
        attrs = re.sub(r'\sdata-duration="[^"]*"', "", attrs)
        attrs = re.sub(r'\sdata-track-index="[^"]*"', "", attrs)
        return f"<div{attrs}>"

    return re.sub(
        r"<div(?P<attrs>[^>]*\bdata-composition-id=\""
        + re.escape(composition_id)
        + r"\"[^>]*)>",
        replace_root,
        content,
        count=1,
        flags=re.IGNORECASE,
    )


def _move_template_styles_inside_root(content: str, *, scoped_id: str) -> str:
    style_blocks = [
        match.group(0)
        for match in re.finditer(
            r"<style\b[^>]*>.*?</style>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    if not style_blocks:
        return content
    without_styles = re.sub(
        r"<style\b[^>]*>.*?</style>\s*",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    styles = "\n".join(style_blocks)
    return re.sub(
        r"(<div[^>]*\bid=\"" + re.escape(scoped_id) + r"\"[^>]*>)",
        r"\1" + "\n" + styles,
        without_styles,
        count=1,
        flags=re.IGNORECASE,
    )


def _scope_composition_document(
    content: str,
    *,
    composition_id: str,
    scoped_id: str,
    start: float,
    duration: float,
    track_index: int,
) -> str:
    content = _scope_common_document_selectors(
        content,
        scoped_id=scoped_id,
    )

    def replace_root(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        attrs = re.sub(r'\bid="root"', f'id="{scoped_id}"', attrs, count=1)
        attrs = re.sub(
            r'\bdata-composition-id="[^"]*"',
            f'data-cinematic-composition-id="{composition_id}"',
            attrs,
            count=1,
        )
        if f'id="{scoped_id}"' not in attrs:
            attrs += f' id="{scoped_id}"'
        attrs = re.sub(r'\bdata-start="[^"]*"', f'data-start="{start:.3f}"', attrs)
        attrs = re.sub(r'\bdata-duration="[^"]*"', f'data-duration="{duration:.3f}"', attrs)
        if "data-track-index=" in attrs:
            attrs = re.sub(r'\bdata-track-index="[^"]*"', f'data-track-index="{track_index}"', attrs)
        else:
            attrs += f' data-track-index="{track_index}"'
        if 'class="' in attrs:
            attrs = re.sub(
                r'class="([^"]*)"',
                lambda class_match: f'class="clip beat-composition inline-composition {class_match.group(1)}"',
                attrs,
                count=1,
            )
        else:
            attrs += ' class="clip beat-composition inline-composition"'
        if 'style="' in attrs:
            attrs = re.sub(
                r'style="([^"]*)"',
                lambda style_match: (
                    'style="position:absolute;inset:0;width:100%;height:100%;'
                    f'overflow:hidden;{style_match.group(1)}"'
                ),
                attrs,
                count=1,
            )
        else:
            attrs += ' style="position:absolute;inset:0;width:100%;height:100%;overflow:hidden;"'
        return f"<div{attrs}>"

    scoped = re.sub(
        r"<div(?P<attrs>[^>]*\bdata-composition-id=\""
        + re.escape(composition_id)
        + r"\"[^>]*)>",
        replace_root,
        content,
        count=1,
        flags=re.IGNORECASE,
    )
    return _append_inline_safe_area_css(scoped, scoped_id=scoped_id)


def _append_inline_safe_area_css(content: str, *, scoped_id: str) -> str:
    selector = f"#{scoped_id}.inline-composition"
    return (
        content
        + "\n<style>\n"
        + f"{selector} .visual-world-canvas {{ inset: 5.5% !important; }}\n"
        + f"{selector} .vw-data-sculpture .vw-data-title {{ left: 7% !important; top: 6.5% !important; max-width: 54% !important; }}\n"
        + f"{selector} .vw-data-sculpture .vw-data-title b {{ font-size: clamp(34px, 4.4vw, 78px) !important; line-height: .96 !important; }}\n"
        + f"{selector} .vw-data-sculpture .vw-masses {{ inset: 32% 9% 11% 9% !important; transform: scale(.86); transform-origin: center; }}\n"
        + f"{selector} .vw-data-sculpture .vw-mass strong {{ font-size: clamp(16px, 1.72vw, 28px) !important; }}\n"
        + f"{selector} .vw-collage .vw-collage-masthead {{ left: 5.8% !important; top: 6.2% !important; max-width: 62% !important; }}\n"
        + f"{selector} .vw-collage .vw-collage-masthead b {{ font-size: clamp(34px, 4.6vw, 76px) !important; line-height: .98 !important; }}\n"
        + f"{selector} .vw-collage .vw-collage-pieces {{ inset: 27% 7% 8% 7% !important; transform: scale(.91); transform-origin: center; }}\n"
        + f"{selector} .vw-collage .vw-collage-piece strong {{ font-size: clamp(19px, 2.35vw, 38px) !important; line-height: .98 !important; }}\n"
        + "</style>\n"
    )


def _scope_common_document_selectors(content: str, *, scoped_id: str) -> str:
    scoped_selector = f"#{scoped_id}"
    content = re.sub(
        r":root\s*\{",
        f"{scoped_selector} {{",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"html\s*,\s*body\s*\{",
        f"{scoped_selector} {{",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"#root\b", scoped_selector, content)
    content = content.replace(
        'document.getElementById("root")',
        f'document.getElementById("{scoped_id}")',
    )
    replacements = {
        'document.querySelectorAll("[data-anim]")': 'root.querySelectorAll("[data-anim]")',
        'document.querySelectorAll("[data-bar]")': 'root.querySelectorAll("[data-bar]")',
        'document.querySelectorAll("[data-line]")': 'root.querySelectorAll("[data-line]")',
        'document.querySelector("[data-route]")': 'root.querySelector("[data-route]")',
        'document.querySelector("[data-route-dot]")': 'root.querySelector("[data-route-dot]")',
        'document.querySelector(".semantic-stage")': 'root.querySelector(".semantic-stage")',
        'root.querySelector(".semantic-stage") !== null': (
            'root.querySelector(".semantic-stage, .visual-world-stage, .visual-world-canvas") !== null'
        ),
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    return content


def evaluate_rendered_cinematography(
    *,
    output_path: Path,
    project_dir: Path,
    request: VideoGenerationRequest,
    beat_graph: BeatGraph,
    root_html: str,
    cinematic_plan: CinematicPlan | None,
    output_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not output_path.is_file() or output_metadata is None:
        return None
    duration = float(output_metadata.get("duration_sec") or beat_graph.duration_sec or 0.0)
    capture_plan = _final_capture_plan(beat_graph)
    frame_paths = extract_quality_frames(
        output_path,
        project_dir / "snapshots" / "final_visual_qa",
        duration_sec=duration,
        frame_count=max(4, min(8, len(capture_plan))),
        capture_plan=capture_plan,
    )
    first_metadata = _first_comp_metadata(cinematic_plan)
    report = analyze_hyperframes_quality(
        video_path=output_path,
        html=root_html,
        frame_paths=frame_paths,
        theme=dict((first_metadata.get("art_direction") or {}).get("theme") or {}),
        design_ir=dict(first_metadata.get("design_ir") or {}),
        min_score=float(config.HYPERFRAMES_MIN_QUALITY_SCORE),
    )
    fingerprint = build_rendered_visual_fingerprint(
        frame_paths,
        visual_world_program=dict(first_metadata.get("visual_world_program") or {}),
    )
    accepted = cinematic_plan.accepted_count if cinematic_plan else 0
    beat_count = len(beat_graph.beats)
    issues, warnings = _split_generated_video_visual_issues(
        report.issues,
        report_score=report.score,
    )
    if beat_count and accepted / beat_count < 0.67:
        issues.append("semantic_cinematographer_coverage_below_67_percent")
    if not frame_paths:
        issues.append("final_visual_qa_captured_no_frames")
    if accepted and _repeats_medium_too_often(cinematic_plan):
        warnings.append("semantic_cinematographer_repeated_medium_family")
    report_passed = report.passed or (
        report.score >= float(config.HYPERFRAMES_MIN_QUALITY_SCORE)
        and not issues
    )
    passed = report_passed and not issues
    return {
        "version": "generated-video-visual-qa-v1",
        "passed": passed,
        "score": report.score,
        "issues": _unique(issues),
        "warnings": _unique(warnings),
        "frame_paths": [str(path) for path in frame_paths],
        "fingerprint": fingerprint,
        "cinematographer": {
            "version": cinematic_plan.version if cinematic_plan else "",
            "accepted_count": accepted,
            "beat_count": beat_count,
            "semantic_coverage": round(accepted / max(beat_count, 1), 4),
            "medium_families": _medium_families(cinematic_plan),
        },
        "quality_report": report.to_dict(),
    }


def _split_generated_video_visual_issues(
    report_issues: list[str],
    *,
    report_score: float,
) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    score_allows_softening = report_score >= float(config.HYPERFRAMES_MIN_QUALITY_SCORE)
    for issue in report_issues:
        normalized = issue.lower()
        if score_allows_softening and (
            "too much visible copy" in normalized
            or "too close to frame edges" in normalized
        ):
            soft.append(issue)
        else:
            hard.append(issue)
    return hard, soft


def _compile_beat(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat: Beat,
    visual_world_history: list[dict[str, Any]],
) -> CinematicBeatComposition:
    errors: list[str] = []
    for candidate in _candidate_specs(
        request=request,
        plan=plan,
        beat=beat,
        visual_world_history=visual_world_history,
    ):
        try:
            compiled = compile_hyperframes_plan(candidate)
        except Exception as exc:  # noqa: BLE001
            errors.append(_short_error(exc))
            continue
        if not compiled.passed:
            errors.extend(compiled.issues)
            continue
        variants = build_variants(compiled.renderer_spec, default_count=4)
        if not variants:
            errors.append("compiler_returned_no_visual_variants")
            continue
        variant = variants[(beat.index - 1) % len(variants)]
        try:
            composition = build_composition(
                variant.spec,
                width=request.width,
                height=request.height,
                fps=request.fps,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(_short_error(exc))
            continue
        return CinematicBeatComposition(
            beat_id=beat.beat_id,
            start=beat.start,
            duration=beat.duration,
            compiler_passed=True,
            spec=variant.spec,
            compiler_issues=[],
            template=str(variant.spec.get("template") or ""),
            scene_type=str((variant.spec.get("visual_explanation_ir") or {}).get("scene_type") or ""),
            variant_id=variant.variant_id,
            variant_index=variant.variant_index,
            composition_id=composition.composition_id,
            composition_html=composition.html,
            metadata=composition.metadata,
        )
    return CinematicBeatComposition(
        beat_id=beat.beat_id,
        start=beat.start,
        duration=beat.duration,
        compiler_passed=False,
        compiler_issues=_unique(errors)[:10],
    )


def _candidate_specs(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat: Beat,
    visual_world_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    context = _context_text(plan=plan, beat=beat)
    source = f"{beat.narration} {context}"
    base = {
        "visual_id": f"generated_{beat.beat_id}",
        "sentence_text": beat.narration,
        "context_text": context,
        "duration": max(1.4, min(float(beat.duration or 0.0), 12.0)),
        "composition_mode": "replace",
        "importance": 0.92,
        "visual_type_hint": _visual_type_hint(request, source),
        "hyperframes_variant_count": 4,
        "visual_world_history": [dict(item) for item in visual_world_history[-3:]],
        "style_pack": request.style,
        "program_context": {
            "source": "generate_video",
            "title": plan.title,
            "prompt": request.prompt,
        },
    }
    candidates: list[dict[str, Any]] = []
    if _is_sparse_attention(source):
        candidates.extend(_sparse_attention_specs(base, source, beat.index))
    candidates.extend(
        [
            _matched_transform_spec(base, source),
            _guided_process_spec(base, source),
            _causal_spec(base, source),
            _quote_spec(base, source),
        ]
    )
    return [candidate for candidate in candidates if candidate]


def _sparse_attention_specs(
    base: dict[str, Any],
    source: str,
    index: int,
) -> list[dict[str, Any]]:
    context = (
        "Before sparse attention, the wall of token links makes every token look at everything. "
        "Then a sparse mask drops irrelevant links. After the mask, useful routes stay active "
        "and form a focused reasoning path. Compute follows signal instead of noise. "
        + source
    )
    specs = [
        {
            **base,
            "context_text": context,
            "semantic_frame": {
                "before_state": "wall of token links",
                "after_state": "focused reasoning path",
                "preserved_constraint": "useful routes stay active",
            },
            "required_labels": [
                "wall of token links",
                "sparse mask",
                "useful routes",
                "focused reasoning path",
            ],
        },
        {
            **base,
            "context_text": context,
            "semantic_frame": {
                "input": "wall of token links",
                "steps": [
                    "apply sparse mask",
                    "drop irrelevant links",
                    "keep useful routes",
                ],
                "result": "focused reasoning path",
            },
            "required_labels": [
                "wall of token links",
                "sparse mask",
                "irrelevant links",
                "focused reasoning path",
            ],
        },
        {
            **base,
            "context_text": (
                "Sparse attention causes irrelevant links to drop away because the sparse mask "
                "keeps useful routes active. The focused reasoning path is the result. "
                + source
            ),
            "semantic_frame": {
                "problem": "irrelevant links",
                "mechanism": "sparse mask",
                "intervention": "useful routes active",
                "result": "focused reasoning path",
            },
            "required_labels": [
                "irrelevant links",
                "sparse mask",
                "useful routes active",
                "focused reasoning path",
            ],
        },
    ]
    offset = (index - 1) % len(specs)
    return specs[offset:] + specs[:offset]


def _matched_transform_spec(base: dict[str, Any], source: str) -> dict[str, Any]:
    before, after = _before_after_labels(source)
    constraint = _best_label(source, fallback="useful signal")
    context = (
        f"Before the change, the visible state is {before}. "
        f"After the intervention, the resolved state is {after}. "
        f"The visual preserves {constraint} while it transforms from before to after. "
        + source
    )
    return {
        **base,
        "context_text": context,
        "semantic_frame": {
            "before_state": before,
            "after_state": after,
            "preserved_constraint": constraint,
        },
        "required_labels": [before, after, constraint],
    }


def _guided_process_spec(base: dict[str, Any], source: str) -> dict[str, Any]:
    labels = _process_labels(source)
    context = (
        f"The process starts with {labels[0]}, then {labels[1]}, then {labels[2]}, "
        f"and finally reaches {labels[3]}. "
        + source
    )
    return {
        **base,
        "context_text": context,
        "semantic_frame": {
            "input": labels[0],
            "steps": labels[1:3],
            "result": labels[3],
        },
        "required_labels": labels,
    }


def _causal_spec(base: dict[str, Any], source: str) -> dict[str, Any]:
    labels = _causal_labels(source)
    context = (
        f"{labels[0]} causes friction because {labels[1]} blocks the path. "
        f"The intervention is {labels[2]}, and therefore the result is {labels[3]}. "
        + source
    )
    return {
        **base,
        "context_text": context,
        "semantic_frame": {
            "problem": labels[0],
            "mechanism": labels[1],
            "intervention": labels[2],
            "result": labels[3],
        },
        "required_labels": labels,
    }


def _quote_spec(base: dict[str, Any], source: str) -> dict[str, Any]:
    quote = _short_sentence(source, max_words=9)
    context = f"The exact quote is {quote}. The decisive phrase is {quote}. " + source
    return {
        **base,
        "context_text": context,
        "semantic_frame": {
            "exact_quote": quote,
        },
        "required_labels": _quote_labels(quote),
    }


def _before_after_labels(source: str) -> tuple[str, str]:
    cleaned = _clean(source, limit=360)
    turn_match = re.search(
        r"\bturns?\s+(?P<before>.+?)\s+into\s+(?P<after>.+?)(?:\s+by\b|[,.;]|\s+using\b|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if turn_match:
        return (
            _label(turn_match.group("before"), fallback="messy starting state"),
            _label(turn_match.group("after"), fallback="clear resolved state"),
        )
    from_to = re.search(
        r"\bfrom\s+(?P<before>.+?)\s+to\s+(?P<after>.+?)(?:[,.;]|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if from_to:
        return (
            _label(from_to.group("before"), fallback="before state"),
            _label(from_to.group("after"), fallback="after state"),
        )
    keywords = _keyword_phrases(cleaned, limit=4)
    before = keywords[0] if keywords else "messy bottleneck"
    after = keywords[-1] if len(keywords) > 1 else "clearer path"
    if _too_generic(before):
        before = "messy bottleneck"
    if _too_generic(after) or _normalize(before) == _normalize(after):
        after = "clearer path"
    return before, after


def _process_labels(source: str) -> list[str]:
    keywords = _keyword_phrases(source, limit=6)
    defaults = [
        "visible input",
        "filter noise",
        "route useful signal",
        "resolved path",
    ]
    values = [item for item in keywords if not _too_generic(item)]
    labels = [*values, *defaults]
    return _unique(labels)[:4]


def _causal_labels(source: str) -> list[str]:
    keywords = _keyword_phrases(source, limit=6)
    defaults = [
        "messy bottleneck",
        "hidden friction",
        "useful intervention",
        "resolved outcome",
    ]
    values = [item for item in keywords if not _too_generic(item)]
    labels = [*values, *defaults]
    return _unique(labels)[:4]


def _quote_labels(quote: str) -> list[str]:
    words = [word for word in _words(quote) if len(word) >= 4]
    if len(words) >= 4:
        return [" ".join(words[:2]), " ".join(words[-2:])]
    return [quote]


def _context_text(*, plan: ScriptPlan, beat: Beat) -> str:
    return _clean(
        " ".join(
            item
            for item in [
                plan.prompt,
                plan.design_direction,
                plan.narration,
                beat.visual_metaphor,
            ]
            if item
        ),
        limit=1200,
    )


def _visual_type_hint(request: VideoGenerationRequest, source: str) -> str:
    text = f"{request.style} {source}".lower()
    if re.search(r"\b(?:ui|interface|screen|dashboard|app|editor)\b", text):
        return "product_ui"
    if re.search(r"\b(?:data|metric|token|attention|graph|network|route|system)\b", text):
        return "data_graphic"
    if re.search(r"\b(?:quote|line|phrase)\b", text):
        return "abstract_motion"
    return "process"


def _is_sparse_attention(source: str) -> bool:
    lower = source.lower()
    return "sparse attention" in lower or (
        "attention" in lower and "token" in lower and "mask" in lower
    )


def _keyword_phrases(source: str, *, limit: int) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9%+./ -]+", " ", source)
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9%+./-]+", cleaned)
        if len(token) >= 4 and token.lower() not in _STOPWORDS
    ]
    phrases: list[str] = []
    for index in range(0, len(tokens), 2):
        phrase = " ".join(tokens[index : index + 2])
        if phrase:
            phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return _unique([_label(item, fallback="") for item in phrases if item])[:limit]


def _best_label(source: str, *, fallback: str) -> str:
    for phrase in _keyword_phrases(source, limit=5):
        if not _too_generic(phrase):
            return phrase
    return fallback


def _short_sentence(source: str, *, max_words: int) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", _clean(source, limit=240))[0]
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words])
    return sentence.strip(" ,.;:") or "The useful pattern becomes visible"


def _label(value: Any, *, fallback: str) -> str:
    cleaned = _clean(value, limit=54).strip(" ,.;:-")
    words = cleaned.split()
    if len(words) > 5:
        cleaned = " ".join(words[:5])
    if not cleaned:
        return fallback
    return cleaned[0].lower() + cleaned[1:]


def _clean(value: Any, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip(" ,.;:") + "..."


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _words(value: Any) -> list[str]:
    return re.findall(r"[A-Za-z0-9%+./-]+", str(value or ""))


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "beat")).strip("-_").lower() or "beat"


def _short_error(exc: Exception) -> str:
    return _clean(str(exc), limit=220)


def _final_capture_plan(beat_graph: BeatGraph) -> list[dict[str, Any]]:
    duration = max(float(beat_graph.duration_sec or 0.0), 0.1)
    captures: list[dict[str, Any]] = []
    for beat in beat_graph.beats[:8]:
        midpoint = beat.start + max(beat.duration * 0.58, 0.12)
        captures.append(
            {
                "capture_id": f"{beat.beat_id}_resolved",
                "fraction": round(max(0.0, min(midpoint / duration, 0.98)), 4),
            }
        )
    if not captures:
        captures = [{"capture_id": "final_midpoint", "fraction": 0.5}]
    return captures


def _first_comp_metadata(cinematic_plan: CinematicPlan | None) -> dict[str, Any]:
    if cinematic_plan is None:
        return {}
    for item in cinematic_plan.beat_compositions:
        if item.compiler_passed and item.metadata:
            return dict(item.metadata)
    return {}


def _medium_families(cinematic_plan: CinematicPlan | None) -> list[str]:
    if cinematic_plan is None:
        return []
    result: list[str] = []
    for item in cinematic_plan.beat_compositions:
        world = dict(item.metadata.get("visual_world_program") or {})
        medium = str(world.get("medium_family") or "")
        if medium:
            result.append(medium)
    return result


def _repeats_medium_too_often(cinematic_plan: CinematicPlan | None) -> bool:
    mediums = _medium_families(cinematic_plan)
    if len(mediums) < 3:
        return False
    return any(
        mediums[index] == mediums[index - 1] == mediums[index - 2]
        for index in range(2, len(mediums))
    )


def _too_generic(label: str) -> bool:
    normalized = _normalize(label)
    return normalized in {
        "action",
        "context",
        "input",
        "output",
        "result",
        "signal",
        "system",
        "workflow",
    }


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    if not spec:
        return {}
    world = dict(spec.get("visual_world_program") or {})
    ir = dict(spec.get("visual_explanation_ir") or {})
    return {
        "visual_id": str(spec.get("visual_id") or ""),
        "template": str(spec.get("template") or ""),
        "scene_type": str(ir.get("scene_type") or ""),
        "headline": str(spec.get("headline") or ""),
        "required_labels": list((spec.get("qa_contract") or {}).get("required_labels") or []),
        "semantic_signature": str((spec.get("qa_contract") or {}).get("semantic_signature") or ""),
        "proof_program_id": str(spec.get("proof_program_id") or ""),
        "visual_world": {
            "world_id": str(world.get("world_id") or ""),
            "medium_family": str(world.get("medium_family") or ""),
            "canvas_system": str(world.get("canvas_system") or ""),
            "background_mode": str(world.get("background_mode") or ""),
            "world_signature": str(world.get("world_signature") or ""),
        },
    }


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    stage = dict(metadata.get("stage") or {})
    world = dict(metadata.get("visual_world_program") or {})
    return {
        "composition_id": str(metadata.get("composition_id") or ""),
        "template": str(metadata.get("template") or ""),
        "duration_sec": float(metadata.get("duration_sec") or 0.0),
        "archetype": str(metadata.get("archetype") or ""),
        "semantic_signature": str(metadata.get("semantic_signature") or ""),
        "stage": {
            "stage_family": str(stage.get("stage_family") or ""),
            "generation_mode": str(stage.get("generation_mode") or ""),
            "object_coverage": stage.get("object_coverage"),
            "relation_coverage": stage.get("relation_coverage"),
            "grounded_copy_ratio": stage.get("grounded_copy_ratio"),
        },
        "visual_world": {
            "medium_family": str(world.get("medium_family") or ""),
            "canvas_system": str(world.get("canvas_system") or ""),
            "background_mode": str(world.get("background_mode") or ""),
            "world_signature": str(world.get("world_signature") or ""),
        },
    }


_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "before",
    "because",
    "but",
    "can",
    "every",
    "from",
    "into",
    "only",
    "that",
    "then",
    "this",
    "through",
    "using",
    "where",
    "with",
    "without",
}


__all__ = [
    "CINEMATOGRAPHER_VERSION",
    "CinematicBeatComposition",
    "CinematicPlan",
    "build_cinematic_plan",
    "evaluate_rendered_cinematography",
    "write_cinematic_compositions",
]
