from __future__ import annotations

import re
import json
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from video_generation.beat_tournament import select_directed_variant
from video_generation.director import DirectorPackage, director_context_for_beat
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
    tournament: dict[str, Any] = field(default_factory=dict)

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
            "tournament": dict(self.tournament),
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
    director_package: DirectorPackage | None = None,
) -> CinematicPlan:
    compositions: list[CinematicBeatComposition] = []
    warnings: list[str] = []
    visual_world_history: list[dict[str, Any]] = []
    used_medium_families: list[str] = []
    used_world_signatures: set[str] = set()
    for beat in beat_graph.beats:
        compiled_item = _compile_beat(
            request=request,
            plan=plan,
            beat=beat,
            visual_world_history=visual_world_history,
            director_package=director_package,
            used_medium_families=used_medium_families,
            used_world_signatures=used_world_signatures,
        )
        compositions.append(compiled_item)
        fingerprint = (
            dict(compiled_item.metadata.get("visual_world_program") or {})
            .get("fingerprint")
        )
        if compiled_item.compiler_passed and isinstance(fingerprint, dict):
            visual_world_history.append(dict(fingerprint))
        if compiled_item.compiler_passed:
            world = dict(compiled_item.metadata.get("visual_world_program") or {})
            medium = str(world.get("medium_family") or "").strip()
            signature = str(world.get("world_signature") or "").strip()
            if medium:
                used_medium_families.append(medium)
            if signature:
                used_world_signatures.add(signature)
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
    motion_plan: Any | None = None,
    width: int = 1920,
    height: int = 1080,
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
        profile = (
            motion_plan.profile_for(item.beat_id)
            if motion_plan is not None and hasattr(motion_plan, "profile_for")
            else None
        )
        path.write_text(
            _external_template_html(
                item,
                motion_profile=profile,
                width=width,
                height=height,
            ),
            encoding="utf-8",
        )
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


def _external_template_html(
    item: CinematicBeatComposition,
    *,
    motion_profile: Any | None = None,
    width: int = 1920,
    height: int = 1080,
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
    template_id = f"{_safe_id(item.composition_id)}-template"
    scoped_id = f"{_safe_id(item.composition_id)}-{_safe_id(item.beat_id)}-template-root"
    content = _scope_external_template_document(
        f"{head_styles}\n{body}",
        composition_id=item.composition_id,
        scoped_id=scoped_id,
    )
    content = _move_template_styles_inside_root(content, scoped_id="root")
    content = _inject_native_motion_metadata(
        content,
        item=item,
        motion_profile=motion_profile,
        width=width,
        height=height,
    )
    content = _inject_native_motion_runtime(
        content,
        item=item,
        motion_profile=motion_profile,
    )
    return (
        f'<template id="{template_id}">\n'
        f"{content}\n"
        "</template>\n"
    )


def _inject_native_motion_metadata(
    html_doc: str,
    *,
    item: CinematicBeatComposition,
    motion_profile: Any | None,
    width: int,
    height: int,
) -> str:
    payload = _motion_profile_payload(motion_profile)
    attrs = (
        ' data-vex-native-composition="true"'
        f' data-vex-beat-id="{_escape_attr(item.beat_id)}"'
        f' data-vex-motion-scope="{_escape_attr(item.composition_id + ":" + item.beat_id)}"'
        f' data-vex-width="{int(width)}"'
        f' data-vex-height="{int(height)}"'
        f' data-duration="{float(item.duration):.3f}"'
        f" data-vex-motion-profile='{_escape_attr(json.dumps(payload, ensure_ascii=True))}'"
    )

    def replace_root(match: re.Match[str]) -> str:
        tag = match.group(0)
        if "data-vex-native-composition=" in tag:
            return tag
        return tag[:-1] + attrs + ">"

    return re.sub(
        r"<div[^>]*\bdata-composition-id=\""
        + re.escape(item.composition_id)
        + r"\"[^>]*>",
        replace_root,
        html_doc,
        count=1,
        flags=re.IGNORECASE,
    )


def _inject_native_motion_runtime(
    html_doc: str,
    *,
    item: CinematicBeatComposition,
    motion_profile: Any | None,
) -> str:
    payload = _motion_profile_payload(motion_profile)
    style = _native_motion_style(payload)
    script = _native_motion_script(
        composition_id=item.composition_id,
        payload=payload,
    )
    injection = f"\n<style>{style}</style>\n{script}\n"
    if re.search(r"</body>", html_doc, flags=re.IGNORECASE):
        return re.sub(
            r"</body>",
            injection + "</body>",
            html_doc,
            count=1,
            flags=re.IGNORECASE,
        )
    return html_doc + injection


def _motion_profile_payload(motion_profile: Any | None) -> dict[str, Any]:
    if motion_profile is None:
        return {
            "version": "native-motion-profile-v1",
            "technique": "semantic_motion_stage",
            "camera_move": "guided_center_push",
            "energy": "calm",
            "effect_stack": ["grain_overlay", "depth_parallax"],
            "capabilities": ["seekable_timeline", "motion_css_variables"],
            "audio_cues": [],
        }
    if hasattr(motion_profile, "to_dict"):
        raw = dict(motion_profile.to_dict())
    elif isinstance(motion_profile, dict):
        raw = dict(motion_profile)
    else:
        raw = {}
    beat_start = _float(raw.get("start"), 0.0)
    duration = _float(raw.get("duration"), 0.0)
    global_cues = [
        dict(item)
        for item in raw.get("audio_cues") or []
        if isinstance(item, dict)
    ]
    local_cues = [_localize_audio_cue(item, beat_start=beat_start) for item in global_cues]
    return {
        "version": "native-motion-profile-v1",
        "beat_id": str(raw.get("beat_id") or ""),
        "start": beat_start,
        "duration": duration,
        "medium_family": str(raw.get("medium_family") or ""),
        "technique": str(raw.get("technique") or "semantic_motion_stage"),
        "camera_move": str(raw.get("camera_move") or "guided_center_push"),
        "transition_in": str(raw.get("transition_in") or ""),
        "transition_out": str(raw.get("transition_out") or ""),
        "caption_style": str(raw.get("caption_style") or ""),
        "energy": str(raw.get("energy") or "calm"),
        "effect_stack": [
            str(item)
            for item in raw.get("effect_stack") or []
            if str(item).strip()
        ],
        "capabilities": [
            str(item)
            for item in raw.get("capabilities") or []
            if str(item).strip()
        ],
        "audio_cues": local_cues,
        "audio_cues_global": global_cues,
    }


def _localize_audio_cue(cue: dict[str, Any], *, beat_start: float) -> dict[str, Any]:
    localized = dict(cue)
    raw_start = _float(localized.get("start"), beat_start)
    raw_end = _float(localized.get("end"), raw_start + 0.12)
    start = raw_start - beat_start
    end = raw_end - beat_start
    localized["start"] = round(max(0.0, start), 3)
    localized["end"] = round(max(localized["start"] + 0.05, end), 3)
    return localized


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _native_motion_style(payload: dict[str, Any]) -> str:
    technique = _safe_id(payload.get("technique"))
    camera = _safe_id(payload.get("camera_move"))
    energy = _safe_id(payload.get("energy"))
    return f"""
      [data-vex-native-composition="true"] {{
        --vex-bass: 0;
        --vex-mids: 0;
        --vex-treble: 0;
        --vex-amp: 0;
        isolation: isolate;
        overflow: visible !important;
      }}
      [data-vex-native-composition="true"]::before,
      [data-vex-native-composition="true"]::after {{
        content: "";
        position: absolute;
        inset: -8%;
        z-index: 0;
        pointer-events: none;
        opacity: calc(.045 + var(--vex-amp, 0) * .12);
        mix-blend-mode: screen;
      }}
      [data-vex-native-composition="true"]::before {{
        background:
          radial-gradient(circle at calc(18% + var(--p, 0) * 64%) calc(20% + var(--vex-mids, 0) * 28%), rgba(45,212,191,.46), transparent 30%),
          linear-gradient(115deg, transparent, rgba(249,115,22,.22), transparent);
        filter: blur(calc(12px + var(--vex-bass, 0) * 18px));
      }}
      [data-vex-native-composition="true"]::after {{
        background:
          repeating-linear-gradient(90deg, rgba(248,250,252,.08) 0 1px, transparent 1px 18px),
          radial-gradient(circle at 72% 56%, rgba(163,230,53,.32), transparent 26%);
        transform: translate3d(calc(var(--vex-treble, 0) * 18px), calc(var(--vex-bass, 0) * -12px), 0);
      }}
      [data-vex-native-composition="true"] .visual-world-canvas {{
        transform:
          perspective(1400px)
          translate3d(0, 0, 0)
          scale(calc(1 + var(--vex-amp, 0) * .008));
        transform-origin: center;
        filter:
          saturate(calc(1 + var(--vex-treble, 0) * .14))
          contrast(calc(1 + var(--vex-bass, 0) * .05));
        z-index: 2;
      }}
      [data-vex-native-composition="true"] .vw-relation,
      [data-vex-native-composition="true"] [data-line] {{
        filter: drop-shadow(0 0 calc(10px + var(--vex-mids, 0) * 20px) var(--accent));
      }}
      [data-vex-native-composition="true"] [data-anim] {{
        will-change: transform, opacity, filter;
      }}
      [data-vex-native-composition="true"][data-vex-motion-profile*="{technique}"] .visual-world-canvas {{
        --vex-technique-profile: "{technique}";
      }}
      [data-vex-native-composition="true"][data-vex-motion-profile*="{camera}"] .visual-world-canvas {{
        --vex-camera-profile: "{camera}";
      }}
      [data-vex-native-composition="true"][data-vex-motion-profile*="{energy}"] {{
        --vex-energy-profile: "{energy}";
      }}
    """


def _native_motion_script(
    *,
    composition_id: str,
    payload: dict[str, Any],
) -> str:
    profile_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    composition_json = json.dumps(composition_id, ensure_ascii=True)
    beat_json = json.dumps(str(payload.get("beat_id") or ""), ensure_ascii=True)
    return f"""
      <script>
      (() => {{
        const compositionId = {composition_json};
        const beatId = {beat_json};
        const profile = {profile_json};
        const nativeRoots = Array.from(document.querySelectorAll('[data-vex-native-composition="true"]'));
        const root = nativeRoots.find((candidate) => candidate.dataset.vexBeatId === beatId)
          || nativeRoots.find((candidate) => candidate.dataset.compositionId === compositionId)
          || document.getElementById("root");
        if (!root) return;
        const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
        const cues = Array.isArray(profile.audio_cues) ? profile.audio_cues : [];
        function cueStrength(time, band) {{
          let value = 0;
          cues.forEach((cue) => {{
            if (cue.band !== band) return;
            const start = Number(cue.start || 0);
            const end = Math.max(Number(cue.end || start + .12), start + .05);
            const p = clamp((Number(time) - start) / (end - start));
            value = Math.max(value, Math.sin(p * Math.PI) * Number(cue.strength || .5));
          }});
          return clamp(value);
        }}
        function setCueVars(time) {{
          const bass = cueStrength(time, "bass");
          const mids = cueStrength(time, "mids");
          const treble = cueStrength(time, "treble");
          const amp = Math.max(cueStrength(time, "amplitude"), bass * .72, mids * .42, treble * .34);
          root.style.setProperty("--vex-bass", bass.toFixed(5));
          root.style.setProperty("--vex-mids", mids.toFixed(5));
          root.style.setProperty("--vex-treble", treble.toFixed(5));
          root.style.setProperty("--vex-amp", amp.toFixed(5));
        }}
        function patchTimeline() {{
          window.__timelines = window.__timelines || {{}};
          const timeline = window.__timelines[compositionId];
          if (!timeline || timeline.__vexNativeMotionPatched) {{
            setCueVars(0);
            return;
          }}
          const originalTime = typeof timeline.time === "function" ? timeline.time.bind(timeline) : null;
          const originalSeek = typeof timeline.seek === "function" ? timeline.seek.bind(timeline) : null;
          const originalProgress = typeof timeline.progress === "function" ? timeline.progress.bind(timeline) : null;
          const duration = typeof timeline.duration === "function"
            ? Number(timeline.duration()) || Number(root.dataset.duration || 0)
            : Number(root.dataset.duration || 0);
          if (originalTime) {{
            timeline.time = (value) => {{
              originalTime(value);
              setCueVars(Number(value) || 0);
              return timeline;
            }};
          }}
          if (originalSeek) {{
            timeline.seek = (value) => {{
              originalSeek(value);
              setCueVars(Number(value) || 0);
              return timeline;
            }};
          }}
          if (originalProgress) {{
            timeline.progress = (value) => {{
              const p = Number(value) || 0;
              originalProgress(p);
              setCueVars(p * duration);
              return timeline;
            }};
          }}
          timeline.__vexNativeMotionPatched = true;
          setCueVars(0);
        }}
        patchTimeline();
      }})();
      </script>
    """


def _escape_attr(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _scope_external_template_document(
    content: str,
    *,
    composition_id: str,
    scoped_id: str,
) -> str:
    content = _scope_common_document_selectors(
        content,
        scoped_id="root",
    )

    def replace_root(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        root_classes = _class_names_from_attrs(attrs)
        attrs = re.sub(r'\bid="[^"]*"', 'id="root"', attrs, count=1)
        if 'id="root"' not in attrs:
            attrs += ' id="root"'
        if "data-vex-template-root-id=" not in attrs:
            attrs += f' data-vex-template-root-id="{_escape_attr(scoped_id)}"'
        attrs = re.sub(r'\sdata-start="[^"]*"', "", attrs)
        attrs = re.sub(r'\sdata-duration="[^"]*"', "", attrs)
        attrs = re.sub(r'\sdata-track-index="[^"]*"', "", attrs)
        return f"<div{attrs}>", root_classes

    root_classes: list[str] = []

    def replace_root_and_capture(match: re.Match[str]) -> str:
        replacement, captured_classes = replace_root(match)
        root_classes.extend(captured_classes)
        return replacement

    scoped = re.sub(
        r"<div(?P<attrs>[^>]*\bdata-composition-id=\""
        + re.escape(composition_id)
        + r"\"[^>]*)>",
        replace_root_and_capture,
        content,
        count=1,
        flags=re.IGNORECASE,
    )
    return _rewrite_template_root_class_selectors(
        scoped,
        root_classes=root_classes,
        root_id="root",
    )


def _class_names_from_attrs(attrs: str) -> list[str]:
    match = re.search(r'\bclass="([^"]*)"', attrs, flags=re.IGNORECASE)
    if not match:
        return []
    return [
        item
        for item in match.group(1).split()
        if item.strip()
    ]


def _rewrite_template_root_class_selectors(
    content: str,
    *,
    root_classes: list[str],
    root_id: str,
) -> str:
    if not root_classes:
        return content
    root_selector = f"#{root_id}"
    class_patterns = [
        re.compile(
            rf"(?<![A-Za-z0-9_-])\.{re.escape(class_name)}(?=[:\s,>.#\[]|$)"
        )
        for class_name in root_classes
    ]

    def rewrite_style(match: re.Match[str]) -> str:
        block = match.group(0)
        for pattern in class_patterns:
            block = pattern.sub(root_selector, block)
        block = re.sub(rf"(?:{re.escape(root_selector)}){{2,}}", root_selector, block)
        block = re.sub(
            rf"{re.escape(root_selector)}\s+{re.escape(root_selector)}\b",
            root_selector,
            block,
        )
        return block

    return re.sub(
        r"<style\b[^>]*>.*?</style>",
        rewrite_style,
        content,
        flags=re.IGNORECASE | re.DOTALL,
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
                    f'overflow:visible;{style_match.group(1)}"'
                ),
                attrs,
                count=1,
            )
        else:
            attrs += ' style="position:absolute;inset:0;width:100%;height:100%;overflow:visible;"'
        if "data-layout-allow-overflow" not in attrs:
            attrs += ' data-layout-allow-overflow="camera-safe-area"'
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
        + f"{selector} {{ --vex-camera-x:0px; --vex-camera-y:0px; --vex-zoom:1; --vex-roll:0deg; --vex-tilt-x:0deg; --vex-tilt-y:0deg; --vex-parallax-x:0px; --vex-parallax-y:0px; --vex-inner-x:0px; --vex-inner-y:0px; --vex-inner-roll:0deg; --vex-counter-roll:0deg; --vex-glow:.3; perspective:1600px; overflow:visible !important; }}\n"
        + f"{selector} .visual-world-canvas {{ transform:perspective(1600px) translate3d(var(--vex-camera-x),var(--vex-camera-y),0) rotateX(var(--vex-tilt-x)) rotateY(var(--vex-tilt-y)) rotateZ(var(--vex-roll)) scale(var(--vex-zoom)) !important; transform-origin:center; filter:saturate(calc(1 + var(--vex-treble,0) * .18)) contrast(calc(1 + var(--vex-bass,0) * .06)) brightness(calc(1 + var(--vex-amp,0) * .025)); will-change:transform,filter; isolation:isolate; }}\n"
        + f"{selector} .visual-world-canvas::before, {selector} .visual-world-canvas::after {{ content:\"\"; position:absolute; inset:-8%; z-index:0; pointer-events:none; opacity:calc(.035 + var(--vex-glow,.3) * .07); mix-blend-mode:screen; }}\n"
        + f"{selector} .visual-world-canvas::before {{ background:linear-gradient(112deg,transparent 0 36%,rgba(45,212,191,.18),rgba(249,115,22,.10),transparent 64% 100%); transform:translate3d(calc(var(--vex-parallax-x) * -1),var(--vex-parallax-y),0); filter:blur(calc(8px + var(--vex-amp,0) * 10px)); }}\n"
        + f"{selector} .visual-world-canvas::after {{ background:radial-gradient(circle at calc(18% + var(--p,0) * 64%) calc(24% + var(--vex-mids,0) * 28%),rgba(163,230,53,.12),transparent 25%),repeating-linear-gradient(90deg,rgba(248,250,252,.035) 0 1px,transparent 1px 28px); transform:translate3d(var(--vex-parallax-x),calc(var(--vex-parallax-y) * -1),0); }}\n"
        + f"{selector} .vw-particle-field, {selector} .vw-type-relations, {selector} .vw-data-relations {{ transform:translate3d(var(--vex-parallax-x),var(--vex-parallax-y),0); will-change:transform; }}\n"
        + f"{selector} .vw-relation, {selector} [data-line] {{ filter:drop-shadow(0 0 calc(8px + var(--vex-glow,.3) * 22px) var(--accent)); stroke-dasharray:1; stroke-dashoffset:calc(1 - var(--line-progress,0)); }}\n"
        + f"{selector} [data-anim] {{ will-change:transform,opacity,filter; }}\n"
        + f"{selector} .visual-world-canvas {{ inset: 5.5% !important; }}\n"
        + f"{selector} .vw-data-title, {selector} .vw-spatial-title, {selector} .vw-collage-masthead, {selector} .vw-system-title, {selector} .vw-partition-title, {selector} .vw-source-panel header {{ display:grid !important; grid-template-rows:auto auto !important; row-gap:24px !important; align-content:start !important; }}\n"
        + f"{selector} .vw-data-title span, {selector} .vw-spatial-title span, {selector} .vw-collage-masthead span, {selector} .vw-system-title span, {selector} .vw-partition-title span, {selector} .vw-source-panel header span {{ display:block !important; margin:0 !important; line-height:1 !important; }}\n"
        + f"{selector} .vw-data-title b, {selector} .vw-spatial-title b, {selector} .vw-collage-masthead b, {selector} .vw-system-title b, {selector} .vw-partition-title b, {selector} .vw-source-panel header b {{ display:block !important; margin:0 !important; }}\n"
        + f"{selector} .vw-data-sculpture .vw-data-title {{ left: 7% !important; top: 6.5% !important; max-width: 54% !important; }}\n"
        + f"{selector} .vw-data-sculpture .vw-data-title b {{ font-size: clamp(34px, 4.4vw, 78px) !important; line-height: .96 !important; }}\n"
        + f"{selector} .vw-data-sculpture .vw-masses {{ inset: 32% 9% 11% 9% !important; transform:translate3d(var(--vex-inner-x),var(--vex-inner-y),0) rotateZ(var(--vex-counter-roll)) scale(calc(.86 + var(--vex-amp,0) * .018)); transform-origin: center; will-change:transform; }}\n"
        + f"{selector} .vw-data-sculpture .vw-mass strong {{ font-size: clamp(16px, 1.72vw, 28px) !important; }}\n"
        + f"{selector} .vw-collage .vw-collage-masthead {{ left: 5.8% !important; top: 6.2% !important; max-width: 62% !important; }}\n"
        + f"{selector} .vw-collage .vw-collage-masthead b {{ font-size: clamp(34px, 4.6vw, 76px) !important; line-height: .98 !important; }}\n"
        + f"{selector} .vw-collage .vw-collage-pieces {{ inset: 27% 7% 8% 7% !important; transform:translate3d(var(--vex-inner-x),var(--vex-inner-y),0) rotateZ(var(--vex-inner-roll)) scale(calc(.91 + var(--vex-amp,0) * .012)); transform-origin: center; will-change:transform; }}\n"
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
    director_package: DirectorPackage | None = None,
    used_medium_families: list[str] | None = None,
    used_world_signatures: set[str] | None = None,
) -> CinematicBeatComposition:
    errors: list[str] = []
    used_medium_families = list(used_medium_families or [])
    used_world_signatures = set(used_world_signatures or set())
    for candidate in _candidate_specs(
        request=request,
        plan=plan,
        beat=beat,
        visual_world_history=visual_world_history,
        director_package=director_package,
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
        variant, tournament = select_directed_variant(
            variants,
            beat=beat,
            request=request,
            plan=plan,
            director_package=director_package,
            used_medium_families=used_medium_families,
            used_world_signatures=used_world_signatures,
        )
        if variant is None:
            errors.extend(tournament.issues or ["no_variant_passed_authoring_tournament"])
            continue
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
            metadata={
                **dict(composition.metadata),
                "beat_tournament": tournament.to_dict(),
                "director_context": director_context_for_beat(
                    director_package,
                    beat.beat_id,
                ),
            },
            tournament=tournament.to_dict(),
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
    director_package: DirectorPackage | None = None,
) -> list[dict[str, Any]]:
    context = _context_text(plan=plan, beat=beat)
    label_source = _label_source(beat)
    source = f"{label_source} {context}"
    director_context = director_context_for_beat(director_package, beat.beat_id)
    contract = dict(director_context.get("beat_contract") or {})
    contract_objects = [
        str(item)
        for item in contract.get("required_objects") or []
        if str(item).strip()
    ]
    base = {
        "visual_id": f"generated_{beat.beat_id}",
        "sentence_text": beat.narration,
        "context_text": _director_context_text(
            context,
            contract=contract,
            objects=contract_objects,
        ),
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
            "director_context": director_context,
        },
    }
    candidates: list[dict[str, Any]] = []
    if _is_sparse_attention(source):
        candidates.extend(_sparse_attention_specs(base, source, beat.index))
    transform_spec = _matched_transform_spec(base, label_source, source)
    process_spec = _guided_process_spec(base, label_source, source)
    causal_spec = _causal_spec(base, label_source, source)
    quote_spec = _quote_spec(base, label_source, source)
    if len(_clause_phrases(label_source, limit=5)) >= 3:
        candidates.extend([process_spec, transform_spec, causal_spec, quote_spec])
    else:
        candidates.extend([transform_spec, process_spec, causal_spec, quote_spec])
    return [
        _apply_director_contract(candidate, contract=contract)
        for candidate in candidates
        if candidate
    ]


def _director_context_text(
    context: str,
    *,
    contract: dict[str, Any],
    objects: list[str],
) -> str:
    if not contract:
        return context
    additions = [
        f"Beat objective: {contract.get('objective', '')}.",
        f"Viewer question: {contract.get('viewer_question', '')}.",
        f"Visual job: {contract.get('visual_job', '')}.",
        f"Required relation: {contract.get('required_relation', '')}.",
        f"Motion intent: {contract.get('motion_intent', '')}.",
    ]
    if objects:
        additions.append("Required visible objects: " + ", ".join(objects[:5]) + ".")
    return _clean(" ".join([context, *additions]), limit=1800)


def _apply_director_contract(
    candidate: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not contract:
        return candidate
    required_objects = [
        str(item).strip()
        for item in contract.get("required_objects") or []
        if str(item).strip()
    ]
    if not required_objects:
        return candidate
    labels = _unique(
        [
            *[str(item) for item in candidate.get("required_labels") or []],
            *required_objects,
        ]
    )[:7]
    semantic_frame = {
        **dict(candidate.get("semantic_frame") or {}),
        "director_objective": str(contract.get("objective") or ""),
        "director_relation": str(contract.get("required_relation") or ""),
        "director_objects": labels[:5],
    }
    qa_contract = {
        **dict(candidate.get("qa_contract") or {}),
        "required_labels": labels,
        "director_visual_job": str(contract.get("visual_job") or ""),
    }
    return {
        **candidate,
        "required_labels": labels,
        "semantic_frame": semantic_frame,
        "qa_contract": qa_contract,
        "director_contract": contract,
    }


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


def _matched_transform_spec(
    base: dict[str, Any],
    label_source: str,
    source: str,
) -> dict[str, Any]:
    pair = _before_after_labels(label_source)
    if pair is None:
        return {}
    before, after = pair
    constraint = _best_label(
        label_source,
        fallback="",
        exclude={before, after},
    )
    if constraint:
        context = (
            f"Before the change, the visible state is {before}. "
            f"After the intervention, the resolved state is {after}. "
            f"The visual preserves {constraint} while it transforms from before to after. "
            + source
        )
        semantic_frame = {
            "before_state": before,
            "after_state": after,
            "preserved_constraint": constraint,
        }
        required_labels = [before, after, constraint]
    else:
        context = (
            f"Before the change, the visible state is {before}. "
            f"After the intervention, the resolved state is {after}. "
            + source
        )
        semantic_frame = {
            "before_state": before,
            "after_state": after,
        }
        required_labels = [before, after]
    return {
        **base,
        "context_text": context,
        "semantic_frame": semantic_frame,
        "headline": after,
        "required_labels": required_labels,
    }


def _guided_process_spec(
    base: dict[str, Any],
    label_source: str,
    source: str,
) -> dict[str, Any]:
    labels = _process_labels(label_source)
    if len(labels) < 4:
        return {}
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
        "headline": labels[-1],
        "required_labels": labels,
    }


def _causal_spec(
    base: dict[str, Any],
    label_source: str,
    source: str,
) -> dict[str, Any]:
    labels = _causal_labels(label_source)
    if len(labels) < 4:
        return {}
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
        "headline": labels[-1],
        "required_labels": labels,
    }


def _quote_spec(
    base: dict[str, Any],
    label_source: str,
    source: str,
) -> dict[str, Any]:
    quote = _short_sentence(label_source, max_words=9)
    context = f"The exact quote is {quote}. The decisive phrase is {quote}. " + source
    return {
        **base,
        "context_text": context,
        "semantic_frame": {
            "exact_quote": quote,
        },
        "headline": quote,
        "required_labels": _quote_labels(quote),
    }


def _before_after_labels(source: str) -> tuple[str, str] | None:
    cleaned = _clean(source, limit=360)
    patterns = (
        r"\b(?:turns?|transforms?|converts?)\s+(?P<before>.+?)\s+into\s+(?P<after>.+?)(?:\s+by\b|[,.;]|\s+using\b|$)",
        r"\b(?:breaks?|splits?|compresses?)\s+(?P<before>.+?)\s+into\s+(?P<after>.+?)(?:\s+by\b|[,.;]|\s+using\b|$)",
        r"(?:^|[.!?:;]\s*)(?P<before>[^.!?:;]{2,90}?)\s+becomes?\s+(?P<after>.+?)(?:[,.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        before = _label(match.group("before"), fallback="")
        after = _label(match.group("after"), fallback="")
        if before and after and _normalize(before) != _normalize(after):
            return before, after
    from_to = re.search(
        r"\bfrom\s+(?P<before>.+?)\s+to\s+(?P<after>.+?)(?:[,.;]|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if from_to:
        before = _label(from_to.group("before"), fallback="")
        after = _label(from_to.group("after"), fallback="")
        if before and after and _normalize(before) != _normalize(after):
            return before, after
    return None


def _process_labels(source: str) -> list[str]:
    keywords = _keyword_phrases(source, limit=6)
    values = [item for item in keywords if not _too_generic(item)]
    return _unique(values)[:4]


def _causal_labels(source: str) -> list[str]:
    keywords = _keyword_phrases(source, limit=6)
    values = [item for item in keywords if not _too_generic(item)]
    return _unique(values)[:4]


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
    return _clause_phrases(source, limit=limit)


def _best_label(
    source: str,
    *,
    fallback: str,
    exclude: set[str] | None = None,
) -> str:
    excluded = {_normalize(item) for item in (exclude or set())}
    for phrase in _keyword_phrases(source, limit=5):
        if (
            not _too_generic(phrase)
            and len(_words(phrase)) >= 2
            and _normalize(phrase) not in excluded
            and not _overlaps_excluded_label(phrase, exclude or set())
        ):
            return phrase
    return fallback


def _short_sentence(source: str, *, max_words: int) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", _clean(source, limit=240))[0]
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words])
    return sentence.strip(" ,.;:") or "The useful pattern becomes visible"


def _label(value: Any, *, fallback: str) -> str:
    cleaned = _strip_label_intro(_clean(value, limit=84)).strip(" ,.;:-")
    words = cleaned.split()
    if len(words) > 5:
        cleaned = " ".join(words[:5])
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    if not cleaned or not _label_is_readable(cleaned):
        return fallback
    return cleaned[0].lower() + cleaned[1:]


def _label_source(beat: Beat) -> str:
    return _clean(beat.narration or beat.caption or beat.title, limit=420)


def _clause_phrases(source: str, *, limit: int) -> list[str]:
    cleaned = _clean(source, limit=420)
    pieces = re.split(r"(?<=[.!?])\s+|[;:]+|,\s*", cleaned)
    phrases: list[str] = []
    for piece in pieces:
        label = _label(piece, fallback="")
        if not label or _too_generic(label):
            continue
        phrases.append(label)
        if len(phrases) >= limit:
            break
    return _unique(phrases)[:limit]


def _strip_label_intro(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-")
    cleaned = re.sub(
        r"^(?:and\s+)?(?:first|second|third|next|then|finally|now|once)\b[:,]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    cleaned = re.sub(
        r"^(?:the\s+)?(?:proof|result|takeaway|payoff)\s+(?:is|becomes)\s+(?:this\s*)?:?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    cleaned = re.sub(
        r"^(?:it|this|that)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    choose_match = re.match(
        r"chooses?\s+where\s+(?P<label>.+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if choose_match:
        cleaned = choose_match.group("label")
    cleaned = re.sub(
        r"\b(?:a|an|the)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    return cleaned


def _label_is_readable(value: str) -> bool:
    cleaned = str(value or "").strip()
    if not cleaned:
        return False
    if re.search(r"[.!?]", cleaned):
        return False
    words = [word.lower() for word in _words(cleaned)]
    if not words:
        return False
    if words[-1] in _BAD_LABEL_ENDINGS:
        return False
    if len(words) == 2 and all(word in _VERBISH_WORDS for word in words):
        return False
    if _normalize(cleaned) in _GENERIC_LABEL_SET:
        return False
    return True


def _clean(value: Any, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").replace("\ufeff", "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip(" ,.;:") + "..."


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9%+./-]+", " ", str(value or "").lower()).strip()


def _words(value: Any) -> list[str]:
    return re.findall(r"[A-Za-z0-9%+./-]+", str(value or ""))


def _overlaps_excluded_label(label: str, excluded: set[str]) -> bool:
    label_tokens = {token.lower() for token in _words(label) if len(token) >= 3}
    if not label_tokens:
        return False
    for item in excluded:
        item_tokens = {token.lower() for token in _words(item) if len(token) >= 3}
        if not item_tokens:
            continue
        overlap = len(label_tokens & item_tokens)
        if overlap / max(len(item_tokens), 1) >= 0.6:
            return True
    return False


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
    return normalized in _GENERIC_LABEL_SET


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

_BAD_LABEL_ENDINGS = {
    "and",
    "are",
    "because",
    "by",
    "for",
    "from",
    "into",
    "is",
    "of",
    "or",
    "that",
    "to",
    "where",
    "which",
    "with",
}

_GENERIC_LABEL_SET = {
    "action",
    "better",
    "clear",
    "context",
    "input",
    "output",
    "result",
    "signal",
    "simple",
    "system",
    "useful",
    "workflow",
}

_VERBISH_WORDS = {
    "adds",
    "applies",
    "becomes",
    "breaks",
    "build",
    "builds",
    "choose",
    "chooses",
    "compile",
    "compiles",
    "create",
    "creates",
    "drop",
    "drops",
    "generate",
    "generates",
    "helps",
    "keep",
    "keeps",
    "make",
    "makes",
    "render",
    "renders",
    "route",
    "routes",
    "run",
    "runs",
    "turn",
    "turns",
}


__all__ = [
    "CINEMATOGRAPHER_VERSION",
    "CinematicBeatComposition",
    "CinematicPlan",
    "build_cinematic_plan",
    "evaluate_rendered_cinematography",
    "write_cinematic_compositions",
]
