from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from video_generation.models import Beat, BeatGraph, ScriptPlan, VideoGenerationRequest
from video_generation.skill_graph import VideoSkillGraph, assignment_payload


MOTION_PLAN_VERSION = "hyperframes-native-motion-plan-v1"
_LOW_VALUE_CUE_WORDS = {
    "a",
    "about",
    "actually",
    "again",
    "also",
    "and",
    "are",
    "as",
    "because",
    "becomes",
    "before",
    "being",
    "but",
    "can",
    "could",
    "does",
    "every",
    "finally",
    "first",
    "from",
    "have",
    "here",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "more",
    "much",
    "need",
    "not",
    "onto",
    "only",
    "or",
    "over",
    "same",
    "simple",
    "some",
    "still",
    "that",
    "the",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "turns",
    "very",
    "when",
    "where",
    "with",
    "without",
    "was",
    "were",
}


@dataclass(frozen=True)
class AudioMotionCue:
    cue_id: str
    beat_id: str
    label: str
    start: float
    end: float
    band: str
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeatMotionProfile:
    beat_id: str
    composition_id: str
    start: float
    duration: float
    medium_family: str
    technique: str
    camera_move: str
    transition_in: str
    transition_out: str
    caption_style: str
    energy: str
    effect_stack: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    audio_cues: list[AudioMotionCue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["audio_cues"] = [cue.to_dict() for cue in self.audio_cues]
        return payload


@dataclass(frozen=True)
class MotionPlan:
    version: str
    source: str
    native_composition_count: int
    advanced_capabilities: list[str]
    global_effects: list[str]
    render_contract: dict[str, Any]
    beat_profiles: list[BeatMotionProfile]
    warnings: list[str] = field(default_factory=list)

    @property
    def audio_cue_count(self) -> int:
        return sum(len(profile.audio_cues) for profile in self.beat_profiles)

    def profile_for(self, beat_id: str) -> BeatMotionProfile | None:
        for profile in self.beat_profiles:
            if profile.beat_id == beat_id:
                return profile
        return None

    def cue_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for profile in self.beat_profiles:
            for cue in profile.audio_cues:
                rows.append(cue.to_dict())
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "native_composition_count": self.native_composition_count,
            "audio_cue_count": self.audio_cue_count,
            "advanced_capabilities": list(self.advanced_capabilities),
            "global_effects": list(self.global_effects),
            "render_contract": dict(self.render_contract),
            "beat_profiles": [profile.to_dict() for profile in self.beat_profiles],
            "warnings": list(self.warnings),
        }


def build_motion_plan(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    cinematic_plan: Any | None = None,
    video_skill_graph: VideoSkillGraph | None = None,
) -> MotionPlan:
    profiles: list[BeatMotionProfile] = []
    warnings: list[str] = []
    capabilities: set[str] = {
        "nested_compositions",
        "registered_seekable_timelines",
        "deterministic_audio_cues",
        "transition_overlays",
        "snapshot_motion_qa",
    }
    native_count = 0

    for beat in beat_graph.beats:
        cinematic = _cinematic_for(cinematic_plan, beat.beat_id)
        skill_assignment = assignment_payload(video_skill_graph, beat.beat_id)
        composition_id = str(getattr(cinematic, "composition_id", "") or "")
        medium = _medium_for(cinematic, beat)
        if composition_id:
            native_count += 1
        technique = _technique_for(
            medium,
            beat,
            request=request,
            skill_assignment=skill_assignment,
        )
        profile_capabilities = _capabilities_for(medium, technique)
        capabilities.update(profile_capabilities)
        profiles.append(
            BeatMotionProfile(
                beat_id=beat.beat_id,
                composition_id=composition_id,
                start=round(float(beat.start), 3),
                duration=round(float(beat.duration), 3),
                medium_family=medium,
                technique=technique,
                camera_move=_camera_move_for(medium, beat, skill_assignment=skill_assignment),
                transition_in=_transition_in(beat, skill_assignment=skill_assignment),
                transition_out=_transition_out(beat, beat_graph, skill_assignment=skill_assignment),
                caption_style=_caption_style_for(request, beat),
                energy=_energy_for(request, plan, beat, skill_assignment=skill_assignment),
                effect_stack=_effect_stack_for(medium, technique, skill_assignment=skill_assignment),
                capabilities=profile_capabilities,
                audio_cues=_audio_cues_for_beat(beat, beat_graph),
            )
        )

    if native_count == 0:
        warnings.append("native_motion_plan_has_no_compiled_compositions")
    if not any(profile.audio_cues for profile in profiles):
        warnings.append("native_motion_plan_has_no_audio_cues")

    return MotionPlan(
        version=MOTION_PLAN_VERSION,
        source="script_audio_cinematography",
        native_composition_count=native_count,
        advanced_capabilities=sorted(capabilities),
        global_effects=[
            "shader_transition_overlays",
            "audio_pulse_variables",
            "depth_parallax_field",
            "kinetic_caption_layer",
        ],
        render_contract={
            "preferred_quality": "high",
            "preferred_fps": max(30, min(int(request.fps), 60)),
            "supports_4k_render_resolution": True,
            "deterministic": True,
            "no_wall_clock_animation": True,
        },
        beat_profiles=profiles,
        warnings=warnings,
    )


def _cinematic_for(cinematic_plan: Any | None, beat_id: str) -> Any | None:
    if cinematic_plan is None:
        return None
    for item in getattr(cinematic_plan, "beat_compositions", []) or []:
        if getattr(item, "beat_id", "") == beat_id and getattr(item, "compiler_passed", False):
            return item
    return None


def _medium_for(cinematic: Any | None, beat: Beat) -> str:
    if cinematic is not None:
        metadata = dict(getattr(cinematic, "metadata", {}) or {})
        world = dict(metadata.get("visual_world_program") or metadata.get("visual_world") or {})
        medium = str(world.get("medium_family") or "").strip()
        if medium:
            return medium
    if beat.scene_type in {"metric", "proof"}:
        return "data_sculpture"
    if beat.scene_type == "process":
        return "diagrammatic_system"
    if beat.scene_type == "contrast":
        return "editorial_collage"
    if beat.scene_type == "hook":
        return "kinetic_typography"
    return "spatial_metaphor"


def _technique_for(
    medium: str,
    beat: Beat,
    *,
    request: VideoGenerationRequest,
    skill_assignment: dict[str, Any] | None = None,
) -> str:
    if skill_assignment and str(skill_assignment.get("motion_technique") or "").strip():
        return str(skill_assignment.get("motion_technique")).strip()
    text = f"{request.style} {beat.title} {beat.narration} {beat.visual_metaphor}".lower()
    if "code" in text or "repo" in text or "terminal" in text:
        return "code_shader_reveal"
    if medium == "data_sculpture":
        return "particle_sculpture_orbit"
    if medium == "editorial_collage":
        return "parallax_editorial_collage"
    if medium == "kinetic_typography":
        return "per_word_kinetic_type"
    if medium == "product_interface":
        return "ui_camera_push_focus"
    if medium == "spatial_metaphor":
        return "pseudo_3d_arc_motion"
    if medium == "diagrammatic_system":
        return "routed_system_trace"
    if medium == "source_media_composite":
        return "source_media_spotlight"
    return "semantic_motion_stage"


def _capabilities_for(medium: str, technique: str) -> list[str]:
    base = ["seekable_timeline", "motion_css_variables"]
    if "shader" in technique:
        base.extend(["shader_dissolve", "chromatic_edge_glow"])
    if "particle" in technique:
        base.extend(["particle_field", "depth_lighting", "bass_pulse"])
    if "collage" in technique:
        base.extend(["parallax_layers", "marker_highlight", "paper_camera"])
    if "kinetic" in technique:
        base.extend(["per_word_emphasis", "scale_pop_captions", "treble_glow"])
    if "3d" in technique:
        base.extend(["css_3d_camera", "arc_motion_path", "auto_rotate_feel"])
    if medium in {"diagrammatic_system", "data_sculpture"}:
        base.extend(["route_trace", "relation_highlight"])
    if medium == "product_interface":
        base.extend(["ui_focus_ring", "shimmer_sweep"])
    return _unique(base)


def _camera_move_for(
    medium: str,
    beat: Beat,
    *,
    skill_assignment: dict[str, Any] | None = None,
) -> str:
    if skill_assignment and str(skill_assignment.get("camera_move") or "").strip():
        return str(skill_assignment.get("camera_move")).strip()
    if medium == "data_sculpture":
        return "slow_orbital_push"
    if medium == "editorial_collage":
        return "paper_parallax_pan"
    if medium == "kinetic_typography":
        return "type_scale_snap"
    if medium == "spatial_metaphor":
        return "low_angle_dolly"
    if medium == "product_interface":
        return "interface_push_in"
    if beat.scene_type == "contrast":
        return "split_reveal_slide"
    return "guided_center_push"


def _transition_in(
    beat: Beat,
    *,
    skill_assignment: dict[str, Any] | None = None,
) -> str:
    if skill_assignment and beat.index > 1:
        intent = str(skill_assignment.get("transition_intent") or "")
        if "morph" in intent:
            return "cross_warp_morph"
        if "handoff" in intent:
            return "whip_pan"
    if beat.index <= 1:
        return "cold_open"
    choices = ["cross_warp_morph", "whip_pan", "light_leak", "cinematic_zoom"]
    return choices[(beat.index - 2) % len(choices)]


def _transition_out(
    beat: Beat,
    beat_graph: BeatGraph,
    *,
    skill_assignment: dict[str, Any] | None = None,
) -> str:
    if skill_assignment:
        intent = str(skill_assignment.get("transition_intent") or "")
        if "final_hold" in intent or "resolve" in intent:
            return "soft_resolve_hold"
        if "morph" in intent:
            return "cross_warp_morph"
        if "handoff" in intent:
            return "whip_pan"
    if beat.index >= len(beat_graph.beats):
        return "soft_resolve_hold"
    choices = ["whip_pan", "ridged_burn", "cross_warp_morph", "flash_through_white"]
    return choices[(beat.index - 1) % len(choices)]


def _caption_style_for(request: VideoGenerationRequest, beat: Beat) -> str:
    text = f"{request.style} {beat.narration}".lower()
    if "hype" in text or "viral" in text:
        return "kinetic_slam"
    if "tutorial" in text or "walkthrough" in text:
        return "monospace_typewriter"
    if beat.scene_type in {"hook", "contrast"}:
        return "scale_pop_keyword"
    return "calm_karaoke_highlight"


def _energy_for(
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat: Beat,
    *,
    skill_assignment: dict[str, Any] | None = None,
) -> str:
    if skill_assignment:
        role = str(skill_assignment.get("arc_role") or "")
        if role in {"hook", "decision", "contrast"}:
            return "medium"
        if role == "payoff":
            return "calm"
    text = f"{request.style} {plan.design_direction} {beat.narration}".lower()
    if re.search(r"\b(?:insane|hype|viral|max|dramatic|cinematic|high contrast)\b", text):
        return "high"
    if beat.duration <= 3.5 or beat.scene_type in {"hook", "contrast"}:
        return "medium"
    return "calm"


def _effect_stack_for(
    medium: str,
    technique: str,
    *,
    skill_assignment: dict[str, Any] | None = None,
) -> list[str]:
    effects = ["grain_overlay", "depth_parallax"]
    if medium == "data_sculpture":
        effects.extend(["orbital_particles", "relation_trails", "specular_sheen"])
    elif medium == "editorial_collage":
        effects.extend(["paper_shadow_parallax", "marker_sweep", "chromatic_snap"])
    elif medium == "kinetic_typography":
        effects.extend(["word_pop", "treble_glow", "difference_blend_sweep"])
    elif medium == "spatial_metaphor":
        effects.extend(["floor_grid_perspective", "arc_motion_path", "camera_dolly"])
    elif medium == "product_interface":
        effects.extend(["focus_ring", "shimmer_sweep", "cursor_trace"])
    elif medium == "diagrammatic_system":
        effects.extend(["route_draw", "node_pulse", "scanline"])
    if "shader" in technique:
        effects.append("shader_dissolve")
    if skill_assignment:
        effects.extend(
            str(item)
            for item in skill_assignment.get("effect_stack") or []
            if str(item).strip()
        )
    return _unique(effects)


def _audio_cues_for_beat(beat: Beat, beat_graph: BeatGraph) -> list[AudioMotionCue]:
    words = [
        word
        for word in beat_graph.words
        if word.start >= beat.start - 0.001 and word.start < beat.end - 0.001
    ]
    if not words:
        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9%+./-]+", beat.narration)
            if len(token) >= 4
        ][:6]
        span = max(beat.duration, 0.1)
        words = [
            type("_CueWord", (), {
                "text": token,
                "start": beat.start + span * ((index + 1) / (len(tokens) + 1)),
                "end": beat.start + span * ((index + 1.35) / (len(tokens) + 1)),
            })()
            for index, token in enumerate(tokens)
        ]
    cues: list[AudioMotionCue] = []
    selected_words = _select_cue_words(words, limit=7)
    for index, word in enumerate(selected_words, start=1):
        label = _clean_label(getattr(word, "text", "cue"))
        start = max(float(getattr(word, "start", beat.start)), beat.start)
        end = min(max(float(getattr(word, "end", start + 0.18)), start + 0.08), beat.end)
        cues.append(
            AudioMotionCue(
                cue_id=f"{beat.beat_id}_cue_{index:02d}",
                beat_id=beat.beat_id,
                label=label,
                start=round(start, 3),
                end=round(end, 3),
                band=_band_for(label, index),
                strength=_strength_for(label, index),
            )
        )
    return cues


def _select_cue_words(words: list[Any], *, limit: int) -> list[Any]:
    candidates = [
        word
        for word in words
        if _word_weight(str(getattr(word, "text", ""))) > 0
    ]
    ranked = sorted(
        candidates,
        key=lambda word: (
            _word_weight(str(getattr(word, "text", ""))),
            float(getattr(word, "end", 0.0)) - float(getattr(word, "start", 0.0)),
        ),
        reverse=True,
    )
    selected = sorted(ranked[:limit], key=lambda word: float(getattr(word, "start", 0.0)))
    return selected


def _word_weight(word: str) -> float:
    cleaned = _clean_label(word)
    if not cleaned:
        return 0.0
    if cleaned.lower() in _LOW_VALUE_CUE_WORDS:
        return 0.0
    weight = min(len(cleaned) / 10.0, 1.0)
    if re.search(r"\d|%|x|ms|gb|mb", cleaned, flags=re.IGNORECASE):
        weight += 0.45
    if cleaned.lower() in {
        "attention",
        "sparse",
        "reasoning",
        "focused",
        "compress",
        "route",
        "mask",
        "signal",
        "proof",
        "breakthrough",
    }:
        weight += 0.35
    return weight


def _band_for(label: str, index: int) -> str:
    lower = label.lower()
    if re.search(r"\d|%|x|ms|gb|mb", lower):
        return "bass"
    if lower in {"reasoning", "attention", "signal", "proof", "focused"}:
        return "mids"
    if index % 3 == 0:
        return "treble"
    if index % 2 == 0:
        return "mids"
    return "amplitude"


def _strength_for(label: str, index: int) -> float:
    base = 0.52 + min(len(label), 12) * 0.025
    if re.search(r"\d|%|x|ms|gb|mb", label, flags=re.IGNORECASE):
        base += 0.14
    base += (index % 3) * 0.035
    return round(max(0.45, min(base, 0.94)), 3)


def _clean_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9%+./-]+", "", str(value or "")).strip(" .,:;")


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


__all__ = [
    "AudioMotionCue",
    "BeatMotionProfile",
    "MOTION_PLAN_VERSION",
    "MotionPlan",
    "build_motion_plan",
]
