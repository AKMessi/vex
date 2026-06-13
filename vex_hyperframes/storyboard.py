from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from visual_explanation import VisualExplanationIR


@dataclass(frozen=True)
class StoryboardPanel:
    panel_id: str
    phase: str
    start_fraction: float
    end_fraction: float
    focus_object_id: str
    visible_object_ids: list[str]
    visual_change: str
    semantic_purpose: str
    camera_instruction: str
    final_frame_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryboardReview:
    passed: bool
    score: float
    fatal_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 3)
        return payload


_CAMERA_BY_SCENE = {
    "architecture_flow": "Follow the active signal without leaving the full service path unreadable.",
    "causal_intervention": "Hold the causal spine; reframe only when the intervention changes the outcome.",
    "decision_branch": "Stay centered on the gate, then bias toward the active branch.",
    "evidence_backed_quote": "Use a restrained push toward the decisive phrase.",
    "grounded_interface_walkthrough": "Punch into the active interface state, then return enough context to show the result.",
    "guided_process": "Track the traveler in reading order while preserving completed steps.",
    "matched_state_transform": "Keep matched geometry registered so the viewer can inspect what changed.",
    "metric_delta": "Keep both measured states visible through the comparison.",
    "metric_intervention": "Hold the before value, reveal the intervention, then land on the after value.",
    "metric_proof": "Move from evidence geometry to the measured claim.",
    "narrative_progression": "Advance through setup, turn, and payoff without resetting the spatial frame.",
    "set_partition": "Keep the original token set visible while it resolves into grouped compressed blocks.",
}


def build_storyboard(ir: VisualExplanationIR) -> list[StoryboardPanel]:
    if ir.render_policy != "render":
        return []
    object_ids = [item.object_id for item in ir.objects]
    object_by_id = {item.object_id: item for item in ir.objects}
    panels: list[StoryboardPanel] = []
    visible: list[str] = []
    for index, beat in enumerate(ir.beats, start=1):
        if beat.subject_id not in visible:
            visible.append(beat.subject_id)
        focus = object_by_id.get(beat.subject_id)
        target = object_by_id.get(beat.target_id)
        change = _visual_change(
            action=beat.action,
            focus_label=focus.label if focus else beat.subject_id,
            target_label=target.label if target else "",
        )
        panels.append(
            StoryboardPanel(
                panel_id=f"panel_{index:02d}",
                phase=beat.phase,
                start_fraction=round(float(beat.start_fraction), 3),
                end_fraction=round(float(beat.end_fraction), 3),
                focus_object_id=beat.subject_id,
                visible_object_ids=list(visible),
                visual_change=change,
                semantic_purpose=_semantic_purpose(ir.scene_type, beat.phase),
                camera_instruction=_CAMERA_BY_SCENE.get(
                    ir.scene_type,
                    "Hold a stable explanatory frame and move attention through object emphasis.",
                ),
                final_frame_requirement=_final_frame_requirement(ir),
            )
        )
    return panels


def review_storyboard(ir: VisualExplanationIR, panels: list[StoryboardPanel]) -> StoryboardReview:
    if ir.render_policy != "render":
        return StoryboardReview(
            passed=True,
            score=1.0,
            strengths=["The unsupported visual was rejected before rendering."],
        )
    fatal: list[str] = []
    warnings: list[str] = []
    strengths: list[str] = []
    object_ids = {item.object_id for item in ir.objects}
    covered_objects = {
        object_id
        for panel in panels
        for object_id in panel.visible_object_ids
    }
    if len(panels) < 2:
        fatal.append("storyboard_has_fewer_than_two_semantic_beats")
    if panels and panels[0].start_fraction > 0.15:
        warnings.append("storyboard_establishes_the_idea_too_late")
    if panels and panels[-1].end_fraction < 0.82:
        warnings.append("storyboard_does_not_hold_the_resolved_state")
    if object_ids - covered_objects:
        fatal.append("storyboard_omits_required_objects")
    if any(panel.end_fraction <= panel.start_fraction for panel in panels):
        fatal.append("storyboard_contains_invalid_time_window")
    if any(panel.focus_object_id not in object_ids for panel in panels):
        fatal.append("storyboard_focuses_unknown_object")
    if len({panel.visual_change for panel in panels}) < min(2, len(panels)):
        warnings.append("storyboard_repeats_one_visual_change")
    if not fatal:
        strengths.append("Every grounded object is introduced by a timed semantic beat.")
    if panels and panels[-1].final_frame_requirement:
        strengths.append("The resolved frame has an explicit screenshot test.")
    score = 1.0 - (0.34 * len(fatal)) - (0.08 * len(warnings))
    score = max(0.0, min(score, 1.0))
    return StoryboardReview(
        passed=not fatal and score >= 0.72,
        score=score,
        fatal_issues=fatal,
        warnings=warnings,
        strengths=strengths,
    )


def _visual_change(*, action: str, focus_label: str, target_label: str) -> str:
    target = f" toward {target_label}" if target_label else ""
    descriptions = {
        "activate_branch": f"Activate the branch labeled {focus_label}{target}.",
        "advance_route": f"Move the route traveler through {focus_label}{target}.",
        "advance_story": f"Transform the story state at {focus_label}{target}.",
        "compare_metric": f"Register the measured state {focus_label}{target}.",
        "focus_interface_state": f"Focus the real interface state {focus_label}{target}.",
        "lock_focus": f"Hold {focus_label} as the resolved takeaway.",
        "propagate_cause": f"Propagate the causal signal from {focus_label}{target}.",
        "reveal": f"Reveal {focus_label} as the first grounded object.",
        "reveal_evidence": f"Reveal the evidence attached to {focus_label}{target}.",
        "route_request": f"Route the active request through {focus_label}{target}.",
        "trace_metric_change": f"Trace the measured change at {focus_label}{target}.",
        "transform_state": f"Morph the registered state {focus_label}{target}.",
        "partition_set": f"Group {focus_label} into the source-backed compressed blocks{target}.",
    }
    return descriptions.get(action, f"Make the semantic change for {focus_label}{target} visible.")


def _semantic_purpose(scene_type: str, phase: str) -> str:
    purpose = {
        "establish": "Establish the concrete state or claim the viewer must inspect.",
        "explain": "Show the mechanism, route, branch, or intervention that changes the state.",
        "resolve": "Land on source-backed proof or a concrete resulting state.",
    }.get(phase, "Advance the explanation without introducing unsupported content.")
    if scene_type == "evidence_backed_quote":
        return "Preserve the exact source language and focus attention on its decisive phrase."
    return purpose


def _final_frame_requirement(ir: VisualExplanationIR) -> str:
    labels = ", ".join(ir.required_labels[:6])
    return (
        f"A paused resolved frame must make the relationship between {labels} understandable "
        "without relying on the transcript."
    )


__all__ = [
    "StoryboardPanel",
    "StoryboardReview",
    "build_storyboard",
    "review_storyboard",
]
