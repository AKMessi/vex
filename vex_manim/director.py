from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from broll_intelligence import call_reasoning_model, extract_json_object, truncate
from vex_manim.briefs import SceneBrief
from vex_manim.scene_library import SceneExample
from vex_manim.skill_pack import SkillSlice, retrieve_skill_slices


@dataclass
class SceneCandidate:
    summary: str
    features: list[str]
    scene_code: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _examples_block(examples: list[SceneExample]) -> str:
    if not examples:
        return "No retrieval examples were found."
    return "\n\n".join(example.to_prompt_block() for example in examples)


def _brief_block(brief: SceneBrief) -> str:
    payload = brief.to_dict()
    return json.dumps(payload, indent=2)


def _skills_block(skills: list[SkillSlice]) -> str:
    if not skills:
        return "No additional skill slices were selected."
    return "\n\n".join(skill.to_prompt_block() for skill in skills)


def _system_prompt() -> str:
    return (
        "You are a principal motion designer and senior Manim engineer writing production-quality animation code. "
        "Use the full expressive power of Manim when it meaningfully improves the scene: camera motion, trackers, transforms, charts, path animation, kinetic typography, morphs, and elegant choreography. "
        "Do not write generic text-on-box layouts unless the brief truly demands restraint. "
        "You must output ONLY a JSON object with keys summary, features, scene_code. "
        "scene_code must define exactly one class named GeneratedScene that subclasses VexGeneratedScene. "
        "Use real Manim constructs and keep the code self-contained. "
        "Forbidden: filesystem access, network access, subprocess calls, os/sys/pathlib/shutil usage, eval/exec/open, or any code outside of animation needs. "
        "Assume SCENE_SPEC and SCENE_BRIEF globals exist, and that VexGeneratedScene already provides themed helpers like apply_house_background, make_title_block, make_pill, make_glass_panel, make_signal_node, make_connector, make_glow_dot, make_orbit_ring, make_route_path, make_focus_beam, make_metric_badge, make_ribbon_label, fit_text, camera_focus, and register_layout_group."
    )


def _user_prompt(
    brief: SceneBrief,
    examples: list[SceneExample],
    skills: list[SkillSlice],
    *,
    previous_code: str | None = None,
    feedback_lines: list[str] | None = None,
) -> str:
    feedback_block = ""
    if previous_code and feedback_lines:
        feedback_block = (
            "\n\nPrevious attempt that needs repair:\n"
            f"{truncate(previous_code, 7000)}\n\n"
            "Fix these issues:\n"
            + "\n".join(f"- {item}" for item in feedback_lines)
        )
    latex_note = ""
    if not bool(brief.render_constraints.get("latex_available", True)):
        latex_note = (
            "\n- LaTeX is NOT available in this runtime. Avoid Tex, MathTex, DecimalNumber, BarChart, Integer, and any TeX-dependent mobjects or default chart labels that route through MathTex."
        )
    return (
        "Scene brief:\n"
        f"{_brief_block(brief)}\n\n"
        "Relevant Manim skill slices:\n"
        f"{_skills_block(skills)}\n\n"
        "Retrieved scene examples:\n"
        f"{_examples_block(examples)}\n\n"
        "Hard requirements:\n"
        "- Start from VexGeneratedScene and build a real animated scene.\n"
        "- Add the title treatment with make_title_block unless the scene has a stronger editorial framing.\n"
        "- Register the principal visible groups with register_layout_group(name, group, role=...) so runtime layout guardrails can keep the scene clean.\n"
        "- Register at least a title/hero group and one or two supporting groups whenever they exist.\n"
        "- Keep the pacing within the target duration.\n"
        "- Use at least two advanced Manim techniques when the brief intensity is medium or high.\n"
        "- For premium replace scenes, default to paths, motion systems, morphs, axes, orbit rings, signal flow, layered depth, or tracked geometry before reaching for glass panels.\n"
        "- Avoid more than two card or panel containers unless the brief is explicitly a product interface scene.\n"
        "- Avoid plain repeated cards or transcript parroting.\n"
        "- Prefer elegant asymmetry, guided focus, and meaningful motion.\n"
        "- scene_code must be valid Python with no markdown fences."
        f"{latex_note}"
        f"{feedback_block}"
    )


def _parse_candidate(raw_text: str) -> SceneCandidate:
    payload = json.loads(extract_json_object(raw_text))
    scene_code = str(payload.get("scene_code") or "").strip()
    summary = truncate(str(payload.get("summary") or "Generated Manim scene."), 220)
    features = [truncate(str(item), 40) for item in payload.get("features", []) if str(item).strip()][:10]
    return SceneCandidate(summary=summary, features=features, scene_code=scene_code)


def request_scene_candidate(
    provider_name: str,
    model_name: str,
    brief: SceneBrief,
    examples: list[SceneExample],
    *,
    previous_code: str | None = None,
    feedback_lines: list[str] | None = None,
) -> SceneCandidate:
    skills = retrieve_skill_slices(brief, limit=3)
    raw_text = call_reasoning_model(
        provider_name,
        model_name,
        _system_prompt(),
        _user_prompt(
            brief,
            examples,
            skills,
            previous_code=previous_code,
            feedback_lines=feedback_lines,
        ),
    )
    return _parse_candidate(raw_text)


def write_generation_report(
    target_path: Path,
    *,
    brief: SceneBrief,
    selected_examples: list[SceneExample],
    attempts: list[dict[str, Any]],
    final_candidate: SceneCandidate | None,
    final_scene_code: str | None,
    quality_score: float | None,
    fallback_used: bool,
) -> None:
    payload = {
        "scene_brief": brief.to_dict(),
        "selected_examples": [
            {
                "example_id": example.example_id,
                "scene_family": example.scene_family,
                "tags": list(example.tags),
                "source": example.source,
            }
            for example in selected_examples
        ],
        "attempts": attempts,
        "summary": final_candidate.summary if final_candidate else "",
        "final_features": list(final_candidate.features) if final_candidate else [],
        "final_scene_code": final_scene_code or "",
        "quality_score": quality_score,
        "fallback_used": fallback_used,
    }
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
