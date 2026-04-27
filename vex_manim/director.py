from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from broll_intelligence import call_reasoning_model, extract_json_object, truncate
from vex_manim.blueprint import SceneBlueprint
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


RUNTIME_HELPER_NAMES = {
    "apply_house_background",
    "make_title_block",
    "make_pill",
    "make_glass_panel",
    "make_signal_node",
    "make_connector",
    "make_glow_dot",
    "make_orbit_ring",
    "make_route_path",
    "make_focus_beam",
    "make_metric_badge",
    "make_ribbon_label",
    "fit_text",
    "theme_color",
    "camera_focus",
    "register_layout_group",
    "register_text_group",
    "register_panel_group",
    "stagger_fade_in",
}

RATE_FUNCTION_NAMES = {
    "ease_in_sine",
    "ease_out_sine",
    "ease_in_out_sine",
    "ease_in_quad",
    "ease_out_quad",
    "ease_in_out_quad",
    "ease_in_cubic",
    "ease_out_cubic",
    "ease_in_out_cubic",
    "ease_in_quart",
    "ease_out_quart",
    "ease_in_out_quart",
    "ease_in_quint",
    "ease_out_quint",
    "ease_in_out_quint",
    "ease_in_expo",
    "ease_out_expo",
    "ease_in_out_expo",
    "ease_in_circ",
    "ease_out_circ",
    "ease_in_out_circ",
    "ease_in_back",
    "ease_out_back",
    "ease_in_out_back",
    "ease_in_bounce",
    "ease_out_bounce",
    "ease_in_out_bounce",
    "ease_in_elastic",
    "ease_out_elastic",
    "ease_in_out_elastic",
}

ANIMATION_CALL_NAMES = {
    "AnimationGroup",
    "Create",
    "FadeIn",
    "FadeOut",
    "FadeTransform",
    "GrowFromCenter",
    "GrowFromEdge",
    "LaggedStart",
    "MoveAlongPath",
    "ReplacementTransform",
    "Succession",
    "Transform",
    "TransformMatchingShapes",
    "Write",
}

UNSUPPORTED_ANIMATION_KWARGS = {
    "accent",
    "accent_color",
    "color",
    "fill",
    "fill_color",
    "fill_opacity",
    "glow_color",
    "glow_opacity",
    "opacity",
    "stroke_color",
    "stroke_opacity",
}

POINT_LITERAL_NAMES = {"UP", "DOWN", "LEFT", "RIGHT", "ORIGIN", "UL", "UR", "DL", "DR"}
POINT_FUNCTION_NAMES = {"interpolate", "midpoint"}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _is_numeric_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_numeric_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_numeric_expr(node.left) and _is_numeric_expr(node.right)
    return False


def _is_point_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in POINT_LITERAL_NAMES
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts) in {2, 3} and all(_is_numeric_expr(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        return _is_point_expression(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _is_point_expression(node.left) and _is_point_expression(node.right)
        if isinstance(node.op, (ast.Mult, ast.Div)):
            return (
                (_is_point_expression(node.left) and _is_numeric_expr(node.right))
                or (_is_numeric_expr(node.left) and _is_point_expression(node.right))
            )
    if isinstance(node, ast.Call):
        call_name = _call_name(node.func)
        return (
            call_name.split(".")[-1] in POINT_FUNCTION_NAMES
            or call_name.endswith(".c2p")
            or call_name.endswith(".point_from_proportion")
        )
    return False


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


def _blueprint_block(blueprint: SceneBlueprint, alternatives: list[SceneBlueprint]) -> str:
    lines = [
        "Selected scene blueprint:",
        blueprint.to_prompt_block(),
    ]
    if alternatives:
        lines.append("\nNearby alternatives for contrast only:")
        lines.extend(
            f"- {item.blueprint_id}: {item.archetype} | focal={item.focal_system} | camera={item.camera_plan}"
            for item in alternatives
        )
    return "\n".join(lines)


def _system_prompt() -> str:
    return (
        "You are a principal motion designer and senior Manim engineer writing production-quality animation code. "
        "Use the full expressive power of Manim when it meaningfully improves the scene: camera motion, trackers, transforms, charts, path animation, kinetic typography, morphs, and elegant choreography. "
        "Compose in layers so the scene has atmosphere, structure, and clear focal annotation. "
        "Do not write generic text-on-box layouts unless the selected blueprint explicitly calls for interface modules. "
        "You must output ONLY a JSON object with keys summary, features, scene_code. "
        "scene_code must define exactly one class named GeneratedScene that subclasses VexGeneratedScene. "
        "Use real Manim constructs and keep the code self-contained. "
        "Every VexGeneratedScene helper must be called as self.helper_name(...), never as a bare helper_name(...). "
        "Forbidden: filesystem access, network access, subprocess calls, os/sys/pathlib/shutil usage, eval/exec/open, or any code outside of animation needs. "
        "Assume SCENE_SPEC and SCENE_BRIEF globals exist, and that VexGeneratedScene already provides themed helpers like apply_house_background, make_title_block, make_pill, make_glass_panel, make_signal_node, make_connector, make_glow_dot, make_orbit_ring, make_route_path, make_focus_beam, make_metric_badge, make_ribbon_label, fit_text, camera_focus, and register_layout_group."
    )


def _user_prompt(
    brief: SceneBrief,
    examples: list[SceneExample],
    skills: list[SkillSlice],
    blueprint: SceneBlueprint,
    *,
    alternative_blueprints: list[SceneBlueprint] | None = None,
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
    contract_block = "\n".join(f"- {item}" for item in brief.scene_contract)
    return (
        "Scene brief:\n"
        f"{_brief_block(brief)}\n\n"
        "Relevant Manim skill slices:\n"
        f"{_skills_block(skills)}\n\n"
        f"{_blueprint_block(blueprint, list(alternative_blueprints or []))}\n\n"
        "Retrieved scene examples:\n"
        f"{_examples_block(examples)}\n\n"
        "Scene contract:\n"
        f"{contract_block}\n\n"
        "Hard requirements:\n"
        "- Start from VexGeneratedScene and build a real animated scene.\n"
        "- Add the title treatment with make_title_block unless the scene has a stronger editorial framing.\n"
        "- Call runtime helpers as self.make_title_block(...), self.make_orbit_ring(...), self.camera_focus(...), and so on; never use bare helper calls.\n"
        "- Honor the selected blueprint's focal system, motion beats, and element roles; do not collapse it into generic panels or stacked text boxes.\n"
        "- If the blueprint uses a route, orbit, bridge, ladder, sweep, or focus lane, that motion spine must remain visible in the final scene.\n"
        "- Register the principal visible groups with register_layout_group(name, group, role=...) so runtime layout guardrails can keep the scene clean.\n"
        "- Register at least a title/hero group and one or two supporting groups whenever they exist.\n"
        "- Keep the pacing within the target duration.\n"
        f"- Keep simultaneous visible copy under roughly {brief.text_budget_words} words; use short labels, badges, and support lines instead of transcript-like paragraphs.\n"
        "- Use at least two advanced Manim techniques when the brief intensity is medium or high.\n"
        f"- Include at least {brief.minimum_dynamic_devices} dynamic devices from this set when appropriate: camera reframing, trackers, redraw-driven motion, morphs, path travel, signal trails, orbit rings, focus beams, or staged transforms.\n"
        "- Make the scene read in three layers: atmosphere, structure, and annotation.\n"
        "- For premium replace scenes, default to paths, motion systems, morphs, axes, orbit rings, signal flow, layered depth, or tracked geometry before reaching for glass panels.\n"
        "- Avoid more than one card or panel container unless the blueprint explicitly calls for interface modules or a bounded surface.\n"
        "- Avoid plain repeated cards or transcript parroting.\n"
        "- Prefer elegant asymmetry, guided focus, and meaningful motion.\n"
        "- scene_code must be valid Python with no markdown fences."
        f"{latex_note}"
        f"{feedback_block}"
    )


def _recover_scene_code_from_raw_text(raw_text: str) -> str:
    code_fence_match = re.search(r"```(?:python)?\s*(?P<code>[\s\S]*?)```", raw_text, re.IGNORECASE)
    if code_fence_match:
        code = str(code_fence_match.group("code") or "").strip()
        if "class GeneratedScene" in code:
            return code
    class_match = re.search(r"(class\s+GeneratedScene\b[\s\S]+)", raw_text)
    if class_match:
        return str(class_match.group(1) or "").strip()
    return ""


def _repair_scene_code(scene_code: str) -> str:
    cleaned = str(scene_code or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^```(?:python)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        try:
            unwrapped = ast.literal_eval(cleaned)
        except Exception:
            unwrapped = None
        if isinstance(unwrapped, str) and "class GeneratedScene" in unwrapped:
            cleaned = unwrapped.strip()
    if "\\n" in cleaned:
        try:
            decoded = bytes(cleaned, "utf-8").decode("unicode_escape").strip()
        except Exception:
            decoded = cleaned
        if "class GeneratedScene" in decoded and decoded.count("\n") >= cleaned.count("\n"):
            cleaned = decoded
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return cleaned

    class HelperQualifier(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and len(node.args) == 1:
                base = node.func.value
                if node.func.attr == "shift" and _is_point_expression(base):
                    return ast.BinOp(left=base, op=ast.Add(), right=node.args[0])
                if node.func.attr == "scale" and _is_point_expression(base) and _is_numeric_expr(node.args[0]):
                    return ast.BinOp(left=base, op=ast.Mult(), right=node.args[0])
            if isinstance(node.func, ast.Name) and node.func.id in RUNTIME_HELPER_NAMES:
                node.func = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=node.func.id, ctx=ast.Load())
            call_name = _call_name(node.func)
            short_name = call_name.split(".")[-1]
            for keyword in node.keywords:
                if keyword.arg == "font_weight":
                    keyword.arg = "weight"
                elif keyword.arg == "font_style":
                    keyword.arg = "slant"
            if short_name in ANIMATION_CALL_NAMES:
                node.keywords = [
                    keyword
                    for keyword in node.keywords
                    if keyword.arg not in UNSUPPORTED_ANIMATION_KWARGS
                ]
            if short_name == "make_pill":
                has_text = bool(node.args) or any(keyword.arg == "text" for keyword in node.keywords)
                if not has_text:
                    node.args.insert(0, ast.Constant(value=""))
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in RATE_FUNCTION_NAMES:
                return ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="manim", ctx=ast.Load()), attr="rate_functions", ctx=ast.Load()),
                    attr=node.id,
                    ctx=ast.Load(),
                )
            return node

    repaired_tree = ast.fix_missing_locations(HelperQualifier().visit(tree))
    try:
        return ast.unparse(repaired_tree).strip()
    except Exception:
        return cleaned


def _parse_candidate(raw_text: str) -> SceneCandidate:
    payload: dict[str, Any] = {}
    extracted_object = ""
    try:
        extracted_object = extract_json_object(raw_text)
    except Exception:
        extracted_object = ""
    if extracted_object:
        try:
            parsed = json.loads(extracted_object)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(extracted_object)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = {}
    scene_code = str(payload.get("scene_code") or "").strip()
    if not scene_code:
        scene_code = _recover_scene_code_from_raw_text(raw_text)
    scene_code = _repair_scene_code(scene_code)
    if not scene_code:
        raise ValueError("Could not recover GeneratedScene code from model response.")
    summary = truncate(str(payload.get("summary") or "Generated Manim scene."), 220)
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        raw_features = []
    features = [truncate(str(item), 40) for item in raw_features if str(item).strip()][:10]
    return SceneCandidate(summary=summary, features=features, scene_code=scene_code)


def request_scene_candidate(
    provider_name: str,
    model_name: str,
    brief: SceneBrief,
    examples: list[SceneExample],
    blueprint: SceneBlueprint,
    *,
    alternative_blueprints: list[SceneBlueprint] | None = None,
    previous_code: str | None = None,
    feedback_lines: list[str] | None = None,
) -> SceneCandidate:
    skill_limit = 2 if brief.animation_intensity == "low" else 4
    if brief.scene_family in {"kinetic_quote", "kinetic_stack", "dashboard_build"}:
        skill_limit = min(skill_limit, 3)
    skills = retrieve_skill_slices(
        brief,
        limit=skill_limit,
        preferred_features=blueprint.suggested_features,
    )
    raw_text = call_reasoning_model(
        provider_name,
        model_name,
        _system_prompt(),
        _user_prompt(
            brief,
            examples,
            skills,
            blueprint,
            alternative_blueprints=alternative_blueprints,
            previous_code=previous_code,
            feedback_lines=feedback_lines,
        ),
    )
    return _parse_candidate(raw_text)


def write_generation_report(
    target_path: Path,
    *,
    brief: SceneBrief,
    blueprint_candidates: list[SceneBlueprint],
    selected_blueprint: SceneBlueprint | None,
    selected_examples: list[SceneExample],
    attempts: list[dict[str, Any]],
    final_candidate: SceneCandidate | None,
    final_scene_code: str | None,
    quality_score: float | None,
    fallback_used: bool,
) -> None:
    payload = {
        "scene_brief": brief.to_dict(),
        "blueprint_candidates": [item.to_dict() for item in blueprint_candidates],
        "selected_blueprint": selected_blueprint.to_dict() if selected_blueprint else None,
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
