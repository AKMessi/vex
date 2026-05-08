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


@dataclass
class PlannedElement:
    element_id: str
    role: str
    treatment: str
    copy_lines: list[str]
    source_hint: str = ""
    layout_intent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannedBeat:
    beat_id: str
    focus: str
    story_goal: str
    motion: str
    camera: str
    visible_elements: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneExecutionPlan:
    summary: str
    intuition_thesis: str
    visual_logic: str
    motion_spine: str
    title_text: str
    deck_text: str
    layout_rules: list[str]
    guardrails: list[str]
    advanced_devices: list[str]
    element_plan: list[PlannedElement]
    beat_plan: list[PlannedBeat]
    source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "intuition_thesis": self.intuition_thesis,
            "visual_logic": self.visual_logic,
            "motion_spine": self.motion_spine,
            "title_text": self.title_text,
            "deck_text": self.deck_text,
            "layout_rules": list(self.layout_rules),
            "guardrails": list(self.guardrails),
            "advanced_devices": list(self.advanced_devices),
            "element_plan": [item.to_dict() for item in self.element_plan],
            "beat_plan": [item.to_dict() for item in self.beat_plan],
            "source": self.source,
        }

    def to_prompt_block(self) -> str:
        lines = [
            f"Summary: {self.summary}",
            f"Intuition thesis: {self.intuition_thesis}",
            f"Visual logic: {self.visual_logic}",
            f"Motion spine: {self.motion_spine}",
            f"Title text: {self.title_text or '(none)'}",
        ]
        if self.deck_text:
            lines.append(f"Deck text: {self.deck_text}")
        lines.append("Layout rules:")
        lines.extend(f"- {item}" for item in self.layout_rules)
        lines.append("Element assignments:")
        lines.extend(
            f"- {item.element_id}: role={item.role}, treatment={item.treatment}, copy={item.copy_lines or ['(none)']}, layout={item.layout_intent or 'n/a'}"
            for item in self.element_plan
        )
        lines.append("Beat plan:")
        lines.extend(
            f"- {item.beat_id}: focus={item.focus}; goal={item.story_goal}; motion={item.motion}; camera={item.camera}; visible={', '.join(item.visible_elements)}"
            for item in self.beat_plan
        )
        if self.guardrails:
            lines.append("Guardrails:")
            lines.extend(f"- {item}" for item in self.guardrails)
        if self.advanced_devices:
            lines.append(f"Advanced devices: {', '.join(self.advanced_devices)}")
        return "\n".join(lines)


RUNTIME_HELPER_NAMES = {
    "apply_house_background",
    "build_layout",
    "layout_slot",
    "slot_point",
    "slot_anchor",
    "layout_route_points",
    "place_in_slot",
    "fit_text_for_slot",
    "route_between_slots",
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
    "sine_in",
    "sine_out",
    "sine_in_out",
}

RATE_FUNCTION_ALIASES = {
    "sine_in": "ease_in_sine",
    "sine_out": "ease_out_sine",
    "sine_in_out": "ease_in_out_sine",
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
POINT_ACCESSOR_SUFFIXES = {
    ".get_center",
    ".get_left",
    ".get_right",
    ".get_top",
    ".get_bottom",
    ".get_corner",
    ".point_at_angle",
}


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
            or any(call_name.endswith(suffix) for suffix in POINT_ACCESSOR_SUFFIXES)
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


def _execution_plan_block(plan: SceneExecutionPlan) -> str:
    return plan.to_prompt_block()


def _intuition_block(brief: SceneBrief) -> str:
    lines = [
        f"Mode: {brief.intuition_mode or 'general'}",
        f"Mental model: {brief.mental_model or brief.objective}",
    ]
    if brief.before_state:
        lines.append(f"Before state: {brief.before_state}")
    if brief.after_state:
        lines.append(f"After state: {brief.after_state}")
    if brief.cause:
        lines.append(f"Cause: {brief.cause}")
    if brief.effect:
        lines.append(f"Effect: {brief.effect}")
    if brief.viewer_takeaway:
        lines.append(f"Viewer takeaway: {brief.viewer_takeaway}")
    if brief.visual_metaphor:
        lines.append(f"Suggested visual metaphor: {brief.visual_metaphor}")
    if brief.story_window:
        lines.append(f"Story window: {truncate(brief.story_window, 220)}")
    return "\n".join(lines)


def _condense_copy(text: str, *, max_words: int, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").replace("\n", " ")).strip(" -:;,")
    cleaned = re.sub(r"\$?\s*\\?(?:right)?arrow\s*\$?", " -> ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,")
    if not cleaned:
        return ""
    fragments = re.split(r"[.;:!?]|(?:\s+-\s+)", cleaned)
    candidate = next((fragment.strip() for fragment in fragments if fragment.strip()), cleaned)
    words = candidate.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words]).strip()
    if len(candidate) > max_chars:
        candidate = candidate[: max_chars - 1].rstrip(" ,;:-") + "…"
    return candidate.strip()


def _unique_nonempty(items: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        values.append(text)
    return values


def _sanitize_plan_text(text: str, *, max_chars: int) -> str:
    cleaned = str(text or "").replace("\\n", " ")
    cleaned = re.sub(r"\$?\s*\\?(?:right)?arrow\s*\$?", " -> ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,")
    return truncate(cleaned, max_chars)


def _copy_candidates(brief: SceneBrief, source_hint: str, role: str) -> list[str]:
    bank = dict(brief.copy_bank or {})
    if brief.intuition_mode == "process_route" and role in {"chart", "diagram", "hero"}:
        route_bits = _unique_nonempty(
            [
                brief.before_state,
                brief.after_state,
                *[str(item).strip() for item in list(bank.get("steps") or []) if str(item).strip()],
                brief.viewer_takeaway,
            ]
        )
        if route_bits:
            return route_bits
    if role in {"chart", "diagram", "hero"} and bank.get("steps"):
        return [str(item).strip() for item in list(bank.get("steps") or []) if str(item).strip()]
    semantic_pairs = {
        "headline": [brief.headline],
        "deck": [brief.deck],
        "left_detail": [bank.get("left_detail") or brief.before_state],
        "right_detail": [bank.get("right_detail") or brief.after_state],
        "sentence_text": [bank.get("sentence_text") or brief.spoken_anchor],
        "supporting_lines": list(bank.get("supporting_lines") or []),
        "steps": list(bank.get("steps") or []),
        "keywords": list(bank.get("keywords") or []),
        "footer_text": [bank.get("footer_text") or brief.viewer_takeaway],
    }
    values = semantic_pairs.get(str(source_hint or "").strip(), [])
    if not values:
        values = [brief.viewer_takeaway, brief.after_state, brief.headline, brief.deck]
    return [str(item).strip() for item in values if str(item or "").strip()]


def _copy_lines_for_element(brief: SceneBrief, role: str, source_hint: str) -> list[str]:
    max_words = 3 if role in {"metric", "label", "chip"} else 5
    if role == "title":
        max_words = min(max(4, brief.text_budget_words // 2), 8)
    if role in {"hero", "diagram"}:
        max_words = 4
    candidates = _copy_candidates(brief, source_hint, role)
    chosen = [_condense_copy(item, max_words=max_words, max_chars=44 if role != "title" else 72) for item in candidates]
    chosen = _unique_nonempty([item for item in chosen if item])
    if role == "title":
        if chosen:
            return chosen[:1]
        return [_condense_copy(brief.headline or brief.viewer_takeaway, max_words=max_words, max_chars=72)]
    if role == "metric":
        return chosen[:1]
    if role in {"hero", "diagram"}:
        return chosen[:2]
    return chosen[:2]


def _role_treatment(role: str, kind: str) -> str:
    if role == "title":
        return "editorial_band"
    if role == "metric":
        return "counter_or_badge"
    if role == "chart":
        return "evidence_geometry"
    if role == "hero" and "state" in kind:
        return "state_cluster"
    if role == "hero":
        return "hero_focus"
    if role == "diagram":
        return "signal_or_route_label"
    if role == "background":
        return "ambient_depth"
    return kind or "support_label"


def _layout_intent(element: Any) -> str:
    placement = str(getattr(element, "placement", "") or "").replace("_", " ")
    motion = str(getattr(element, "motion", "") or "").replace("_", " ")
    return truncate(f"{placement}; motion={motion}", 80)


def build_deterministic_execution_plan(brief: SceneBrief, blueprint: SceneBlueprint) -> SceneExecutionPlan:
    title_text = _condense_copy(brief.headline or brief.viewer_takeaway or brief.spoken_anchor, max_words=min(max(4, brief.text_budget_words // 2), 8), max_chars=72)
    deck_basis = brief.viewer_takeaway or brief.after_state or brief.deck
    deck_text = _condense_copy(deck_basis, max_words=8, max_chars=76)
    element_plan: list[PlannedElement] = []
    for element in blueprint.elements:
        if str(element.role or "").strip().lower() == "background":
            copy_lines: list[str] = []
        else:
            copy_lines = _copy_lines_for_element(brief, str(element.role or "").strip().lower(), str(element.copy_source or "").strip())
        if str(element.role or "").strip().lower() == "title" and title_text:
            copy_lines = [title_text]
        element_plan.append(
            PlannedElement(
                element_id=str(element.element_id or ""),
                role=str(element.role or ""),
                treatment=_role_treatment(str(element.role or ""), str(element.kind or "")),
                copy_lines=copy_lines,
                source_hint=str(element.copy_source or ""),
                layout_intent=_layout_intent(element),
            )
        )
    beat_plan = [
        PlannedBeat(
            beat_id=str(beat.beat_id or f"beat_{index + 1}"),
            focus=str(beat.focus or ""),
            story_goal=_condense_copy(str(beat.purpose or ""), max_words=10, max_chars=90),
            motion=_condense_copy(" / ".join(str(item).replace("_", " ") for item in beat.actions), max_words=12, max_chars=90),
            camera=_condense_copy(blueprint.camera_plan, max_words=12, max_chars=90),
            visible_elements=[
                item.element_id
                for item in element_plan
                if item.element_id == beat.focus or item.role in {"title", "hero", "metric"}
            ][:4],
        )
        for index, beat in enumerate(blueprint.motion_beats)
    ]
    layout_rules = [
        "Keep the title compact and anchored; do not let it drift into the evidence geometry.",
        "Let the motion spine stay visible throughout the scene, even while labels animate.",
        "Use short labels only; prefer chips, badges, and side labels over transcript fragments.",
    ]
    if brief.before_state and brief.after_state:
        layout_rules.append("Make the before/after contrast readable at a glance with clear spatial separation.")
    guardrails = [
        "Never put paragraph-length copy inside small shapes.",
        "Do not repeat the same sentence in the title and support labels.",
        "Do not let background depth objects compete with the focal system.",
    ]
    return SceneExecutionPlan(
        summary=truncate(f"{brief.objective} through {blueprint.archetype.replace('_', ' ')}", 180),
        intuition_thesis=brief.mental_model or brief.viewer_takeaway or brief.objective,
        visual_logic=blueprint.rationale,
        motion_spine=blueprint.focal_system,
        title_text=title_text,
        deck_text=deck_text,
        layout_rules=layout_rules,
        guardrails=guardrails,
        advanced_devices=list(blueprint.dynamic_devices[: max(brief.minimum_dynamic_devices, 2)]),
        element_plan=element_plan,
        beat_plan=beat_plan,
        source="deterministic",
    )


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
        "Assume SCENE_SPEC and SCENE_BRIEF globals exist, and that VexGeneratedScene already provides themed helpers like apply_house_background, make_title_block, make_pill, make_glass_panel, make_signal_node, make_connector, make_glow_dot, make_orbit_ring, make_route_path, make_focus_beam, make_metric_badge, make_ribbon_label, fit_text, camera_focus, and register_layout_group. "
        "For all principal objects, use the aspect-aware layout helpers: self.place_in_slot(...), self.fit_text_for_slot(...), self.slot_point(...), self.layout_route_points(...), and self.route_between_slots(...). "
        "Do not anchor main scene groups with hard-coded LEFT * 4 / RIGHT * 4 style frame-edge coordinates; raw move_to is only acceptable for internal geometry inside a group before that group is placed in a layout slot."
    )


def _user_prompt(
    brief: SceneBrief,
    examples: list[SceneExample],
    skills: list[SkillSlice],
    blueprint: SceneBlueprint,
    execution_plan: SceneExecutionPlan,
    *,
    alternative_blueprints: list[SceneBlueprint] | None = None,
    previous_code: str | None = None,
    feedback_lines: list[str] | None = None,
    storyboard_context: str | None = None,
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
            "\n- LaTeX is NOT available in this runtime. Avoid Tex, MathTex, BarChart, Matrix, Variable, and any TeX-dependent labels."
            "\n- Vex provides runtime-safe DecimalNumber and Integer text shims, so numeric badges/counters are still fine."
        )
    contract_block = "\n".join(f"- {item}" for item in brief.scene_contract)
    storyboard_block = ""
    if str(storyboard_context or "").strip():
        storyboard_block = (
            "\n\nVisual IR and storyboard contract:\n"
            f"{str(storyboard_context).strip()}\n"
        )
    return (
        "Scene brief:\n"
        f"{_brief_block(brief)}\n\n"
        "Intuition target:\n"
        f"{_intuition_block(brief)}\n\n"
        "Relevant Manim skill slices:\n"
        f"{_skills_block(skills)}\n\n"
        f"{_blueprint_block(blueprint, list(alternative_blueprints or []))}\n\n"
        "Execution plan:\n"
        f"{_execution_plan_block(execution_plan)}\n\n"
        "Retrieved scene examples:\n"
        f"{_examples_block(examples)}\n\n"
        "Scene contract:\n"
        f"{contract_block}\n\n"
        f"{storyboard_block}"
        "Hard requirements:\n"
        "- Start from VexGeneratedScene and build a real animated scene.\n"
        "- Treat the Visual IR and storyboard as the source of truth for what the visual must teach.\n"
        "- Every visible object should map to the storyboard contract or the selected blueprint; delete decorative filler that does not teach the intuition.\n"
        "- Add the title treatment with make_title_block unless the scene has a stronger editorial framing.\n"
        "- Call runtime helpers as self.make_title_block(...), self.make_orbit_ring(...), self.camera_focus(...), and so on; never use bare helper calls.\n"
        "- Honor the selected blueprint's focal system, motion beats, and element roles; do not collapse it into generic panels or stacked text boxes.\n"
        "- Execute the provided plan faithfully: use its title text, element assignments, and beat sequence as the scene's concrete recipe.\n"
        "- Do not expand the planned copy into longer prose. If a planned element has compact copy, keep it compact in code.\n"
        "- Make the mental-model shift legible: the viewer should understand why the idea works, not just read the subtitle again.\n"
        "- Use the before/after/cause/effect context when present; convert it into visual logic, not extra prose.\n"
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
        @staticmethod
        def _center_point(node: ast.AST) -> ast.Call:
            return ast.Call(
                func=ast.Attribute(value=node, attr="get_center", ctx=ast.Load()),
                args=[],
                keywords=[],
            )

        def _coerce_point_like(self, node: ast.AST) -> ast.AST:
            if _is_point_expression(node) or _is_numeric_expr(node):
                return node
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    return ast.BinOp(
                        left=self._coerce_point_like(node.left),
                        op=node.op,
                        right=self._coerce_point_like(node.right),
                    )
                if isinstance(node.op, (ast.Mult, ast.Div)):
                    return ast.BinOp(
                        left=self._coerce_point_like(node.left),
                        op=node.op,
                        right=self._coerce_point_like(node.right),
                    )
            if isinstance(node, ast.UnaryOp):
                return ast.UnaryOp(op=node.op, operand=self._coerce_point_like(node.operand))
            if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
                return self._center_point(node)
            return node

        @staticmethod
        def _rate_function_attribute(name: str) -> ast.Attribute:
            canonical = RATE_FUNCTION_ALIASES.get(name, name)
            return ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="manim", ctx=ast.Load()), attr="rate_functions", ctx=ast.Load()),
                attr=canonical,
                ctx=ast.Load(),
            )

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            short_name = _call_name(node.func).split(".")[-1]
            if short_name == "move_to" and node.args and isinstance(node.args[0], (ast.BinOp, ast.UnaryOp)):
                node.args[0] = self._coerce_point_like(node.args[0])
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
                return self._rate_function_attribute(node.id)
            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            self.generic_visit(node)
            short_name = node.attr
            if short_name not in RATE_FUNCTION_NAMES:
                return node
            qualified_name = _call_name(node)
            if qualified_name in {
                f"utils.{short_name}",
                f"rate_functions.{short_name}",
                f"manim.utils.{short_name}",
                f"manim.rate_functions.{short_name}",
            }:
                return self._rate_function_attribute(short_name)
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


def _execution_plan_system_prompt() -> str:
    return (
        "You are a principal motion designer planning a Manim scene before code is written. "
        "Your job is to translate the transcript beat into a precise scene recipe with compact copy, clear visual logic, and a concrete beat-by-beat motion sequence. "
        "Output ONLY a JSON object with keys summary, intuition_thesis, visual_logic, motion_spine, title_text, deck_text, layout_rules, guardrails, advanced_devices, element_plan, beat_plan. "
        "Each element_plan item must have: element_id, role, treatment, copy_lines, source_hint, layout_intent. "
        "Each beat_plan item must have: beat_id, focus, story_goal, motion, camera, visible_elements. "
        "Keep text compact: title_text <= 10 words, deck_text <= 10 words, and most element copy_lines <= 6 words. "
        "Prefer intuition, causality, and motion logic over transcript repetition."
    )


def _execution_plan_user_prompt(
    brief: SceneBrief,
    blueprint: SceneBlueprint,
    alternatives: list[SceneBlueprint] | None = None,
    storyboard_context: str | None = None,
) -> str:
    storyboard_block = ""
    if str(storyboard_context or "").strip():
        storyboard_block = (
            "\n\nVisual IR and storyboard contract:\n"
            f"{str(storyboard_context).strip()}\n"
        )
    return (
        "Scene brief:\n"
        f"{_brief_block(brief)}\n\n"
        "Intuition target:\n"
        f"{_intuition_block(brief)}\n\n"
        f"{_blueprint_block(blueprint, list(alternatives or []))}\n\n"
        f"{storyboard_block}"
        "Planning requirements:\n"
        "- Translate the beat into a scene the viewer can understand at a glance.\n"
        "- Plan against the Visual IR, not only the transcript. The storyboard frames are the required beginning, mechanism, and payoff states.\n"
        "- Assign short, readable copy to the blueprint elements instead of reusing transcript fragments.\n"
        "- Make the motion spine explicit so the later codegen phase can execute it cleanly.\n"
        "- Use the before/after/cause/effect fields when present.\n"
        "- Keep title and support text compact enough to fit cleanly inside Manim layouts.\n"
        "- Principal objects must be assigned to layout slots, not placed by fixed frame-edge coordinates.\n"
        "- If the scene family implies a route, system, morph, or interface walkthrough, reflect that in both the element assignments and beat sequence.\n"
    )


def _parse_execution_plan(raw_text: str, brief: SceneBrief, blueprint: SceneBlueprint) -> SceneExecutionPlan:
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
    fallback = build_deterministic_execution_plan(brief, blueprint)
    if not payload:
        return fallback

    title_text = _condense_copy(str(payload.get("title_text") or fallback.title_text or ""), max_words=10, max_chars=72)
    deck_text = _condense_copy(str(payload.get("deck_text") or fallback.deck_text or ""), max_words=10, max_chars=76)
    raw_elements = payload.get("element_plan")
    element_plan: list[PlannedElement] = []
    if isinstance(raw_elements, list):
        blueprint_roles = {
            str(element.element_id): str(element.role or "")
            for element in blueprint.elements
        }
        for item in raw_elements[:12]:
            if not isinstance(item, dict):
                continue
            element_id = str(item.get("element_id") or "").strip()
            if not element_id or element_id not in blueprint_roles:
                continue
            raw_lines = item.get("copy_lines")
            if isinstance(raw_lines, str):
                raw_lines = [raw_lines]
            lines = [
                _condense_copy(
                    _sanitize_plan_text(str(line or ""), max_chars=72 if blueprint_roles[element_id] == "title" else 44),
                    max_words=10 if blueprint_roles[element_id] == "title" else 6,
                    max_chars=72 if blueprint_roles[element_id] == "title" else 44,
                )
                for line in list(raw_lines or [])
            ]
            lines = _unique_nonempty([line for line in lines if line])
            if blueprint_roles[element_id] == "title" and title_text:
                lines = [title_text]
            element_plan.append(
                PlannedElement(
                    element_id=element_id,
                    role=str(item.get("role") or blueprint_roles[element_id] or ""),
                    treatment=_sanitize_plan_text(str(item.get("treatment") or "focused_component"), max_chars=40),
                    copy_lines=lines,
                    source_hint=_sanitize_plan_text(str(item.get("source_hint") or ""), max_chars=40),
                    layout_intent=_sanitize_plan_text(str(item.get("layout_intent") or ""), max_chars=90),
                )
            )
    if not element_plan:
        element_plan = fallback.element_plan
    else:
        existing_ids = {item.element_id for item in element_plan}
        for item in fallback.element_plan:
            if item.element_id not in existing_ids:
                element_plan.append(item)

    raw_beats = payload.get("beat_plan")
    beat_plan: list[PlannedBeat] = []
    if isinstance(raw_beats, list):
        for item in raw_beats[:8]:
            if not isinstance(item, dict):
                continue
            beat_plan.append(
                PlannedBeat(
                    beat_id=_sanitize_plan_text(str(item.get("beat_id") or f"beat_{len(beat_plan) + 1}"), max_chars=32),
                    focus=_sanitize_plan_text(str(item.get("focus") or ""), max_chars=40),
                    story_goal=_sanitize_plan_text(str(item.get("story_goal") or ""), max_chars=120),
                    motion=_sanitize_plan_text(str(item.get("motion") or ""), max_chars=120),
                    camera=_sanitize_plan_text(str(item.get("camera") or ""), max_chars=80),
                    visible_elements=[
                        _sanitize_plan_text(str(elem), max_chars=40)
                        for elem in list(item.get("visible_elements") or [])
                        if str(elem).strip()
                    ][:6],
                )
            )
    if not beat_plan:
        beat_plan = fallback.beat_plan

    layout_rules = [
        _sanitize_plan_text(str(item), max_chars=110)
        for item in list(payload.get("layout_rules") or [])
        if str(item).strip()
    ][:6] or fallback.layout_rules
    guardrails = [
        _sanitize_plan_text(str(item), max_chars=110)
        for item in list(payload.get("guardrails") or [])
        if str(item).strip()
    ][:6] or fallback.guardrails
    advanced_devices = [
        _sanitize_plan_text(str(item), max_chars=40)
        for item in list(payload.get("advanced_devices") or [])
        if str(item).strip()
    ][: max(brief.minimum_dynamic_devices + 1, 4)] or fallback.advanced_devices
    return SceneExecutionPlan(
        summary=_sanitize_plan_text(str(payload.get("summary") or fallback.summary), max_chars=220),
        intuition_thesis=_sanitize_plan_text(str(payload.get("intuition_thesis") or fallback.intuition_thesis), max_chars=180),
        visual_logic=_sanitize_plan_text(str(payload.get("visual_logic") or fallback.visual_logic), max_chars=220),
        motion_spine=_sanitize_plan_text(str(payload.get("motion_spine") or fallback.motion_spine), max_chars=120),
        title_text=title_text or fallback.title_text,
        deck_text=deck_text,
        layout_rules=layout_rules,
        guardrails=guardrails,
        advanced_devices=advanced_devices,
        element_plan=element_plan,
        beat_plan=beat_plan,
        source="llm_plan",
    )


def request_scene_execution_plan(
    provider_name: str,
    model_name: str,
    brief: SceneBrief,
    blueprint: SceneBlueprint,
    *,
    alternative_blueprints: list[SceneBlueprint] | None = None,
    storyboard_context: str | None = None,
) -> SceneExecutionPlan:
    try:
        raw_text = call_reasoning_model(
            provider_name,
            model_name,
            _execution_plan_system_prompt(),
            _execution_plan_user_prompt(
                brief,
                blueprint,
                alternatives=alternative_blueprints,
                storyboard_context=storyboard_context,
            ),
        )
        return _parse_execution_plan(raw_text, brief, blueprint)
    except Exception:
        return build_deterministic_execution_plan(brief, blueprint)


def request_scene_candidate(
    provider_name: str,
    model_name: str,
    brief: SceneBrief,
    examples: list[SceneExample],
    blueprint: SceneBlueprint,
    execution_plan: SceneExecutionPlan,
    *,
    alternative_blueprints: list[SceneBlueprint] | None = None,
    previous_code: str | None = None,
    feedback_lines: list[str] | None = None,
    storyboard_context: str | None = None,
) -> SceneCandidate:
    skill_limit = 2 if brief.animation_intensity == "low" else 4
    if float(getattr(brief, "duration_sec", 0.0) or 0.0) <= 3.8:
        skill_limit = min(skill_limit, 3)
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
            execution_plan,
            alternative_blueprints=alternative_blueprints,
            previous_code=previous_code,
            feedback_lines=feedback_lines,
            storyboard_context=storyboard_context,
        ),
    )
    return _parse_candidate(raw_text)


def write_generation_report(
    target_path: Path,
    *,
    brief: SceneBrief,
    blueprint_candidates: list[SceneBlueprint],
    selected_blueprint: SceneBlueprint | None,
    selected_execution_plan: SceneExecutionPlan | None,
    selected_examples: list[SceneExample],
    attempts: list[dict[str, Any]],
    final_candidate: SceneCandidate | None,
    final_scene_code: str | None,
    quality_score: float | None,
    fallback_used: bool,
    visual_explanation_ir: dict[str, Any] | None = None,
    storyboard_frames: list[dict[str, Any]] | None = None,
    storyboard_critique: dict[str, Any] | None = None,
    storyboard_candidates: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "scene_brief": brief.to_dict(),
        "visual_explanation_ir": visual_explanation_ir or {},
        "storyboard_frames": storyboard_frames or [],
        "storyboard_critique": storyboard_critique or {},
        "storyboard_candidates": storyboard_candidates or [],
        "blueprint_candidates": [item.to_dict() for item in blueprint_candidates],
        "selected_blueprint": selected_blueprint.to_dict() if selected_blueprint else None,
        "selected_execution_plan": selected_execution_plan.to_dict() if selected_execution_plan else None,
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
