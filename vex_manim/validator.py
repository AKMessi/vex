from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from vex_manim.briefs import SceneBrief


ALLOWED_IMPORT_PREFIXES = {
    "manim",
    "math",
    "random",
    "itertools",
    "collections",
    "numpy",
    "vex_manim.runtime",
}

FORBIDDEN_ROOT_NAMES = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "httpx",
}

FORBIDDEN_CALL_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "__import__",
}

ADVANCED_FEATURE_NAMES = {
    "MovingCameraScene",
    "ValueTracker",
    "always_redraw",
    "TransformMatchingShapes",
    "ReplacementTransform",
    "FadeTransform",
    "LaggedStart",
    "AnimationGroup",
    "Succession",
    "MoveAlongPath",
    "TracedPath",
    "Axes",
    "BarChart",
    "NumberLine",
    "Code",
    "MathTex",
    "SVGMobject",
    "ImageMobject",
    "CurvedArrow",
    "SurroundingRectangle",
}

TEX_DEPENDENT_NAMES = {
    "BarChart",
    "DecimalTable",
    "MathTex",
    "MathTable",
    "Tex",
    "SingleStringMathTex",
    "Matrix",
    "Variable",
}

PRIMITIVE_NAMES = {
    "Text",
    "Paragraph",
    "Rectangle",
    "RoundedRectangle",
    "Circle",
    "Square",
    "Dot",
    "Line",
    "Arrow",
    "VGroup",
}

PANEL_HEAVY_HELPERS = {
    "make_glass_panel",
    "make_pill",
}

PREMIUM_MOTION_HELPERS = {
    "make_glow_dot",
    "make_orbit_ring",
    "make_route_path",
    "make_focus_beam",
    "make_metric_badge",
    "make_ribbon_label",
    "make_connector",
    "make_signal_node",
    "place_in_slot",
    "fit_text_for_slot",
    "route_between_slots",
    "layout_route_points",
}

VISIBLE_TEXT_CALL_NAMES = {
    "Text",
    "fit_text",
    "make_pill",
    "make_metric_badge",
    "make_ribbon_label",
    "make_signal_node",
}

DYNAMIC_VISUAL_FEATURES = {
    "MovingCameraScene",
    "ValueTracker",
    "always_redraw",
    "TransformMatchingShapes",
    "ReplacementTransform",
    "FadeTransform",
    "LaggedStart",
    "AnimationGroup",
    "Succession",
    "MoveAlongPath",
    "TracedPath",
    "Axes",
    "BarChart",
    "NumberLine",
    "CurvedArrow",
}


def _visible_word_count_from_text(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'./%-]*", value))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


@dataclass
class CodeProfile:
    advanced_features: list[str] = field(default_factory=list)
    primitive_features: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    play_calls: int = 0
    wait_calls: int = 0
    layout_registration_calls: int = 0
    layout_slot_api_calls: int = 0
    frame_edge_coordinate_mentions: int = 0
    panel_helper_calls: int = 0
    premium_helper_calls: int = 0
    title_helper_calls: int = 0
    visible_text_word_count: int = 0
    long_visible_text_literals: int = 0
    dynamic_device_count: int = 0
    camera_move_mentions: int = 0
    class_names: list[str] = field(default_factory=list)
    line_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str]
    warnings: list[str]
    profile: CodeProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "profile": self.profile.to_dict(),
        }

    def feedback_lines(self) -> list[str]:
        return [*self.errors, *self.warnings]


def _allowed_import(module_name: str) -> bool:
    if not module_name:
        return True
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in ALLOWED_IMPORT_PREFIXES)


def profile_scene_code(scene_code: str) -> CodeProfile:
    tree = ast.parse(scene_code)
    profile = CodeProfile(
        line_count=len(scene_code.splitlines()),
        camera_move_mentions=scene_code.count("camera.frame.animate") + scene_code.count("camera_focus("),
        frame_edge_coordinate_mentions=len(
            re.findall(
                r"\b(?:LEFT|RIGHT)\s*\*\s*(?:[3-9](?:\.\d+)?|\d{2,}(?:\.\d+)?)"
                r"|\b(?:UP|DOWN)\s*\*\s*(?:2\.[5-9]|[3-9](?:\.\d+)?|\d{2,}(?:\.\d+)?)",
                scene_code,
            )
        ),
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                profile.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            profile.imports.append(node.module or "")
        elif isinstance(node, ast.ClassDef):
            profile.class_names.append(node.name)
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            short_name = call_name.split(".")[-1]
            if short_name in ADVANCED_FEATURE_NAMES and short_name not in profile.advanced_features:
                profile.advanced_features.append(short_name)
            if short_name in PRIMITIVE_NAMES and short_name not in profile.primitive_features:
                profile.primitive_features.append(short_name)
            if call_name.endswith(".play") or short_name == "play":
                profile.play_calls += 1
            if call_name.endswith(".wait") or short_name == "wait":
                profile.wait_calls += 1
            if call_name.endswith(".register_layout_group") or short_name in {"register_layout_group", "register_text_group", "register_panel_group"}:
                profile.layout_registration_calls += 1
            if short_name in {"place_in_slot", "fit_text_for_slot", "route_between_slots", "layout_route_points", "slot_point", "slot_anchor", "layout_slot"}:
                profile.layout_slot_api_calls += 1
            if short_name in PANEL_HEAVY_HELPERS:
                profile.panel_helper_calls += 1
            if short_name in PREMIUM_MOTION_HELPERS:
                profile.premium_helper_calls += 1
            if short_name == "make_title_block":
                profile.title_helper_calls += 1
            if short_name in VISIBLE_TEXT_CALL_NAMES:
                text_value = None
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    text_value = node.args[0].value
                else:
                    for keyword in node.keywords:
                        if keyword.arg in {"text", "label", "headline", "deck"} and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                            text_value = keyword.value.value
                            break
                if text_value:
                    word_count = _visible_word_count_from_text(text_value)
                    profile.visible_text_word_count += word_count
                    if word_count >= 8:
                        profile.long_visible_text_literals += 1
    profile.dynamic_device_count = len(set(profile.advanced_features) & DYNAMIC_VISUAL_FEATURES) + profile.premium_helper_calls
    return profile


def validate_generated_scene_code(
    scene_code: str,
    *,
    expected_class_name: str = "GeneratedScene",
    latex_available: bool = True,
    brief: SceneBrief | None = None,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        tree = ast.parse(scene_code)
    except SyntaxError as exc:
        return ValidationReport(
            valid=False,
            errors=[f"Python syntax error on line {exc.lineno}: {exc.msg}"],
            warnings=[],
            profile=CodeProfile(line_count=len(scene_code.splitlines())),
        )

    profile = profile_scene_code(scene_code)
    class_node: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _allowed_import(alias.name):
                    errors.append(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if not _allowed_import(node.module or ""):
                errors.append(f"Import not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            short_name = call_name.split(".")[-1]
            root_name = call_name.split(".")[0]
            if short_name in FORBIDDEN_CALL_NAMES:
                errors.append(f"Forbidden call used: {short_name}")
            if root_name in FORBIDDEN_ROOT_NAMES:
                errors.append(f"Forbidden module usage: {root_name}")
            if not latex_available and short_name in TEX_DEPENDENT_NAMES:
                errors.append(
                    f"{short_name} requires a LaTeX toolchain, which is not available in this environment."
                )
        elif isinstance(node, ast.ClassDef) and node.name == expected_class_name:
            class_node = node

    if class_node is None:
        errors.append(f"Scene code must define class {expected_class_name}.")
    else:
        has_construct = any(isinstance(item, ast.FunctionDef) and item.name == "construct" for item in class_node.body)
        if not has_construct:
            errors.append(f"{expected_class_name} must define construct(self).")
        bases = [_call_name(base) for base in class_node.bases]
        if "VexGeneratedScene" not in bases:
            errors.append(f"{expected_class_name} must subclass VexGeneratedScene.")

    if profile.play_calls == 0:
        errors.append("Scene code must animate with at least one self.play(...).")
    if profile.line_count > 280:
        warnings.append("Scene code is long; prefer a tighter scene with fewer moving parts.")
    if len(profile.advanced_features) == 0:
        warnings.append("No advanced Manim features detected; the scene may still feel generic.")
    if profile.layout_registration_calls == 0:
        warnings.append("Register the principal layout groups with register_layout_group(...) so the runtime can protect the composition.")
    if brief is not None and getattr(brief, "composition_mode", "") == "replace" and profile.layout_slot_api_calls == 0:
        warnings.append("Use aspect-aware layout slot helpers for principal objects instead of hand-placing the whole frame.")
    if profile.layout_slot_api_calls == 0 and profile.frame_edge_coordinate_mentions >= 2:
        warnings.append("Hard-coded frame-edge coordinates detected; place principal groups through layout slots for aspect-safe rendering.")
    primitive_count = len(profile.primitive_features)
    advanced_count = len(profile.advanced_features)
    if primitive_count >= 4 and advanced_count == 0:
        warnings.append("The scene leans heavily on primitive shapes without richer Manim choreography.")
    if profile.panel_helper_calls >= 3 and profile.premium_helper_calls == 0 and advanced_count < 3:
        warnings.append("The scene still leans too heavily on panels and pills instead of richer spatial motion.")
    text_budget = int(getattr(brief, "text_budget_words", 0) or 0)
    if text_budget > 0 and profile.visible_text_word_count > text_budget + 6:
        warnings.append(
            f"The scene likely puts too much copy on screen ({profile.visible_text_word_count} visible words for a target of about {text_budget})."
        )
    if profile.long_visible_text_literals >= 2:
        warnings.append("Multiple long text literals were detected; compress copy into shorter labels or cues.")
    minimum_dynamic_devices = int(getattr(brief, "minimum_dynamic_devices", 0) or 0)
    if minimum_dynamic_devices > 0 and profile.dynamic_device_count < minimum_dynamic_devices:
        warnings.append(
            f"The scene does not yet show enough dynamic visual devices ({profile.dynamic_device_count} found; target at least {minimum_dynamic_devices})."
        )
    if getattr(brief, "camera_style", "") in {"guided", "punch_in"} and profile.camera_move_mentions == 0:
        warnings.append("The brief calls for camera language, but no camera movement or focus helper was detected.")

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, profile=profile)
