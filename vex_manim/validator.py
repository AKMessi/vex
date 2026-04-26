from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any


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
    "DecimalNumber",
    "DecimalTable",
    "Integer",
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
}


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
    panel_helper_calls: int = 0
    premium_helper_calls: int = 0
    title_helper_calls: int = 0
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
    profile = CodeProfile(line_count=len(scene_code.splitlines()))
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
            if short_name in PANEL_HEAVY_HELPERS:
                profile.panel_helper_calls += 1
            if short_name in PREMIUM_MOTION_HELPERS:
                profile.premium_helper_calls += 1
            if short_name == "make_title_block":
                profile.title_helper_calls += 1
    return profile


def validate_generated_scene_code(
    scene_code: str,
    *,
    expected_class_name: str = "GeneratedScene",
    latex_available: bool = True,
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
    primitive_count = len(profile.primitive_features)
    advanced_count = len(profile.advanced_features)
    if primitive_count >= 4 and advanced_count == 0:
        warnings.append("The scene leans heavily on primitive shapes without richer Manim choreography.")
    if profile.panel_helper_calls >= 3 and profile.premium_helper_calls == 0 and advanced_count < 3:
        warnings.append("The scene still leans too heavily on panels and pills instead of richer spatial motion.")

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, profile=profile)
