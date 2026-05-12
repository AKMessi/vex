from __future__ import annotations

from vex_manim.validator import validate_generated_scene_code


def test_validator_rejects_builtins_dunder_import_bypass() -> None:
    code = """
from manim import *
from vex_manim.runtime import VexGeneratedScene


class GeneratedScene(VexGeneratedScene):
    def construct(self):
        importer = getattr(__builtins__, "__import__")
        importer("subprocess").run(["echo", "owned"])
        self.play(FadeIn(Text("safe")))
"""

    report = validate_generated_scene_code(code)

    assert not report.valid
    assert "Forbidden call used: getattr" in report.errors
    assert "Forbidden dunder name used: __builtins__" in report.errors
    assert "Forbidden callable reference: __import__" in report.errors
    assert "Forbidden module reference: subprocess" in report.errors


def test_validator_rejects_dunder_class_mro_escape() -> None:
    code = """
from manim import *
from vex_manim.runtime import VexGeneratedScene


class GeneratedScene(VexGeneratedScene):
    def construct(self):
        classes = Text.__class__.__mro__
        self.play(FadeIn(Text(str(len(classes)))))
"""

    report = validate_generated_scene_code(code)

    assert not report.valid
    assert "Forbidden dunder attribute used: __class__" in report.errors
    assert "Forbidden dunder attribute used: __mro__" in report.errors
