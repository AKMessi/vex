"""Typed semantic and open-program compilation for Vex's Remotion renderer."""

from vex_remotion.compiler import (
    REMOTION_SCENE_PROGRAM_VERSION,
    RemotionCompilationResult,
    RemotionSceneProgram,
    compile_remotion_scene_program,
)
from vex_remotion.qa import RemotionRenderQA, evaluate_remotion_render
from vex_remotion.structural_qa import (
    RemotionStructuralQA,
    evaluate_remotion_structure,
)

__all__ = [
    "REMOTION_SCENE_PROGRAM_VERSION",
    "RemotionCompilationResult",
    "RemotionRenderQA",
    "RemotionStructuralQA",
    "RemotionSceneProgram",
    "compile_remotion_scene_program",
    "evaluate_remotion_render",
    "evaluate_remotion_structure",
]
