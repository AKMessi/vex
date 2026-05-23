from vex_manim.briefs import SceneBrief, build_scene_brief

__all__ = ["SceneBrief", "VexGeneratedScene", "build_scene_brief"]


def __getattr__(name: str):
    if name == "VexGeneratedScene":
        from vex_manim.runtime import VexGeneratedScene

        return VexGeneratedScene
    raise AttributeError(f"module 'vex_manim' has no attribute {name!r}")
