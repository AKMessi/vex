from __future__ import annotations

import config
import renderers.manim_renderer as manim_renderer
from renderers.manim_renderer import ManimRenderer
from vex_manim.qa import PreviewReport


def test_deterministic_blueprint_compiler_does_not_require_llm_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "MANIM_ALLOW_LLM_CODEGEN", False)

    def fail_execution_plan(*_args, **_kwargs):
        raise AssertionError("LLM execution planning should not run when Manim codegen is disabled.")

    fake_preview = tmp_path / "preview.mp4"
    fake_preview.write_bytes(b"")

    monkeypatch.setattr(manim_renderer, "request_scene_execution_plan", fail_execution_plan)
    monkeypatch.setattr(manim_renderer, "retrieve_scene_examples", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(manim_renderer, "_render_script", lambda *_args, **_kwargs: fake_preview)
    monkeypatch.setattr(
        manim_renderer,
        "probe_video",
        lambda _path: {"duration_sec": 5.0, "width": 640, "height": 360, "has_audio": False},
    )
    monkeypatch.setattr(manim_renderer, "extract_preview_frames", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        manim_renderer,
        "analyze_preview",
        lambda *_args, **_kwargs: PreviewReport(
            preview_video_path=str(fake_preview),
            duration_sec=5.0,
            mean_contrast=34.0,
            mean_occupancy=0.12,
            motion_delta=0.05,
        ),
    )

    script_path, metadata, artifact_paths = ManimRenderer()._attempt_generated_scene(
        _comparison_spec(),
        job_dir=tmp_path / "job",
        width=640,
        height=360,
        fps=15,
        latex_available=False,
    )

    assert script_path.is_file()
    assert metadata["scene_generation_mode"] == "blueprint_compiler"
    assert metadata["production_contract_passed"] is True
    assert artifact_paths["production_contract_path"].endswith("production_contract.json")


def _comparison_spec() -> dict:
    return {
        "visual_id": "deterministic_manim_smoke",
        "template": "spotlight_compare",
        "renderer_hint": "manim",
        "composition_mode": "replace",
        "headline": "Build First Study Later",
        "deck": "Inverted learning loop",
        "sentence_text": "You do not learn hard things by watching tutorials for ten hours.",
        "context_text": "Pick a small project, get stuck, then study exactly what blocks you.",
        "left_detail": "Tutorial binge",
        "right_detail": "Build then study",
        "supporting_lines": ["Get stuck", "Targeted study", "Build first"],
        "duration": 5.0,
        "importance": 0.86,
        "semantic_frame": {
            "intuition_mode": "misconception_flip",
            "mental_model": "Active building exposes the exact gaps passive watching hides.",
            "viewer_takeaway": "Build first, study the blocker.",
            "before_state": "Tutorial binge",
            "after_state": "Build then study",
            "cause": "Passive watching hides gaps",
            "effect": "Getting stuck reveals the next lesson",
            "story_window": "Tutorials feel productive until a project exposes the missing skill.",
        },
    }
