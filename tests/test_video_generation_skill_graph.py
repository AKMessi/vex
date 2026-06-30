from __future__ import annotations

import json
from pathlib import Path

from video_generation import generate_video, normalize_generation_request
from video_generation.beat_graph import build_initial_beat_graph
from video_generation.cinematographer import build_cinematic_plan
from video_generation.director import build_director_package, direct_script_plan
from video_generation.script_planner import build_script_plan
from video_generation.skill_graph import (
    VIDEO_GENERATION_SKILL_GRAPH_VERSION,
    build_video_skill_graph,
)


def _graph_for(script: str, *, prompt: str = "generate an architecture demo"):
    request = normalize_generation_request(
        {
            "prompt": prompt,
            "script": script,
            "render": False,
            "generate_audio": False,
            "duration_sec": 12,
        }
    )
    initial_plan = build_script_plan(request)
    plan = direct_script_plan(request, initial_plan)
    beat_graph = build_initial_beat_graph(plan, target_duration_sec=12.0, voice_speed=1.0)
    director_package = build_director_package(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        script_rewrite_applied=plan.narration != initial_plan.narration,
    )
    return request, plan, beat_graph, director_package, build_video_skill_graph(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        director_package=director_package,
    )


def test_video_skill_graph_routes_architecture_demo() -> None:
    request, plan, beat_graph, director_package, graph = _graph_for(
        (
            "The API gateway receives a request. "
            "The planner service routes it to the renderer worker. "
            "The renderer returns the finished video response."
        ),
        prompt="generate a premium architecture video about an API gateway planner service and renderer worker",
    )

    assert graph.version == VIDEO_GENERATION_SKILL_GRAPH_VERSION
    assert graph.passed is True
    assert graph.production_skill_id == "architecture-demo"
    assert graph.coverage == 1.0
    assert graph.assignment_count == len(beat_graph.beats)
    assert any(item.scene_type == "architecture_flow" for item in graph.beat_assignments)
    assert all(item.renderer_route == "hyperframes" for item in graph.beat_assignments)
    assert "Video Generation Skill Graph" in graph.prompt_block

    rebuilt = build_video_skill_graph(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        director_package=director_package,
    )
    assert rebuilt.to_dict() == graph.to_dict()


def test_video_skill_graph_does_not_treat_duration_as_metric() -> None:
    request = normalize_generation_request(
        {
            "prompt": "generate a portrait hyperframes video about sparse attention in 12 seconds",
            "render": False,
            "generate_audio": False,
            "duration_sec": 12,
            "aspect": "portrait",
        }
    )
    initial_plan = build_script_plan(request)
    plan = direct_script_plan(request, initial_plan)
    beat_graph = build_initial_beat_graph(plan, target_duration_sec=12.0, voice_speed=1.0)
    director_package = build_director_package(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        script_rewrite_applied=plan.narration != initial_plan.narration,
    )

    graph = build_video_skill_graph(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        director_package=director_package,
    )

    assert graph.production_skill_id == "technical-explainer"
    assert graph.passed is True
    assert all(not item.metric_facts for item in graph.beat_assignments)


def test_cinematographer_carries_video_skill_assignment_into_metadata() -> None:
    request, plan, beat_graph, director_package, graph = _graph_for(
        (
            "The API gateway receives a request, then the planner service routes it to the renderer worker. "
            "The renderer returns the finished video response."
        ),
        prompt="generate an architecture video about API gateway planner service renderer worker",
    )

    cinematic = build_cinematic_plan(
        request=request,
        plan=plan,
        beat_graph=beat_graph,
        director_package=director_package,
        video_skill_graph=graph,
    )

    assert cinematic.accepted_count == len(beat_graph.beats)
    for item in cinematic.beat_compositions:
        assignment = item.metadata.get("video_skill_assignment")
        assert assignment
        assert assignment["beat_id"] == item.beat_id
        assert item.spec["video_generation_skill"]["beat_id"] == item.beat_id
        assert item.tournament["records"]
        assert any(
            "video_skill" in " ".join(record["reasons"])
            for record in item.tournament["records"]
        )


def test_generate_video_writes_video_skill_graph_artifacts(tmp_path: Path) -> None:
    result = generate_video(
        {
            "prompt": "generate a video about retrieval augmented generation",
            "script": (
                "Retrieval adds evidence before generation. "
                "The generator reads the evidence and returns a grounded answer."
            ),
            "output_dir": str(tmp_path),
            "render": False,
            "generate_audio": False,
            "duration_sec": 9,
        }
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    skill_graph_path = Path(manifest["artifacts"]["video_skill_graph_path"])
    skill_graph = json.loads(skill_graph_path.read_text(encoding="utf-8"))

    assert result.qa_passed is True
    assert skill_graph["version"] == VIDEO_GENERATION_SKILL_GRAPH_VERSION
    assert skill_graph["passed"] is True
    assert skill_graph["assignment_count"] == len(manifest["beat_graph"]["beats"])
    assert manifest["video_skill_graph"]["version"] == VIDEO_GENERATION_SKILL_GRAPH_VERSION
    assert manifest["qa"]["evidence"]["video_skill_graph"]["passed"] is True
    assert "Video generation skill graph" in Path(result.design_path).read_text(encoding="utf-8")
    assert "Video skill:" in Path(result.storyboard_path).read_text(encoding="utf-8")
