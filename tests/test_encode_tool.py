from __future__ import annotations

from pathlib import Path

import encode_planner
from state import ProjectState, utc_now_iso
from tools import encode


def test_plan_encode_stores_pending_plan(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(encode_planner, "available_ffmpeg_encoders", lambda: {"libx264", "aac"})
    state = _state(tmp_path)

    result = encode.execute_plan({"raw_request": "convert to mp4 and compress it"}, state)

    assert result["success"]
    pending = state.artifacts["pending_encode"]
    assert pending["plan_id"] == result["plan_id"]
    assert pending["mode"] == "quality_crf"
    assert pending["source_fingerprint"]["timeline_count"] == 0
    assert "ffmpeg" in pending["display_command"]


def test_run_pending_encode_rejects_stale_plan(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(encode_planner, "available_ffmpeg_encoders", lambda: {"libx264", "aac"})
    state = _state(tmp_path)
    plan_result = encode.execute_plan({"raw_request": "convert to mp4"}, state)
    state.timeline.append({"op": "trim_clip"})

    result = encode.execute_run_pending({"plan_id": plan_result["plan_id"]}, state)

    assert not result["success"]
    assert "stale" in result["message"].lower()
    assert "pending_encode" not in state.artifacts


def test_run_pending_encode_records_latest_encode(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(encode_planner, "available_ffmpeg_encoders", lambda: {"libx264", "aac"})
    state = _state(tmp_path)
    plan_result = encode.execute_plan({"raw_request": "convert to mp4"}, state)
    output_path = Path(state.artifacts["pending_encode"]["output_path"])

    def fake_run_encode_plan(_plan: dict) -> dict:
        output_path.write_bytes(b"encoded")
        return {
            "output_path": str(output_path),
            "output_metadata": _metadata(size_bytes=7),
            "output_size_bytes": 7,
            "validation": {"ok": True, "warnings": []},
        }

    monkeypatch.setattr(encode, "run_encode_plan", fake_run_encode_plan)
    result = encode.execute_run_pending({"plan_id": plan_result["plan_id"]}, state)

    assert result["success"]
    assert "pending_encode" not in state.artifacts
    assert state.artifacts["latest_encode"]["output_path"] == str(output_path)


def _state(tmp_path: Path) -> ProjectState:
    source = tmp_path / "working.mov"
    source.write_bytes(b"source-video")
    now = utc_now_iso()
    return ProjectState(
        project_id="test-project",
        project_name="Encode Test",
        created_at=now,
        updated_at=now,
        source_files=[str(source)],
        working_file=str(source),
        working_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        metadata=_metadata(size_bytes=source.stat().st_size),
        provider="test",
        model="test-model",
    )


def _metadata(*, size_bytes: int) -> dict[str, object]:
    return {
        "duration_sec": 10.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": "h264",
        "video_bit_rate": 8_000_000,
        "has_audio": True,
        "audio_codec": "aac",
        "audio_bit_rate": 128_000,
        "audio_channels": 2,
        "size_bytes": size_bytes,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
    }
