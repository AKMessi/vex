from __future__ import annotations

from pathlib import Path

from encode_planner import build_encode_plan


def test_plans_balanced_h264_compression_for_plain_english_request(tmp_path: Path) -> None:
    plan = build_encode_plan(
        "input.mov",
        str(tmp_path),
        "Launch Cut",
        {"raw_request": "convert this mov file to mp4 and compress the size without losing much quality"},
        metadata=_metadata(codec="prores", audio_codec="pcm_s16le"),
        available_encoders={"libx264", "aac"},
    )

    command = plan.commands[0]
    assert plan.mode == "quality_crf"
    assert plan.output_path.endswith(".mp4")
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "23"
    assert command[command.index("-preset") + 1] == "slow"
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-movflags" in command


def test_uses_stream_copy_for_safe_mov_to_mp4_remux(tmp_path: Path) -> None:
    plan = build_encode_plan(
        "input.mov",
        str(tmp_path),
        "Camera Clip",
        {"raw_request": "convert to mp4"},
        metadata=_metadata(codec="h264", audio_codec="aac"),
        available_encoders={"libx264", "aac"},
    )

    command = plan.commands[0]
    assert plan.mode == "stream_copy"
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "copy"


def test_target_size_uses_two_pass_encode(tmp_path: Path) -> None:
    plan = build_encode_plan(
        "input.mov",
        str(tmp_path),
        "Tiny Cut",
        {"raw_request": "compress this under 50 mb"},
        metadata=_metadata(codec="h264", audio_codec="aac", duration_sec=120.0),
        available_encoders={"libx264", "aac"},
    )

    assert plan.mode == "two_pass_target_size"
    assert len(plan.commands) == 2
    assert plan.estimated_size_bytes == 50 * 1024 * 1024
    assert plan.commands[0][plan.commands[0].index("-pass") + 1] == "1"
    assert plan.commands[1][plan.commands[1].index("-pass") + 1] == "2"
    assert "-b:v" in plan.commands[1]


def _metadata(
    *,
    codec: str,
    audio_codec: str,
    duration_sec: float = 30.0,
) -> dict[str, object]:
    return {
        "duration_sec": duration_sec,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "codec": codec,
        "video_bit_rate": 8_000_000,
        "has_audio": True,
        "audio_codec": audio_codec,
        "audio_bit_rate": 128_000,
        "audio_channels": 2,
        "size_bytes": 100 * 1024 * 1024,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
    }
