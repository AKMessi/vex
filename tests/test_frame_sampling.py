from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vex_visuals.frame_sampling import extract_native_frames


def test_native_frame_sampling_uses_one_forward_decode(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):  # noqa: ANN001
        commands.append(command)
        select_filter = command[command.index("-vf") + 1]
        output_pattern = str(command[-1])
        for index in range(1, select_filter.count("eq(n\\,") + 1):
            output = Path(output_pattern.replace("%04d", f"{index:04d}"))
            Image.new("RGB", (32, 18), color=(index * 30, 20, 40)).save(output)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("vex_visuals.frame_sampling.subprocess.run", fake_run)
    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    final = tmp_path / "final.png"

    paths, errors = extract_native_frames(
        tmp_path / "input.mp4",
        [(first, 0.1), (duplicate, 0.1), (final, 0.9)],
        fps=30,
        duration_sec=1.0,
    )

    assert errors == []
    assert paths == [first, duplicate, final]
    assert len(commands) == 1
    command = commands[0]
    assert "-ss" not in command
    assert command[command.index("-vf") + 1] == "select=eq(n\\,3)+eq(n\\,27)"
    assert first.read_bytes() == duplicate.read_bytes()
    assert first.read_bytes() != final.read_bytes()
