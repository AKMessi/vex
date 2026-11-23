"""Cache verified narration takes for preview and delivery."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from engine import probe_video
from vex_runtime.locking import exclusive_file_lock
from vex_visuals.evidence import file_digest
from video_generation.renderer import HyperframesVideoRuntime


def materialize_speech(root: Path, filename: str, spec: dict) -> Path:
    if root.resolve() != root:
        raise ValueError("The narration cache must stay inside the production workspace.")
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    receipt = target.with_suffix(".sha256")
    with exclusive_file_lock(target.with_suffix(".lock"), timeout_sec=1800):
        if target.resolve() != target or receipt.resolve() != receipt:
            raise ValueError("Narration assets must stay inside the production cache.")
        if target.is_file() and receipt.is_file() and receipt.read_text(encoding="ascii") == file_digest(target):
            return target
        take = Path(tempfile.mkdtemp(prefix="take-", dir=root))
        try:
            text = take / "line.txt"
            text.write_text(spec["text"], encoding="utf-8")
            wav = take / "line.wav"
            HyperframesVideoRuntime(auto_install=False).tts(text_path=text, output_path=wav, voice=spec["voice"], speed=spec["speed"])
            metadata = probe_video(str(wav))
            if not metadata["has_audio"] or not 0 < metadata["duration_sec"] <= spec["frames"] / spec["fps"]:
                raise ValueError("The recorded sentence does not fit the shot. Shorten the line and preview it again.")
            digest = file_digest(wav)
            wav.replace(target)
            hash_file = take / "digest"
            hash_file.write_text(digest, encoding="ascii")
            hash_file.replace(receipt)
        finally:
            # A Windows TTS child can briefly keep its working directory open.
            shutil.rmtree(take, ignore_errors=True)
    return target
