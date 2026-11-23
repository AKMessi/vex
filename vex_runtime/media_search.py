"""Search the project catalog and timed speech without contacting a service."""
from __future__ import annotations

import math
import re
from pathlib import Path

from asset_registry import load_asset_registry
from state import ProjectState
from tools.transcript_utils import load_transcript_bundle
from vex_runtime.production_edits import inspect_production


def search_media(state: ProjectState, query: str, *, limit: int = 20) -> list[dict]:
    words = set(re.findall(r"\w+", query.casefold())) if isinstance(query, str) and len(query) <= 180 else set()
    if not words:
        raise ValueError("Enter a search term of up to 180 characters.")
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Search returns up to 50 results.")
    hits = []

    def add(kind: str, text: str, **fields) -> None:
        tokens = set(re.findall(r"\w+", text.casefold()))
        matches = sum(any(token.startswith(word) for token in tokens) for word in words)
        if matches:
            hits.append((matches, {"kind": kind, "text": text, **fields}))

    for source in state.source_files:
        add("source", Path(source).name)
    for record in load_asset_registry(state.working_dir)["assets"]:
        tags = dict(record.get("metadata") or {}).get("tags") or []
        name = Path(record["path"]).name
        add("asset", " ".join([name, record["kind"], record["role"], *(str(tag) for tag in tags if isinstance(tags, list))]),
            asset_id=record["asset_id"], name=name)

    transcript = state.artifacts.get("latest_transcript") or {}
    if transcript.get("working_file") == state.working_file:
        bundle = load_transcript_bundle(state.working_dir)
        for row in bundle["sentences"] or bundle["segments"]:
            if not isinstance(row, dict):
                continue
            start, end, text = row.get("start"), row.get("end"), row.get("text")
            if (type(start) in (int, float) and type(end) in (int, float)
                    and math.isfinite(end) and 0 <= start < end and isinstance(text, str)):
                add("transcript", text, start_sec=start, end_sec=end)
    try:
        production = inspect_production(state)
    except (OSError, ValueError, KeyError, TypeError):
        production = {}
    for shot in production.get("shots", []):
        add("shot", shot["narration"], shot_id=shot["id"], start_sec=shot["start_sec"], end_sec=shot["end_sec"])
    for cue in production.get("captions", []):
        fps = production["fps"]
        add("caption", cue["text"], cue_id=cue["id"], start_sec=cue["start_frame"] / fps,
            end_sec=(cue["start_frame"] + cue["duration_frames"]) / fps)
    hits.sort(key=lambda pair: (-pair[0], pair[1].get("start_sec", float("inf"))))
    return [row for _, row in hits[:limit]]
