"""Project-owned, sanitized visual references for directing shots."""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from state import ProjectState
from vex_visuals.evidence import file_digest


MAX_REFERENCE_BYTES = 8 * 1024 * 1024
MAX_REFERENCES = 4
HASH = re.compile(r"[a-f0-9]{64}")


def validate_references(value: object) -> list[dict]:
    if (not isinstance(value, list) or len(value) > MAX_REFERENCES or any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str) or not HASH.fullmatch(item["id"])
        or not isinstance(item.get("note"), str) or not 1 <= len(item["note"].strip()) <= 240
        for item in value
    ) or len({item["id"] for item in value}) != len(value)):
        raise ValueError("The project reference list is invalid.")
    return [{"id": row["id"], "note": row["note"]} for row in value]


def references(state: ProjectState) -> list[dict]:
    return validate_references(state.artifacts.get("production_references", []))


def reference_path(workspace: str | Path, identity: str) -> Path:
    if not isinstance(identity, str) or not HASH.fullmatch(identity):
        raise ValueError("Choose an attached reference image.")
    path = Path(workspace).resolve() / "production_references" / f"{identity}.png"
    if path.resolve() != path:
        raise ValueError("Reference images must stay in the project workspace.")
    return path


def open_reference(state: ProjectState, identity: str) -> Path:
    if identity not in {row["id"] for row in references(state)}:
        raise ValueError("Choose an attached reference image.")
    path = reference_path(state.working_dir, identity)
    if not path.is_file() or path.stat().st_size > MAX_REFERENCE_BYTES or file_digest(path) != identity:
        raise ValueError("The reference image is missing or has changed.")
    try:
        with Image.open(path, formats=("PNG",)) as image:
            if max(image.size) > 1024 or getattr(image, "is_animated", False):
                raise ValueError("The reference image exceeds its saved dimensions.")
            image.verify()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The saved reference is not a valid still PNG.") from exc
    return path


def save_reference(state: ProjectState, path: Path, note: str) -> list[dict]:
    if not isinstance(note, str) or not 1 <= len(note.strip()) <= 240:
        raise ValueError("Describe what the reference shows in 1 to 240 characters.")
    if not 0 < path.stat().st_size <= MAX_REFERENCE_BYTES:
        raise ValueError("Choose a reference image up to 8 MiB.")
    try:
        with Image.open(path, formats=("PNG", "JPEG", "WEBP")) as image:
            if image.width * image.height > 24_000_000 or getattr(image, "is_animated", False):
                raise ValueError("Use a still image up to 24 megapixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            # Carry pixel data only, including no metadata identifying the original file.
            clean = Image.new("RGB", image.size)
            clean.paste(image)
            clean.save(encoded, format="PNG")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("Choose a valid still PNG, JPEG or WebP reference.") from exc
    blob = encoded.getvalue()
    identity = hashlib.sha256(blob).hexdigest()
    rows = [row for row in references(state) if row["id"] != identity]
    if len(rows) >= MAX_REFERENCES:
        raise ValueError(f"A project can hold up to {MAX_REFERENCES} reference images.")
    target = reference_path(state.working_dir, identity)
    target.parent.mkdir(exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".reference-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(blob)
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    state.artifacts["production_references"] = [*rows, {"id": identity, "note": note.strip()}]
    state.save()
    return references(state)


def remove_reference(state: ProjectState, identity: str) -> list[dict]:
    rows = references(state)
    retained = [row for row in rows if row["id"] != identity]
    if len(rows) == len(retained):
        raise ValueError("Choose an attached reference image.")
    state.artifacts["production_references"] = retained
    state.save()
    return retained
