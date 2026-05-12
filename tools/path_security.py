from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


TRUSTED_OUTPUT_PATH_TOKEN = object()


class UnsafeOutputPathError(ValueError):
    pass


def is_trusted_output_path_request(params: dict) -> bool:
    return params.get("_trusted_output_path_token") is TRUSTED_OUTPUT_PATH_TOKEN


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([_norm(path), _norm(root)]) == _norm(root)
    except ValueError:
        return False


def resolve_output_path(
    requested_path: str,
    *,
    default_root: str | Path,
    allowed_roots: Iterable[str | Path],
    trusted: bool = False,
    allow_overwrite: bool = False,
    allowed_suffixes: set[str] | None = None,
) -> Path:
    raw = str(requested_path or "").strip()
    if not raw:
        raise UnsafeOutputPathError("Output path is empty.")

    root = Path(default_root).expanduser().resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()

    if allowed_suffixes is not None:
        normalized_suffixes = {suffix.lower() for suffix in allowed_suffixes}
        if candidate.suffix.lower() not in normalized_suffixes:
            raise UnsafeOutputPathError(
                "Output path must use one of these extensions: "
                + ", ".join(sorted(normalized_suffixes))
            )

    if not trusted:
        roots = [Path(item).expanduser().resolve() for item in allowed_roots]
        if not any(_is_within(candidate, allowed_root) for allowed_root in roots):
            allowed_text = ", ".join(str(path) for path in roots)
            raise UnsafeOutputPathError(
                f"Custom output paths from agent tools must stay inside: {allowed_text}"
            )
        allow_overwrite = False

    if candidate.exists() and not allow_overwrite:
        raise UnsafeOutputPathError(f"Output already exists: {candidate}")

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate
