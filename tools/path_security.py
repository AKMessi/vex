from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


TRUSTED_OUTPUT_PATH_TOKEN = object()


class UnsafeOutputPathError(ValueError):
    pass


class UnsafeInputPathError(ValueError):
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


def project_input_roots(state: object) -> list[Path]:
    roots: list[Path] = []
    for value in (
        getattr(state, "working_dir", None),
        getattr(state, "output_dir", None),
    ):
        if value:
            roots.append(Path(str(value)))
    for source in getattr(state, "source_files", None) or []:
        if source:
            roots.append(Path(str(source)).expanduser().resolve(strict=False).parent)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        normalized = _norm(root.expanduser().resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(root)
    return deduped


def resolve_existing_input_path(
    requested_path: str,
    *,
    default_root: str | Path,
    allowed_roots: Iterable[str | Path],
    allowed_suffixes: set[str] | None = None,
    max_size_bytes: int | None = None,
) -> Path:
    raw = str(requested_path or "").strip()
    if not raw:
        raise UnsafeInputPathError("Input path is empty.")

    root = Path(default_root).expanduser().resolve(strict=False)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise UnsafeInputPathError(f"Input file not found: {candidate}") from exc

    roots = [Path(item).expanduser().resolve(strict=False) for item in allowed_roots]
    if not roots or not any(_is_within(candidate, allowed_root) for allowed_root in roots):
        allowed_text = ", ".join(str(path) for path in roots) or str(root)
        raise UnsafeInputPathError(
            f"Input paths from agent tools must stay inside: {allowed_text}"
        )

    if not candidate.is_file():
        raise UnsafeInputPathError(f"Input path is not a file: {candidate}")

    if allowed_suffixes is not None:
        normalized_suffixes = {suffix.lower() for suffix in allowed_suffixes}
        if candidate.suffix.lower() not in normalized_suffixes:
            raise UnsafeInputPathError(
                "Input path must use one of these extensions: "
                + ", ".join(sorted(normalized_suffixes))
            )

    if max_size_bytes is not None and candidate.stat().st_size > max_size_bytes:
        raise UnsafeInputPathError(
            f"Input file is too large for this operation: {candidate}"
        )

    return candidate


def resolve_existing_project_file(
    requested_path: str,
    state: object,
    *,
    allowed_suffixes: set[str] | None = None,
    max_size_bytes: int | None = None,
) -> Path:
    return resolve_existing_input_path(
        requested_path,
        default_root=getattr(state, "working_dir", "."),
        allowed_roots=project_input_roots(state),
        allowed_suffixes=allowed_suffixes,
        max_size_bytes=max_size_bytes,
    )


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
