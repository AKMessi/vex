from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vex_runtime import __version__

PROJECT_NAME = "vex-video"
NORMALIZED_NAME = "vex_video"
LICENSE_EXPRESSION = "LicenseRef-PolyForm-Noncommercial-1.0.0"
REQUIRED_WHEEL_FILES = {
    "main.py",
    "config.py",
    "visual_opportunity.py",
    "visual_program.py",
    "presets/export_presets.json",
    "tools/auto_visuals.py",
    "vex_runtime/__init__.py",
    "vex_runtime/resources/config/.env.example",
    "vex_runtime/resources/hyperframes/package.json",
    "vex_runtime/resources/hyperframes/package-lock.json",
}
FORBIDDEN_PATH_PARTS = {"node_modules", "__pycache__", ".git"}


class ReleaseValidationError(RuntimeError):
    pass


def is_prerelease(version: str) -> bool:
    return re.search(r"(?:a|b|rc)\d+$", version, flags=re.IGNORECASE) is not None


def validate_tag(tag: str, version: str = __version__) -> None:
    expected = f"v{version}"
    if tag != expected:
        raise ReleaseValidationError(
            f"Release tag/version mismatch: expected {expected!r}, received {tag!r}."
        )


def _distribution_files(dist_dir: Path) -> tuple[Path, Path]:
    wheels = sorted(dist_dir.glob(f"{NORMALIZED_NAME}-*.whl"))
    sdists = sorted(dist_dir.glob(f"{NORMALIZED_NAME}-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseValidationError(
            "Expected exactly one vex-video wheel and one source distribution in "
            f"{dist_dir}; found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)."
        )
    expected_prefix = f"{NORMALIZED_NAME}-{__version__}"
    if not wheels[0].name.startswith(expected_prefix):
        raise ReleaseValidationError(
            f"Wheel filename does not contain version {__version__}: {wheels[0].name}"
        )
    if sdists[0].name != f"{expected_prefix}.tar.gz":
        raise ReleaseValidationError(
            f"Source distribution filename does not match version {__version__}: "
            f"{sdists[0].name}"
        )
    return wheels[0], sdists[0]


def _validate_archive_paths(paths: set[str], *, archive_name: str) -> None:
    for path in paths:
        parts = set(Path(path).parts)
        if parts & FORBIDDEN_PATH_PARTS:
            raise ReleaseValidationError(
                f"{archive_name} contains forbidden build/runtime content: {path}"
            )
        if Path(path).name == ".env":
            raise ReleaseValidationError(
                f"{archive_name} contains a local .env secret file: {path}"
            )


def validate_wheel(wheel_path: Path, root: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        paths = set(archive.namelist())
        _validate_archive_paths(paths, archive_name=wheel_path.name)
        missing = REQUIRED_WHEEL_FILES - paths
        if missing:
            raise ReleaseValidationError(
                f"Wheel is missing required runtime files: {', '.join(sorted(missing))}"
            )

        metadata_paths = [
            path for path in paths if path.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ReleaseValidationError(
                f"Wheel must contain one METADATA file; found {len(metadata_paths)}."
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if metadata.get("Name") != PROJECT_NAME:
            raise ReleaseValidationError(
                f"Wheel project name is {metadata.get('Name')!r}, expected {PROJECT_NAME!r}."
            )
        if metadata.get("Version") != __version__:
            raise ReleaseValidationError(
                f"Wheel version is {metadata.get('Version')!r}, expected {__version__!r}."
            )
        if metadata.get("License-Expression") != LICENSE_EXPRESSION:
            raise ReleaseValidationError(
                "Wheel license expression is "
                f"{metadata.get('License-Expression')!r}, expected {LICENSE_EXPRESSION!r}."
            )
        if not any(path.endswith(".dist-info/licenses/LICENSE") for path in paths):
            raise ReleaseValidationError("Wheel does not contain the authoritative LICENSE file.")
        entry_point_paths = [
            path for path in paths if path.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_point_paths) != 1:
            raise ReleaseValidationError(
                f"Wheel must contain one entry_points.txt file; found {len(entry_point_paths)}."
            )
        entry_points = archive.read(entry_point_paths[0]).decode("utf-8")
        if "vex = main:app" not in entry_points.splitlines():
            raise ReleaseValidationError(
                "Wheel does not expose the expected `vex = main:app` console command."
            )

        resource_pairs = {
            "vex_runtime/resources/config/.env.example": root / ".env.example",
            "vex_runtime/resources/hyperframes/package.json": root / "package.json",
            "vex_runtime/resources/hyperframes/package-lock.json": root / "package-lock.json",
        }
        for archive_path, source_path in resource_pairs.items():
            archive_bytes = archive.read(archive_path)
            source_bytes = source_path.read_bytes()
            if archive_path.endswith(".json"):
                matches = json.loads(archive_bytes) == json.loads(source_bytes)
            else:
                matches = archive_bytes.decode("utf-8").splitlines() == (
                    source_bytes.decode("utf-8").splitlines()
                )
            if not matches:
                raise ReleaseValidationError(
                    f"Packaged resource does not match repository source: {archive_path}"
                )


def validate_sdist(sdist_path: Path) -> None:
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        paths = {member.name for member in archive.getmembers()}
        _validate_archive_paths(paths, archive_name=sdist_path.name)
        expected_root = f"{NORMALIZED_NAME}-{__version__}"
        required = {
            f"{expected_root}/pyproject.toml",
            f"{expected_root}/README.md",
            f"{expected_root}/LICENSE",
            f"{expected_root}/visual_opportunity.py",
            f"{expected_root}/visual_program.py",
            f"{expected_root}/tools/auto_visuals.py",
            f"{expected_root}/vex_runtime/resources/config/.env.example",
            f"{expected_root}/vex_runtime/resources/hyperframes/package-lock.json",
        }
        missing = required - paths
        if missing:
            raise ReleaseValidationError(
                f"Source distribution is missing required files: {', '.join(sorted(missing))}"
            )


def write_checksums(dist_dir: Path, files: tuple[Path, Path]) -> Path:
    checksum_path = dist_dir / "SHA256SUMS"
    lines = []
    for path in sorted(files, key=lambda item: item.name):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return checksum_path


def write_github_output(path: Path) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"version={__version__}\n")
        output.write(f"prerelease={'true' if is_prerelease(__version__) else 'false'}\n")


def validate_release(
    dist_dir: Path,
    *,
    tag: str | None = None,
    checksums: bool = False,
    github_output: Path | None = None,
) -> tuple[Path, Path]:
    if tag:
        validate_tag(tag)
    wheel, sdist = _distribution_files(dist_dir)
    validate_wheel(wheel, ROOT)
    validate_sdist(sdist)
    if checksums:
        write_checksums(dist_dir, (wheel, sdist))
    if github_output:
        write_github_output(github_output)
    return wheel, sdist


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Vex release artifacts before publication."
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--tag",
        help="Release tag to validate. Omit for non-release CI package checks.",
    )
    parser.add_argument("--write-checksums", action="store_true")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    try:
        wheel, sdist = validate_release(
            args.dist_dir.resolve(),
            tag=args.tag,
            checksums=args.write_checksums,
            github_output=args.github_output,
        )
    except (OSError, ReleaseValidationError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"release validation failed: {exc}\n")
    print(f"Validated {wheel.name} and {sdist.name} for Vex {__version__}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
