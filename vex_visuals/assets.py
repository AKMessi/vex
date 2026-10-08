"""Resolve model-selected assets only through a trusted project registry."""
from __future__ import annotations
import base64
import copy
import mimetypes
from pathlib import Path
from PIL import Image
from vex_visuals.evidence import file_digest
from vex_visuals.open_visual_program import sign_open_visual_program


def bind_program_assets(spec: dict) -> dict:
    result = copy.deepcopy(spec)
    program = result.get("open_visual_program")
    if not isinstance(program, dict):
        return result
    registry = {str(asset.get("asset_id")): asset for asset in result.get("visual_asset_registry") or [] if isinstance(asset, dict)}
    roots = [Path(path).resolve() for path in result.get("allowed_asset_roots") or []]
    changed = False
    for element in program.get("elements") or []:
        if element.get("type") != "image":
            continue
        reference = element.get("asset") or {}
        if reference.get("data_uri"):
            continue
        record = registry.get(str(reference.get("asset_id") or ""))
        if record is None:
            raise ValueError("Image asset is not registered in the project")
        path = Path(record["path"]).resolve(strict=True)
        if not roots or not any(path.is_relative_to(root) for root in roots) or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Image asset escapes the project or exceeds its size budget")
        digest = file_digest(path)
        if record.get("checksum_sha256") and record["checksum_sha256"] != digest:
            raise ValueError("Image asset content changed after registration")
        with Image.open(path) as image:
            image.verify()
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("Unsupported registered image format")
        element["asset"] = {"asset_id": record["asset_id"], "data_uri": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii"), "content_hash": digest, "provenance": record.get("source") or "project_asset", "fit": reference.get("fit") or "contain"}
        changed = True
    if changed:
        result["open_visual_program"] = sign_open_visual_program(program)
        result["open_visual_program_candidates"] = [result["open_visual_program"] if item.get("program_id") == program.get("program_id") else item for item in result.get("open_visual_program_candidates") or []]
        result["open_visual_tournament"] = {}
    return result
