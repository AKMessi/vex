from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HyperframesVariant:
    variant_id: str
    variant_index: int
    spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_index": self.variant_index,
            "spec": dict(self.spec),
        }


def _coerce_count(value: Any, default: int) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = default
    return max(1, min(count, 5))


def build_variants(spec: dict[str, Any], *, default_count: int = 3) -> list[HyperframesVariant]:
    proof_programs = [
        dict(item)
        for item in (spec.get("visual_proof_programs") or [])
        if isinstance(item, dict)
    ]
    if proof_programs:
        count = min(len(proof_programs), 8)
        if str(spec.get("composition_mode") or "").strip().lower() == "picture_in_picture":
            count = min(count, 2)
        variants: list[HyperframesVariant] = []
        for index, program in enumerate(proof_programs[:count]):
            variant_id = str(program.get("proof_program_id") or f"proof_{index + 1:02d}")
            variant_spec = {
                **dict(spec),
                **program,
                "qa_contract": {
                    **dict(spec.get("qa_contract") or {}),
                    **dict(program.get("qa_contract") or {}),
                },
                "hyperframes_variant_id": variant_id,
                "hyperframes_variant_index": index,
            }
            variants.append(
                HyperframesVariant(
                    variant_id=variant_id,
                    variant_index=index,
                    spec=variant_spec,
                )
            )
        return variants

    requested = spec.get("hyperframes_variant_count", spec.get("variant_count", default_count))
    count = _coerce_count(requested, default_count)
    if str(spec.get("composition_mode") or "").strip().lower() == "picture_in_picture":
        count = min(count, 2)
    variants: list[HyperframesVariant] = []
    for index in range(count):
        variant_id = f"variant_{index + 1:02d}"
        variant_spec = dict(spec)
        variant_spec["hyperframes_variant_id"] = variant_id
        variant_spec["hyperframes_variant_index"] = index
        variants.append(
            HyperframesVariant(
                variant_id=variant_id,
                variant_index=index,
                spec=variant_spec,
            )
        )
    return variants


def select_best_variant(
    records: list[dict[str, Any]],
    *,
    require_passing: bool = True,
) -> dict[str, Any] | None:
    successful = [record for record in records if record.get("asset_path") and not record.get("render_error")]
    if require_passing:
        successful = [
            record
            for record in successful
            if bool((record.get("qa") or {}).get("passed"))
        ]
    if not successful:
        return None
    return max(
        successful,
        key=lambda record: (
            1 if bool(((record.get("qa") or {}).get("passed"))) else 0,
            float(((record.get("qa") or {}).get("score") or 0.0)),
            -int(record.get("variant_index") or 0),
        ),
    )
