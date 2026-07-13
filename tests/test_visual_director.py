from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from tests.test_visual_communication_contract import _ir
from tests.test_visual_verifier import _payload
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.director import direct_rendered_visual
from vex_visuals.generative_authoring import compile_open_visual_program_for_spec
from vex_visuals.verifier import VisualQualityState, reset_verifier_circuits


@dataclass(frozen=True)
class _LocalQA:
    passed: bool
    score: float
    issues: list[str]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": list(self.issues),
        }


def _spec() -> dict:
    compiled, result = compile_open_visual_program_for_spec(
        {"visual_id": "visual_001", "duration": 4.8},
        ir=_ir(),
        width=1920,
        height=1080,
        fps=60,
        enable_model_authoring=False,
        candidate_count=4,
    )
    assert result.passed
    return compiled


def _asset(name: str = "initial") -> SimpleNamespace:
    return SimpleNamespace(
        renderer="remotion",
        asset_path=f"{name}.mp4",
        metadata={},
    )


def _frames(tmp_path: Path):  # noqa: ANN202
    def extract(_spec, _asset, candidate_id):  # noqa: ANN001, ANN202
        target = tmp_path / f"{candidate_id.replace(':', '_')}.png"
        target.write_bytes(b"rendered-frame-evidence")
        return [target]

    return extract


def _never_render(_spec, _round):  # noqa: ANN001, ANN202
    raise AssertionError("A verified candidate must not trigger repair rendering.")


def test_verified_render_is_published_without_repair(tmp_path: Path) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())
    outcome = direct_rendered_visual(
        _spec(),
        _asset(),
        "initial render",
        ir=_ir(),
        contract=contract.to_dict(),
        render_candidate=_never_render,
        evaluate_local_quality=lambda *_: _LocalQA(True, 0.86, []),
        extract_candidate_frames=_frames(tmp_path),
        strict=True,
        provider_models=[("test", "vision")],
        vision_request=lambda *_: _payload(contract),
        cache_dir=tmp_path / "cache",
    )

    assert outcome.passed
    assert outcome.selected.verification.state == VisualQualityState.VERIFIED
    assert len(outcome.candidates) == 1


def test_verified_multimodal_proof_can_override_soft_motion_heuristic(tmp_path: Path) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())
    outcome = direct_rendered_visual(
        _spec(),
        _asset(),
        "initial render",
        ir=_ir(),
        contract=contract.to_dict(),
        render_candidate=_never_render,
        evaluate_local_quality=lambda *_: _LocalQA(
            False,
            0.7,
            ["remotion_render_has_no_meaningful_motion"],
        ),
        extract_candidate_frames=_frames(tmp_path),
        strict=True,
        provider_models=[("test", "vision")],
        vision_request=lambda *_: _payload(contract),
    )

    assert outcome.passed
    assert not outcome.selected.hard_local_issues


def test_balanced_mode_publishes_local_proof_as_explicitly_degraded_on_outage(
    tmp_path: Path,
) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())

    def unavailable(*_args):  # noqa: ANN002, ANN202
        raise TimeoutError("provider overloaded")

    outcome = direct_rendered_visual(
        _spec(),
        _asset(),
        "initial render",
        ir=_ir(),
        contract=contract.to_dict(),
        render_candidate=_never_render,
        evaluate_local_quality=lambda *_: _LocalQA(True, 0.84, []),
        extract_candidate_frames=_frames(tmp_path),
        strict=False,
        provider_models=[("test", "vision")],
        vision_request=unavailable,
    )

    assert outcome.passed
    assert outcome.selected.verification.state == VisualQualityState.DEGRADED
    assert "visual_published_with_explicit_degraded_verification" in outcome.warnings


def test_rejected_candidate_is_repaired_rendered_and_reverified(tmp_path: Path) -> None:
    reset_verifier_circuits()
    contract = build_communication_contract(_ir())
    calls = 0
    renders = 0

    def verify(*_args):  # noqa: ANN002, ANN202
        nonlocal calls
        calls += 1
        payload = _payload(contract)
        if calls == 1:
            payload["answers"] = {}
            payload["design"]["hierarchy"] = 0.4
            payload["temporal"]["meaningful_motion"] = 0.35
        return payload

    def render(_spec, round_index):  # noqa: ANN001, ANN202
        nonlocal renders
        renders += 1
        return _asset(f"repair-{round_index}"), "typed repair render"

    outcome = direct_rendered_visual(
        _spec(),
        _asset(),
        "initial render",
        ir=_ir(),
        contract=contract.to_dict(),
        render_candidate=render,
        evaluate_local_quality=lambda *_: _LocalQA(True, 0.88, []),
        extract_candidate_frames=_frames(tmp_path),
        strict=True,
        max_repair_rounds=2,
        provider_models=[("test", "vision")],
        vision_request=verify,
    )

    assert outcome.passed
    assert renders == 1
    assert len(outcome.candidates) == 2
    assert outcome.selected.round_index == 1
    assert outcome.selected.verification.state == VisualQualityState.VERIFIED
    assert outcome.repair_history[0]["accepted_for_next_round"]
