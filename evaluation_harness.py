from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from intent_compiler import compile_intent
from state import ProjectState, utc_now_iso


EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    instruction: str
    expected_tools: list[str]
    expected_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    min_confidence: float = 0.78
    requires_project: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    instruction: str
    passed: bool
    expected_tools: list[str]
    actual_tools: list[str]
    confidence: float
    issues: list[str]
    plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    created_at: str
    suite: str
    passed: bool
    score: float
    passed_count: int
    failed_count: int
    case_count: int
    cases: list[EvaluationCaseResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [case.to_dict() for case in self.cases]
        return payload


def built_in_intent_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="trim_export_chain",
            instruction="trim the first 30 seconds and export it for instagram",
            expected_tools=["trim_clip", "export_video"],
            expected_params={
                "trim_clip": {"start": "0", "end": "30"},
                "export_video": {"preset_name": "instagram_reels"},
            },
            tags=["editing", "export"],
        ),
        EvaluationCase(
            case_id="subtitle_chain",
            instruction="add subtitles at the bottom",
            expected_tools=["burn_subtitles"],
            expected_params={"burn_subtitles": {"position": "bottom"}},
            requires_project=False,
            tags=["subtitles"],
        ),
        EvaluationCase(
            case_id="auto_visuals_hyperframes",
            instruction="add 3 generated visuals with hyperframes",
            expected_tools=["add_auto_visuals"],
            expected_params={
                "add_auto_visuals": {
                    "max_visuals": 3,
                    "requested_count": 3,
                    "coverage_policy": "target_count",
                    "renderer": "hyperframes",
                }
            },
            tags=["visuals", "renderers"],
        ),
        EvaluationCase(
            case_id="encode_plan",
            instruction="convert this mov file to mp4 and compress it",
            expected_tools=["plan_encode"],
            tags=["encode"],
        ),
        EvaluationCase(
            case_id="generate_video",
            instruction="generate a portrait hyperframes video about sparse attention in 20 seconds",
            expected_tools=["generate_video"],
            expected_params={"generate_video": {"aspect": "portrait", "duration_sec": 20}},
            requires_project=False,
            tags=["generation"],
        ),
    ]


def run_intent_evaluation(
    cases: Iterable[EvaluationCase] | None = None,
    *,
    state: ProjectState | None = None,
    suite: str = "intent_compiler_builtin",
) -> EvaluationReport:
    case_results = [
        evaluate_intent_case(case, state=state)
        for case in (cases or built_in_intent_cases())
    ]
    passed_count = sum(1 for result in case_results if result.passed)
    failed_count = len(case_results) - passed_count
    score = passed_count / len(case_results) if case_results else 1.0
    return EvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        created_at=utc_now_iso(),
        suite=suite,
        passed=failed_count == 0,
        score=round(score, 4),
        passed_count=passed_count,
        failed_count=failed_count,
        case_count=len(case_results),
        cases=case_results,
    )


def evaluate_intent_case(case: EvaluationCase, *, state: ProjectState | None = None) -> EvaluationCaseResult:
    effective_state = state if case.requires_project else None
    plan = compile_intent(case.instruction, effective_state)
    issues: list[str] = []
    if plan is None:
        return EvaluationCaseResult(
            case_id=case.case_id,
            instruction=case.instruction,
            passed=False,
            expected_tools=case.expected_tools,
            actual_tools=[],
            confidence=0.0,
            issues=["compiler returned no deterministic plan"],
            plan=None,
        )
    actual_tools = [step.tool for step in plan.steps]
    if actual_tools != case.expected_tools:
        issues.append(f"expected tools {case.expected_tools}, got {actual_tools}")
    if plan.confidence < case.min_confidence:
        issues.append(f"confidence {plan.confidence:.3f} below {case.min_confidence:.3f}")
    for tool_name, expected_subset in case.expected_params.items():
        matching_steps = [step for step in plan.steps if step.tool == tool_name]
        if not matching_steps:
            issues.append(f"missing expected params for absent tool {tool_name}")
            continue
        params = matching_steps[0].params
        missing = {
            key: value
            for key, value in expected_subset.items()
            if params.get(key) != value
        }
        if missing:
            issues.append(f"{tool_name} param mismatch: {missing}")
    return EvaluationCaseResult(
        case_id=case.case_id,
        instruction=case.instruction,
        passed=not issues,
        expected_tools=case.expected_tools,
        actual_tools=actual_tools,
        confidence=plan.confidence,
        issues=issues,
        plan=plan.to_dict(),
    )


def write_evaluation_report(report: EvaluationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    _atomic_write_json(path, report.to_dict())
    return path


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
