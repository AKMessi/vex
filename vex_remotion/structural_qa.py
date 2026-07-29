from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from vex_visuals.scene_graph import validate_scene_graph


REMOTION_STRUCTURAL_QA_VERSION = "remotion-structural-qa-v1"


@dataclass(frozen=True)
class RemotionStructuralQA:
    version: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    resolved_layout: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Rect:
    x: float
    y: float
    width: float
    height: float
    z_index: int

    def copy(self) -> _Rect:
        return _Rect(self.x, self.y, self.width, self.height, self.z_index)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "width": round(self.width, 6),
            "height": round(self.height, 6),
            "z_index": self.z_index,
        }


def evaluate_remotion_structure(program: dict[str, Any]) -> RemotionStructuralQA:
    graph = dict(program.get("scene_graph") or {})
    if not graph:
        return RemotionStructuralQA(
            version=REMOTION_STRUCTURAL_QA_VERSION,
            passed=False,
            score=0.0,
            issues=["remotion_structural_qa_scene_graph_missing"],
        )

    issues: list[str] = []
    warnings: list[str] = []
    graph_validation = validate_scene_graph(graph)
    if not graph_validation.passed:
        issues.extend(
            f"remotion_structural_qa_graph_invalid:{item}"
            for item in graph_validation.errors[:12]
        )
    warnings.extend(
        f"remotion_structural_qa_graph_warning:{item}"
        for item in graph_validation.warnings[:8]
    )

    nodes = [
        dict(item)
        for item in graph.get("nodes") or []
        if isinstance(item, dict)
    ]
    relations = [
        dict(item)
        for item in graph.get("relations") or []
        if isinstance(item, dict)
    ]
    rects = solve_scene_graph_layout(graph)
    safe_area = _safe_area(graph)
    canvas = dict(graph.get("canvas") or {})
    width = max(1, int(_number(canvas.get("width"), program.get("width") or 1920)))
    height = max(1, int(_number(canvas.get("height"), program.get("height") or 1080)))

    unsafe_nodes: list[str] = []
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        rect = rects.get(node_id)
        if rect is None or not _inside_safe_area(rect, safe_area):
            unsafe_nodes.append(node_id)
    if unsafe_nodes:
        issues.extend(
            f"remotion_structural_qa_unsafe_area_intrusion:{node_id}"
            for node_id in unsafe_nodes[:12]
        )

    constraint_violations = _constraint_violations(graph, rects)
    issues.extend(constraint_violations)

    quality_contract = dict(graph.get("quality_contract") or {})
    required_objects = {
        str(item)
        for item in quality_contract.get("required_object_ids") or []
        if str(item)
    }
    required_relations = {
        str(item)
        for item in quality_contract.get("required_relation_ids") or []
        if str(item)
    }
    represented_objects = {
        str((node.get("binding") or {}).get("id") or "")
        for node in nodes
        if str((node.get("binding") or {}).get("kind") or "") == "object"
    }
    represented_relations = {
        str((relation.get("binding") or {}).get("id") or "")
        for relation in relations
        if str((relation.get("binding") or {}).get("kind") or "") == "relation"
    }
    missing_objects = sorted(required_objects - represented_objects)
    missing_relations = sorted(required_relations - represented_relations)
    issues.extend(
        f"remotion_structural_qa_missing_semantic_binding:object:{item}"
        for item in missing_objects[:12]
    )
    issues.extend(
        f"remotion_structural_qa_missing_semantic_binding:relation:{item}"
        for item in missing_relations[:12]
    )

    text_reports = [
        _text_fit_report(
            node,
            rects.get(str(node.get("node_id") or "")),
            width=width,
            height=height,
            typography=dict((graph.get("design_system") or {}).get("typography") or {}),
        )
        for node in nodes
        if str((node.get("content") or {}).get("text") or "").strip()
        and not bool(node.get("decorative"))
    ]
    text_reports = [item for item in text_reports if item is not None]
    for report in text_reports:
        if bool(report["overflow"]):
            issues.append(
                "remotion_structural_qa_clipped_text:"
                + str(report["node_id"])
            )
        elif float(report["capacity_ratio"]) > 0.88:
            warnings.append(
                "remotion_structural_qa_text_near_capacity:"
                + str(report["node_id"])
            )

    detached_relations: list[str] = []
    overlapping_relations: list[str] = []
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        source = rects.get(str(relation.get("source_id") or ""))
        target = rects.get(str(relation.get("target_id") or ""))
        if source is None or target is None:
            detached_relations.append(relation_id)
            continue
        intersection_x, intersection_y = _overlap(source, target)
        if intersection_x > 1e-6 and intersection_y > 1e-6:
            overlapping_relations.append(relation_id)
            continue
        start, end = _relation_endpoints(source, target)
        if (
            _distance_to_rect(start, source) > 1e-6
            or _distance_to_rect(end, target) > 1e-6
        ):
            detached_relations.append(relation_id)
    issues.extend(
        f"remotion_structural_qa_relation_endpoint_detached:{item}"
        for item in detached_relations[:12]
    )
    issues.extend(
        f"remotion_structural_qa_relation_endpoints_overlap:{item}"
        for item in overlapping_relations[:12]
    )

    motion_intrusions, maximum_simultaneous_motion = _motion_safety(
        graph,
        rects,
        safe_area,
    )
    issues.extend(
        f"remotion_structural_qa_motion_exits_safe_area:{node_id}@{timestamp:.4f}"
        for node_id, timestamp in motion_intrusions[:12]
    )
    allowed_simultaneous = max(
        1,
        int(
            _number(
                (graph.get("motion_graph") or {}).get("maximum_simultaneous_motion"),
                3,
            )
        ),
    )
    if maximum_simultaneous_motion > allowed_simultaneous:
        warnings.append(
            "remotion_structural_qa_motion_concurrency_exceeded:"
            f"{maximum_simultaneous_motion}>{allowed_simultaneous}"
        )

    global_overlap_pairs = _semantic_overlap_pairs(nodes, rects)
    warnings.extend(
        f"remotion_structural_qa_semantic_overlap:{left}:{right}"
        for left, right in global_overlap_pairs[:8]
    )

    node_count = max(len(nodes), 1)
    text_count = max(len(text_reports), 1)
    binding_total = max(len(required_objects) + len(required_relations), 1)
    binding_missing = len(missing_objects) + len(missing_relations)
    safe_score = 1.0 - min(len(unsafe_nodes) / node_count, 1.0)
    constraint_score = 1.0 - min(
        len(constraint_violations) / max(len(graph.get("constraints") or []), 1),
        1.0,
    )
    binding_score = 1.0 - min(binding_missing / binding_total, 1.0)
    text_score = 1.0 - min(
        len([item for item in text_reports if bool(item["overflow"])]) / text_count,
        1.0,
    )
    relation_score = 1.0 - min(
        (len(detached_relations) + len(overlapping_relations))
        / max(len(relations), 1),
        1.0,
    )
    motion_score = 1.0 - min(len(motion_intrusions) / node_count, 1.0)
    graph_score = 1.0 if graph_validation.passed else 0.0
    score = (
        graph_score * 0.18
        + safe_score * 0.16
        + constraint_score * 0.16
        + binding_score * 0.18
        + text_score * 0.16
        + relation_score * 0.08
        + motion_score * 0.08
    )
    metrics = {
        "graph_validation": graph_validation.to_dict(),
        "canvas": {"width": width, "height": height},
        "safe_area": safe_area,
        "node_count": len(nodes),
        "relation_count": len(relations),
        "constraint_count": len(graph.get("constraints") or []),
        "unsafe_node_count": len(unsafe_nodes),
        "constraint_violation_count": len(constraint_violations),
        "required_object_binding_count": len(required_objects),
        "required_relation_binding_count": len(required_relations),
        "missing_object_bindings": missing_objects,
        "missing_relation_bindings": missing_relations,
        "text_fit": text_reports,
        "detached_relation_count": len(detached_relations),
        "overlapping_relation_count": len(overlapping_relations),
        "motion_safe_area_intrusion_count": len(motion_intrusions),
        "maximum_simultaneous_motion": maximum_simultaneous_motion,
        "allowed_simultaneous_motion": allowed_simultaneous,
        "semantic_overlap_pair_count": len(global_overlap_pairs),
    }
    return RemotionStructuralQA(
        version=REMOTION_STRUCTURAL_QA_VERSION,
        passed=not issues,
        score=round(max(0.0, min(score, 1.0)), 4),
        issues=_unique(issues),
        warnings=_unique(warnings),
        metrics=metrics,
        resolved_layout={
            node_id: rect.to_dict()
            for node_id, rect in sorted(rects.items())
        },
    )


def solve_scene_graph_layout(graph: dict[str, Any]) -> dict[str, _Rect]:
    nodes = [
        dict(item)
        for item in graph.get("nodes") or []
        if isinstance(item, dict)
    ]
    safe_area = _safe_area(graph)
    rects = {
        str(node.get("node_id") or ""): _anchor_adjusted(
            dict(node.get("layout") or {})
        )
        for node in nodes
        if str(node.get("node_id") or "")
    }
    constraints = sorted(
        [
            dict(item)
            for item in graph.get("constraints") or []
            if isinstance(item, dict)
        ],
        key=lambda item: _number(item.get("priority"), 100.0),
        reverse=True,
    )
    for constraint in constraints:
        targets = [
            str(item)
            for item in constraint.get("targets") or []
            if str(item) in rects
        ]
        constraint_type = str(constraint.get("type") or "")
        axis = (
            str(constraint.get("axis") or "")
            if str(constraint.get("axis") or "") in {"x", "y"}
            else "both"
        )
        gap = _clamp(_number(constraint.get("gap"), 0.02), 0.0, 0.5)
        padding = _clamp(
            _number(constraint.get("padding"), 0.0),
            0.0,
            0.25,
        )
        if not targets:
            continue
        if constraint_type == "keep_inside_safe_area":
            for target in targets:
                rects[target] = _clamp_to_safe_area(rects[target], safe_area)
            continue
        if constraint_type == "contain" and len(targets) >= 2:
            _solve_containment(rects, targets, padding, safe_area)
            continue
        if constraint_type == "align" and len(targets) >= 2:
            center_x = sum(
                rects[target].x + rects[target].width / 2 for target in targets
            ) / len(targets)
            center_y = sum(
                rects[target].y + rects[target].height / 2 for target in targets
            ) / len(targets)
            for target in targets:
                rect = rects[target].copy()
                if axis != "y":
                    rect.x = center_x - rect.width / 2
                if axis != "x":
                    rect.y = center_y - rect.height / 2
                rects[target] = _clamp_to_safe_area(rect, safe_area)
            continue
        if constraint_type == "distribute" and len(targets) >= 3:
            use_x = axis != "y"
            for group in _distribution_groups(rects, targets, axis):
                if len(group) < 3:
                    continue
                ordered = sorted(
                    group,
                    key=lambda target: (
                        rects[target].x + rects[target].width / 2
                        if use_x
                        else rects[target].y + rects[target].height / 2
                    ),
                )
                first = rects[ordered[0]]
                last = rects[ordered[-1]]
                start = (
                    first.x + first.width / 2
                    if use_x
                    else first.y + first.height / 2
                )
                end = (
                    last.x + last.width / 2
                    if use_x
                    else last.y + last.height / 2
                )
                for index, target in enumerate(ordered):
                    rect = rects[target].copy()
                    center = start + (end - start) * index / max(len(ordered) - 1, 1)
                    if use_x:
                        rect.x = center - rect.width / 2
                    else:
                        rect.y = center - rect.height / 2
                    rects[target] = _clamp_to_safe_area(rect, safe_area)
            continue
        if constraint_type in {"avoid_overlap", "minimum_gap"}:
            _solve_separation(rects, targets, axis, gap, safe_area)
    for node_id, rect in list(rects.items()):
        rects[node_id] = _clamp_to_safe_area(rect, safe_area)
    return rects


def _safe_area(graph: dict[str, Any]) -> dict[str, float]:
    source = dict((graph.get("canvas") or {}).get("safe_area") or {})
    return {
        "left": _clamp(_number(source.get("left"), 0.04), 0.0, 0.2),
        "right": _clamp(_number(source.get("right"), 0.04), 0.0, 0.2),
        "top": _clamp(_number(source.get("top"), 0.04), 0.0, 0.2),
        "bottom": _clamp(_number(source.get("bottom"), 0.04), 0.0, 0.2),
    }


def _anchor_adjusted(layout: dict[str, Any]) -> _Rect:
    width = _clamp(_number(layout.get("width"), 0.1), 0.001, 1.0)
    height = _clamp(_number(layout.get("height"), 0.1), 0.001, 1.0)
    x = _number(layout.get("x"), 0.0)
    y = _number(layout.get("y"), 0.0)
    anchor = str(layout.get("anchor") or "top_left")
    if anchor == "center":
        x -= width / 2
        y -= height / 2
    elif anchor == "top_right":
        x -= width
    elif anchor == "bottom_left":
        y -= height
    elif anchor == "bottom_right":
        x -= width
        y -= height
    return _Rect(
        x=x,
        y=y,
        width=width,
        height=height,
        z_index=int(_clamp(round(_number(layout.get("z_index"), 4)), 0, 100)),
    )


def _clamp_to_safe_area(
    rect: _Rect,
    safe_area: dict[str, float],
) -> _Rect:
    left = safe_area["left"]
    right = 1.0 - safe_area["right"]
    top = safe_area["top"]
    bottom = 1.0 - safe_area["bottom"]
    width = min(rect.width, max(right - left, 0.001))
    height = min(rect.height, max(bottom - top, 0.001))
    return _Rect(
        x=_clamp(rect.x, left, max(left, right - width)),
        y=_clamp(rect.y, top, max(top, bottom - height)),
        width=width,
        height=height,
        z_index=rect.z_index,
    )


def _solve_separation(
    rects: dict[str, _Rect],
    targets: list[str],
    axis: str,
    gap: float,
    safe_area: dict[str, float],
) -> None:
    for _pass in range(5):
        changed = False
        for left_index, left_id in enumerate(targets):
            for right_id in targets[left_index + 1 :]:
                left = rects[left_id]
                right = rects[right_id]
                intersection_x, intersection_y = _overlap(left, right, gap / 2)
                if intersection_x <= 0 or intersection_y <= 0:
                    continue
                use_x = axis == "x" or (
                    axis == "both" and intersection_x <= intersection_y
                )
                shift = (
                    (intersection_x if use_x else intersection_y) + gap
                ) / 2
                left_next = left.copy()
                right_next = right.copy()
                if use_x:
                    left_next.x -= shift
                    right_next.x += shift
                else:
                    left_next.y -= shift
                    right_next.y += shift
                rects[left_id] = _clamp_to_safe_area(left_next, safe_area)
                rects[right_id] = _clamp_to_safe_area(right_next, safe_area)
                changed = True
        if not changed:
            break


def _solve_containment(
    rects: dict[str, _Rect],
    targets: list[str],
    padding: float,
    safe_area: dict[str, float],
) -> None:
    container_id = targets[0]
    container = _clamp_to_safe_area(rects[container_id], safe_area)
    rects[container_id] = container
    left = container.x + padding
    right = container.x + container.width - padding
    top = container.y + padding
    bottom = container.y + container.height - padding
    available_width = max(right - left, 0.001)
    available_height = max(bottom - top, 0.001)
    for child_id in targets[1:]:
        child = rects[child_id]
        width = min(child.width, available_width)
        height = min(child.height, available_height)
        rects[child_id] = _Rect(
            x=_clamp(child.x, left, max(left, right - width)),
            y=_clamp(child.y, top, max(top, bottom - height)),
            width=width,
            height=height,
            z_index=child.z_index,
        )


def _distribution_groups(
    rects: dict[str, _Rect],
    targets: list[str],
    axis: str,
) -> list[list[str]]:
    use_x = axis != "y"
    ordered = sorted(
        targets,
        key=lambda target: (
            rects[target].y + rects[target].height / 2
            if use_x
            else rects[target].x + rects[target].width / 2
        ),
    )
    groups: list[dict[str, Any]] = []
    for target in ordered:
        rect = rects[target]
        center = (
            rect.y + rect.height / 2
            if use_x
            else rect.x + rect.width / 2
        )
        size = rect.height if use_x else rect.width
        current = groups[-1] if groups else None
        if current is None or abs(center - float(current["center"])) > max(
            0.04,
            min(size, float(current["size"])) * 0.45,
        ):
            groups.append({"center": center, "size": size, "targets": [target]})
            continue
        current_targets = list(current["targets"])
        current_targets.append(target)
        current["targets"] = current_targets
        current["center"] = (
            float(current["center"]) * (len(current_targets) - 1) + center
        ) / len(current_targets)
        current["size"] = max(float(current["size"]), size)
    return [list(group["targets"]) for group in groups]


def _constraint_violations(
    graph: dict[str, Any],
    rects: dict[str, _Rect],
) -> list[str]:
    issues: list[str] = []
    for constraint in graph.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        constraint_type = str(constraint.get("type") or "")
        constraint_id = str(constraint.get("constraint_id") or "")
        targets = [
            str(item)
            for item in constraint.get("targets") or []
            if str(item) in rects
        ]
        if constraint_type == "contain" and len(targets) >= 2:
            container_id = targets[0]
            container = rects[container_id]
            padding = _clamp(
                _number(constraint.get("padding"), 0.0),
                0.0,
                0.25,
            )
            left = container.x + padding
            right = container.x + container.width - padding
            top = container.y + padding
            bottom = container.y + container.height - padding
            for child_id in targets[1:]:
                child = rects[child_id]
                if (
                    child.x < left - 1e-5
                    or child.y < top - 1e-5
                    or child.x + child.width > right + 1e-5
                    or child.y + child.height > bottom + 1e-5
                ):
                    issues.append(
                        "remotion_structural_qa_constraint_violation:"
                        f"{constraint_id}:{container_id}:{child_id}"
                    )
            continue
        if constraint_type not in {"avoid_overlap", "minimum_gap"}:
            continue
        gap = _clamp(_number(constraint.get("gap"), 0.02), 0.0, 0.5)
        for index, left_id in enumerate(targets):
            for right_id in targets[index + 1 :]:
                overlap_x, overlap_y = _overlap(
                    rects[left_id],
                    rects[right_id],
                    gap / 2,
                )
                if overlap_x > 1e-5 and overlap_y > 1e-5:
                    issues.append(
                        "remotion_structural_qa_constraint_violation:"
                        f"{constraint_id}:{left_id}:{right_id}"
                    )
    return _unique(issues)


def _text_fit_report(
    node: dict[str, Any],
    rect: _Rect | None,
    *,
    width: int,
    height: int,
    typography: dict[str, Any],
) -> dict[str, Any] | None:
    if rect is None:
        return None
    content = " ".join(str((node.get("content") or {}).get("text") or "").split())
    if not content:
        return None
    primitive = str(node.get("primitive") or "")
    framed = primitive in {
        "data_chart",
        "graph_node",
        "masked_media",
        "metric_mark",
        "semantic_token",
    }
    horizontal_padding = (
        2 * max(10, round(width * 0.012))
        if framed
        else 0
    )
    vertical_padding = horizontal_padding
    available_width = max(2.0, rect.width * width - horizontal_padding)
    available_height = max(2.0, rect.height * height - vertical_padding)
    minimum_font = _number(typography.get("minimum_font_px"), 18.0)
    role = str(node.get("role") or "")
    minimum_font = (
        max(30.0, minimum_font * 1.7)
        if role == "title"
        else max(18.0, minimum_font)
    )
    requested_font = _clamp(
        _number(
            (node.get("style") or {}).get("font_size"),
            68.0 if role == "title" else 30.0,
        ),
        minimum_font,
        128.0,
    )
    words = content.split()
    estimated_single_line_width = max(
        minimum_font,
        sum(len(word) * requested_font * 0.555 for word in words)
        + max(len(words) - 1, 0) * requested_font * 0.32,
    )
    line_height = _clamp(
        _number(typography.get("line_height"), 1.12),
        0.8,
        2.0,
    )
    fit_font = _clamp(
        min(
            requested_font * available_width / estimated_single_line_width,
            available_height / line_height,
        ),
        minimum_font,
        requested_font,
    )
    line_count = 1
    current_width = 0.0
    max_word_width = 0.0
    for word in words:
        word_width = len(word) * fit_font * 0.555
        max_word_width = max(max_word_width, word_width)
        spacing = fit_font * 0.32 if current_width else 0.0
        if (
            current_width
            and current_width + spacing + word_width
            > available_width + 1e-6
        ):
            line_count += 1
            current_width = word_width
        else:
            current_width += spacing + word_width
    required_height = line_count * fit_font * line_height
    width_ratio = max_word_width / max(available_width, 1.0)
    height_ratio = required_height / max(available_height, 1.0)
    capacity_ratio = max(width_ratio, height_ratio)
    return {
        "node_id": str(node.get("node_id") or ""),
        "font_size_px": round(fit_font, 3),
        "minimum_font_px": round(minimum_font, 3),
        "available_width_px": round(available_width, 3),
        "available_height_px": round(available_height, 3),
        "estimated_line_count": line_count,
        "required_height_px": round(required_height, 3),
        "capacity_ratio": round(capacity_ratio, 4),
        "overflow": bool(width_ratio > 1.02 or height_ratio > 1.05),
    }


def _motion_safety(
    graph: dict[str, Any],
    rects: dict[str, _Rect],
    safe_area: dict[str, float],
) -> tuple[list[tuple[str, float]], int]:
    motion_graph = dict(graph.get("motion_graph") or {})
    tracks = [
        dict(item)
        for item in motion_graph.get("tracks") or []
        if isinstance(item, dict)
    ]
    tracks_by_target: dict[str, list[dict[str, Any]]] = {}
    for track in tracks:
        tracks_by_target.setdefault(str(track.get("target_id") or ""), []).append(track)
    sample_times = [
        _clamp(_number(item, 0.0), 0.0, 1.0)
        for item in (graph.get("telemetry_contract") or {}).get("sample_times") or []
    ]
    sample_times = sorted({0.0, 1.0, *sample_times})
    intrusions: list[tuple[str, float]] = []
    maximum_simultaneous = 0
    for timestamp in sample_times:
        active_nodes = 0
        for node_id, rect in rects.items():
            target_tracks = tracks_by_target.get(node_id, [])
            opacity = _track_value(target_tracks, "opacity", timestamp, 1.0)
            translate_x = _track_value(
                target_tracks,
                "translate_x",
                timestamp,
                0.0,
            )
            translate_y = _track_value(
                target_tracks,
                "translate_y",
                timestamp,
                0.0,
            )
            scale = max(
                0.001,
                _track_value(target_tracks, "scale", timestamp, 1.0),
            )
            transformed = _transform_rect(
                rect,
                translate_x=translate_x,
                translate_y=translate_y,
                scale=scale,
            )
            if opacity > 0.08 and not _inside_safe_area(
                transformed,
                safe_area,
                tolerance=0.004,
            ):
                intrusions.append((node_id, timestamp))
            if any(_track_is_changing(track, timestamp) for track in target_tracks):
                active_nodes += 1
        maximum_simultaneous = max(maximum_simultaneous, active_nodes)
    return list(dict.fromkeys(intrusions)), maximum_simultaneous


def _track_value(
    tracks: Iterable[dict[str, Any]],
    property_name: str,
    timestamp: float,
    fallback: float,
) -> float:
    track = next(
        (
            item
            for item in tracks
            if str(item.get("property") or "") == property_name
        ),
        None,
    )
    if track is None:
        return fallback
    keyframes = sorted(
        [
            dict(item)
            for item in track.get("keyframes") or []
            if isinstance(item, dict)
            and math.isfinite(_number(item.get("t"), math.nan))
            and math.isfinite(_number(item.get("value"), math.nan))
        ],
        key=lambda item: _number(item.get("t"), 0.0),
    )
    if not keyframes:
        return fallback
    if timestamp <= _number(keyframes[0].get("t"), 0.0):
        return _number(keyframes[0].get("value"), fallback)
    if timestamp >= _number(keyframes[-1].get("t"), 1.0):
        return _number(keyframes[-1].get("value"), fallback)
    right_index = next(
        index
        for index, item in enumerate(keyframes)
        if _number(item.get("t"), 0.0) >= timestamp
    )
    left = keyframes[max(0, right_index - 1)]
    right = keyframes[right_index]
    left_t = _number(left.get("t"), 0.0)
    right_t = _number(right.get("t"), 1.0)
    local = (timestamp - left_t) / max(right_t - left_t, 0.0001)
    eased = _easing_value(
        local,
        str(right.get("easing") or left.get("easing") or "linear"),
    )
    left_value = _number(left.get("value"), fallback)
    right_value = _number(right.get("value"), fallback)
    return left_value + (right_value - left_value) * eased


def _track_is_changing(track: dict[str, Any], timestamp: float) -> bool:
    keyframes = sorted(
        [
            dict(item)
            for item in track.get("keyframes") or []
            if isinstance(item, dict)
        ],
        key=lambda item: _number(item.get("t"), 0.0),
    )
    for left, right in zip(keyframes, keyframes[1:]):
        left_t = _number(left.get("t"), 0.0)
        right_t = _number(right.get("t"), 0.0)
        if (
            left_t <= timestamp <= right_t
            and abs(
                _number(right.get("value"), 0.0)
                - _number(left.get("value"), 0.0)
            )
            > 1e-5
        ):
            return True
    return False


def _transform_rect(
    rect: _Rect,
    *,
    translate_x: float,
    translate_y: float,
    scale: float,
) -> _Rect:
    width = rect.width * scale
    height = rect.height * scale
    center_x = rect.x + rect.width / 2 + translate_x
    center_y = rect.y + rect.height / 2 + translate_y
    return _Rect(
        x=center_x - width / 2,
        y=center_y - height / 2,
        width=width,
        height=height,
        z_index=rect.z_index,
    )


def _semantic_overlap_pairs(
    nodes: list[dict[str, Any]],
    rects: dict[str, _Rect],
) -> list[tuple[str, str]]:
    semantic = [
        str(node.get("node_id") or "")
        for node in nodes
        if not bool(node.get("decorative"))
        and str(node.get("node_id") or "") in rects
        and not str(node.get("parent_id") or "")
    ]
    pairs: list[tuple[str, str]] = []
    for index, left_id in enumerate(semantic):
        for right_id in semantic[index + 1 :]:
            left = rects[left_id]
            right = rects[right_id]
            intersection_x, intersection_y = _overlap(left, right)
            if intersection_x <= 0 or intersection_y <= 0:
                continue
            intersection = intersection_x * intersection_y
            smaller_area = min(
                left.width * left.height,
                right.width * right.height,
            )
            if intersection / max(smaller_area, 1e-8) > 0.35:
                pairs.append((left_id, right_id))
    return pairs


def _relation_endpoints(
    source: _Rect,
    target: _Rect,
) -> tuple[tuple[float, float], tuple[float, float]]:
    source_center = (
        source.x + source.width / 2,
        source.y + source.height / 2,
    )
    target_center = (
        target.x + target.width / 2,
        target.y + target.height / 2,
    )
    delta_x = target_center[0] - source_center[0]
    delta_y = target_center[1] - source_center[1]
    if abs(delta_x) >= abs(delta_y):
        return (
            (
                source.x + source.width if delta_x >= 0 else source.x,
                source_center[1],
            ),
            (
                target.x if delta_x >= 0 else target.x + target.width,
                target_center[1],
            ),
        )
    return (
        (
            source_center[0],
            source.y + source.height if delta_y >= 0 else source.y,
        ),
        (
            target_center[0],
            target.y if delta_y >= 0 else target.y + target.height,
        ),
    )


def _distance_to_rect(point: tuple[float, float], rect: _Rect) -> float:
    x, y = point
    inside_x = rect.x - 1e-8 <= x <= rect.x + rect.width + 1e-8
    inside_y = rect.y - 1e-8 <= y <= rect.y + rect.height + 1e-8
    on_edge = (
        abs(x - rect.x) <= 1e-7
        or abs(x - (rect.x + rect.width)) <= 1e-7
        or abs(y - rect.y) <= 1e-7
        or abs(y - (rect.y + rect.height)) <= 1e-7
    )
    return 0.0 if inside_x and inside_y and on_edge else 1.0


def _inside_safe_area(
    rect: _Rect,
    safe_area: dict[str, float],
    *,
    tolerance: float = 1e-7,
) -> bool:
    return (
        rect.x >= safe_area["left"] - tolerance
        and rect.y >= safe_area["top"] - tolerance
        and rect.x + rect.width <= 1.0 - safe_area["right"] + tolerance
        and rect.y + rect.height <= 1.0 - safe_area["bottom"] + tolerance
    )


def _overlap(
    left: _Rect,
    right: _Rect,
    gap: float = 0.0,
) -> tuple[float, float]:
    return (
        min(left.x + left.width + gap, right.x + right.width + gap)
        - max(left.x - gap, right.x - gap),
        min(left.y + left.height + gap, right.y + right.height + gap)
        - max(left.y - gap, right.y - gap),
    )


def _easing_value(progress: float, easing: str) -> float:
    value = _clamp(progress, 0.0, 1.0)
    if easing in {"ease_out", "spring_snappy"}:
        return 1.0 - (1.0 - value) ** 3
    if easing == "ease_in":
        return value**3
    if easing in {"ease_in_out", "spring_gentle"}:
        return value * value * (3.0 - 2.0 * value)
    return value


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = [
    "REMOTION_STRUCTURAL_QA_VERSION",
    "RemotionStructuralQA",
    "evaluate_remotion_structure",
    "solve_scene_graph_layout",
]
