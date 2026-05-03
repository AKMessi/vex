from __future__ import annotations

import re
from typing import Any

from manim import (
    AnimationGroup,
    Axes,
    Create,
    Dot,
    DOWN,
    FadeIn,
    FadeTransform,
    LaggedStart,
    LEFT,
    Line,
    MoveAlongPath,
    ORIGIN,
    RIGHT,
    Succession,
    UP,
    VGroup,
    VMobject,
)


def _duration(spec: dict[str, Any], brief: dict[str, Any]) -> tuple[float, float, float, float]:
    total = max(float(spec.get("duration") or brief.get("duration_sec") or 2.4), 1.0)
    intro = min(max(total * 0.24, 0.35), 0.9)
    develop = min(max(total * 0.34, 0.45), 1.2)
    resolve = min(max(total * 0.24, 0.35), 1.0)
    settle = max(total - intro - develop - resolve, 0.12)
    return intro, develop, resolve, settle


def _visual_ir(spec: dict[str, Any]) -> dict[str, Any]:
    payload = spec.get("visual_explanation_ir")
    return dict(payload) if isinstance(payload, dict) else {}


def _ir_copy_terms(spec: dict[str, Any], *, roles: set[str] | None = None, limit: int = 4) -> list[str]:
    ir = _visual_ir(spec)
    objects = ir.get("objects") if isinstance(ir.get("objects"), list) else []
    terms: list[str] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if roles and role not in roles:
            continue
        for line in list(item.get("copy") or []):
            compact = _compact_phrase(line, max_words=4, max_chars=28)
            if compact and compact.lower() not in {term.lower() for term in terms}:
                terms.append(compact)
                if len(terms) >= limit:
                    return terms
    for key in ["correct_model", "proof_signal", "claim", "misconception"]:
        compact = _compact_phrase(ir.get(key), max_words=4, max_chars=28)
        if compact and compact.lower() not in {term.lower() for term in terms}:
            terms.append(compact)
            if len(terms) >= limit:
                break
    return terms


def _unique_terms(spec: dict[str, Any], *, limit: int = 4) -> list[str]:
    items: list[str] = []
    for candidate in [
        *_ir_copy_terms(spec, limit=limit),
        *(spec.get("steps") or []),
        *(spec.get("supporting_lines") or []),
        *(spec.get("keywords") or []),
        spec.get("left_detail"),
        spec.get("right_detail"),
        spec.get("deck"),
        spec.get("headline"),
        spec.get("sentence_text"),
    ]:
        text = str(candidate or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {item.lower() for item in items}:
            continue
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _compact_phrase(text: Any, *, max_words: int = 3, max_chars: int = 22) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" -,\n\t")
    if not cleaned:
        return ""
    tokens = re.findall(r"[A-Za-z0-9%+.-]+(?:'[A-Za-z0-9%+.-]+)*", cleaned)
    if not tokens:
        return cleaned[:max_chars]
    filler = {
        "the", "a", "an", "this", "that", "these", "those", "we", "you", "it", "they", "our", "your", "their",
        "to", "for", "with", "by", "in", "on", "of", "but", "so", "because",
    }
    trailing = filler | {"is", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did"}
    kept: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if not kept and lowered in filler:
            continue
        kept.append(token)
        if len(kept) >= max_words:
            break
    candidate = " ".join(kept).strip() or " ".join(tokens[:max_words]).strip()
    while candidate:
        tail = candidate.split()[-1].lower()
        if tail not in trailing:
            break
        candidate = " ".join(candidate.split()[:-1]).strip()
    if len(candidate) > max_chars:
        candidate = candidate[:max_chars].rstrip()
    return candidate or cleaned[:max_chars]


def _process_terms(spec: dict[str, Any], *, limit: int = 4) -> list[str]:
    terms: list[str] = []
    for candidate in [
        *_ir_copy_terms(spec, limit=limit),
        *(spec.get("steps") or []),
        *(spec.get("supporting_lines") or []),
        spec.get("headline"),
        spec.get("deck"),
        spec.get("sentence_text"),
        spec.get("context_text"),
    ]:
        compact = _compact_phrase(candidate, max_words=3, max_chars=20)
        lowered = compact.lower()
        if not compact or lowered in {item.lower() for item in terms}:
            continue
        terms.append(compact)
        if len(terms) >= limit:
            break
    return terms


def _title(scene, spec: dict[str, Any]):
    ir = _visual_ir(spec)
    title = scene.make_title_block(
        eyebrow=str(spec.get("eyebrow") or ""),
        headline=str(spec.get("headline") or ir.get("claim") or ""),
        deck=str(spec.get("deck") or ir.get("correct_model") or ir.get("proof_signal") or ""),
        max_width=8.8,
    )
    if len(title) > 0:
        scene.register_layout_group("title_group", title, role="title")
    return title


def _node_label_fallback(spec: dict[str, Any], index: int, default: str) -> str:
    terms = _process_terms(spec, limit=4)
    if index < len(terms):
        return str(terms[index])
    return _compact_phrase(default, max_words=2, max_chars=16) or default


def _metric_story(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    ir = _visual_ir(spec)
    numbers = []
    evidence = ir.get("evidence") if isinstance(ir.get("evidence"), dict) else {}
    if isinstance(evidence.get("numbers"), list):
        numbers = [str(item) for item in evidence.get("numbers") if str(item).strip()]
    badge = scene.make_metric_badge(
        str(numbers[0] if numbers else spec.get("emphasis_text") or spec.get("headline") or ir.get("proof_signal") or "Key signal"),
        label=str(spec.get("deck") or ir.get("correct_model") or ""),
        width=3.0,
    )
    badge.move_to(LEFT * 4.1 + DOWN * 0.1)
    scene.register_layout_group("hero_metric", badge, role="metric")

    axes = Axes(
        x_range=[0, 4, 1],
        y_range=[0, 4, 1],
        x_length=4.6,
        y_length=2.7,
        axis_config={"include_ticks": False, "include_numbers": False, "stroke_opacity": 0.42},
    )
    axes.set_stroke(color=scene.theme_color("grid"), opacity=0.42)
    axes.move_to(RIGHT * 2.7 + DOWN * 0.25)
    graph_points = [
        axes.c2p(0.2, 0.5),
        axes.c2p(1.2, 1.0),
        axes.c2p(2.1, 1.8),
        axes.c2p(3.0, 2.4),
        axes.c2p(3.7, 3.2),
    ]
    graph = VMobject()
    graph.set_points_smoothly(graph_points)
    graph.set_stroke(scene.theme_color("accent_secondary"), width=5, opacity=0.92)
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(graph_points[0])
    scene.register_layout_group("metric_graph", VGroup(axes, graph, pulse), role="chart")

    terms = [_compact_phrase(term, max_words=3, max_chars=22) for term in _unique_terms(spec, limit=2)]
    support = VGroup(
        *[scene.make_ribbon_label(term, max_width=2.6) for term in terms if term]
    )
    if len(support) > 0:
        support.arrange(DOWN, buff=0.36, aligned_edge=LEFT)
        support.move_to(LEFT * 3.9 + DOWN * 2.15)
        scene.register_layout_group("support_stack", support, role="support")

    scene.play(FadeIn(title, shift=DOWN * 0.16), FadeIn(badge, shift=RIGHT * 0.16), run_time=intro)
    scene.play(Create(axes), Create(graph), run_time=develop * 0.7)
    scene.play(
        MoveAlongPath(pulse, graph),
        scene.camera_focus(axes, scale=0.92, run_time=develop),
        run_time=develop,
    )
    if len(support) > 0:
        scene.play(scene.stagger_fade_in(list(support), shift=RIGHT * 0.12, lag_ratio=0.16), run_time=resolve)
    else:
        scene.play(FadeIn(pulse, scale=1.05), run_time=resolve * 0.7)
    scene.wait(settle)


def _system_map(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    terms = _process_terms(spec, limit=4)
    source_label = terms[0] if len(terms) >= 1 else _node_label_fallback(spec, 0, "Start")
    hub_label = _compact_phrase(str(spec.get("headline") or "Core Loop"), max_words=3, max_chars=20) or "Core Loop"
    destination_label = terms[1] if len(terms) >= 2 else _node_label_fallback(spec, 1, "Outcome")
    source = scene.make_signal_node(source_label, radius=0.62, color=scene.theme_color("accent_secondary"))
    hub = scene.make_signal_node(hub_label, radius=0.86, color=scene.theme_color("panel_stroke"))
    destination = scene.make_signal_node(destination_label, radius=0.62, color=scene.theme_color("accent"))
    source.move_to(LEFT * 4.25 + DOWN * 1.35)
    hub.move_to(ORIGIN + DOWN * 0.05)
    destination.move_to(RIGHT * 4.0 + UP * 1.05)
    ring = scene.make_orbit_ring(radius=1.18, color=scene.theme_color("glow"), opacity=0.28).move_to(hub.get_center())
    beam = scene.make_focus_beam(length=4.8, center=hub.get_center() + UP * 0.15, color=scene.theme_color("glow"), opacity=0.14)
    path_a = scene.make_route_path(source.get_right() + RIGHT * 0.12, hub.get_left() + LEFT * 0.08, bend=-0.28, color=scene.theme_color("accent_secondary"))
    path_b = scene.make_route_path(hub.get_right() + RIGHT * 0.08, destination.get_left() + LEFT * 0.12, bend=0.24, color=scene.theme_color("accent"))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(source.get_right() + RIGHT * 0.12)

    scene.register_layout_group("network_nodes", VGroup(source, hub, destination), role="hero")
    scene.register_layout_group("network_paths", VGroup(path_a, path_b, ring, beam, pulse), role="diagram")

    scene.play(
        FadeIn(title, shift=DOWN * 0.16),
        FadeIn(beam),
        LaggedStart(FadeIn(source, scale=0.9), FadeIn(hub, scale=0.92), FadeIn(destination, scale=0.9), lag_ratio=0.14),
        Create(path_a),
        Create(path_b),
        FadeIn(ring),
        run_time=intro,
    )
    scene.play(MoveAlongPath(pulse, path_a), run_time=develop * 0.55)
    scene.play(
        MoveAlongPath(pulse, path_b),
        scene.camera_focus(hub, scale=0.9, run_time=develop),
        run_time=develop,
    )
    scene.play(FadeIn(ring.copy().scale(1.06), scale=1.01), run_time=resolve * 0.72)
    scene.wait(settle)


def _comparison(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    ir = _visual_ir(spec)
    left_panel = scene.make_glass_panel(3.2, 2.25, stroke=scene.theme_color("panel_stroke"), fill=scene.theme_color("panel_fill"))
    right_panel = scene.make_glass_panel(3.2, 2.25, stroke=scene.theme_color("accent"), fill=scene.theme_color("panel_fill"))
    left_text = scene.fit_text(str(spec.get("left_detail") or ir.get("misconception") or "Before"), max_width=2.4, max_font_size=24, min_font_size=14, max_lines=3)
    right_text = scene.fit_text(str(spec.get("right_detail") or ir.get("correct_model") or "After"), max_width=2.4, max_font_size=24, min_font_size=14, max_lines=3)
    left_group = VGroup(left_panel, left_text.move_to(left_panel.get_center()))
    right_group = VGroup(right_panel, right_text.move_to(right_panel.get_center()))
    left_group.move_to(LEFT * 3.0 + DOWN * 0.1)
    right_group.move_to(RIGHT * 3.0 + DOWN * 0.1)
    bridge = scene.make_route_path(left_group.get_right() + RIGHT * 0.12, right_group.get_left() + LEFT * 0.12, bend=-0.12, color=scene.theme_color("accent_secondary"))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(left_group.get_right() + RIGHT * 0.12)
    verdict = scene.make_metric_badge(str(spec.get("headline") or ir.get("claim") or "Upgrade"), label=str(spec.get("deck") or ir.get("proof_signal") or ""), width=3.1)
    verdict.move_to(ORIGIN + DOWN * 2.1)

    scene.register_layout_group("comparison_left", left_group, role="hero")
    scene.register_layout_group("comparison_right", right_group, role="hero")
    scene.register_layout_group("comparison_bridge", VGroup(bridge, pulse), role="diagram")
    scene.register_layout_group("comparison_verdict", verdict, role="support")

    scene.play(FadeIn(title, shift=DOWN * 0.16), FadeIn(left_group, shift=RIGHT * 0.12), FadeIn(right_group, shift=LEFT * 0.12), run_time=intro)
    scene.play(Create(bridge), run_time=develop * 0.4)
    scene.play(MoveAlongPath(pulse, bridge), run_time=develop * 0.6)
    scene.play(FadeTransform(left_text.copy(), right_text), scene.camera_focus(right_group, scale=0.9, run_time=resolve), FadeIn(verdict, shift=UP * 0.14), run_time=resolve)
    scene.wait(settle)


def _timeline(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    terms = _process_terms(spec, limit=4) or ["Start", "Build", "Learn", "Ship"]
    anchors = [
        LEFT * 4.6 + DOWN * 1.55,
        LEFT * 1.7 + DOWN * 0.85,
        RIGHT * 1.1 + DOWN * 1.45,
        RIGHT * 4.35 + DOWN * 0.55,
    ][: len(terms)]
    route = scene.make_route_path(points=anchors, color=scene.theme_color("accent_secondary"))
    nodes = VGroup()
    labels = VGroup()
    for index, (anchor, term) in enumerate(zip(anchors, terms), start=1):
        node = scene.make_signal_node("", number=index, radius=0.38, color=scene.theme_color("panel_stroke"))
        node.move_to(anchor)
        label_direction = UP
        rail = Line(anchor, anchor + label_direction * 0.9, color=scene.theme_color("panel_stroke"), stroke_width=3, stroke_opacity=0.72)
        label = scene.make_ribbon_label(term, max_width=2.6)
        label.next_to(rail.get_end(), label_direction, buff=0.18)
        nodes.add(node)
        labels.add(VGroup(rail, label))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(anchors[0])
    footer_text = str(spec.get("deck") or spec.get("footer_text") or "").strip()
    dedupe_terms = {term.lower() for term in terms}
    footer = VGroup()
    if footer_text and footer_text.lower() not in dedupe_terms:
        footer = scene.fit_text(
            footer_text,
            max_width=8.2,
            max_font_size=20,
            min_font_size=14,
            max_lines=2,
            color=scene.theme_color("text_secondary"),
        )
        footer.move_to(DOWN * 2.85)
    scene.register_layout_group("timeline_route", VGroup(route, nodes, labels, pulse), role="diagram")
    if len(footer) > 0:
        scene.register_layout_group("timeline_footer", footer, role="footer")

    scene.play(FadeIn(title, shift=DOWN * 0.16), Create(route), FadeIn(nodes, scale=0.92), run_time=intro)
    scene.play(scene.stagger_fade_in(list(labels), shift=UP * 0.1, lag_ratio=0.14), run_time=develop * 0.45)
    scene.play(MoveAlongPath(pulse, route), scene.camera_focus(nodes[-1], scale=0.92, run_time=develop), run_time=develop)
    if len(footer) > 0:
        scene.play(FadeIn(footer, shift=UP * 0.1), run_time=resolve)
    scene.wait(settle)


def _kinetic(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    emphasis = scene.fit_text(
        str(spec.get("headline") or spec.get("quote_text") or "Key idea"),
        max_width=8.8,
        max_font_size=40,
        min_font_size=24,
        color=scene.theme_color("text_primary"),
    )
    emphasis.move_to(UP * 0.65)
    ribbon_path = scene.make_route_path(
        points=[
            LEFT * 4.4 + DOWN * 1.5,
            LEFT * 1.6 + DOWN * 0.7,
            RIGHT * 1.1 + DOWN * 1.35,
            RIGHT * 4.2 + DOWN * 0.45,
        ],
        color=scene.theme_color("accent_secondary"),
    )
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(ribbon_path.get_start())
    terms = _process_terms(spec, limit=3) or [_compact_phrase(str(spec.get("emphasis_text") or "Build"), max_words=3, max_chars=18)]
    ribbons = VGroup()
    for idx, term in enumerate(terms):
        label = scene.make_ribbon_label(term, max_width=2.5)
        anchor = ribbon_path.point_from_proportion(min(0.2 + idx * 0.28, 0.86))
        label.move_to(anchor + UP * (0.72 if idx % 2 == 0 else -0.62))
        ribbons.add(label)
    beam = scene.make_focus_beam(length=6.4, center=emphasis.get_center() + DOWN * 0.12, color=scene.theme_color("glow"), opacity=0.12)
    scene.register_layout_group("kinetic_emphasis", emphasis, role="hero")
    scene.register_layout_group("kinetic_spine", VGroup(ribbon_path, ribbons, pulse, beam), role="diagram")

    scene.play(FadeIn(title, shift=DOWN * 0.16), FadeIn(beam), FadeIn(emphasis, shift=UP * 0.12), Create(ribbon_path), run_time=intro)
    if len(ribbons) > 0:
        scene.play(scene.stagger_fade_in(list(ribbons), shift=UP * 0.08, lag_ratio=0.14), run_time=develop * 0.45)
    scene.play(MoveAlongPath(pulse, ribbon_path), run_time=develop)
    scene.play(scene.camera_focus(emphasis, scale=0.92, run_time=resolve), run_time=resolve)
    scene.wait(settle)


def _interface(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    terms = _process_terms(spec, limit=3) or ["Capture", "Refine", "Ship"]
    modules = VGroup()
    for index, term in enumerate(terms):
        panel = scene.make_glass_panel(2.55, 1.62, stroke=scene.theme_color("panel_stroke"), fill=scene.theme_color("panel_fill"))
        label = scene.fit_text(term, max_width=1.8, max_font_size=22, min_font_size=13, max_lines=3)
        stack = VGroup(panel, label.move_to(panel.get_center()))
        stack.move_to(LEFT * (3.2 - index * 3.2) + DOWN * (0.1 if index == 1 else 0.45))
        modules.add(stack)
    focus = scene.make_focus_beam(length=3.1, center=modules[1].get_center() + DOWN * 0.58, color=scene.theme_color("glow"), opacity=0.16)
    connector_a = scene.make_connector(modules[0], modules[1], curved=False, color=scene.theme_color("accent_secondary"))
    connector_b = scene.make_connector(modules[1], modules[2], curved=False, color=scene.theme_color("accent_secondary"))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(connector_a.get_start())
    scene.register_layout_group("interface_modules", modules, role="hero")
    scene.register_layout_group("interface_connectors", VGroup(connector_a, connector_b, pulse, focus), role="diagram")

    scene.play(FadeIn(title, shift=DOWN * 0.16), FadeIn(focus), scene.stagger_fade_in(list(modules), shift=UP * 0.08, lag_ratio=0.12), run_time=intro)
    scene.play(Create(connector_a), Create(connector_b), run_time=develop * 0.35)
    scene.play(Succession(MoveAlongPath(pulse, connector_a), MoveAlongPath(pulse, connector_b)), run_time=develop)
    scene.play(scene.camera_focus(modules[-1], scale=0.9, run_time=resolve), run_time=resolve)
    scene.wait(settle)


def run_premium_blueprint_scene(scene, spec: dict[str, Any], brief: dict[str, Any], blueprint: dict[str, Any]) -> None:
    family = str(brief.get("scene_family") or "")
    if family == "system_map":
        return _system_map(scene, spec, brief)
    if family in {"metric_story", "dashboard_build"}:
        return _metric_story(scene, spec, brief)
    if family == "comparison_morph":
        return _comparison(scene, spec, brief)
    if family == "timeline_journey":
        return _timeline(scene, spec, brief)
    if family == "interface_focus":
        return _interface(scene, spec, brief)
    return _kinetic(scene, spec, brief)
