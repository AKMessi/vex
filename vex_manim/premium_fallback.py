from __future__ import annotations

import re
from typing import Any

from manim import (
    Axes,
    Create,
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


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _ir_copy_terms(spec: dict[str, Any], *, roles: set[str] | None = None, limit: int = 4) -> list[str]:
    ir = _visual_ir(spec)
    objects = _list_value(ir.get("objects"))
    terms: list[str] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if roles and role not in roles:
            continue
        for line in _list_value(item.get("copy")):
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
        *_list_value(spec.get("steps")),
        *_list_value(spec.get("supporting_lines")),
        *_list_value(spec.get("keywords")),
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
        *_list_value(spec.get("steps")),
        *_list_value(spec.get("supporting_lines")),
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
    title_slot = scene.layout_slot("title")
    headline_text = str(spec.get("headline") or ir.get("claim") or "")
    deck_text = str(spec.get("deck") or ir.get("correct_model") or ir.get("proof_signal") or "")
    if title_slot.inner_width < 5.2:
        headline_text = _compact_phrase(headline_text, max_words=4, max_chars=26) or headline_text
        deck_text = ""
    title = scene.make_title_block(
        eyebrow=str(spec.get("eyebrow") or ""),
        headline=headline_text,
        deck=deck_text,
        max_width=max(title_slot.inner_width, 1.2),
    )
    if len(title) > 0:
        scene.place_in_slot("title_group", title, "title", role="title", x_align="left", y_align="top")
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
    if isinstance(evidence, dict):
        numbers = [str(item) for item in _list_value(evidence.get("numbers")) if str(item).strip()]
    badge = scene.make_metric_badge(
        str(numbers[0] if numbers else spec.get("emphasis_text") or spec.get("headline") or ir.get("proof_signal") or "Key signal"),
        label=str(spec.get("deck") or ir.get("correct_model") or ""),
        width=min(max(scene.layout_slot("metric").inner_width * 0.78, 1.8), 3.0),
    )
    scene.place_in_slot("hero_metric", badge, "metric", role="metric")

    chart_slot = scene.layout_slot("chart")
    axes = Axes(
        x_range=[0, 4, 1],
        y_range=[0, 4, 1],
        x_length=max(chart_slot.inner_width * 0.82, 1.4),
        y_length=max(chart_slot.inner_height * 0.68, 1.0),
        axis_config={"include_ticks": False, "include_numbers": False, "stroke_opacity": 0.42},
    )
    axes.set_stroke(color=scene.theme_color("grid"), opacity=0.42)
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
    graph_group = VGroup(axes, graph, pulse)
    scene.place_in_slot("metric_graph", graph_group, "chart", role="chart")

    terms = [_compact_phrase(term, max_words=3, max_chars=22) for term in _unique_terms(spec, limit=2)]
    support = VGroup(
        *[scene.make_ribbon_label(term, max_width=2.6) for term in terms if term]
    )
    if len(support) > 0:
        support.arrange(DOWN, buff=0.36, aligned_edge=LEFT)
        scene.place_in_slot("support_stack", support, "support", role="support", x_align="left")

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
    source_slot = scene.layout_slot("source")
    hub_slot = scene.layout_slot("hub")
    outcome_slot = scene.layout_slot("outcome")
    if scene.layout_spec.aspect_class == "vertical":
        source = scene.make_metric_badge(source_label, label="Input", width=min(max(source_slot.inner_width * 0.72, 2.0), 3.2), fill=scene.theme_color("panel_fill"), text_color=scene.theme_color("text_primary"))
        hub = scene.make_metric_badge(hub_label, label="Mechanism", width=min(max(hub_slot.inner_width * 0.78, 2.1), 3.3), fill=scene.theme_color("panel_stroke"), text_color=scene.theme_color("background"))
        destination = scene.make_metric_badge(destination_label, label="Outcome", width=min(max(outcome_slot.inner_width * 0.72, 2.0), 3.2), fill=scene.theme_color("accent"), text_color=scene.theme_color("background"))
    else:
        source = scene.make_signal_node(source_label, radius=min(max(min(source_slot.inner_width, source_slot.inner_height) * 0.36, 0.46), 0.68), color=scene.theme_color("accent_secondary"))
        hub = scene.make_signal_node(hub_label, radius=min(max(min(hub_slot.inner_width, hub_slot.inner_height) * 0.44, 0.54), 0.86), color=scene.theme_color("panel_stroke"))
        destination = scene.make_signal_node(destination_label, radius=min(max(min(outcome_slot.inner_width, outcome_slot.inner_height) * 0.36, 0.46), 0.68), color=scene.theme_color("accent"))
    scene.place_in_slot("system_source", source, "source", fit=True, register=False)
    scene.place_in_slot("system_hub", hub, "hub", fit=True, register=False)
    scene.place_in_slot("system_outcome", destination, "outcome", fit=True, register=False)
    ring = scene.make_orbit_ring(radius=1.18, color=scene.theme_color("glow"), opacity=0.28).move_to(hub.get_center())
    beam = scene.make_focus_beam(length=4.8, center=hub.get_center() + UP * 0.15, color=scene.theme_color("glow"), opacity=0.14)
    path_a = scene.route_between_slots("source", "hub", bend=-0.22, color=scene.theme_color("accent_secondary"))
    path_b = scene.route_between_slots("hub", "outcome", bend=0.22, color=scene.theme_color("accent"))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(path_a.get_start())

    scene.register_layout_group("network_nodes", VGroup(source, hub, destination), role="system", slot_id="full")
    scene.register_layout_group("network_paths", VGroup(path_a, path_b, ring, beam, pulse), role="connector", slot_id="motion_spine")

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
    before_label = _compact_phrase(str(spec.get("left_detail") or ir.get("misconception") or "Before"), max_words=3, max_chars=22)
    after_label = _compact_phrase(str(spec.get("right_detail") or ir.get("correct_model") or "After"), max_words=3, max_chars=22)
    proof_label = _compact_phrase(str(spec.get("deck") or ir.get("proof_signal") or ""), max_words=5, max_chars=34)
    before_slot = scene.layout_slot("before")
    after_slot = scene.layout_slot("after")
    if scene.layout_spec.aspect_class == "vertical":
        left_group = scene.make_metric_badge(
            before_label or "Before",
            label="Before",
            width=min(max(before_slot.inner_width * 0.72, 2.0), 3.2),
            fill=scene.theme_color("panel_fill"),
            text_color=scene.theme_color("text_primary"),
        )
        right_group = scene.make_metric_badge(
            after_label or "After",
            label="After",
            width=min(max(after_slot.inner_width * 0.72, 2.0), 3.2),
            fill=scene.theme_color("accent"),
            text_color=scene.theme_color("background"),
        )
    else:
        left_group = scene.make_signal_node(before_label or "Before", radius=min(max(min(before_slot.inner_width, before_slot.inner_height) * 0.38, 0.48), 0.94), color=scene.theme_color("panel_stroke"))
        right_group = scene.make_signal_node(after_label or "After", radius=min(max(min(after_slot.inner_width, after_slot.inner_height) * 0.42, 0.52), 1.12), color=scene.theme_color("accent"))
    left_group.set_opacity(0.72)
    scene.place_in_slot("comparison_before_state", left_group, "before", fit=True, register=False)
    scene.place_in_slot("comparison_after_state", right_group, "after", fit=True, register=False)
    bridge = scene.route_between_slots("before", "after", bend=-0.2, color=scene.theme_color("accent_secondary"), stroke_width=5.0)
    pulse = scene.make_glow_dot(radius=0.13, color=scene.theme_color("accent")).move_to(bridge.get_start())
    ring = scene.make_orbit_ring(radius=1.28, color=scene.theme_color("accent"), opacity=0.22).move_to(right_group.get_center())
    beam = scene.make_focus_beam(length=5.2, center=right_group.get_center() + DOWN * 0.08, color=scene.theme_color("glow"), opacity=0.12)
    verdict_text = _compact_phrase(str(spec.get("headline") or ir.get("claim") or "Upgrade"), max_words=4, max_chars=28)
    verdict = scene.make_metric_badge(verdict_text or "Upgrade", label=proof_label, width=min(max(scene.layout_slot("support").inner_width * 0.68, 2.0), 3.3))
    scene.place_in_slot("comparison_verdict", verdict, "support", role="support")

    scene.register_layout_group("comparison_before_state", left_group, role="diagram")
    scene.register_layout_group("comparison_after_state", right_group, role="diagram")
    scene.register_layout_group("comparison_motion_spine", VGroup(bridge, pulse, ring, beam), role="connector")

    scene.play(FadeIn(title, shift=DOWN * 0.16), FadeIn(beam), FadeIn(left_group, shift=RIGHT * 0.12), Create(bridge), run_time=intro)
    scene.play(MoveAlongPath(pulse, bridge), FadeIn(ring, scale=1.04), run_time=develop)
    scene.play(
        FadeTransform(left_group.copy(), right_group),
        left_group.animate.set_opacity(0.32).scale(0.9),
        scene.camera_focus(right_group, scale=0.92, run_time=resolve),
        FadeIn(verdict, shift=UP * 0.14),
        run_time=resolve,
    )
    scene.wait(settle)


def _timeline(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    terms = _process_terms(spec, limit=4) or ["Start", "Build", "Learn", "Ship"]
    anchors = scene.layout_route_points("main")[: len(terms)]
    route = scene.make_route_path(points=anchors, color=scene.theme_color("accent_secondary"))
    nodes = VGroup()
    labels = VGroup()
    for index, (anchor, term) in enumerate(zip(anchors, terms), start=1):
        node = scene.make_signal_node("", number=index, radius=0.38, color=scene.theme_color("panel_stroke"))
        node.move_to(anchor)
        label = scene.make_ribbon_label(term, max_width=2.6)
        scene.place_in_slot(f"timeline_step_{index}", label, f"step_{index}", role="label", register=False)
        rail = Line(anchor, label.get_bottom(), color=scene.theme_color("panel_stroke"), stroke_width=3, stroke_opacity=0.72)
        nodes.add(node)
        labels.add(VGroup(rail, label))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(anchors[0])
    footer_text = str(spec.get("deck") or spec.get("footer_text") or "").strip()
    dedupe_terms = {term.lower() for term in terms}
    footer = VGroup()
    if footer_text and footer_text.lower() not in dedupe_terms:
        footer = scene.fit_text(
            footer_text,
            max_width=scene.layout_slot("support").inner_width,
            max_font_size=20,
            min_font_size=14,
            max_lines=2,
            color=scene.theme_color("text_secondary"),
        )
        scene.place_in_slot("timeline_footer", footer, "support", role="footer")
    scene.register_layout_group("timeline_route", VGroup(route, nodes, labels, pulse), role="diagram", slot_id="route")

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
        max_width=scene.layout_slot("hero").inner_width,
        max_font_size=40,
        min_font_size=24,
        color=scene.theme_color("text_primary"),
    )
    scene.place_in_slot("kinetic_emphasis", emphasis, "hero", role="hero")
    route_points = scene.layout_route_points("main")
    ribbon_path = scene.make_route_path(points=route_points, color=scene.theme_color("accent_secondary"))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(ribbon_path.get_start())
    terms = _process_terms(spec, limit=3) or [_compact_phrase(str(spec.get("emphasis_text") or "Build"), max_words=3, max_chars=18)]
    ribbons = VGroup()
    spine_slot = scene.layout_slot("motion_spine")
    for idx, term in enumerate(terms):
        label = scene.make_ribbon_label(term, max_width=2.5)
        anchor = ribbon_path.point_from_proportion(min(0.2 + idx * 0.28, 0.86))
        offset = UP * (0.46 if idx % 2 == 0 else -0.42)
        label.move_to(anchor + offset)
        label_x = min(
            max(float(label.get_center()[0]), spine_slot.left + float(label.width) / 2.0),
            spine_slot.right - float(label.width) / 2.0,
        )
        label_y = min(
            max(float(label.get_center()[1]), spine_slot.bottom + float(label.height) / 2.0),
            spine_slot.top - float(label.height) / 2.0,
        )
        label.move_to(RIGHT * label_x + UP * label_y)
        ribbons.add(label)
    beam = scene.make_focus_beam(length=min(scene.layout_slot("hero").inner_width, 6.4), center=emphasis.get_center() + DOWN * 0.12, color=scene.theme_color("glow"), opacity=0.12)
    scene.register_layout_group("kinetic_spine", VGroup(ribbon_path, ribbons, pulse, beam), role="diagram", slot_id="motion_spine")

    scene.play(FadeIn(title, shift=DOWN * 0.16), FadeIn(beam), FadeIn(emphasis, shift=UP * 0.12), Create(ribbon_path), run_time=intro)
    if len(ribbons) > 0:
        scene.play(scene.stagger_fade_in(list(ribbons), shift=UP * 0.08, lag_ratio=0.14), run_time=develop * 0.45)
    scene.play(MoveAlongPath(pulse, ribbon_path), run_time=develop)
    scene.play(scene.camera_focus(emphasis, scale=0.92, run_time=resolve), run_time=resolve)
    scene.wait(settle)


def _interface(scene, spec: dict[str, Any], brief: dict[str, Any]) -> None:
    intro, develop, resolve, settle = _duration(spec, brief)
    title = _title(scene, spec)
    terms = _process_terms(spec, limit=3) or []
    for fallback in ["Capture", "Refine", "Ship"]:
        if len(terms) >= 3:
            break
        if fallback.lower() not in {term.lower() for term in terms}:
            terms.append(fallback)
    modules = VGroup()
    for index, term in enumerate(terms):
        slot_id = f"module_{index + 1}"
        slot = scene.layout_slot(slot_id)
        panel = scene.make_glass_panel(
            min(slot.inner_width, 2.55),
            min(slot.inner_height, 1.62),
            stroke=scene.theme_color("panel_stroke"),
            fill=scene.theme_color("panel_fill"),
        )
        label = scene.fit_text(term, max_width=max(min(slot.inner_width - 0.44, 1.8), 0.8), max_font_size=22, min_font_size=13, max_lines=3)
        stack = VGroup(panel, label.move_to(panel.get_center()))
        scene.place_in_slot(f"interface_module_{index + 1}", stack, slot_id, fit=True, register=False)
        modules.add(stack)
    focus = scene.make_focus_beam(length=3.1, center=modules[1].get_center() + DOWN * 0.58, color=scene.theme_color("glow"), opacity=0.16)
    connector_a = scene.make_connector(modules[0], modules[1], curved=False, color=scene.theme_color("accent_secondary"))
    connector_b = scene.make_connector(modules[1], modules[2], curved=False, color=scene.theme_color("accent_secondary"))
    pulse = scene.make_glow_dot(color=scene.theme_color("accent")).move_to(connector_a.get_start())
    scene.register_layout_group("interface_modules", modules, role="panel", allow_scale_down=False, avoid_safe_bottom=False, slot_id="full")
    scene.register_layout_group("interface_connectors", VGroup(connector_a, connector_b, pulse, focus), role="connector", slot_id="motion_spine")

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
