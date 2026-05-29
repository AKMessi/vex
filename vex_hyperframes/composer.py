from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any

from vex_hyperframes.design import DesignIR, build_design_ir, root_class_names
from vex_hyperframes.skill_pack import retrieve_skill_slices


SUPPORTED_TEMPLATES = {
    "data_journey",
    "signal_network",
    "kinetic_route",
    "spotlight_compare",
    "interface_cascade",
    "ribbon_quote",
    "causal_chain",
    "flywheel_loop",
    "decision_matrix",
    "anatomy_cutaway",
    "stack_ranking",
    "contrast_ladder",
    "proof_sequence",
    "narrative_arc",
    "concept_map",
    "problem_solution",
    "myth_buster",
    "checklist_reveal",
    "risk_radar",
    "opportunity_map",
    "scorecard",
    "pipeline_xray",
    "decision_tree",
    "momentum_wave",
    "focus_ring",
    "timeline_filmstrip",
    "quote_breakdown",
    "market_map",
    "mechanism_blueprint",
    "data_pulse",
    "metric_callout",
    "keyword_stack",
    "timeline_steps",
    "comparison_split",
    "quote_focus",
    "system_flow",
    "stat_grid",
}


@dataclass(frozen=True)
class HyperframesComposition:
    composition_id: str
    html: str
    metadata: dict[str, Any]


def _clean_id(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "visual")).strip("-_").lower()
    return cleaned or "visual"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _theme_defaults(spec: dict[str, Any]) -> dict[str, str]:
    theme = dict(spec.get("theme") or {})
    defaults = {
        "background": "#08111F",
        "panel_fill": "#101E33",
        "panel_stroke": "#5BC0EB",
        "accent": "#F59E0B",
        "accent_secondary": "#38BDF8",
        "glow": "#1D4ED8",
        "eyebrow_fill": "#14324D",
        "eyebrow_text": "#E0F2FE",
        "grid": "#244760",
        "text_primary": "#F8FAFC",
        "text_secondary": "#D6E3F3",
    }
    defaults.update({key: str(value) for key, value in theme.items() if value})
    return defaults


def _text(value: Any, *, fallback: str = "", max_chars: int = 90) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or fallback or "")).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip(" ,.;:-")
    return cleaned


def _html(value: Any, *, fallback: str = "", max_chars: int = 90) -> str:
    return html.escape(_text(value, fallback=fallback, max_chars=max_chars), quote=True)


def _list(value: Any, *, fallback: list[str] | None = None, limit: int = 4, max_chars: int = 38) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    items = [_text(item, max_chars=max_chars) for item in raw_items if _text(item, max_chars=max_chars)]
    if not items and fallback:
        items = [_text(item, max_chars=max_chars) for item in fallback if _text(item, max_chars=max_chars)]
    return items[:limit]


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def _numeric_seed(spec: dict[str, Any]) -> int:
    source = "|".join(
        [
            str(spec.get("headline") or ""),
            str(spec.get("emphasis_text") or ""),
            str(spec.get("template") or ""),
            str(spec.get("visual_id") or ""),
        ]
    )
    return sum((index + 1) * ord(char) for index, char in enumerate(source)) % 997


def _clip(track: int, duration: float, *, class_name: str = "") -> str:
    classes = f"clip {class_name}".strip()
    return f'class="{classes}" data-start="0" data-duration="{duration:.3f}" data-track-index="{track}"'


def _animate_attrs(name: str, delay: float, span: float = 0.74, *, y: int = 36, scale: float = 0.985) -> str:
    return (
        f'data-anim="{name}" data-delay="{delay:.3f}" data-span="{span:.3f}" '
        f'data-y="{y}" data-scale="{scale:.3f}"'
    )


def _header(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int]:
    eyebrow = _html(spec.get("eyebrow") or spec.get("visual_type_hint") or "INSIGHT", max_chars=22)
    headline = _html(spec.get("headline") or spec.get("sentence_text") or "Key idea", max_chars=54)
    deck = _html(spec.get("deck") or spec.get("footer_text") or spec.get("context_text") or "", max_chars=74)
    html_block = f"""
      <section id="hf-header" {_clip(track, duration, class_name="hf-header")} {_animate_attrs("rise", 0.04, 0.58, y=28)}>
        <div class="eyebrow">{eyebrow}</div>
        <h1>{headline}</h1>
        <p>{deck}</p>
      </section>
    """
    return html_block, track + 1


def _stage_background(duration: float, track: int) -> tuple[str, int]:
    html_block = f"""
      <div id="hf-bg-grid" {_clip(track, duration, class_name="bg-grid")} aria-hidden="true"></div>
      <div id="hf-bg-wash" {_clip(track + 1, duration, class_name="bg-wash")} aria-hidden="true"></div>
      <div id="hf-bg-rails" {_clip(track + 2, duration, class_name="bg-rails")} aria-hidden="true">
        <span></span><span></span><span></span>
      </div>
    """
    return html_block, track + 3


def _metric_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    emphasis = _html(spec.get("emphasis_text") or spec.get("headline") or "Proof", max_chars=24)
    support = _list(
        spec.get("supporting_lines"),
        fallback=[spec.get("footer_text") or spec.get("context_text") or "Signal becomes visible"],
        limit=3,
        max_chars=40,
    )
    metric_facts = [
        item
        for item in (spec.get("metric_facts") or [])
        if isinstance(item, dict) and str(item.get("value") or "").strip()
    ][:3]
    seed = _numeric_seed(spec)
    values = [0.38 + (((seed + index * 17) % 44) / 100.0) for index in range(5)]
    if metric_facts:
        support = [
            _text(item.get("label") or item.get("value"), max_chars=40)
            for item in metric_facts
        ] + support
    support = list(dict.fromkeys(item for item in support if item))[:3]
    card_labels = support or [_text(spec.get("headline"), fallback="Source-backed signal", max_chars=40)]
    cards = "\n".join(
        f"""
          <div class="stat-card" {_animate_attrs("rise", 0.24 + index * 0.07, 0.52, y=30)}>
            <b>{html.escape(label, quote=True)}</b>
            {f'<span>{html.escape(str(metric_facts[index]["value"]), quote=True)}</span>' if index < len(metric_facts) else ''}
          </div>
        """
        for index, label in enumerate(card_labels)
    )
    bars = "\n".join(
        f'<span class="bar" data-bar="{value:.3f}" style="--bar-target:{value:.3f}"></span>'
        for value in values
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage metric-stage")}>
        <div class="metric-hero" {_animate_attrs("scale", 0.16, 0.68, y=18, scale=0.92)}>
          <div class="metric-label">MEASURED SHIFT</div>
          <div class="metric-value">{emphasis}</div>
          <div class="metric-bars">{bars}</div>
        </div>
        <aside class="stat-stack">{cards}</aside>
      </main>
    """
    return html_block, track + 1, {"bar_count": len(values)}


def _flow_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    steps = _list(
        spec.get("steps"),
        fallback=spec.get("keywords") if isinstance(spec.get("keywords"), list) else None,
        limit=4,
        max_chars=26,
    )
    if len(steps) < 2:
        steps = [_text(spec.get("headline"), fallback="Input", max_chars=24), _text(spec.get("footer_text"), fallback="Outcome", max_chars=24)]
    nodes = "\n".join(
        f"""
          <div class="flow-node flow-node-{index + 1}" {_animate_attrs("pop", 0.18 + index * 0.16, 0.48, y=22, scale=0.88)}>
            <span>{index + 1:02d}</span>
            <b>{html.escape(label, quote=True)}</b>
          </div>
        """
        for index, label in enumerate(steps[:4])
    )
    lines = "\n".join(
        f'<span class="flow-line flow-line-{index + 1}" data-line data-delay="{0.28 + index * 0.16:.3f}"></span>'
        for index in range(max(len(steps[:4]) - 1, 1))
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage flow-stage")}>
        <div class="flow-lines">{lines}</div>
        <div class="flow-nodes">{nodes}</div>
        <div class="signal-pulse" aria-hidden="true"></div>
      </main>
    """
    return html_block, track + 1, {"node_count": len(steps[:4])}


def _route_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    steps = _list(spec.get("steps"), fallback=spec.get("keywords") if isinstance(spec.get("keywords"), list) else None, limit=4, max_chars=24)
    if len(steps) < 2:
        steps = ["Start", _text(spec.get("headline"), fallback="Focus", max_chars=24), "Result"]
    labels = "\n".join(
        f'<li {_animate_attrs("rise", 0.22 + index * 0.12, 0.46, y=26)}><span>{index + 1}</span><b>{html.escape(label, quote=True)}</b></li>'
        for index, label in enumerate(steps[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage route-stage")}>
        <svg class="route-svg" viewBox="0 0 1280 420" aria-hidden="true">
          <path class="route-shadow" pathLength="1" d="M80,300 C260,90 475,370 640,210 S990,80 1200,250" />
          <path class="route-path" data-route pathLength="1" d="M80,300 C260,90 475,370 640,210 S990,80 1200,250" />
        </svg>
        <span class="route-tracer" data-route-dot></span>
        <ol class="route-labels">{labels}</ol>
      </main>
    """
    return html_block, track + 1, {"step_count": len(steps[:4])}


def _compare_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    left_label = _html(spec.get("left_label") or "Before", max_chars=24)
    right_label = _html(spec.get("right_label") or "After", max_chars=24)
    left_detail = _html(spec.get("left_detail") or spec.get("sentence_text") or "Scattered signal", max_chars=70)
    right_detail = _html(spec.get("right_detail") or spec.get("context_text") or "Focused outcome", max_chars=70)
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage compare-stage")}>
        <article class="compare-card before-card" {_animate_attrs("slide-left", 0.14, 0.58, y=0)}>
          <span>01</span>
          <h2>{left_label}</h2>
          <p>{left_detail}</p>
        </article>
        <div class="compare-bridge" data-line data-delay="0.460"></div>
        <article class="compare-card after-card" {_animate_attrs("slide-right", 0.36, 0.58, y=0)}>
          <span>02</span>
          <h2>{right_label}</h2>
          <p>{right_detail}</p>
        </article>
      </main>
    """
    return html_block, track + 1, {"comparison": True}


def _interface_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    steps = _list(spec.get("steps"), fallback=spec.get("supporting_lines"), limit=3, max_chars=30)
    if not steps:
        steps = ["Input captured", "Model scores context", "Action rendered"]
    rows = "\n".join(
        f"""
          <div class="ui-row" {_animate_attrs("slide-right", 0.22 + index * 0.1, 0.48, y=0)}>
            <span></span><b>{html.escape(label, quote=True)}</b><i>{82 + index * 5}%</i>
          </div>
        """
        for index, label in enumerate(steps[:3])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage interface-stage")}>
        <section class="browser-card primary-browser" {_animate_attrs("rise", 0.16, 0.62, y=36)}>
          <div class="chrome-dots"><span></span><span></span><span></span></div>
          <div class="ui-title">{_html(spec.get("headline") or "Workflow", max_chars=36)}</div>
          <div class="ui-rows">{rows}</div>
        </section>
        <section class="browser-card rear-card rear-one"></section>
        <section class="browser-card rear-card rear-two"></section>
      </main>
    """
    return html_block, track + 1, {"ui_rows": len(steps[:3])}


def _quote_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    quote = _html(spec.get("quote_text") or spec.get("headline") or spec.get("sentence_text") or "Make the idea visible.", max_chars=86)
    keywords = _list(spec.get("keywords"), fallback=[spec.get("emphasis_text") or "Focus"], limit=4, max_chars=20)
    chips = "\n".join(
        f'<span {_animate_attrs("pop", 0.48 + index * 0.08, 0.38, y=18, scale=0.86)}>{html.escape(keyword, quote=True)}</span>'
        for index, keyword in enumerate(keywords)
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage quote-stage")}>
        <div class="ribbon ribbon-one" aria-hidden="true"></div>
        <div class="ribbon ribbon-two" aria-hidden="true"></div>
        <blockquote {_animate_attrs("scale", 0.16, 0.68, y=18, scale=0.94)}>{quote}</blockquote>
        <div class="keyword-row">{chips}</div>
      </main>
    """
    return html_block, track + 1, {"keyword_count": len(keywords)}


def _causal_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    semantic = dict(spec.get("semantic_frame") or {})
    cause = _text(semantic.get("cause") or spec.get("left_detail") or spec.get("sentence_text"), fallback="Cause", max_chars=42)
    effect = _text(semantic.get("effect") or spec.get("right_detail") or spec.get("deck"), fallback="Effect", max_chars=42)
    mechanism = _text(semantic.get("mental_model") or spec.get("footer_text") or spec.get("headline"), fallback="Mechanism", max_chars=62)
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage causal-stage")}>
        <article class="causal-card cause-card" {_animate_attrs("slide-left", 0.12, 0.54, y=0)}>
          <span>CAUSE</span><b>{html.escape(cause, quote=True)}</b>
        </article>
        <div class="causal-spine">
          <i data-line data-delay="0.280"></i>
          <strong {_animate_attrs("pop", 0.42, 0.46, y=14, scale=0.84)}>{html.escape(mechanism, quote=True)}</strong>
        </div>
        <article class="causal-card effect-card" {_animate_attrs("slide-right", 0.34, 0.54, y=0)}>
          <span>EFFECT</span><b>{html.escape(effect, quote=True)}</b>
        </article>
      </main>
    """
    return html_block, track + 1, {"causal": True}


def _flywheel_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    steps = _list(spec.get("steps"), fallback=spec.get("keywords"), limit=4, max_chars=24)
    if len(steps) < 3:
        steps = ["Input", _text(spec.get("headline"), fallback="Action", max_chars=22), "Feedback", "Improve"]
    nodes = "\n".join(
        f"""
          <li class="loop-node loop-node-{index + 1}" {_animate_attrs("pop", 0.18 + index * 0.12, 0.46, y=18, scale=0.86)}>
            <span>{index + 1}</span><b>{html.escape(label, quote=True)}</b>
          </li>
        """
        for index, label in enumerate(steps[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage flywheel-stage")}>
        <div class="loop-orbit" aria-hidden="true"></div>
        <div class="loop-core" {_animate_attrs("scale", 0.14, 0.62, y=16, scale=0.9)}>{_html(spec.get("emphasis_text") or spec.get("headline"), fallback="Loop", max_chars=28)}</div>
        <ol class="loop-nodes">{nodes}</ol>
      </main>
    """
    return html_block, track + 1, {"loop_nodes": len(steps[:4])}


def _matrix_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    criteria = _list(spec.get("supporting_lines"), fallback=spec.get("keywords"), limit=3, max_chars=24)
    if len(criteria) < 3:
        criteria = ["Speed", "Clarity", "Compounding"]
    left = _text(spec.get("left_label") or "Old path", max_chars=24)
    right = _text(spec.get("right_label") or "Better path", max_chars=24)
    rows = "\n".join(
        f"""
          <div class="matrix-row" {_animate_attrs("rise", 0.22 + index * 0.1, 0.48, y=20)}>
            <b>{html.escape(label, quote=True)}</b>
            <span class="weak"></span>
            <span class="strong"></span>
          </div>
        """
        for index, label in enumerate(criteria[:3])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage matrix-stage")}>
        <section class="matrix-card" {_animate_attrs("scale", 0.14, 0.6, y=20, scale=0.94)}>
          <div class="matrix-head"><i></i><b>{html.escape(left, quote=True)}</b><b>{html.escape(right, quote=True)}</b></div>
          {rows}
        </section>
      </main>
    """
    return html_block, track + 1, {"criteria": len(criteria[:3])}


def _anatomy_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    layers = _list(spec.get("steps"), fallback=spec.get("supporting_lines") or spec.get("keywords"), limit=4, max_chars=26)
    if len(layers) < 3:
        layers = ["Surface signal", "Hidden mechanism", "Decision layer", "Outcome"]
    layer_html = "\n".join(
        f"""
          <div class="anatomy-layer anatomy-layer-{index + 1}" {_animate_attrs("slide-right", 0.16 + index * 0.12, 0.52, y=0)}>
            <span>{index + 1:02d}</span><b>{html.escape(label, quote=True)}</b>
          </div>
        """
        for index, label in enumerate(layers[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage anatomy-stage")}>
        <div class="anatomy-core" {_animate_attrs("scale", 0.12, 0.62, y=18, scale=0.92)}>{_html(spec.get("headline"), fallback="System", max_chars=32)}</div>
        <section class="anatomy-layers">{layer_html}</section>
      </main>
    """
    return html_block, track + 1, {"layers": len(layers[:4])}


def _ranking_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    items = _list(spec.get("steps"), fallback=spec.get("supporting_lines") or spec.get("keywords"), limit=4, max_chars=28)
    if len(items) < 3:
        items = [_text(spec.get("headline"), fallback="Primary signal", max_chars=24), "Context", "Timing"]
    rows = "\n".join(
        f"""
          <li {_animate_attrs("slide-right", 0.16 + index * 0.1, 0.5, y=0)}>
            <span>{index + 1}</span><b>{html.escape(label, quote=True)}</b><i style="--rank:{1 - index * 0.17:.2f}"></i>
          </li>
        """
        for index, label in enumerate(items[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage ranking-stage")}>
        <ol class="ranking-list">{rows}</ol>
      </main>
    """
    return html_block, track + 1, {"ranked_items": len(items[:4])}


def _ladder_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    left = _text(spec.get("left_detail") or spec.get("left_label") or "Before", max_chars=36)
    right = _text(spec.get("right_detail") or spec.get("right_label") or "After", max_chars=36)
    middle = _list(spec.get("steps"), fallback=spec.get("supporting_lines"), limit=2, max_chars=26)
    labels = [left, *(middle[:2] or ["Pressure", "Shift"]), right]
    rungs = "\n".join(
        f"""
          <li class="ladder-rung ladder-rung-{index + 1}" {_animate_attrs("rise", 0.16 + index * 0.12, 0.48, y=22)}>
            <span>{index + 1}</span><b>{html.escape(label, quote=True)}</b>
          </li>
        """
        for index, label in enumerate(labels[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage ladder-stage")}>
        <ol class="ladder-list">{rungs}</ol>
        <div class="ladder-line" data-line data-delay="0.240"></div>
      </main>
    """
    return html_block, track + 1, {"rungs": len(labels[:4])}


def _proof_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    points = _list(spec.get("supporting_lines"), fallback=spec.get("keywords"), limit=4, max_chars=32)
    if len(points) < 3:
        points = ["Claim", "Evidence", "Pattern", "Payoff"]
    cards = "\n".join(
        f"""
          <article class="proof-card proof-card-{index + 1}" {_animate_attrs("rise", 0.16 + index * 0.1, 0.5, y=24)}>
            <span>{index + 1:02d}</span><b>{html.escape(label, quote=True)}</b>
          </article>
        """
        for index, label in enumerate(points[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage proof-stage")}>
        <div class="proof-chain">{cards}</div>
      </main>
    """
    return html_block, track + 1, {"proof_points": len(points[:4])}


def _arc_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    beats = _list(spec.get("steps"), fallback=spec.get("supporting_lines") or spec.get("keywords"), limit=3, max_chars=28)
    if len(beats) < 3:
        beats = ["Setup", _text(spec.get("headline"), fallback="Tension", max_chars=24), "Payoff"]
    labels = "\n".join(
        f'<li {_animate_attrs("pop", 0.2 + index * 0.16, 0.46, y=18, scale=0.86)}><span>{index + 1}</span><b>{html.escape(label, quote=True)}</b></li>'
        for index, label in enumerate(beats[:3])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage arc-stage")}>
        <svg class="arc-svg" viewBox="0 0 1180 380" aria-hidden="true">
          <path class="arc-shadow" d="M80,300 C320,40 780,40 1100,300" pathLength="1" />
          <path class="arc-path" data-route d="M80,300 C320,40 780,40 1100,300" pathLength="1" />
        </svg>
        <ol class="arc-labels">{labels}</ol>
      </main>
    """
    return html_block, track + 1, {"arc_beats": len(beats[:3])}


def _map_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    nodes = _list(
        spec.get("steps"),
        fallback=spec.get("supporting_lines") or spec.get("keywords"),
        limit=5,
        max_chars=26,
    )
    if len(nodes) < 4:
        nodes = [
            _text(spec.get("headline"), fallback="Core idea", max_chars=24),
            "Signal",
            "Context",
            "Action",
            "Outcome",
        ]
    node_html = "\n".join(
        f"""
          <li class="map-node map-node-{index + 1}" {_animate_attrs("pop", 0.16 + index * 0.08, 0.46, y=18, scale=0.86)}>
            <span>{index + 1:02d}</span><b>{html.escape(label, quote=True)}</b>
          </li>
        """
        for index, label in enumerate(nodes[:5])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage map-stage")}>
        <div class="map-core" {_animate_attrs("scale", 0.12, 0.6, y=18, scale=0.9)}>{_html(spec.get("headline"), fallback="Concept", max_chars=34)}</div>
        <ol class="map-nodes">{node_html}</ol>
        <div class="map-links" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>
      </main>
    """
    return html_block, track + 1, {"map_nodes": len(nodes[:5])}


def _checklist_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    items = _list(spec.get("steps"), fallback=spec.get("supporting_lines") or spec.get("keywords"), limit=5, max_chars=30)
    if len(items) < 3:
        items = [_text(spec.get("headline"), fallback="Define target", max_chars=28), "Check signal", "Ship next step"]
    rows = "\n".join(
        f"""
          <li {_animate_attrs("slide-right", 0.14 + index * 0.09, 0.44, y=0)}>
            <span>{index + 1}</span><b>{html.escape(label, quote=True)}</b><i></i>
          </li>
        """
        for index, label in enumerate(items[:5])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage checklist-stage")}>
        <ol class="checklist">{rows}</ol>
      </main>
    """
    return html_block, track + 1, {"check_items": len(items[:5])}


def _radar_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    signals = _list(spec.get("supporting_lines"), fallback=spec.get("keywords"), limit=4, max_chars=26)
    if len(signals) < 3:
        signals = ["Weak signal", "High leverage", "Hidden risk", "Next move"]
    blips = "\n".join(f'<span class="radar-blip radar-blip-{index + 1}"></span>' for index in range(4))
    rows = "\n".join(
        f'<li {_animate_attrs("rise", 0.24 + index * 0.08, 0.42, y=20)}><span>{index + 1:02d}</span><b>{html.escape(label, quote=True)}</b></li>'
        for index, label in enumerate(signals[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage radar-stage")}>
        <div class="radar-disc" {_animate_attrs("scale", 0.12, 0.62, y=12, scale=0.9)}>
          <i></i>{blips}
        </div>
        <ol class="radar-list">{rows}</ol>
      </main>
    """
    return html_block, track + 1, {"radar_signals": len(signals[:4])}


def _xray_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    layers = _list(spec.get("steps"), fallback=spec.get("supporting_lines") or spec.get("keywords"), limit=5, max_chars=28)
    if len(layers) < 3:
        layers = ["Input", "Hidden layer", "Decision", "Output"]
    rows = "\n".join(
        f"""
          <li class="xray-layer xray-layer-{index + 1}" {_animate_attrs("slide-right", 0.14 + index * 0.08, 0.48, y=0)}>
            <span>{index + 1:02d}</span><b>{html.escape(label, quote=True)}</b><i></i>
          </li>
        """
        for index, label in enumerate(layers[:5])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage xray-stage")}>
        <div class="xray-title" {_animate_attrs("scale", 0.12, 0.58, y=14, scale=0.92)}>{_html(spec.get("headline"), fallback="Blueprint", max_chars=36)}</div>
        <ol class="xray-layers">{rows}</ol>
      </main>
    """
    return html_block, track + 1, {"xray_layers": len(layers[:5])}


def _tree_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    branches = _list(spec.get("steps"), fallback=spec.get("supporting_lines") or spec.get("keywords"), limit=4, max_chars=28)
    if len(branches) < 3:
        branches = ["If signal is clear", "Commit", "If not", "Refine"]
    branch_html = "\n".join(
        f"""
          <li class="tree-node tree-node-{index + 1}" {_animate_attrs("pop", 0.24 + index * 0.1, 0.42, y=16, scale=0.86)}>
            <span>{index + 1}</span><b>{html.escape(label, quote=True)}</b>
          </li>
        """
        for index, label in enumerate(branches[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage tree-stage")}>
        <div class="tree-root" {_animate_attrs("scale", 0.12, 0.56, y=16, scale=0.9)}>{_html(spec.get("headline"), fallback="Decision", max_chars=32)}</div>
        <div class="tree-lines" aria-hidden="true"><span data-line data-delay="0.220"></span><span data-line data-delay="0.300"></span><span data-line data-delay="0.300"></span></div>
        <ol class="tree-nodes">{branch_html}</ol>
      </main>
    """
    return html_block, track + 1, {"branches": len(branches[:4])}


def _wave_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    support = _list(spec.get("supporting_lines"), fallback=spec.get("keywords"), limit=3, max_chars=28)
    if len(support) < 2:
        support = ["Signal rises", "Momentum compounds", "Outcome lands"]
    chips = "\n".join(
        f'<span {_animate_attrs("rise", 0.34 + index * 0.08, 0.4, y=20)}>{html.escape(label, quote=True)}</span>'
        for index, label in enumerate(support[:3])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage wave-stage")}>
        <svg class="wave-svg" viewBox="0 0 1200 420" aria-hidden="true">
          <path class="wave-shadow" d="M60,310 C250,270 290,110 470,150 S720,390 910,210 1030,90 1140,120" pathLength="1" />
          <path class="wave-path" data-route d="M60,310 C250,270 290,110 470,150 S720,390 910,210 1030,90 1140,120" pathLength="1" />
        </svg>
        <div class="wave-value" {_animate_attrs("scale", 0.14, 0.62, y=16, scale=0.9)}>{_html(spec.get("emphasis_text") or spec.get("headline"), fallback="Momentum", max_chars=26)}</div>
        <div class="wave-chips">{chips}</div>
      </main>
    """
    return html_block, track + 1, {"wave_labels": len(support[:3])}


def _filmstrip_stage(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    beats = _list(spec.get("steps"), fallback=spec.get("supporting_lines") or spec.get("keywords"), limit=4, max_chars=26)
    if len(beats) < 3:
        beats = ["Setup", "Shift", "Proof", "Payoff"]
    frames = "\n".join(
        f"""
          <li {_animate_attrs("rise", 0.16 + index * 0.1, 0.44, y=20)}>
            <span>{index + 1:02d}</span><b>{html.escape(label, quote=True)}</b>
          </li>
        """
        for index, label in enumerate(beats[:4])
    )
    html_block = f"""
      <main id="hf-stage" {_clip(track, duration, class_name="stage filmstrip-stage")}>
        <ol class="filmstrip">{frames}</ol>
      </main>
    """
    return html_block, track + 1, {"frames": len(beats[:4])}


def _stage_for_template(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    template = str(spec.get("template") or "ribbon_quote").strip().lower()
    if template in {"concept_map", "opportunity_map", "market_map"}:
        return _map_stage(spec, duration, track)
    if template in {"checklist_reveal"}:
        return _checklist_stage(spec, duration, track)
    if template in {"risk_radar", "scorecard"}:
        return _radar_stage(spec, duration, track)
    if template in {"pipeline_xray", "quote_breakdown", "mechanism_blueprint"}:
        return _xray_stage(spec, duration, track)
    if template == "decision_tree":
        return _tree_stage(spec, duration, track)
    if template in {"momentum_wave", "focus_ring", "data_pulse"}:
        return _wave_stage(spec, duration, track)
    if template == "timeline_filmstrip":
        return _filmstrip_stage(spec, duration, track)
    if template in {"problem_solution", "myth_buster"}:
        return _compare_stage(spec, duration, track)
    if template == "causal_chain":
        return _causal_stage(spec, duration, track)
    if template == "flywheel_loop":
        return _flywheel_stage(spec, duration, track)
    if template == "decision_matrix":
        return _matrix_stage(spec, duration, track)
    if template == "anatomy_cutaway":
        return _anatomy_stage(spec, duration, track)
    if template == "stack_ranking":
        return _ranking_stage(spec, duration, track)
    if template == "contrast_ladder":
        return _ladder_stage(spec, duration, track)
    if template == "proof_sequence":
        return _proof_stage(spec, duration, track)
    if template == "narrative_arc":
        return _arc_stage(spec, duration, track)
    if template in {"data_journey", "metric_callout", "stat_grid"}:
        return _metric_stage(spec, duration, track)
    if template in {"signal_network", "system_flow"}:
        return _flow_stage(spec, duration, track)
    if template in {"kinetic_route", "timeline_steps"}:
        return _route_stage(spec, duration, track)
    if template in {"spotlight_compare", "comparison_split"}:
        return _compare_stage(spec, duration, track)
    if template == "interface_cascade":
        return _interface_stage(spec, duration, track)
    return _quote_stage(spec, duration, track)


def _css(theme: dict[str, str], width: int, height: int, ir: DesignIR) -> str:
    return f"""
    :root {{
      --bg: {theme["background"]};
      --panel: {theme["panel_fill"]};
      --stroke: {theme["panel_stroke"]};
      --accent: {theme["accent"]};
      --accent-2: {theme["accent_secondary"]};
      --glow: {theme["glow"]};
      --grid: {theme["grid"]};
      --text: {theme["text_primary"]};
      --muted: {theme["text_secondary"]};
      --eyebrow-bg: {theme["eyebrow_fill"]};
      --eyebrow-text: {theme["eyebrow_text"]};
      --safe-x: {ir.safe_margin_px}px;
      --safe-bottom: {ir.subtitle_safe_px}px;
      --contrast-target: {ir.art_direction.contrast_target:.2f};
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; width: {width}px; height: {height}px; overflow: hidden; background: #000; }}
    body {{ font-family: "Inter", "Segoe UI", Arial, sans-serif; }}
    #root {{
      position: relative;
      width: {width}px;
      height: {height}px;
      overflow: hidden;
      color: var(--text);
      background:
        linear-gradient(130deg, color-mix(in srgb, var(--bg) 92%, black), var(--bg) 45%, color-mix(in srgb, var(--panel) 72%, black)),
        var(--bg);
      isolation: isolate;
      letter-spacing: 0;
    }}
    #root::before {{
      content: "";
      position: absolute;
      inset: 0;
      z-index: 1;
      pointer-events: none;
      background:
        radial-gradient(1200px 720px at 74% 18%, color-mix(in srgb, var(--glow) 18%, transparent), transparent 56%),
        linear-gradient(115deg, transparent 0%, color-mix(in srgb, var(--accent) 8%, transparent) 42%, transparent 72%);
      opacity: .9;
      mix-blend-mode: screen;
    }}
    #root::after {{
      content: "";
      position: absolute;
      inset: 0;
      z-index: 20;
      pointer-events: none;
      background-image:
        repeating-linear-gradient(0deg, color-mix(in srgb, white 6%, transparent) 0 1px, transparent 1px 5px),
        repeating-linear-gradient(90deg, color-mix(in srgb, black 10%, transparent) 0 1px, transparent 1px 7px);
      opacity: .035;
    }}
    .ad-cinematic_editorial::before {{
      background:
        linear-gradient(105deg, color-mix(in srgb, var(--accent) 18%, transparent), transparent 36%, color-mix(in srgb, var(--accent-2) 14%, transparent)),
        radial-gradient(1100px 680px at 20% 78%, color-mix(in srgb, var(--glow) 22%, transparent), transparent 62%);
    }}
    .ad-product_ui::before {{
      background:
        linear-gradient(145deg, color-mix(in srgb, var(--glow) 16%, transparent), transparent 46%),
        repeating-linear-gradient(90deg, transparent 0 58px, color-mix(in srgb, var(--stroke) 12%, transparent) 58px 60px);
    }}
    .ad-data_proof::before {{
      background:
        linear-gradient(90deg, color-mix(in srgb, var(--accent-2) 16%, transparent), transparent 34%),
        repeating-linear-gradient(0deg, transparent 0 54px, color-mix(in srgb, var(--accent) 10%, transparent) 54px 56px);
    }}
    .clip {{ position: absolute; }}
    .bg-grid {{
      inset: 0;
      opacity: 0.42;
      background-image:
        linear-gradient(to right, color-mix(in srgb, var(--grid) 62%, transparent) 1px, transparent 1px),
        linear-gradient(to bottom, color-mix(in srgb, var(--grid) 46%, transparent) 1px, transparent 1px);
      background-size: 96px 96px;
      mask-image: linear-gradient(120deg, transparent 0%, #000 16%, #000 76%, transparent 100%);
      transform: translate3d(calc(var(--p, 0) * -30px), calc(var(--p, 0) * -18px), 0);
    }}
    .bg-wash {{
      inset: 0;
      background:
        linear-gradient(90deg, color-mix(in srgb, var(--accent) 22%, transparent), transparent 34%, color-mix(in srgb, var(--accent-2) 18%, transparent)),
        linear-gradient(180deg, color-mix(in srgb, var(--stroke) 14%, transparent), transparent 38%, color-mix(in srgb, var(--glow) 12%, transparent));
      opacity: 0.78;
    }}
    .bg-rails {{ inset: 0; pointer-events: none; }}
    .bg-rails span {{
      position: absolute;
      left: -12%;
      width: 124%;
      height: 2px;
      background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--stroke) 50%, transparent), transparent);
      transform: translateX(calc((var(--p, 0) - .5) * 180px));
      opacity: .42;
    }}
    .bg-rails span:nth-child(1) {{ top: 16%; }}
    .bg-rails span:nth-child(2) {{ top: 50%; opacity: .22; }}
    .bg-rails span:nth-child(3) {{ top: 82%; }}
    .hf-header {{
      top: 72px;
      left: var(--safe-x);
      width: min(940px, calc(100% - var(--safe-x) * 2));
      z-index: 10;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      min-height: 42px;
      padding: 0 18px;
      border: 1px solid color-mix(in srgb, var(--stroke) 48%, transparent);
      background: color-mix(in srgb, var(--eyebrow-bg) 82%, transparent);
      color: var(--eyebrow-text);
      font-size: 20px;
      font-weight: 800;
      line-height: 1;
      text-transform: uppercase;
      overflow-wrap: anywhere;
    }}
    h1 {{
      max-width: 940px;
      margin: 24px 0 0;
      font-size: 72px;
      line-height: .98;
      font-weight: 850;
      text-wrap: balance;
      overflow-wrap: anywhere;
    }}
    .hf-header p {{
      max-width: 760px;
      margin: 22px 0 0;
      color: var(--muted);
      font-size: 28px;
      line-height: 1.24;
      overflow-wrap: anywhere;
    }}
    .stage {{
      left: var(--safe-x);
      right: var(--safe-x);
      top: 320px;
      bottom: var(--safe-bottom);
      z-index: 6;
    }}
    .density-minimal .stage {{ top: 300px; }}
    .density-dense .stage {{ top: 340px; }}
    .motion-high .bg-grid {{ transform: translate3d(calc(var(--p, 0) * -48px), calc(var(--p, 0) * -28px), 0); }}
    .metric-stage {{
      display: grid;
      grid-template-columns: 1.15fr .85fr;
      gap: 42px;
      align-items: center;
    }}
    .metric-hero {{
      position: relative;
      min-height: 460px;
      padding: 54px 60px;
      border: 1px solid color-mix(in srgb, var(--stroke) 64%, transparent);
      background: linear-gradient(145deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--bg) 68%, transparent));
      box-shadow: 0 32px 80px color-mix(in srgb, black 36%, transparent);
    }}
    .metric-label {{ color: var(--accent-2); font-size: 22px; font-weight: 850; }}
    .metric-value {{
      margin-top: 18px;
      font-size: 150px;
      line-height: .9;
      font-weight: 900;
      overflow-wrap: anywhere;
    }}
    .metric-bars {{ position: absolute; left: 60px; right: 60px; bottom: 56px; display: flex; align-items: end; gap: 18px; height: 120px; }}
    .bar {{ flex: 1; height: calc(18px + var(--bar-progress, 0) * 110px); background: linear-gradient(180deg, var(--accent), var(--accent-2)); }}
    .stat-stack {{ display: grid; gap: 20px; }}
    .stat-card {{
      min-height: 126px;
      padding: 26px 30px;
      border-left: 6px solid var(--accent);
      background: color-mix(in srgb, var(--panel) 78%, transparent);
      box-shadow: 0 24px 60px color-mix(in srgb, black 28%, transparent);
    }}
    .stat-card b, .flow-node b, .route-labels b {{ display: block; font-size: 28px; line-height: 1.12; overflow-wrap: anywhere; }}
    .stat-card span {{ display: block; margin-top: 8px; color: var(--accent-2); font-size: 34px; font-weight: 850; }}
    .flow-stage {{ display: grid; place-items: center; }}
    .flow-lines, .flow-nodes {{ position: absolute; inset: 0; }}
    .flow-line {{
      position: absolute;
      top: 50%;
      width: 19%;
      height: 5px;
      transform-origin: left center;
      transform: scaleX(var(--line-progress, 0));
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    .flow-line-1 {{ left: 24%; }}
    .flow-line-2 {{ left: 43%; }}
    .flow-line-3 {{ left: 62%; }}
    .flow-node {{
      position: absolute;
      top: 50%;
      width: 250px;
      min-height: 180px;
      padding: 26px;
      translate: -50% -50%;
      border: 1px solid color-mix(in srgb, var(--stroke) 58%, transparent);
      background: linear-gradient(160deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--bg) 70%, transparent));
      box-shadow: 0 28px 72px color-mix(in srgb, black 34%, transparent);
    }}
    .flow-node span {{ color: var(--accent); font-size: 24px; font-weight: 900; }}
    .flow-node-1 {{ left: 13%; }} .flow-node-2 {{ left: 38%; }} .flow-node-3 {{ left: 63%; }} .flow-node-4 {{ left: 88%; }}
    .signal-pulse {{
      width: 140px;
      height: 140px;
      border: 3px solid color-mix(in srgb, var(--accent-2) 76%, transparent);
      opacity: calc(.12 + var(--pulse, 0) * .22);
      transform: scale(calc(.7 + var(--pulse, 0) * 2.4));
    }}
    .route-stage {{ display: grid; place-items: center; }}
    .route-svg {{ position: absolute; inset: 40px 80px 170px 80px; width: calc(100% - 160px); height: calc(100% - 210px); overflow: visible; }}
    .route-shadow, .route-path {{ fill: none; stroke-linecap: round; }}
    .route-shadow {{ stroke: color-mix(in srgb, black 44%, transparent); stroke-width: 26; }}
    .route-path {{
      stroke: var(--accent-2);
      stroke-width: 12;
      stroke-dasharray: 1;
      stroke-dashoffset: calc(1 - var(--route-progress, 0));
      pathLength: 1;
    }}
    .route-tracer {{
      position: absolute;
      left: calc(9% + var(--route-progress, 0) * 78%);
      top: 48%;
      width: 28px;
      height: 28px;
      border: 7px solid var(--accent);
      background: var(--text);
      transform: translate(-50%, -50%);
      box-shadow: 0 0 36px color-mix(in srgb, var(--accent) 72%, transparent);
    }}
    .route-labels {{ position: absolute; left: 70px; right: 70px; bottom: 10px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 0; margin: 0; list-style: none; }}
    .route-labels li {{ min-height: 128px; padding: 24px; background: color-mix(in srgb, var(--panel) 78%, transparent); border-top: 4px solid var(--accent); }}
    .route-labels span {{ display: block; color: var(--accent-2); font-weight: 900; margin-bottom: 10px; }}
    .compare-stage {{ display: grid; grid-template-columns: 1fr 160px 1fr; gap: 20px; align-items: center; }}
    .compare-card {{
      min-height: 430px;
      padding: 48px;
      border: 1px solid color-mix(in srgb, var(--stroke) 58%, transparent);
      background: color-mix(in srgb, var(--panel) 84%, transparent);
      box-shadow: 0 30px 90px color-mix(in srgb, black 38%, transparent);
    }}
    .compare-card span {{ color: var(--accent); font-size: 24px; font-weight: 900; }}
    .compare-card h2 {{ margin: 22px 0 18px; font-size: 62px; line-height: .98; overflow-wrap: anywhere; }}
    .compare-card p {{ color: var(--muted); font-size: 30px; line-height: 1.22; overflow-wrap: anywhere; }}
    .compare-bridge {{ height: 7px; transform-origin: left center; transform: scaleX(var(--line-progress, 0)); background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .interface-stage {{ display: grid; place-items: center; perspective: 1600px; }}
    .browser-card {{
      position: absolute;
      width: 780px;
      min-height: 440px;
      padding: 28px;
      border: 1px solid color-mix(in srgb, var(--stroke) 58%, transparent);
      background: linear-gradient(145deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--bg) 68%, transparent));
      box-shadow: 0 34px 90px color-mix(in srgb, black 40%, transparent);
    }}
    .primary-browser {{ transform: translateZ(90px); z-index: 3; }}
    .rear-card {{ opacity: .45; }}
    .rear-one {{ transform: translate(110px, 74px) rotateZ(3deg); }}
    .rear-two {{ transform: translate(-120px, -60px) rotateZ(-4deg); }}
    .chrome-dots {{ display: flex; gap: 10px; margin-bottom: 28px; }}
    .chrome-dots span {{ width: 14px; height: 14px; background: var(--accent); }}
    .ui-title {{ font-size: 44px; font-weight: 850; margin-bottom: 26px; overflow-wrap: anywhere; }}
    .ui-rows {{ display: grid; gap: 16px; }}
    .ui-row {{ display: grid; grid-template-columns: 20px 1fr auto; gap: 18px; align-items: center; min-height: 74px; padding: 18px; background: color-mix(in srgb, var(--bg) 54%, transparent); }}
    .ui-row span {{ width: 16px; height: 16px; background: var(--accent-2); }}
    .ui-row b {{ font-size: 24px; overflow-wrap: anywhere; }}
    .ui-row i {{ color: var(--accent); font-style: normal; font-weight: 900; }}
    .quote-stage {{ display: grid; place-items: center; text-align: center; }}
    blockquote {{ max-width: 1080px; margin: 0; font-size: 78px; line-height: 1.02; font-weight: 880; text-wrap: balance; overflow-wrap: anywhere; }}
    .ribbon {{ position: absolute; width: 86%; height: 64px; background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--accent) 52%, transparent), color-mix(in srgb, var(--accent-2) 42%, transparent), transparent); transform: scaleX(var(--ribbon-progress, 0)); opacity: .62; }}
    .ribbon-one {{ top: 23%; }} .ribbon-two {{ bottom: 24%; transform-origin: right center; }}
    .keyword-row {{ position: absolute; bottom: 80px; display: flex; justify-content: center; gap: 16px; max-width: 100%; flex-wrap: wrap; }}
    .keyword-row span {{ padding: 16px 22px; border: 1px solid color-mix(in srgb, var(--stroke) 58%, transparent); background: color-mix(in srgb, var(--panel) 78%, transparent); color: var(--muted); font-size: 24px; font-weight: 780; }}
    .ad-cinematic_editorial blockquote {{ font-size: 84px; max-width: 1160px; }}
    .ad-product_ui .browser-card, .ad-product_ui .compare-card, .ad-product_ui .metric-hero {{
      border-radius: 0;
      border-color: color-mix(in srgb, var(--stroke) 72%, transparent);
    }}
    .ad-data_proof .metric-value {{ font-variant-numeric: tabular-nums; letter-spacing: 0; }}
    .ad-system_flow .flow-line, .ad-system_flow .route-path {{ filter: drop-shadow(0 0 12px color-mix(in srgb, var(--accent-2) 54%, transparent)); }}
    .causal-stage, .matrix-stage, .anatomy-stage, .ranking-stage, .ladder-stage, .proof-stage, .arc-stage {{ display: grid; place-items: center; }}
    .causal-stage {{ grid-template-columns: 1fr 220px 1fr; gap: 22px; align-items: center; }}
    .causal-card {{
      min-height: 350px;
      padding: 44px;
      border: 1px solid color-mix(in srgb, var(--stroke) 62%, transparent);
      background: linear-gradient(155deg, color-mix(in srgb, var(--panel) 88%, transparent), color-mix(in srgb, var(--bg) 64%, transparent));
      box-shadow: 0 30px 86px color-mix(in srgb, black 34%, transparent);
    }}
    .causal-card span, .proof-card span, .anatomy-layer span, .ranking-list span, .ladder-list span {{ color: var(--accent); font-size: 22px; font-weight: 900; }}
    .causal-card b {{ display: block; margin-top: 24px; font-size: 48px; line-height: 1.02; overflow-wrap: anywhere; }}
    .causal-spine {{ display: grid; place-items: center; gap: 24px; color: var(--muted); text-align: center; }}
    .causal-spine i {{ width: 190px; height: 7px; transform-origin: left center; transform: scaleX(var(--line-progress, 0)); background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .causal-spine strong {{ font-size: 24px; line-height: 1.12; overflow-wrap: anywhere; }}
    .flywheel-stage {{ display: grid; place-items: center; }}
    .loop-orbit {{
      width: min(600px, 58vh);
      aspect-ratio: 1;
      border: 9px solid color-mix(in srgb, var(--stroke) 58%, transparent);
      border-right-color: var(--accent);
      border-bottom-color: var(--accent-2);
      border-radius: 50%;
      transform: rotate(calc(var(--p, 0) * 260deg));
      box-shadow: 0 0 80px color-mix(in srgb, var(--glow) 28%, transparent);
    }}
    .loop-core {{ position: absolute; display: grid; place-items: center; width: 230px; min-height: 120px; padding: 22px; background: var(--panel); border: 1px solid color-mix(in srgb, var(--stroke) 60%, transparent); font-size: 34px; font-weight: 900; text-align: center; overflow-wrap: anywhere; }}
    .loop-nodes {{ position: absolute; inset: 0; margin: 0; padding: 0; list-style: none; }}
    .loop-node {{ position: absolute; width: 220px; min-height: 116px; padding: 22px; background: color-mix(in srgb, var(--panel) 82%, transparent); border-top: 4px solid var(--accent); }}
    .loop-node b, .ranking-list b, .ladder-list b, .proof-card b, .anatomy-layer b {{ display: block; font-size: 26px; line-height: 1.12; overflow-wrap: anywhere; }}
    .loop-node-1 {{ left: 50%; top: 3%; translate: -50% 0; }} .loop-node-2 {{ right: 5%; top: 48%; translate: 0 -50%; }} .loop-node-3 {{ left: 50%; bottom: 3%; translate: -50% 0; }} .loop-node-4 {{ left: 5%; top: 48%; translate: 0 -50%; }}
    .matrix-card {{ width: min(1040px, 100%); padding: 34px; background: color-mix(in srgb, var(--panel) 82%, transparent); border: 1px solid color-mix(in srgb, var(--stroke) 62%, transparent); box-shadow: 0 30px 86px color-mix(in srgb, black 34%, transparent); }}
    .matrix-head, .matrix-row {{ display: grid; grid-template-columns: 1fr 220px 220px; gap: 18px; align-items: center; }}
    .matrix-head {{ color: var(--muted); font-size: 23px; font-weight: 850; margin-bottom: 18px; }}
    .matrix-row {{ min-height: 104px; border-top: 1px solid color-mix(in srgb, var(--stroke) 24%, transparent); }}
    .matrix-row b {{ font-size: 30px; overflow-wrap: anywhere; }}
    .matrix-row span {{ height: 40px; }}
    .matrix-row .weak {{ background: linear-gradient(90deg, color-mix(in srgb, var(--muted) 32%, transparent), color-mix(in srgb, var(--muted) 12%, transparent)); }}
    .matrix-row .strong {{ background: linear-gradient(90deg, var(--accent), var(--accent-2)); box-shadow: 0 0 34px color-mix(in srgb, var(--accent) 34%, transparent); }}
    .anatomy-core {{ position: absolute; left: 8%; top: 26%; width: 360px; min-height: 230px; display: grid; place-items: center; padding: 38px; background: radial-gradient(circle at 50% 50%, color-mix(in srgb, var(--accent) 28%, var(--panel)), var(--panel)); border: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent); font-size: 42px; font-weight: 900; text-align: center; overflow-wrap: anywhere; }}
    .anatomy-layers {{ position: absolute; left: 45%; right: 4%; top: 8%; bottom: 8%; display: grid; gap: 18px; }}
    .anatomy-layer {{ display: grid; grid-template-columns: 70px 1fr; gap: 22px; align-items: center; padding: 24px 28px; background: color-mix(in srgb, var(--panel) 78%, transparent); border-left: 5px solid var(--accent); }}
    .ranking-list, .ladder-list {{ width: min(1040px, 100%); display: grid; gap: 18px; margin: 0; padding: 0; list-style: none; }}
    .ranking-list li, .ladder-list li {{ display: grid; grid-template-columns: 72px 1fr 38%; gap: 24px; align-items: center; min-height: 104px; padding: 24px 30px; background: color-mix(in srgb, var(--panel) 80%, transparent); border: 1px solid color-mix(in srgb, var(--stroke) 42%, transparent); }}
    .ranking-list i {{ height: 20px; transform-origin: left center; transform: scaleX(calc(var(--rank, .6) * var(--p, 0))); background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .ladder-stage {{ align-items: end; }}
    .ladder-list {{ grid-template-columns: repeat(4, 1fr); align-items: end; gap: 18px; }}
    .ladder-list li {{ grid-template-columns: 1fr; align-content: start; min-height: calc(116px + var(--p, 0) * 120px); border-top: 5px solid var(--accent); }}
    .ladder-rung-1 {{ margin-top: 180px; }} .ladder-rung-2 {{ margin-top: 120px; }} .ladder-rung-3 {{ margin-top: 60px; }} .ladder-rung-4 {{ margin-top: 0; }}
    .ladder-line {{ position: absolute; left: 8%; right: 8%; bottom: 24%; height: 6px; transform-origin: left center; transform: scaleX(var(--line-progress, 0)); background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .proof-chain {{ width: min(1080px, 100%); display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }}
    .proof-card {{ min-height: 260px; padding: 28px; background: color-mix(in srgb, var(--panel) 82%, transparent); border-bottom: 6px solid var(--accent); box-shadow: 0 28px 70px color-mix(in srgb, black 30%, transparent); }}
    .proof-card b {{ margin-top: 54px; }}
    .arc-svg {{ position: absolute; inset: 20px 70px 170px 70px; width: calc(100% - 140px); height: calc(100% - 190px); overflow: visible; }}
    .arc-shadow, .arc-path {{ fill: none; stroke-linecap: round; }}
    .arc-shadow {{ stroke: color-mix(in srgb, black 46%, transparent); stroke-width: 28; }}
    .arc-path {{ stroke: var(--accent-2); stroke-width: 12; stroke-dasharray: 1; stroke-dashoffset: calc(1 - var(--route-progress, 0)); pathLength: 1; }}
    .arc-labels {{ position: absolute; left: 9%; right: 9%; bottom: 18px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; margin: 0; padding: 0; list-style: none; }}
    .arc-labels li {{ min-height: 138px; padding: 24px; background: color-mix(in srgb, var(--panel) 80%, transparent); border-top: 4px solid var(--accent); }}
    .arc-labels span {{ color: var(--accent-2); font-weight: 900; }}
    .arc-labels b {{ display: block; margin-top: 12px; font-size: 28px; line-height: 1.12; overflow-wrap: anywhere; }}
    .map-stage, .radar-stage, .xray-stage, .tree-stage, .wave-stage, .filmstrip-stage, .checklist-stage {{ display: grid; place-items: center; }}
    .map-core {{
      position: absolute;
      z-index: 3;
      width: 340px;
      min-height: 170px;
      display: grid;
      place-items: center;
      padding: 30px;
      text-align: center;
      font-size: 38px;
      line-height: 1.02;
      font-weight: 900;
      background: radial-gradient(circle at 50% 40%, color-mix(in srgb, var(--accent) 24%, var(--panel)), var(--panel));
      border: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
      box-shadow: 0 30px 82px color-mix(in srgb, black 34%, transparent);
      overflow-wrap: anywhere;
    }}
    .map-nodes, .tree-nodes {{ position: absolute; inset: 0; margin: 0; padding: 0; list-style: none; }}
    .map-node {{
      position: absolute;
      width: 230px;
      min-height: 112px;
      padding: 20px 22px;
      background: color-mix(in srgb, var(--panel) 82%, transparent);
      border-top: 4px solid var(--accent);
      box-shadow: 0 24px 66px color-mix(in srgb, black 26%, transparent);
    }}
    .map-node span, .tree-node span, .checklist span, .xray-layer span, .radar-list span, .filmstrip span {{ color: var(--accent-2); font-size: 20px; font-weight: 900; }}
    .map-node b, .tree-node b, .checklist b, .xray-layer b, .radar-list b, .filmstrip b {{ display: block; margin-top: 8px; font-size: 24px; line-height: 1.12; overflow-wrap: anywhere; }}
    .map-node-1 {{ left: 50%; top: 2%; translate: -50% 0; }} .map-node-2 {{ right: 7%; top: 30%; }} .map-node-3 {{ right: 18%; bottom: 4%; }} .map-node-4 {{ left: 18%; bottom: 4%; }} .map-node-5 {{ left: 7%; top: 30%; }}
    .map-links {{ position: absolute; inset: 0; pointer-events: none; opacity: calc(.26 + var(--p, 0) * .32); }}
    .map-links span {{ position: absolute; left: 22%; right: 22%; top: 50%; height: 3px; background: linear-gradient(90deg, transparent, var(--accent-2), transparent); transform: rotate(calc((var(--p, 0) * 24deg) + 18deg)); transform-origin: center; }}
    .map-links span:nth-child(2) {{ transform: rotate(calc((var(--p, 0) * -18deg) - 20deg)); }} .map-links span:nth-child(3) {{ transform: rotate(58deg); }} .map-links span:nth-child(4) {{ transform: rotate(-58deg); }} .map-links span:nth-child(5) {{ transform: rotate(90deg); }}
    .checklist {{ width: min(1040px, 100%); display: grid; gap: 16px; margin: 0; padding: 0; list-style: none; }}
    .checklist li {{
      display: grid;
      grid-template-columns: 64px 1fr 54px;
      align-items: center;
      min-height: 88px;
      padding: 22px 26px;
      background: color-mix(in srgb, var(--panel) 82%, transparent);
      border-left: 5px solid var(--accent);
    }}
    .checklist i {{ width: 34px; height: 34px; border: 4px solid var(--accent-2); box-shadow: inset 0 0 0 9px color-mix(in srgb, var(--accent) 34%, transparent); transform: scale(calc(.74 + var(--p, 0) * .26)); }}
    .radar-stage {{ grid-template-columns: .9fr .75fr; gap: 48px; align-items: center; }}
    .radar-disc {{
      position: relative;
      width: min(560px, 54vh);
      aspect-ratio: 1;
      border-radius: 50%;
      border: 3px solid color-mix(in srgb, var(--stroke) 70%, transparent);
      background:
        radial-gradient(circle at 50% 50%, transparent 0 24%, color-mix(in srgb, var(--stroke) 20%, transparent) 24% 25%, transparent 25% 49%, color-mix(in srgb, var(--stroke) 18%, transparent) 49% 50%, transparent 50% 74%, color-mix(in srgb, var(--stroke) 16%, transparent) 74% 75%, transparent 75%),
        linear-gradient(90deg, transparent 49.6%, color-mix(in srgb, var(--stroke) 28%, transparent) 49.6% 50.4%, transparent 50.4%),
        linear-gradient(0deg, transparent 49.6%, color-mix(in srgb, var(--stroke) 28%, transparent) 49.6% 50.4%, transparent 50.4%);
      box-shadow: 0 0 90px color-mix(in srgb, var(--glow) 28%, transparent);
    }}
    .radar-disc i {{ position: absolute; left: 50%; top: 50%; width: 50%; height: 4px; background: linear-gradient(90deg, var(--accent), transparent); transform-origin: left center; transform: rotate(calc(var(--p, 0) * 320deg)); }}
    .radar-blip {{ position: absolute; width: 22px; height: 22px; background: var(--accent); box-shadow: 0 0 28px color-mix(in srgb, var(--accent) 70%, transparent); }}
    .radar-blip-1 {{ left: 62%; top: 25%; }} .radar-blip-2 {{ left: 35%; top: 42%; }} .radar-blip-3 {{ left: 72%; top: 67%; }} .radar-blip-4 {{ left: 24%; top: 70%; }}
    .radar-list {{ display: grid; gap: 16px; margin: 0; padding: 0; list-style: none; }}
    .radar-list li {{ min-height: 94px; padding: 20px 24px; background: color-mix(in srgb, var(--panel) 82%, transparent); border-left: 4px solid var(--accent); }}
    .xray-title {{ position: absolute; left: 6%; top: 12%; width: 320px; min-height: 170px; display: grid; place-items: center; padding: 28px; text-align: center; font-size: 38px; font-weight: 900; line-height: 1.04; background: color-mix(in srgb, var(--panel) 84%, transparent); border: 1px solid color-mix(in srgb, var(--stroke) 62%, transparent); overflow-wrap: anywhere; }}
    .xray-layers {{ position: absolute; left: 40%; right: 5%; top: 4%; bottom: 4%; display: grid; gap: 14px; margin: 0; padding: 0; list-style: none; }}
    .xray-layer {{ display: grid; grid-template-columns: 64px 1fr 140px; gap: 18px; align-items: center; padding: 18px 22px; background: linear-gradient(90deg, color-mix(in srgb, var(--panel) 86%, transparent), color-mix(in srgb, var(--bg) 56%, transparent)); border-left: 5px solid var(--accent); }}
    .xray-layer i {{ height: 15px; transform-origin: left center; transform: scaleX(calc(.2 + var(--p, 0) * .8)); background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .tree-root {{ position: absolute; top: 6%; left: 50%; translate: -50% 0; width: 360px; min-height: 120px; display: grid; place-items: center; padding: 24px; text-align: center; font-size: 34px; font-weight: 900; background: var(--panel); border: 1px solid color-mix(in srgb, var(--stroke) 64%, transparent); overflow-wrap: anywhere; }}
    .tree-lines {{ position: absolute; inset: 0; pointer-events: none; }}
    .tree-lines span {{ position: absolute; left: 50%; top: 24%; width: 4px; height: 44%; background: linear-gradient(var(--accent), var(--accent-2)); transform-origin: top center; transform: scaleY(var(--line-progress, 0)); }}
    .tree-lines span:nth-child(2) {{ transform: rotate(44deg) scaleY(var(--line-progress, 0)); }} .tree-lines span:nth-child(3) {{ transform: rotate(-44deg) scaleY(var(--line-progress, 0)); }}
    .tree-node {{ position: absolute; width: 260px; min-height: 120px; padding: 22px; background: color-mix(in srgb, var(--panel) 82%, transparent); border-top: 4px solid var(--accent); }}
    .tree-node-1 {{ left: 16%; top: 48%; }} .tree-node-2 {{ right: 16%; top: 48%; }} .tree-node-3 {{ left: 28%; bottom: 4%; }} .tree-node-4 {{ right: 28%; bottom: 4%; }}
    .wave-svg {{ position: absolute; inset: 70px 70px 180px 70px; width: calc(100% - 140px); height: calc(100% - 250px); overflow: visible; }}
    .wave-shadow, .wave-path {{ fill: none; stroke-linecap: round; }}
    .wave-shadow {{ stroke: color-mix(in srgb, black 46%, transparent); stroke-width: 30; }}
    .wave-path {{ stroke: var(--accent-2); stroke-width: 13; stroke-dasharray: 1; stroke-dashoffset: calc(1 - var(--route-progress, 0)); filter: drop-shadow(0 0 16px color-mix(in srgb, var(--accent-2) 48%, transparent)); }}
    .wave-value {{ position: absolute; left: 8%; bottom: 8%; font-size: 92px; line-height: .92; font-weight: 900; color: var(--text); overflow-wrap: anywhere; max-width: 560px; }}
    .wave-chips {{ position: absolute; right: 7%; bottom: 10%; display: grid; gap: 14px; width: 360px; }}
    .wave-chips span {{ padding: 18px 22px; background: color-mix(in srgb, var(--panel) 82%, transparent); border-left: 4px solid var(--accent); font-size: 24px; font-weight: 820; overflow-wrap: anywhere; }}
    .filmstrip {{ width: min(1120px, 100%); display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin: 0; padding: 0; list-style: none; }}
    .filmstrip li {{ position: relative; min-height: 330px; padding: 28px; display: grid; align-content: end; background: color-mix(in srgb, var(--panel) 82%, transparent); border: 1px solid color-mix(in srgb, var(--stroke) 42%, transparent); box-shadow: 0 24px 70px color-mix(in srgb, black 28%, transparent); }}
    .filmstrip li::before {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 42px; background: repeating-linear-gradient(90deg, color-mix(in srgb, var(--accent) 54%, transparent) 0 22px, transparent 22px 40px); opacity: .62; }}
    [data-anim] {{ opacity: 0; will-change: transform, opacity; }}
    """


def _timeline_script(composition_id: str, duration: float) -> str:
    return f"""
    <script>
    (() => {{
      const compositionId = {_script_json(composition_id)};
      const duration = {duration:.6f};
      const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
      const easeOut = value => 1 - Math.pow(1 - clamp(value), 3);
      const easeInOut = value => {{
        const p = clamp(value);
        return p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      }};
      const root = document.getElementById("root");
      const actors = Array.from(document.querySelectorAll("[data-anim]")).map((el) => ({{
        el,
        mode: el.dataset.anim || "rise",
        delay: Number(el.dataset.delay || 0),
        span: Math.max(Number(el.dataset.span || .55), .05),
        y: Number(el.dataset.y || 28),
        scale: Number(el.dataset.scale || .96)
      }}));
      const bars = Array.from(document.querySelectorAll("[data-bar]"));
      const lines = Array.from(document.querySelectorAll("[data-line]"));
      const route = document.querySelector("[data-route]");
      const dot = document.querySelector("[data-route-dot]");
      function transformFor(actor, eased) {{
        const travel = (1 - eased) * actor.y;
        if (actor.mode === "scale" || actor.mode === "pop") {{
          return `translate3d(0, ${{travel}}px, 0) scale(${{actor.scale + (1 - actor.scale) * eased}})`;
        }}
        if (actor.mode === "slide-left") {{
          return `translate3d(${{(1 - eased) * -60}}px, 0, 0) scale(${{actor.scale + (1 - actor.scale) * eased}})`;
        }}
        if (actor.mode === "slide-right") {{
          return `translate3d(${{(1 - eased) * 60}}px, 0, 0) scale(${{actor.scale + (1 - actor.scale) * eased}})`;
        }}
        return `translate3d(0, ${{travel}}px, 0) scale(${{actor.scale + (1 - actor.scale) * eased}})`;
      }}
      function renderAt(time) {{
        const t = clamp(Number(time) || 0, 0, duration);
        const p = duration > 0 ? clamp(t / duration) : 1;
        const exit = clamp((duration - t) / Math.min(.55, duration * .28));
        root.style.setProperty("--p", p.toFixed(5));
        root.style.setProperty("--pulse", (0.5 + Math.sin(p * Math.PI * 2.0) * 0.5).toFixed(5));
        root.style.setProperty("--ribbon-progress", easeInOut(clamp((p - .08) / .68)).toFixed(5));
        root.style.setProperty("--route-progress", easeInOut(clamp((p - .12) / .74)).toFixed(5));
        actors.forEach((actor) => {{
          const local = clamp((p - actor.delay) / actor.span);
          const eased = easeOut(local);
          actor.el.style.opacity = String(clamp(eased * exit));
          actor.el.style.transform = transformFor(actor, eased);
        }});
        bars.forEach((bar, index) => {{
          const target = Number(bar.dataset.bar || .65);
          const local = easeOut(clamp((p - .22 - index * .045) / .55));
          bar.style.setProperty("--bar-progress", (target * local).toFixed(5));
        }});
        lines.forEach((line) => {{
          const delay = Number(line.dataset.delay || .25);
          line.style.setProperty("--line-progress", easeInOut(clamp((p - delay) / .42)).toFixed(5));
        }});
        if (route) {{
          route.style.strokeDasharray = "1";
          route.style.strokeDashoffset = String(1 - easeInOut(clamp((p - .12) / .74)));
        }}
        if (dot) {{
          const rp = easeInOut(clamp((p - .12) / .74));
          dot.style.left = `${{9 + rp * 78}}%`;
          dot.style.top = `${{51 + Math.sin(rp * Math.PI * 2.1) * 9}}%`;
          dot.style.opacity = String(clamp((p - .08) / .18) * exit);
        }}
      }}
      const timeline = {{
        duration: () => duration,
        time: (value) => {{ renderAt(value); return timeline; }},
        seek: (value) => {{ renderAt(value); return timeline; }},
        progress: (value) => {{ renderAt((Number(value) || 0) * duration); return timeline; }},
        pause: () => timeline,
        play: () => timeline
      }};
      window.__timelines = window.__timelines || {{}};
      window.__timelines["{composition_id}"] = timeline;
      renderAt(0);
    }})();
    </script>
    """


def build_composition(
    spec: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
) -> HyperframesComposition:
    template = str(spec.get("template") or "ribbon_quote").strip().lower()
    if template not in SUPPORTED_TEMPLATES:
        template = "ribbon_quote"
    spec_id = _clean_id(spec.get("visual_id") or spec.get("id") or "visual")
    composition_id = f"vex-{spec_id}"
    duration = _clamp(float(spec.get("duration") or 2.8), 1.0, 12.0)
    variant_index = int(spec.get("hyperframes_variant_index") or spec.get("variant_index") or 0)
    design_ir = build_design_ir(
        {**spec, "template": template, "duration": duration},
        width=width,
        height=height,
        fps=fps,
        variant_index=variant_index,
    )
    theme = design_ir.art_direction.theme
    track = 0
    background_html, track = _stage_background(duration, track)
    header_html, track = _header(spec, duration, track)
    stage_html, track, stage_metadata = _stage_for_template({**spec, "template": template}, duration, track)
    skill_slices = retrieve_skill_slices(template)
    metadata = {
        "composition_id": composition_id,
        "template": template,
        "duration_sec": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "design_ir": design_ir.to_dict(),
        "art_direction": design_ir.art_direction.to_dict(),
        "archetype": design_ir.archetype,
        "variant_index": variant_index,
        "skill_slices": [skill.to_dict() for skill in skill_slices],
        "stage": stage_metadata,
        "program_context": dict(spec.get("program_context") or {}),
        "episode_context": dict(spec.get("episode_context") or {}),
        "visual_beats": list(spec.get("visual_beats") or []),
        "continuity_group": str(spec.get("continuity_group") or ""),
        "concept_ids": list(spec.get("concept_ids") or []),
        "transition_in": dict(spec.get("transition_in") or {}),
        "transition_out": dict(spec.get("transition_out") or {}),
        "qa_contract": dict(spec.get("qa_contract") or {}),
    }
    rendered_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width={width}, height={height}, initial-scale=1">
  <title>{html.escape(composition_id, quote=True)}</title>
  <style>{_css(theme, width, height, design_ir)}</style>
</head>
<body>
  <div id="root" class="{root_class_names(design_ir)}" data-composition-id="{composition_id}" data-start="0" data-duration="{duration:.3f}" data-width="{width}" data-height="{height}">
    {background_html}
    {header_html}
    {stage_html}
    {_timeline_script(composition_id, duration)}
  </div>
</body>
</html>
"""
    return HyperframesComposition(composition_id=composition_id, html=rendered_html, metadata=metadata)
