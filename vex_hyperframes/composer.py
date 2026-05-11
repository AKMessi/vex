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
    seed = _numeric_seed(spec)
    values = [46 + ((seed + index * 17) % 42) for index in range(5)]
    cards = "\n".join(
        f"""
          <div class="stat-card" {_animate_attrs("rise", 0.24 + index * 0.07, 0.52, y=30)}>
            <b>{html.escape(label, quote=True)}</b>
            <span>{value}%</span>
          </div>
        """
        for index, (label, value) in enumerate(zip((support + ["Momentum", "Clarity", "Focus"])[:3], values[:3]))
    )
    bars = "\n".join(
        f'<span class="bar" data-bar="{value / 100:.3f}" style="--bar-target:{value / 100:.3f}"></span>'
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


def _stage_for_template(spec: dict[str, Any], duration: float, track: int) -> tuple[str, int, dict[str, Any]]:
    template = str(spec.get("template") or "ribbon_quote").strip().lower()
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
