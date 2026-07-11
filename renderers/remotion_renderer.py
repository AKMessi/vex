from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import (
    RenderedAsset,
    RendererStatus,
    VisualRenderer,
    VisualRendererError,
    safe_render_job_dir,
)
from vex_runtime.hyperframes import node_major_version
from vex_runtime.paths import hyperframes_runtime_dir


REMOTION_COMPOSITION_ID = "VexAutoVisual"
REMOTION_PACKAGE_VERSION = "4.0.487"

SUPPORTED_REMOTION_TEMPLATES = {
    "semantic_architecture",
    "semantic_causal",
    "semantic_decision",
    "semantic_interface",
    "semantic_metric",
    "semantic_narrative",
    "semantic_quote",
    "semantic_route",
    "semantic_transform",
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


def _safe_scene_name(spec_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(spec_id or "visual")).strip("_")
    return cleaned or "auto_visual"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runner_path() -> Path:
    return Path(__file__).resolve().with_name("remotion_runner.mjs")


def _node_major_version() -> int | None:
    return node_major_version()


def _remotion_timeout_sec() -> int | None:
    try:
        timeout = int(getattr(config, "REMOTION_RENDER_TIMEOUT_SEC", 0))
    except (TypeError, ValueError):
        timeout = 0
    if timeout <= 0:
        return None
    return max(30, timeout)


def _remotion_timeout_ms() -> int:
    timeout = _remotion_timeout_sec()
    if timeout is None:
        return 30000
    return max(10000, int(timeout * 1000))


def _remotion_concurrency() -> str:
    return str(getattr(config, "REMOTION_RENDER_CONCURRENCY", "") or "").strip()


def _write_command_log(path: Path, command: list[str], result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                f"exit_code={result.returncode}",
                "",
                "[stdout]",
                result.stdout or "",
                "",
                "[stderr]",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )


def _candidate_node_roots() -> list[Path]:
    candidates: list[Path] = []
    for candidate in (_repo_root(), hyperframes_runtime_dir()):
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def _probe_node_packages_at(root: Path) -> tuple[bool, str]:
    node_path = shutil.which("node")
    if not node_path:
        return False, "Node.js is not available in PATH."
    if not (root / "node_modules").is_dir():
        return False, f"{root} does not contain node_modules. Run `npm ci` or install the managed renderer runtime."
    probe_script = (
        "const packages = ['remotion','@remotion/renderer','@remotion/bundler','react','react-dom'];"
        "for (const name of packages) { require.resolve(name); }"
    )
    try:
        result = subprocess.run(
            [node_path, "-e", probe_script],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Could not probe Remotion packages: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, f"Remotion packages are not installed. Run `npm ci`. {detail}".strip()
    return True, ""


def _find_remotion_node_root() -> tuple[Path | None, str]:
    reasons: list[str] = []
    for candidate in _candidate_node_roots():
        ok, reason = _probe_node_packages_at(candidate)
        if ok:
            return candidate, ""
        if reason:
            reasons.append(reason)
    detail = "; ".join(reasons[:3])
    return (
        None,
        (
            "Remotion packages are not installed. Run `npm ci` in a source "
            "checkout or `vex renderers install remotion` for the managed "
            f"renderer runtime.{f' Details: {detail}' if detail else ''}"
        ),
    )


def _run_node_package_probe() -> tuple[bool, str]:
    root, reason = _find_remotion_node_root()
    return root is not None, reason


def _node_platform_arch() -> tuple[str | None, str | None, str]:
    node_path = shutil.which("node")
    if not node_path:
        return None, None, "Node.js is not available in PATH."
    try:
        result = subprocess.run(
            [node_path, "-p", "process.platform + ' ' + process.arch"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, None, f"Could not inspect Node.js platform: {exc}"
    if result.returncode != 0:
        return None, None, (result.stderr or result.stdout or "").strip()
    parts = (result.stdout or "").strip().split()
    if len(parts) != 2:
        return None, None, "Node.js platform probe returned an unexpected value."
    return parts[0], parts[1], ""


def _remotion_platform_blocker(platform: str | None, arch: str | None) -> str:
    if not platform or not arch:
        return ""
    if platform == "win32" and arch != "x64":
        return (
            "Remotion 4.0.487 does not ship a Windows "
            f"{arch} compositor package. Use x64 Node/npm on Windows ARM, "
            "run from WSL/Linux, or use another renderer on this machine."
        )
    if platform not in {"win32", "linux", "darwin"}:
        return f"Remotion local rendering is not supported on Node platform {platform!r}."
    return ""


def _text_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned[:limit]


def _build_input_props(
    spec: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    duration = float(spec.get("duration") or 3.0)
    normalized_spec = dict(spec)
    normalized_spec["duration"] = max(0.5, min(duration, 30.0))
    normalized_spec["supporting_lines"] = _text_list(spec.get("supporting_lines"), limit=4)
    normalized_spec["steps"] = _text_list(spec.get("steps"), limit=5)
    normalized_spec["keywords"] = _text_list(spec.get("keywords"), limit=6)
    normalized_spec["metric_facts"] = [
        dict(item)
        for item in (spec.get("metric_facts") or [])
        if isinstance(item, dict)
    ][:4]
    normalized_spec["visual_beats"] = [
        dict(item)
        for item in (spec.get("visual_beats") or [])
        if isinstance(item, dict)
    ][:6]
    return {
        "spec": normalized_spec,
        "width": int(width),
        "height": int(height),
        "fps": float(fps or 30.0),
        "durationSec": normalized_spec["duration"],
        "compositionId": REMOTION_COMPOSITION_ID,
    }


def _template_family(template: str, intent_type: str) -> str:
    value = f"{template} {intent_type}".lower()
    if re.search(r"data|metric|score|risk|proof|stat", value):
        return "data"
    if re.search(r"compare|contrast|decision|myth|problem", value):
        return "contrast"
    if re.search(r"timeline|sequence|narrative|route|journey", value):
        return "timeline"
    if re.search(r"interface|ui", value):
        return "interface"
    if re.search(r"quote|keyword|focus|emphasis", value):
        return "emphasis"
    return "mechanism"


def _estimated_quality_score(spec: dict[str, Any]) -> float:
    template = str(spec.get("template") or "").strip().lower()
    intent_type = str(spec.get("visual_intent_type") or "").strip().lower()
    composition = str(spec.get("composition_mode") or "").strip().lower()
    importance = 0.5
    try:
        importance = float(spec.get("importance") or 0.5)
    except (TypeError, ValueError):
        importance = 0.5
    premium_templates = {
        "data_journey",
        "signal_network",
        "kinetic_route",
        "spotlight_compare",
        "interface_cascade",
        "causal_chain",
        "flywheel_loop",
        "decision_matrix",
        "proof_sequence",
        "narrative_arc",
        "concept_map",
        "problem_solution",
        "risk_radar",
        "scorecard",
        "pipeline_xray",
        "decision_tree",
        "timeline_filmstrip",
        "mechanism_blueprint",
        "data_pulse",
    }
    score = 0.64
    if template in premium_templates or template.startswith("semantic_"):
        score += 0.08
    if composition == "replace":
        score += 0.04
    if intent_type in {"data_proof", "mechanism", "contrast", "sequence", "ui_callout"}:
        score += 0.04
    if intent_type in {"math_or_formula", "spatial_3d"}:
        score -= 0.12
    score += max(0.0, min(importance, 1.0)) * 0.03
    return round(max(0.45, min(score, 0.9)), 4)


def _remotion_entry_source() -> str:
    return r"""import React from 'react';
import {
  AbsoluteFill,
  Composition,
  interpolate,
  registerRoot,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const PALETTES = {
  editorial_clean: {
    background: '#101418',
    panel: '#F6F1E8',
    panelDark: '#1F2933',
    text: '#F8FAFC',
    muted: '#A8B3C2',
    ink: '#111827',
    accent: '#E11D48',
    accent2: '#0891B2',
    accent3: '#F59E0B',
  },
  bold_tech: {
    background: '#090B10',
    panel: '#E9F8FF',
    panelDark: '#111827',
    text: '#F8FAFC',
    muted: '#AEBAC8',
    ink: '#07111F',
    accent: '#22C55E',
    accent2: '#38BDF8',
    accent3: '#F97316',
  },
  documentary_kinetic: {
    background: '#15120E',
    panel: '#F3EFE3',
    panelDark: '#29231A',
    text: '#FFF7ED',
    muted: '#D5C7B1',
    ink: '#1C1917',
    accent: '#DC2626',
    accent2: '#2563EB',
    accent3: '#D97706',
  },
  product_ui: {
    background: '#0C1116',
    panel: '#F8FAFC',
    panelDark: '#17202A',
    text: '#F8FAFC',
    muted: '#B6C2CF',
    ink: '#0F172A',
    accent: '#2563EB',
    accent2: '#14B8A6',
    accent3: '#F43F5E',
  },
  cinematic_night: {
    background: '#06080C',
    panel: '#E5E7EB',
    panelDark: '#111827',
    text: '#F9FAFB',
    muted: '#9CA3AF',
    ink: '#030712',
    accent: '#FBBF24',
    accent2: '#06B6D4',
    accent3: '#EF4444',
  },
  signal_lab: {
    background: '#0A0F0D',
    panel: '#ECFDF5',
    panelDark: '#10231E',
    text: '#F0FDF4',
    muted: '#A7F3D0',
    ink: '#052E2B',
    accent: '#10B981',
    accent2: '#F59E0B',
    accent3: '#3B82F6',
  },
  magazine_luxe: {
    background: '#14110F',
    panel: '#F7F0E3',
    panelDark: '#27211C',
    text: '#FEF3C7',
    muted: '#D6C2A4',
    ink: '#1C1917',
    accent: '#B91C1C',
    accent2: '#0F766E',
    accent3: '#A16207',
  },
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const asText = (value, fallback = '') => String(value || fallback).trim();
const rawList = (value) => (Array.isArray(value) ? value : []);
const list = (value) => (Array.isArray(value) ? value.filter(Boolean).map((item) => String(item).trim()).filter(Boolean) : []);

const pickPalette = (spec) => {
  const base = PALETTES[asText(spec.style_pack, 'editorial_clean')] || PALETTES.editorial_clean;
  const theme = spec.theme || {};
  return {
    ...base,
    background: asText(theme.background, base.background),
    panel: asText(theme.panel_fill, base.panel),
    panelDark: asText(theme.panel_dark, base.panelDark),
    text: asText(theme.text_primary, base.text),
    muted: asText(theme.text_secondary, base.muted),
    accent: asText(theme.accent, base.accent),
    accent2: asText(theme.accent_secondary, base.accent2),
  };
};

const familyFor = (spec) => {
  const value = `${asText(spec.template)} ${asText(spec.visual_intent_type)}`.toLowerCase();
  if (/data|metric|score|risk|proof|stat/.test(value)) return 'data';
  if (/compare|contrast|decision|myth|problem/.test(value)) return 'contrast';
  if (/timeline|sequence|narrative|route|journey/.test(value)) return 'timeline';
  if (/interface|ui/.test(value)) return 'interface';
  if (/quote|keyword|focus|emphasis/.test(value)) return 'emphasis';
  return 'mechanism';
};

const fitText = (text, maxChars, maxWords) => {
  const words = asText(text).replace(/\s+/g, ' ').split(' ').filter(Boolean).slice(0, maxWords);
  const joined = words.join(' ');
  return joined.length > maxChars ? `${joined.slice(0, Math.max(0, maxChars - 1)).trim()}` : joined;
};

const bgPattern = (palette) => ({
  backgroundColor: palette.background,
  backgroundImage: `
    radial-gradient(circle at 18% 20%, ${palette.accent}33, transparent 24%),
    radial-gradient(circle at 86% 74%, ${palette.accent2}2E, transparent 28%),
    linear-gradient(120deg, transparent 0%, ${palette.panelDark}80 52%, transparent 100%),
    linear-gradient(${palette.accent2}18 1px, transparent 1px),
    linear-gradient(90deg, ${palette.accent2}14 1px, transparent 1px)
  `,
  backgroundSize: '100% 100%, 100% 100%, 100% 100%, 72px 72px, 72px 72px',
});

const Shell = ({children, spec, palette}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const open = spring({frame, fps, config: {damping: 18, stiffness: 92}});
  const fade = interpolate(frame, [0, 14], [0, 1], {extrapolateRight: 'clamp'});
  const eyebrow = fitText(spec.eyebrow || spec.visual_intent_type || 'Visual', 18, 3).toUpperCase();
  return (
    <AbsoluteFill style={{...bgPattern(palette), color: palette.text, fontFamily: 'Inter, Arial, sans-serif', overflow: 'hidden'}}>
      <div style={{position: 'absolute', inset: 44, opacity: fade, transform: `translateY(${(1 - open) * 28}px)`}}>
        <div style={{display: 'inline-flex', alignItems: 'center', gap: 10, padding: '8px 12px', border: `1px solid ${palette.accent2}80`, background: `${palette.panelDark}CC`, color: palette.text, fontSize: 20, fontWeight: 800, letterSpacing: 0, borderRadius: 6}}>
          <span style={{width: 9, height: 9, borderRadius: 99, background: palette.accent}} />
          {eyebrow}
        </div>
      </div>
      {children}
      <div style={{position: 'absolute', left: 48, right: 48, bottom: 34, height: 2, background: `linear-gradient(90deg, ${palette.accent}, ${palette.accent2}, transparent)`, opacity: 0.72}} />
    </AbsoluteFill>
  );
};

const HeadlineBlock = ({spec, palette, narrow = false}) => {
  const frame = useCurrentFrame();
  const title = fitText(spec.headline || spec.emphasis_text || 'Key idea', narrow ? 34 : 48, narrow ? 6 : 7);
  const deck = fitText(spec.deck || spec.footer_text || spec.context_text, narrow ? 56 : 74, 12);
  const titleOpacity = interpolate(frame, [8, 20], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <div style={{maxWidth: narrow ? 560 : 820, opacity: titleOpacity}}>
      <div style={{fontSize: narrow ? 58 : 74, lineHeight: 0.94, fontWeight: 900, letterSpacing: 0, color: palette.text, textWrap: 'balance'}}>
        {title}
      </div>
      {deck ? (
        <div style={{marginTop: 18, fontSize: narrow ? 24 : 28, lineHeight: 1.2, fontWeight: 600, color: palette.muted, maxWidth: narrow ? 510 : 760}}>
          {deck}
        </div>
      ) : null}
    </div>
  );
};

const MetricScene = ({spec, palette}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const pulse = spring({frame: frame - 14, fps, config: {damping: 16, stiffness: 110}});
  const metricFacts = rawList(spec.metric_facts).map((item) => asText(item.value || item.label)).filter(Boolean);
  const value = fitText(spec.emphasis_text || metricFacts[0] || spec.headline, 18, 3);
  const supporting = [...metricFacts.slice(1), ...list(spec.supporting_lines), ...list(spec.keywords)].slice(0, 3);
  return (
    <Shell spec={spec} palette={palette}>
      <div style={{position: 'absolute', left: 76, top: 138}}>
        <HeadlineBlock spec={spec} palette={palette} narrow />
      </div>
      <div style={{position: 'absolute', right: 72, top: 142, width: 560, height: 620, transform: `scale(${0.92 + pulse * 0.08})`, transformOrigin: 'center'}}>
        <div style={{position: 'absolute', inset: 0, background: palette.panel, color: palette.ink, borderRadius: 8, boxShadow: `0 30px 90px ${palette.accent2}29`, border: `2px solid ${palette.accent2}99`}} />
        <div style={{position: 'absolute', left: 44, top: 42, right: 44}}>
          <div style={{fontSize: 22, fontWeight: 900, color: palette.accent2}}>MEASURABLE SIGNAL</div>
          <div style={{fontSize: 108, lineHeight: 0.92, marginTop: 26, fontWeight: 950, color: palette.ink}}>{value}</div>
        </div>
        <div style={{position: 'absolute', left: 44, right: 44, bottom: 46, display: 'grid', gap: 16}}>
          {(supporting.length ? supporting : list(spec.keywords).slice(0, 3)).map((item, index) => (
            <div key={item} style={{display: 'flex', alignItems: 'center', gap: 14, padding: '16px 18px', background: '#FFFFFF', border: `1px solid ${palette.accent2}44`, borderRadius: 7}}>
              <div style={{width: 34, height: 34, display: 'grid', placeItems: 'center', borderRadius: 99, background: index === 0 ? palette.accent : palette.accent2, color: '#fff', fontWeight: 900}}>{index + 1}</div>
              <div style={{fontSize: 24, lineHeight: 1.1, fontWeight: 800}}>{fitText(item, 38, 7)}</div>
            </div>
          ))}
        </div>
      </div>
    </Shell>
  );
};

const MechanismScene = ({spec, palette}) => {
  const frame = useCurrentFrame();
  const steps = (list(spec.steps).length ? list(spec.steps) : rawList(spec.visual_beats).map((beat) => beat.text)).slice(0, 4);
  const labels = (steps.length ? steps : list(spec.keywords)).slice(0, 4);
  return (
    <Shell spec={spec} palette={palette}>
      <div style={{position: 'absolute', left: 74, top: 136}}>
        <HeadlineBlock spec={spec} palette={palette} />
      </div>
      <div style={{position: 'absolute', left: 94, right: 94, bottom: 122, height: 252}}>
        <div style={{position: 'absolute', left: 110, right: 110, top: 122, height: 4, background: `linear-gradient(90deg, ${palette.accent}, ${palette.accent2})`, opacity: 0.78}} />
        <div style={{display: 'grid', gridTemplateColumns: `repeat(${Math.max(labels.length, 1)}, 1fr)`, gap: 22}}>
          {labels.map((label, index) => {
            const reveal = interpolate(frame, [14 + index * 8, 28 + index * 8], [0, 1], {extrapolateRight: 'clamp'});
            return (
              <div key={`${label}-${index}`} style={{opacity: reveal, transform: `translateY(${(1 - reveal) * 26}px)`, minHeight: 224, background: `${palette.panelDark}E6`, border: `1px solid ${index % 2 ? palette.accent2 : palette.accent}88`, borderRadius: 8, padding: 24, boxShadow: `0 20px 60px ${palette.accent2}22`}}>
                <div style={{width: 48, height: 48, borderRadius: 99, background: index % 2 ? palette.accent2 : palette.accent, display: 'grid', placeItems: 'center', fontSize: 22, fontWeight: 900}}>{index + 1}</div>
                <div style={{marginTop: 28, fontSize: 30, lineHeight: 1.05, fontWeight: 900}}>{fitText(label, 34, 5)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </Shell>
  );
};

const ContrastScene = ({spec, palette}) => {
  const frame = useCurrentFrame();
  const split = interpolate(frame, [10, 28], [0, 1], {extrapolateRight: 'clamp'});
  const left = fitText(spec.left_label || list(spec.keywords)[0] || 'Before', 24, 4);
  const right = fitText(spec.right_label || list(spec.keywords)[1] || 'After', 24, 4);
  return (
    <Shell spec={spec} palette={palette}>
      <div style={{position: 'absolute', left: 74, top: 136, right: 74}}>
        <HeadlineBlock spec={spec} palette={palette} />
      </div>
      <div style={{position: 'absolute', left: 80, right: 80, bottom: 106, height: 350, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24}}>
        {[left, right].map((label, index) => (
          <div key={label} style={{opacity: split, transform: `translateX(${(index === 0 ? -1 : 1) * (1 - split) * 44}px)`, background: index === 0 ? `${palette.panelDark}EF` : palette.panel, color: index === 0 ? palette.text : palette.ink, borderRadius: 8, border: `2px solid ${index === 0 ? palette.accent : palette.accent2}`, padding: 34}}>
            <div style={{fontSize: 22, fontWeight: 900, color: index === 0 ? palette.accent : palette.accent2}}>{index === 0 ? 'WITHOUT' : 'WITH'}</div>
            <div style={{marginTop: 30, fontSize: 52, lineHeight: 1, fontWeight: 950}}>{label}</div>
            <div style={{marginTop: 22, fontSize: 25, lineHeight: 1.18, fontWeight: 650, color: index === 0 ? palette.muted : '#334155'}}>
              {fitText(index === 0 ? spec.left_detail || spec.deck : spec.right_detail || spec.footer_text, 66, 11)}
            </div>
          </div>
        ))}
      </div>
    </Shell>
  );
};

const TimelineScene = ({spec, palette}) => {
  const frame = useCurrentFrame();
  const items = (list(spec.steps).length ? list(spec.steps) : list(spec.keywords)).slice(0, 5);
  const labels = items.length ? items : ['Signal', 'Shift', 'Outcome'];
  return (
    <Shell spec={spec} palette={palette}>
      <div style={{position: 'absolute', left: 76, top: 134}}>
        <HeadlineBlock spec={spec} palette={palette} />
      </div>
      <div style={{position: 'absolute', left: 86, right: 86, bottom: 114, height: 320}}>
        <div style={{position: 'absolute', left: 36, right: 36, top: 142, height: 5, background: `${palette.accent2}55`}} />
        {labels.map((label, index) => {
          const x = `${(index / Math.max(labels.length - 1, 1)) * 88 + 6}%`;
          const reveal = interpolate(frame, [12 + index * 7, 24 + index * 7], [0, 1], {extrapolateRight: 'clamp'});
          return (
            <div key={`${label}-${index}`} style={{position: 'absolute', left: x, top: 36, width: 206, transform: `translateX(-50%) translateY(${(1 - reveal) * 22}px)`, opacity: reveal}}>
              <div style={{width: 58, height: 58, borderRadius: 99, margin: '0 auto', background: index % 2 ? palette.accent2 : palette.accent, display: 'grid', placeItems: 'center', fontSize: 24, fontWeight: 900, boxShadow: `0 0 40px ${palette.accent2}66`}}>{index + 1}</div>
              <div style={{marginTop: 34, minHeight: 112, background: `${palette.panelDark}E6`, border: `1px solid ${palette.accent2}77`, borderRadius: 8, padding: 18, textAlign: 'center', fontSize: 25, lineHeight: 1.08, fontWeight: 850}}>
                {fitText(label, 28, 4)}
              </div>
            </div>
          );
        })}
      </div>
    </Shell>
  );
};

const InterfaceScene = ({spec, palette}) => {
  const frame = useCurrentFrame();
  const slide = interpolate(frame, [10, 26], [42, 0], {extrapolateRight: 'clamp'});
  const lines = list(spec.supporting_lines).slice(0, 3);
  return (
    <Shell spec={spec} palette={palette}>
      <div style={{position: 'absolute', left: 72, top: 136, width: 600}}>
        <HeadlineBlock spec={spec} palette={palette} narrow />
      </div>
      <div style={{position: 'absolute', right: 74, top: 134, width: 650, height: 620, transform: `translateX(${slide}px)`, background: palette.panel, color: palette.ink, borderRadius: 8, border: `2px solid ${palette.accent2}77`, boxShadow: `0 40px 120px ${palette.accent2}25`, overflow: 'hidden'}}>
        <div style={{height: 58, background: '#E2E8F0', display: 'flex', alignItems: 'center', gap: 10, padding: '0 20px'}}>
          <span style={{width: 14, height: 14, borderRadius: 99, background: '#EF4444'}} />
          <span style={{width: 14, height: 14, borderRadius: 99, background: '#F59E0B'}} />
          <span style={{width: 14, height: 14, borderRadius: 99, background: '#22C55E'}} />
        </div>
        <div style={{padding: 32, display: 'grid', gap: 20}}>
          {(lines.length ? lines : list(spec.keywords).slice(0, 3)).map((line, index) => (
            <div key={line} style={{display: 'grid', gridTemplateColumns: '76px 1fr', gap: 18, alignItems: 'center', padding: 18, borderRadius: 7, background: index === 0 ? `${palette.accent2}24` : '#FFFFFF', border: `1px solid ${palette.accent2}33`}}>
              <div style={{height: 54, borderRadius: 6, background: index % 2 ? palette.accent : palette.accent2}} />
              <div>
                <div style={{fontSize: 25, fontWeight: 900}}>{fitText(line, 36, 6)}</div>
                <div style={{height: 8, width: `${70 - index * 12}%`, marginTop: 12, background: '#CBD5E1', borderRadius: 99}} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </Shell>
  );
};

const EmphasisScene = ({spec, palette}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const pop = spring({frame: frame - 8, fps, config: {damping: 12, stiffness: 120}});
  const words = list(spec.keywords).slice(0, 5);
  return (
    <Shell spec={spec} palette={palette}>
      <div style={{position: 'absolute', left: 82, right: 82, top: 176, textAlign: 'center', transform: `scale(${0.88 + pop * 0.12})`}}>
        <div style={{fontSize: 86, lineHeight: 0.94, fontWeight: 950, textWrap: 'balance'}}>{fitText(spec.headline || spec.quote_text || spec.sentence_text, 58, 8)}</div>
        <div style={{margin: '34px auto 0', maxWidth: 850, fontSize: 32, lineHeight: 1.18, fontWeight: 650, color: palette.muted}}>{fitText(spec.deck || spec.footer_text || spec.context_text, 88, 14)}</div>
      </div>
      <div style={{position: 'absolute', left: 110, right: 110, bottom: 132, display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap'}}>
        {words.map((word, index) => {
          const reveal = interpolate(frame, [22 + index * 5, 32 + index * 5], [0, 1], {extrapolateRight: 'clamp'});
          return (
            <div key={word} style={{opacity: reveal, padding: '14px 18px', background: index % 2 ? palette.accent2 : palette.accent, color: '#fff', borderRadius: 6, fontSize: 24, fontWeight: 900}}>
              {fitText(word, 18, 2)}
            </div>
          );
        })}
      </div>
    </Shell>
  );
};

const VexAutoVisual = (props) => {
  const spec = props.spec || {};
  const palette = pickPalette(spec);
  const family = familyFor(spec);
  if (family === 'data') return <MetricScene spec={spec} palette={palette} />;
  if (family === 'contrast') return <ContrastScene spec={spec} palette={palette} />;
  if (family === 'timeline') return <TimelineScene spec={spec} palette={palette} />;
  if (family === 'interface') return <InterfaceScene spec={spec} palette={palette} />;
  if (family === 'emphasis') return <EmphasisScene spec={spec} palette={palette} />;
  return <MechanismScene spec={spec} palette={palette} />;
};

const Root = () => {
  return (
    <Composition
      id="VexAutoVisual"
      component={VexAutoVisual}
      durationInFrames={90}
      fps={30}
      width={1280}
      height={720}
      defaultProps={{spec: {}, width: 1280, height: 720, fps: 30, durationSec: 3}}
      calculateMetadata={({props}) => {
        const fps = clamp(Number(props.fps) || 30, 15, 120);
        const width = Math.max(320, Math.round(Number(props.width) || 1280));
        const height = Math.max(240, Math.round(Number(props.height) || 720));
        const durationSec = clamp(Number(props.durationSec) || Number(props.spec?.duration) || 3, 0.5, 30);
        return {
          durationInFrames: Math.max(1, Math.round(durationSec * fps)),
          fps,
          width,
          height,
        };
      }}
    />
  );
};

registerRoot(Root);
"""


class RemotionRenderer(VisualRenderer):
    name = "remotion"
    supported_templates = SUPPORTED_REMOTION_TEMPLATES

    def availability(self) -> RendererStatus:
        runner = _runner_path()
        if not runner.is_file():
            return RendererStatus(False, "Remotion runner is missing from the Vex installation.")
        node_major = _node_major_version()
        if node_major is None:
            return RendererStatus(False, "Node.js is not available in PATH; Remotion rendering requires Node.js.")
        if node_major < 22:
            return RendererStatus(False, f"Node.js {node_major} is too old for Vex's renderer runtime; install Node.js 22+.")
        platform, arch, platform_reason = _node_platform_arch()
        if platform_reason:
            return RendererStatus(False, platform_reason)
        platform_blocker = _remotion_platform_blocker(platform, arch)
        if platform_blocker:
            return RendererStatus(False, platform_blocker)
        packages_ok, package_reason = _run_node_package_probe()
        if not packages_ok:
            return RendererStatus(False, package_reason)
        return RendererStatus(True, "")

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        template = str(spec.get("template") or "").strip().lower()
        intent_type = str(spec.get("visual_intent_type") or "").strip().lower()
        renderer_hint = str(spec.get("renderer_hint") or "").strip().lower()
        visual_hint = str(spec.get("visual_type_hint") or "").strip().lower()
        composition = str(spec.get("composition_mode") or "").strip().lower()
        if intent_type == "spatial_3d" and renderer_hint != "remotion":
            return -1.0
        score = 0.92
        if renderer_hint == "remotion":
            score += 0.22
        if template.startswith("semantic_"):
            score += 0.16
        if _template_family(template, intent_type) in {"data", "mechanism", "contrast", "timeline", "interface"}:
            score += 0.18
        if visual_hint in {"product_ui", "process", "abstract_motion", "data_graphic"}:
            score += 0.08
        if composition == "replace":
            score += 0.08
        if intent_type == "math_or_formula" and renderer_hint != "remotion":
            score -= 0.3
        if intent_type == "spatial_3d":
            score -= 0.24
        return round(score, 3)

    def capability_summary(self) -> dict[str, Any]:
        base = super().capability_summary()
        base["render_model"] = "local_remotion_ssr"
        base["package_version"] = REMOTION_PACKAGE_VERSION
        base["composition_id"] = REMOTION_COMPOSITION_ID
        return base

    def render(
        self,
        spec: dict[str, Any],
        render_root: Path,
        width: int,
        height: int,
        fps: float,
    ) -> RenderedAsset:
        status = self.availability()
        if not status.available:
            raise VisualRendererError(status.reason)

        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        scene_name = _safe_scene_name(spec_id)
        job_dir = safe_render_job_dir(render_root, spec_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / "visual.mp4"
        entry_path = job_dir / "entry.jsx"
        spec_path = job_dir / "remotion_spec.json"
        input_props_path = job_dir / "input_props.json"
        request_path = job_dir / "render_request.json"
        result_path = job_dir / "remotion_result.json"
        log_path = job_dir / "remotion_render.log"
        metadata_path = job_dir / "remotion_metadata.json"

        input_props = _build_input_props(spec, width=width, height=height, fps=fps)
        entry_path.write_text(_remotion_entry_source(), encoding="utf-8")
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        input_props_path.write_text(json.dumps(input_props, indent=2), encoding="utf-8")
        request_path.write_text(
            json.dumps(
                {
                    "composition_id": REMOTION_COMPOSITION_ID,
                    "output_path": str(output_path),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "timeout_sec": _remotion_timeout_sec(),
                    "concurrency": _remotion_concurrency(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        node_path = shutil.which("node")
        if not node_path:
            raise VisualRendererError("Node.js is not available in PATH.")
        node_root, node_root_reason = _find_remotion_node_root()
        if node_root is None:
            raise VisualRendererError(node_root_reason)
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        request_payload["node_root"] = str(node_root)
        request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
        command = [node_path, str(_runner_path()), str(job_dir)]
        env = os.environ.copy()
        env["VEX_REMOTION_NODE_ROOT"] = str(node_root)
        env["VEX_REMOTION_TIMEOUT_MS"] = str(_remotion_timeout_ms())
        if _remotion_concurrency():
            env["VEX_REMOTION_CONCURRENCY"] = _remotion_concurrency()
        process_timeout = _remotion_timeout_sec()
        if process_timeout is not None:
            process_timeout += 45
        try:
            result = subprocess.run(
                command,
                cwd=str(node_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=process_timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VisualRendererError(f"Remotion render process failed for {spec_id}: {exc}") from exc
        _write_command_log(log_path, command, result)
        if result.returncode != 0 or not output_path.is_file():
            detail = (result.stderr or result.stdout or "").strip()
            if result_path.is_file():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    detail = str(payload.get("error") or detail)
                except (OSError, json.JSONDecodeError):
                    pass
            raise VisualRendererError(f"Remotion render failed for {spec_id}: {detail}")

        try:
            render_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            render_result = {"ok": True}
        video_metadata = probe_video(str(output_path))
        metadata = {
            **video_metadata,
            "renderer": self.name,
            "render_pipeline": "remotion_ssr_local",
            "remotion_version": REMOTION_PACKAGE_VERSION,
            "composition_id": REMOTION_COMPOSITION_ID,
            "template": str(spec.get("template") or ""),
            "template_family": _template_family(
                str(spec.get("template") or ""),
                str(spec.get("visual_intent_type") or ""),
            ),
            "scene_name": scene_name,
            "quality_score": _estimated_quality_score(spec),
            "quality_passed": True,
            "remotion_render": render_result,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(video_metadata.get("width") or width),
            height=int(video_metadata.get("height") or height),
            duration_sec=float(video_metadata.get("duration_sec") or input_props["durationSec"]),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(entry_path),
            artifact_paths={
                "entry_path": str(entry_path),
                "spec_path": str(spec_path),
                "input_props_path": str(input_props_path),
                "request_path": str(request_path),
                "result_path": str(result_path),
                "render_log_path": str(log_path),
                "metadata_path": str(metadata_path),
            },
            metadata=metadata,
        )
