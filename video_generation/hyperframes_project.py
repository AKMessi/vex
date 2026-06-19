from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from video_generation.cinematographer import (
    CinematicPlan,
    inline_cinematic_composition,
    write_cinematic_compositions,
)
from video_generation.models import Beat, BeatGraph, ScriptPlan, VideoGenerationRequest
from video_generation.motion import MotionPlan


HTML_WRITER_VERSION = "hyperframes-project-writer-v1"


def write_generation_project(
    *,
    project_dir: Path,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    audio_path: Path | None = None,
    transcript_path: Path | None = None,
    background_music_path: Path | None = None,
    cinematic_plan: CinematicPlan | None = None,
    motion_plan: MotionPlan | None = None,
) -> dict[str, str]:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "audio").mkdir(exist_ok=True)
    (project_dir / "renders").mkdir(exist_ok=True)
    (project_dir / "snapshots").mkdir(exist_ok=True)
    (project_dir / "compositions").mkdir(exist_ok=True)

    script_path = project_dir / "SCRIPT.md"
    storyboard_path = project_dir / "STORYBOARD.md"
    design_path = project_dir / "DESIGN.md"
    beat_graph_path = project_dir / "beat_graph.json"
    motion_plan_path = project_dir / "MOTION_PLAN.json"
    motion_cues_path = project_dir / "motion_cues.json"
    index_path = project_dir / "index.html"

    script_path.write_text(_script_markdown(plan), encoding="utf-8")
    storyboard_path.write_text(_storyboard_markdown(beat_graph), encoding="utf-8")
    design_path.write_text(_design_markdown(request, plan), encoding="utf-8")
    beat_graph_path.write_text(json.dumps(beat_graph.to_dict(), indent=2), encoding="utf-8")
    write_cinematic_compositions(
        cinematic_plan,
        compositions_dir=project_dir / "compositions",
        motion_plan=motion_plan,
        width=request.width,
        height=request.height,
    )
    cinematography_path = project_dir / "CINEMATOGRAPHY.json"
    if cinematic_plan is not None:
        cinematography_path.write_text(
            json.dumps(cinematic_plan.to_dict(), indent=2),
            encoding="utf-8",
        )
    if motion_plan is not None:
        motion_plan_path.write_text(
            json.dumps(motion_plan.to_dict(), indent=2),
            encoding="utf-8",
        )
        motion_cues_path.write_text(
            json.dumps(
                {
                    "version": motion_plan.version,
                    "source": motion_plan.source,
                    "cues": motion_plan.cue_rows(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    index_path.write_text(
        build_index_html(
            request=request,
            plan=plan,
            beat_graph=beat_graph,
            audio_path=audio_path,
            background_music_path=background_music_path,
            cinematic_plan=cinematic_plan,
            motion_plan=motion_plan,
        ),
        encoding="utf-8",
    )

    return {
        "script_path": str(script_path),
        "storyboard_path": str(storyboard_path),
        "design_path": str(design_path),
        "beat_graph_path": str(beat_graph_path),
        "index_path": str(index_path),
        "transcript_path": str(transcript_path or ""),
        "cinematography_path": str(cinematography_path if cinematic_plan is not None else ""),
        "motion_plan_path": str(motion_plan_path if motion_plan is not None else ""),
        "motion_cues_path": str(motion_cues_path if motion_plan is not None else ""),
    }


def copy_background_music(source_path: str, project_dir: Path) -> Path | None:
    if not source_path:
        return None
    source = Path(source_path).expanduser().resolve(strict=True)
    if not source.is_file():
        return None
    suffix = source.suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}:
        return None
    target = project_dir / "audio" / f"background_music{suffix}"
    shutil.copy2(source, target)
    return target


def build_index_html(
    *,
    request: VideoGenerationRequest,
    plan: ScriptPlan,
    beat_graph: BeatGraph,
    audio_path: Path | None = None,
    background_music_path: Path | None = None,
    cinematic_plan: CinematicPlan | None = None,
    motion_plan: MotionPlan | None = None,
) -> str:
    duration = max(beat_graph.duration_sec, 1.0)
    audio_markup = _audio_markup(
        request=request,
        project_dir=None,
        duration=duration,
        audio_path=audio_path,
        background_music_path=background_music_path,
    )
    scene_markup = "\n".join(
        _timeline_scene_markup(
            beat,
            request=request,
            cinematic_plan=cinematic_plan,
            motion_plan=motion_plan,
        )
        for beat in beat_graph.beats
    )
    transition_markup = _transition_markup(beat_graph, motion_plan=motion_plan)
    caption_markup = "\n".join(
        _caption_markup(beat, cinematic_plan=cinematic_plan, motion_plan=motion_plan)
        for beat in beat_graph.beats
    )
    metadata = {
        "version": HTML_WRITER_VERSION,
        "title": plan.title,
        "duration_sec": round(duration, 3),
        "beat_count": len(beat_graph.beats),
        "audio_first": True,
        "width": request.width,
        "height": request.height,
        "fps": request.fps,
        "cinematographer": cinematic_plan.version if cinematic_plan is not None else "",
        "cinematic_beat_count": cinematic_plan.accepted_count if cinematic_plan is not None else 0,
        "native_motion": motion_plan.version if motion_plan is not None else "",
        "native_motion_beat_count": motion_plan.native_composition_count if motion_plan is not None else 0,
        "audio_motion_cue_count": motion_plan.audio_cue_count if motion_plan is not None else 0,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width={request.width}, height={request.height}, initial-scale=1">
  <title>{_h(plan.title)}</title>
  <style>{_css(request)}</style>
</head>
<body>
  <div id="root" data-composition-id="root" data-start="0" data-duration="{duration:.3f}" data-width="{request.width}" data-height="{request.height}" data-generation-metadata='{_json_attr(metadata)}'>
    <div id="global-bg" class="clip global-bg" data-start="0" data-duration="{duration:.3f}" data-track-index="1" aria-hidden="true">
      <span></span><span></span><span></span><span></span>
    </div>
    <div id="motion-field" class="clip motion-field" data-start="0" data-duration="{duration:.3f}" data-track-index="2" aria-hidden="true">
      <i></i><i></i><i></i><i></i><i></i>
    </div>
    {audio_markup}
    <main class="stage-stack">
      {scene_markup}
    </main>
    <section class="transition-layer" aria-hidden="true">
      {transition_markup}
    </section>
    <section class="caption-layer" aria-label="Timed captions">
      {caption_markup}
    </section>
    <script id="vex-motion-plan" type="application/json">{_script_json(motion_plan.to_dict() if motion_plan is not None else {})}</script>
  </div>
  {_timeline_script(duration, motion_plan=motion_plan)}
</body>
</html>
"""


def _timeline_scene_markup(
    beat: Beat,
    *,
    request: VideoGenerationRequest,
    cinematic_plan: CinematicPlan | None,
    motion_plan: MotionPlan | None,
) -> str:
    cinematic = _cinematic_beat(cinematic_plan, beat.beat_id)
    if cinematic is not None and cinematic.compiler_passed and cinematic.composition_html:
        return _cinematic_scene_markup(
            beat,
            request=request,
            cinematic=cinematic,
            motion_plan=motion_plan,
        )
    return _scene_markup(beat, request=request)


def _cinematic_beat(cinematic_plan: CinematicPlan | None, beat_id: str):
    if cinematic_plan is None:
        return None
    for item in cinematic_plan.beat_compositions:
        if item.beat_id == beat_id:
            return item
    return None


def _cinematic_scene_markup(
    beat: Beat,
    *,
    request: VideoGenerationRequest,
    cinematic,
    motion_plan: MotionPlan | None,
) -> str:
    profile = motion_plan.profile_for(beat.beat_id) if motion_plan is not None else None
    if cinematic.composition_src:
        profile_dict = profile.to_dict() if profile is not None else {}
        markup = inline_cinematic_composition(
            cinematic,
            start=beat.start,
            duration=beat.duration,
            track_index=10,
        )
        native_classes = [
            "native-hyperframes-beat",
            f"native-medium-{_clean_css_class(profile.medium_family if profile else cinematic.template)}",
            f"native-energy-{_clean_css_class(profile.energy if profile else 'calm')}",
        ]
        native_attrs = (
            f' data-external-composition-src="{_h(cinematic.composition_src)}"'
            ' data-native-hyperframes-motion="true"'
            ' data-vex-native-composition="true"'
            f' data-vex-beat-id="{_h(beat.beat_id)}"'
            f' data-vex-motion-scope="{_h(cinematic.composition_id)}:{_h(beat.beat_id)}"'
            f' data-vex-width="{request.width}"'
            f' data-vex-height="{request.height}"'
            f" data-motion-profile='{_json_attr(profile_dict)}'"
        )
        markup = re.sub(
            r'class="([^"]*\bbeat-composition\b[^"]*)"',
            lambda match: f'class="{match.group(1)} {" ".join(native_classes)}"',
            markup,
            count=1,
        )
        return re.sub(
            r"(<div[^>]*\bdata-cinematic-composition-id=\""
            + re.escape(cinematic.composition_id)
            + r"\"[^>]*)(>)",
            r"\1" + native_attrs + r"\2",
            markup,
            count=1,
        )
    return inline_cinematic_composition(
        cinematic,
        start=beat.start,
        duration=beat.duration,
        track_index=10,
    )


def _audio_markup(
    *,
    request: VideoGenerationRequest,
    project_dir: Path | None,
    duration: float,
    audio_path: Path | None,
    background_music_path: Path | None,
) -> str:
    items: list[str] = []
    if audio_path is not None:
        src = _relative_audio_src(audio_path)
        items.append(
            f'<audio id="narration-audio" class="clip" data-start="0" data-duration="{duration:.3f}" '
            f'data-track-index="0" data-volume="1" src="{_h(src)}"></audio>'
        )
    if background_music_path is not None:
        src = _relative_audio_src(background_music_path)
        items.append(
            f'<audio id="background-music" class="clip" data-start="0" data-duration="{duration:.3f}" '
            f'data-track-index="0" data-volume="{request.music_volume:.3f}" src="{_h(src)}"></audio>'
        )
    return "\n    ".join(items)


def _relative_audio_src(path: Path) -> str:
    parts = path.parts
    if "audio" in parts:
        audio_index = parts.index("audio")
        return "/".join(parts[audio_index:])
    return path.name


def _scene_markup(beat: Beat, *, request: VideoGenerationRequest) -> str:
    classes = f"clip scene scene-{_clean_class(beat.scene_type)}"
    keywords = beat.keywords[:4] or ["idea", "signal", "motion"]
    keyword_markup = "\n".join(
        f'<li style="--i:{index}">{_h(keyword)}</li>'
        for index, keyword in enumerate(keywords, start=1)
    )
    body = _scene_body(beat)
    return f"""
      <section id="{beat.beat_id}" class="{classes}" data-start="{beat.start:.3f}" data-duration="{beat.duration:.3f}" data-track-index="10" data-scene-type="{_h(beat.scene_type)}">
        <div class="scene-chrome">
          <span>{beat.index:02d}</span>
          <b>{_h(beat.scene_type.replace("_", " "))}</b>
        </div>
        <div class="scene-copy">
          <p>{_h(beat.visual_metaphor)}</p>
          <h1>{_h(beat.title)}</h1>
        </div>
        <ul class="keyword-strip">{keyword_markup}</ul>
        {body}
      </section>
    """


def _scene_body(beat: Beat) -> str:
    if beat.scene_type == "metric":
        metric = _first_metric(beat.narration) or f"{max(beat.index * 17, 12)}%"
        return f"""
        <div class="metric-body">
          <strong>{_h(metric)}</strong>
          <div class="meter-grid">{''.join(f'<i style="--i:{i}"></i>' for i in range(1, 13))}</div>
          <small>{_h(beat.caption)}</small>
        </div>
        """
    if beat.scene_type == "contrast":
        return f"""
        <div class="contrast-body">
          <article><span>Before</span><b>{_h(_short_phrase(beat.narration, 0))}</b></article>
          <div class="hinge"><i></i></div>
          <article><span>After</span><b>{_h(_short_phrase(beat.narration, 1))}</b></article>
        </div>
        """
    if beat.scene_type == "process":
        labels = beat.keywords[:5] or ["input", "route", "decision", "output"]
        nodes = "".join(
            f'<li style="--i:{index}"><span>{index}</span><b>{_h(label)}</b></li>'
            for index, label in enumerate(labels, start=1)
        )
        return f"""
        <div class="process-body">
          <svg viewBox="0 0 900 220" aria-hidden="true">
            <path class="route-shadow" d="M40 120 C210 15 330 220 470 120 S720 25 860 120" pathLength="1"></path>
            <path class="route-line" d="M40 120 C210 15 330 220 470 120 S720 25 860 120" pathLength="1"></path>
          </svg>
          <ol>{nodes}</ol>
        </div>
        """
    if beat.scene_type == "proof":
        layers = beat.keywords[:4] or ["claim", "evidence", "mechanism", "takeaway"]
        return f"""
        <div class="proof-body">
          {''.join(f'<article style="--i:{index}"><span>{index:02d}</span><b>{_h(label)}</b></article>' for index, label in enumerate(layers, start=1))}
          <em>{_h(_short_phrase(beat.narration, 0))}</em>
        </div>
        """
    if beat.scene_type == "hook":
        return f"""
        <div class="hook-body">
          <div class="signal-sweep"></div>
          <strong>{_h(_short_phrase(beat.narration, 0))}</strong>
          <span>{_h(_short_phrase(beat.narration, 1))}</span>
        </div>
        """
    tiles = beat.keywords[:5] or ["context", "pattern", "signal"]
    return f"""
        <div class="concept-body">
          {''.join(f'<article style="--i:{index}"><i></i><b>{_h(label)}</b></article>' for index, label in enumerate(tiles, start=1))}
        </div>
    """


def _transition_markup(beat_graph: BeatGraph, *, motion_plan: MotionPlan | None) -> str:
    items: list[str] = []
    for beat in beat_graph.beats:
        if beat.index >= len(beat_graph.beats):
            continue
        profile = motion_plan.profile_for(beat.beat_id) if motion_plan is not None else None
        transition = profile.transition_out if profile is not None else "whip_pan"
        duration = min(0.72, max(0.36, beat.duration * 0.18))
        start = max(0.0, beat.end - duration * 0.55)
        items.append(
            f"""
      <div id="transition-{_h(beat.beat_id)}"
        class="clip scene-transition transition-{_clean_css_class(transition)}"
        data-start="{start:.3f}"
        data-duration="{duration:.3f}"
        data-track-index="90"
        data-transition-kind="{_h(transition)}">
        <span></span><span></span><span></span>
      </div>
            """
        )
    return "\n".join(items)


def _caption_markup(
    beat: Beat,
    *,
    cinematic_plan: CinematicPlan | None = None,
    motion_plan: MotionPlan | None = None,
) -> str:
    cinematic = _cinematic_beat(cinematic_plan, beat.beat_id)
    profile = motion_plan.profile_for(beat.beat_id) if motion_plan is not None else None
    caption_style = profile.caption_style if profile is not None else "calm_karaoke_highlight"
    native_class = "native-caption" if cinematic is not None and cinematic.compiler_passed else "fallback-caption"
    return (
        f'<p id="caption-{beat.beat_id}" class="clip caption {native_class} caption-{_clean_css_class(caption_style)}" '
        f'data-start="{beat.start:.3f}" data-duration="{beat.duration:.3f}" '
        f'data-track-index="70" data-caption-style="{_h(caption_style)}">'
        f'{_h(_compact_caption(beat.caption))}</p>'
    )


def _compact_caption(text: str) -> str:
    words = str(text or "").split()
    if len(words) <= 12:
        return str(text or "")
    return " ".join(words[:12]).rstrip(" ,.;:") + "..."


def _timeline_script(duration: float, *, motion_plan: MotionPlan | None = None) -> str:
    cue_payload = motion_plan.cue_rows() if motion_plan is not None else []
    return f"""
  <script>
  (() => {{
    const duration = {duration:.6f};
    const audioCues = {_script_json(cue_payload)};
    const root = document.getElementById("root");
    const scenes = Array.from(document.querySelectorAll(".scene"));
    const beatCompositions = Array.from(document.querySelectorAll(".beat-composition"));
    const captions = Array.from(document.querySelectorAll(".caption"));
    const transitions = Array.from(document.querySelectorAll(".scene-transition"));
    const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
    const ease = value => 1 - Math.pow(1 - clamp(value), 3);
    const easeInOut = value => {{
      const p = clamp(value);
      return p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
    }};
    function localProgress(el, time) {{
      const start = Number(el.dataset.start || 0);
      const span = Math.max(Number(el.dataset.duration || 0), 0.001);
      return clamp((time - start) / span);
    }}
    function applyTimedElement(el, time, activeScale = 1) {{
      const p = localProgress(el, time);
      const e = ease(p);
      const start = Number(el.dataset.start || 0);
      const end = start + Number(el.dataset.duration || 0);
      const active = time >= start && time <= end;
      el.style.setProperty("--local-p", p.toFixed(5));
      el.style.setProperty("--ease-p", e.toFixed(5));
      el.style.opacity = active ? String(activeScale) : "0";
    }}
    function seekNestedComposition(el, time) {{
      const compId = el.dataset.compositionId || "";
      if (!compId || compId === "root") return;
      const start = Number(el.dataset.start || 0);
      const span = Math.max(Number(el.dataset.duration || 0), 0.001);
      const localTime = clamp(time - start, 0, span);
      const nested = (window.__timelines || {{}})[compId];
      if (!nested || nested === timeline) return;
      try {{
        if (typeof nested.time === "function") {{
          nested.time(localTime);
          return;
        }}
        if (typeof nested.seek === "function") {{
          nested.seek(localTime);
          return;
        }}
        if (typeof nested.progress === "function") {{
          nested.progress(span > 0 ? clamp(localTime / span) : 0);
        }}
      }} catch (_) {{}}
    }}
    function cueStrength(time, band) {{
      let value = 0;
      audioCues.forEach((cue) => {{
        if (cue.band !== band) return;
        const start = Number(cue.start || 0);
        const end = Math.max(Number(cue.end || start + .12), start + .05);
        const p = clamp((time - start) / (end - start));
        const envelope = Math.sin(p * Math.PI);
        value = Math.max(value, envelope * Number(cue.strength || .5));
      }});
      return clamp(value);
    }}
    function applyTransition(el, time) {{
      const p = localProgress(el, time);
      const e = easeInOut(p);
      const start = Number(el.dataset.start || 0);
      const end = start + Number(el.dataset.duration || 0);
      const active = time >= start && time <= end;
      el.style.setProperty("--transition-p", p.toFixed(5));
      el.style.setProperty("--transition-ease", e.toFixed(5));
      el.style.opacity = active ? String(Math.sin(p * Math.PI).toFixed(5)) : "0";
    }}
    function renderAt(rawTime) {{
      const time = clamp(Number(rawTime) || 0, 0, duration);
      const p = duration > 0 ? clamp(time / duration) : 1;
      const bass = cueStrength(time, "bass");
      const mids = cueStrength(time, "mids");
      const treble = cueStrength(time, "treble");
      const amp = Math.max(cueStrength(time, "amplitude"), bass * .72, mids * .42);
      root.style.setProperty("--p", p.toFixed(5));
      root.style.setProperty("--pulse", (0.5 + Math.sin(p * Math.PI * 2) * 0.5).toFixed(5));
      root.style.setProperty("--bass", bass.toFixed(5));
      root.style.setProperty("--mids", mids.toFixed(5));
      root.style.setProperty("--treble", treble.toFixed(5));
      root.style.setProperty("--amp", amp.toFixed(5));
      scenes.forEach((scene) => applyTimedElement(scene, time, 1));
      beatCompositions.forEach((composition) => {{
        applyTimedElement(composition, time, 1);
        seekNestedComposition(composition, time);
      }});
      captions.forEach((caption) => applyTimedElement(caption, time, 1));
      transitions.forEach((transition) => applyTransition(transition, time));
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
    window.__timelines.root = timeline;
    window.__timelines["root"] = timeline;
    renderAt(0);
  }})();
  </script>
    """


def _css(request: VideoGenerationRequest) -> str:
    return f"""
    :root {{
      color-scheme: dark;
      --bg: #080A0F;
      --panel: #111827;
      --panel-2: #172033;
      --text: #F8FAFC;
      --muted: #C7D2FE;
      --accent: #2DD4BF;
      --accent-2: #F97316;
      --accent-3: #A3E635;
      --stroke: rgba(248, 250, 252, 0.20);
      --shadow: rgba(0, 0, 0, 0.36);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      width: {request.width}px;
      height: {request.height}px;
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      font-family: Inter, system-ui, sans-serif;
      color: var(--text);
      letter-spacing: 0;
    }}
    #root {{
      position: relative;
      width: {request.width}px;
      height: {request.height}px;
      overflow: hidden;
      background:
        linear-gradient(135deg, #080A0F 0%, #101827 44%, #17181D 100%);
    }}
    .global-bg, .motion-field, .stage-stack, .transition-layer, .caption-layer {{
      position: absolute;
      inset: 0;
    }}
    .beat-composition {{
      position: absolute;
      inset: 0;
      width: {request.width}px;
      height: {request.height}px;
      overflow: hidden;
      background: var(--bg);
      opacity: 1;
    }}
    .native-hyperframes-beat {{
      transform: scale(calc(1 + var(--amp, 0) * .006));
      transform-origin: center;
      filter: saturate(calc(1 + var(--treble, 0) * .16)) contrast(calc(1 + var(--bass, 0) * .05));
    }}
    .global-bg {{
      background:
        linear-gradient(90deg, rgba(45, 212, 191, .10) 1px, transparent 1px),
        linear-gradient(0deg, rgba(248, 250, 252, .055) 1px, transparent 1px);
      background-size: 88px 88px;
      opacity: .78;
      transform: translate3d(calc(var(--p, 0) * -42px), calc(var(--p, 0) * -28px), 0);
    }}
    .global-bg span {{
      position: absolute;
      display: block;
      width: 42%;
      height: 2px;
      background: linear-gradient(90deg, transparent, rgba(45, 212, 191, .68), transparent);
      transform: rotate(calc(var(--p, 0) * 40deg));
      opacity: .34;
    }}
    .global-bg span:nth-child(1) {{ left: 4%; top: 17%; }}
    .global-bg span:nth-child(2) {{ right: 2%; top: 36%; background: linear-gradient(90deg, transparent, rgba(249, 115, 22, .62), transparent); }}
    .global-bg span:nth-child(3) {{ left: 12%; bottom: 22%; }}
    .global-bg span:nth-child(4) {{ right: 10%; bottom: 12%; background: linear-gradient(90deg, transparent, rgba(163, 230, 53, .56), transparent); }}
    .motion-field {{
      pointer-events: none;
      opacity: .78;
      filter: drop-shadow(0 0 24px rgba(45, 212, 191, .18));
    }}
    .motion-field i {{
      position: absolute;
      display: block;
      width: 9px;
      height: 9px;
      background: var(--accent);
      transform: translate3d(calc(var(--p, 0) * 420px), calc((var(--pulse, .5) * 80px) + (var(--bass, 0) * 42px)), 0) rotate(45deg) scale(calc(1 + var(--amp, 0) * 1.4));
    }}
    .motion-field i:nth-child(1) {{ left: 8%; top: 14%; }}
    .motion-field i:nth-child(2) {{ left: 25%; top: 78%; background: var(--accent-2); }}
    .motion-field i:nth-child(3) {{ right: 20%; top: 22%; background: var(--accent-3); }}
    .motion-field i:nth-child(4) {{ right: 8%; bottom: 24%; }}
    .motion-field i:nth-child(5) {{ left: 48%; top: 48%; background: var(--accent-2); }}
    .scene {{
      position: absolute;
      inset: 6.5% 6%;
      display: grid;
      grid-template-columns: minmax(280px, .54fr) minmax(420px, 1fr);
      grid-template-rows: auto 1fr auto;
      gap: 26px 44px;
      opacity: 0;
      transform: translate3d(0, calc((1 - var(--ease-p, 0)) * 26px), 0) scale(calc(.985 + var(--ease-p, 0) * .015));
      transition: none;
    }}
    .scene-chrome {{
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 14px;
      height: 46px;
      color: var(--muted);
      font-size: 15px;
      font-weight: 850;
      text-transform: uppercase;
    }}
    .scene-chrome span {{
      display: grid;
      place-items: center;
      width: 46px;
      height: 46px;
      background: var(--text);
      color: #090A0F;
      font-weight: 950;
    }}
    .scene-copy {{
      min-width: 0;
      align-self: center;
    }}
    .scene-copy p {{
      max-width: 520px;
      margin: 0 0 24px;
      color: var(--accent);
      font-size: clamp(18px, 1.6vw, 30px);
      line-height: 1.15;
      font-weight: 760;
    }}
    .scene-copy h1 {{
      max-width: 720px;
      margin: 0;
      color: var(--text);
      font-size: clamp(52px, 6.2vw, 118px);
      line-height: .92;
      font-weight: 940;
      overflow-wrap: anywhere;
    }}
    .keyword-strip {{
      grid-column: 1 / -1;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .keyword-strip li {{
      padding: 12px 16px;
      border-top: 3px solid var(--accent);
      background: rgba(17, 24, 39, .78);
      color: var(--text);
      font-size: 20px;
      font-weight: 820;
      transform: translateY(calc((1 - var(--ease-p, 0)) * (14px + var(--i) * 3px)));
      opacity: var(--ease-p, 0);
    }}
    .hook-body, .metric-body, .contrast-body, .process-body, .proof-body, .concept-body {{
      position: relative;
      min-width: 0;
      align-self: center;
      min-height: 54%;
    }}
    .hook-body {{
      display: grid;
      place-items: center;
      border-left: 8px solid var(--accent-2);
      background: linear-gradient(90deg, rgba(249, 115, 22, .18), rgba(17, 24, 39, .18));
      overflow: hidden;
    }}
    .hook-body strong {{
      width: 80%;
      font-size: clamp(44px, 5.4vw, 104px);
      line-height: .95;
      text-align: center;
      overflow-wrap: anywhere;
    }}
    .hook-body span {{
      position: absolute;
      left: 28px;
      bottom: 24px;
      max-width: 70%;
      color: var(--muted);
      font-size: 26px;
      font-weight: 760;
    }}
    .signal-sweep {{
      position: absolute;
      inset: 0;
      background: linear-gradient(110deg, transparent 0 38%, rgba(45, 212, 191, .28) 48%, transparent 58% 100%);
      transform: translateX(calc(-60% + var(--local-p, 0) * 130%));
    }}
    .metric-body {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 24px;
      padding: 36px;
      background: rgba(17, 24, 39, .76);
      border: 1px solid var(--stroke);
      box-shadow: 0 28px 90px var(--shadow);
    }}
    .metric-body strong {{
      color: var(--accent-3);
      font-size: clamp(76px, 11vw, 190px);
      line-height: .82;
      font-weight: 960;
    }}
    .meter-grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 10px;
      align-items: end;
    }}
    .meter-grid i {{
      display: block;
      height: calc(36px + var(--i) * 12px * var(--ease-p, 0));
      background: linear-gradient(180deg, var(--accent-3), var(--accent));
    }}
    .metric-body small {{
      color: var(--muted);
      font-size: 24px;
      line-height: 1.2;
    }}
    .contrast-body {{
      display: grid;
      grid-template-columns: 1fr 74px 1fr;
      gap: 22px;
      align-items: stretch;
    }}
    .contrast-body article {{
      display: grid;
      align-content: end;
      min-width: 0;
      padding: 34px;
      background: rgba(17, 24, 39, .78);
      border: 1px solid var(--stroke);
    }}
    .contrast-body article:last-child {{
      border-top: 7px solid var(--accent);
      transform: translateY(calc((1 - var(--ease-p, 0)) * -24px));
    }}
    .contrast-body span {{
      color: var(--accent-2);
      font-size: 18px;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .contrast-body b {{
      margin-top: 18px;
      font-size: clamp(30px, 3.2vw, 58px);
      line-height: 1.02;
      overflow-wrap: anywhere;
    }}
    .hinge {{
      display: grid;
      place-items: center;
    }}
    .hinge i {{
      display: block;
      width: 6px;
      height: calc(120px + var(--ease-p, 0) * 260px);
      background: var(--accent);
      box-shadow: 0 0 38px rgba(45, 212, 191, .42);
    }}
    .process-body svg {{
      position: absolute;
      inset: 4% 2% auto;
      width: 96%;
      height: 52%;
      overflow: visible;
    }}
    .route-shadow, .route-line {{
      fill: none;
      stroke-linecap: square;
    }}
    .route-shadow {{
      stroke: rgba(0, 0, 0, .42);
      stroke-width: 26;
    }}
    .route-line {{
      stroke: var(--accent);
      stroke-width: 9;
      stroke-dasharray: 1;
      stroke-dashoffset: calc(1 - var(--ease-p, 0));
      filter: drop-shadow(0 0 20px rgba(45, 212, 191, .38));
    }}
    .process-body ol {{
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .process-body li {{
      min-height: 128px;
      padding: 20px;
      background: rgba(17, 24, 39, .78);
      border-top: 5px solid var(--accent);
      transform: translateY(calc((1 - var(--ease-p, 0)) * var(--i) * 7px));
    }}
    .process-body span {{
      color: var(--accent);
      font-size: 18px;
      font-weight: 900;
    }}
    .process-body b {{
      display: block;
      margin-top: 14px;
      font-size: 24px;
      line-height: 1.08;
      overflow-wrap: anywhere;
    }}
    .proof-body {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      align-items: end;
    }}
    .proof-body article {{
      min-height: calc(160px + var(--i) * 34px);
      padding: 24px;
      background: rgba(17, 24, 39, .78);
      border-bottom: 6px solid var(--accent-3);
      transform: translateY(calc((1 - var(--ease-p, 0)) * (42px - var(--i) * 4px)));
    }}
    .proof-body span {{
      color: var(--accent-3);
      font-size: 20px;
      font-weight: 950;
    }}
    .proof-body b {{
      display: block;
      margin-top: 22px;
      font-size: 28px;
      line-height: 1.08;
      overflow-wrap: anywhere;
    }}
    .proof-body em {{
      position: absolute;
      left: 0;
      right: 0;
      bottom: -56px;
      color: var(--muted);
      font-size: 24px;
      font-style: normal;
      font-weight: 760;
    }}
    .concept-body {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      align-content: center;
    }}
    .concept-body article {{
      min-height: 180px;
      display: grid;
      align-content: end;
      padding: 26px;
      background: rgba(17, 24, 39, .76);
      border: 1px solid var(--stroke);
      transform: translate3d(calc((1 - var(--ease-p, 0)) * var(--i) * 8px), calc((1 - var(--ease-p, 0)) * 20px), 0);
    }}
    .concept-body i {{
      width: 42px;
      height: 42px;
      margin-bottom: 28px;
      background: var(--accent-2);
      transform: rotate(calc(45deg + var(--local-p, 0) * 90deg));
    }}
    .concept-body b {{
      font-size: 28px;
      line-height: 1.06;
      overflow-wrap: anywhere;
    }}
    .caption-layer {{
      z-index: 70;
      pointer-events: none;
      display: grid;
      place-items: end center;
      padding: 0 7% 5%;
    }}
    .caption {{
      max-width: min(980px, 78%);
      margin: 0;
      padding: 11px 16px;
      background: rgba(8, 10, 15, .72);
      border-top: 2px solid var(--accent);
      color: var(--text);
      font-size: clamp(20px, 1.55vw, 30px);
      line-height: 1.1;
      font-weight: 780;
      text-align: center;
      overflow-wrap: anywhere;
      opacity: 0;
      transform: translateY(calc((1 - var(--ease-p, 0)) * 18px)) scale(calc(1 + var(--bass, 0) * .035));
      box-shadow: 0 18px 52px rgba(0,0,0,.28);
    }}
    .caption-scale-pop-keyword {{
      border-top-color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .caption-kinetic-slam {{
      border-top-color: var(--accent-3);
      font-weight: 920;
    }}
    .caption-monospace-typewriter {{
      font-family: monospace;
      text-align: left;
    }}
    .transition-layer {{
      z-index: 90;
      pointer-events: none;
    }}
    .scene-transition {{
      position: absolute;
      inset: -8%;
      opacity: 0;
      overflow: hidden;
      mix-blend-mode: screen;
    }}
    .scene-transition span {{
      position: absolute;
      inset: 0;
      display: block;
      transform: translate3d(calc((-120% + var(--transition-ease, 0) * 240%)), 0, 0) skewX(-18deg);
      background: linear-gradient(90deg, transparent, rgba(45,212,191,.72), rgba(249,115,22,.58), transparent);
      filter: blur(18px);
    }}
    .scene-transition span:nth-child(2) {{
      transform: translate3d(calc((120% - var(--transition-ease, 0) * 240%)), 0, 0) skewX(22deg);
      background: linear-gradient(90deg, transparent, rgba(163,230,53,.52), rgba(248,250,252,.65), transparent);
      filter: blur(28px);
    }}
    .scene-transition span:nth-child(3) {{
      left: 48%;
      width: 10%;
      background: rgba(248,250,252,.86);
      transform: scaleX(calc(.1 + var(--transition-ease, 0) * 8));
      filter: blur(22px);
    }}
    .transition-ridged-burn {{
      mix-blend-mode: color-dodge;
      background:
        repeating-linear-gradient(90deg, rgba(249,115,22,.22) 0 16px, transparent 16px 34px),
        radial-gradient(circle at calc(var(--transition-ease, 0) * 100%) 50%, rgba(249,115,22,.7), transparent 42%);
    }}
    .transition-cross-warp-morph {{
      backdrop-filter: blur(calc(var(--transition-ease, 0) * 16px));
      background: radial-gradient(circle at 50% 50%, rgba(45,212,191,.38), transparent 52%);
    }}
    .transition-flash-through-white {{
      mix-blend-mode: normal;
      background: rgba(248,250,252,calc(var(--transition-ease, 0) * .72));
    }}
    .transition-light-leak {{
      background:
        radial-gradient(circle at 10% 45%, rgba(249,115,22,.72), transparent 34%),
        radial-gradient(circle at 82% 52%, rgba(45,212,191,.52), transparent 32%);
    }}
    @media (max-aspect-ratio: 1/1) {{
      .scene {{
        inset: 5% 6%;
        grid-template-columns: 1fr;
        grid-template-rows: auto auto 1fr auto;
        gap: 22px;
      }}
      .scene-copy h1 {{
        font-size: clamp(54px, 10vw, 118px);
      }}
      .contrast-body, .proof-body, .concept-body {{
        grid-template-columns: 1fr;
      }}
      .process-body ol {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}
    """


def _script_markdown(plan: ScriptPlan) -> str:
    return f"# {plan.title}\n\n{plan.narration}\n"


def _storyboard_markdown(beat_graph: BeatGraph) -> str:
    lines = ["# Storyboard", ""]
    for beat in beat_graph.beats:
        lines.extend(
            [
                f"## {beat.index:02d}. {beat.title}",
                f"- Time: {beat.start:.2f}s to {beat.end:.2f}s",
                f"- Scene type: {beat.scene_type}",
                f"- Visual: {beat.visual_metaphor}",
                f"- Narration: {beat.narration}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _design_markdown(request: VideoGenerationRequest, plan: ScriptPlan) -> str:
    return (
        "# Design\n\n"
        f"- Direction: {plan.design_direction}\n"
        f"- Frame: {request.width}x{request.height} at {request.fps}fps\n"
        f"- Aspect: {request.aspect}\n"
        "- Rule: audio timing is source of truth; visuals follow beat_graph.json.\n"
    )


def _json_attr(value: dict[str, Any]) -> str:
    return html.escape(json.dumps(value, ensure_ascii=True), quote=True)


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def _h(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _clean_class(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-") or "concept"


def _clean_css_class(value: object) -> str:
    return _clean_class(str(value or "")).replace("_", "-")


def _first_metric(text: str) -> str | None:
    match = re.search(
        r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|seconds?|tokens?|users?|gb|mb|billion|million)?\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else None


def _short_phrase(text: str, variant: int) -> str:
    chunks = [
        chunk.strip(" ,.;:")
        for chunk in re.split(r"\b(?:but|instead|then|so|because|and)\b|[,.;:]", text, flags=re.IGNORECASE)
        if len(chunk.strip().split()) >= 2
    ]
    if not chunks:
        chunks = [text]
    selected = chunks[min(variant, len(chunks) - 1)]
    words = selected.split()
    if len(words) > 8:
        selected = " ".join(words[:8])
    return selected
