from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from vex_hyperframes.safety import validate_authored_html_safety
from vex_hyperframes.scene_program import validate_scene_program
from vex_hyperframes.visual_world import validate_visual_world_program


@dataclass(frozen=True)
class CompiledVisualWorldStage:
    html: str
    metadata: dict[str, Any]


def compile_visual_world_stage(
    program: dict[str, Any],
    *,
    scene_program: dict[str, Any],
    ir: dict[str, Any],
    claim_graph: dict[str, Any],
    source_asset_data_uri: str = "",
) -> CompiledVisualWorldStage:
    world_validation = validate_visual_world_program(
        program,
        scene_program=scene_program,
    )
    if not world_validation.passed:
        raise ValueError(
            "Unsafe Visual World Program: "
            + "; ".join(world_validation.errors)
        )
    scene_validation = validate_scene_program(
        scene_program,
        ir=ir,
        claim_graph=claim_graph,
    )
    if not scene_validation.passed:
        raise ValueError(
            "Unsafe Scene Program V2: "
            + "; ".join(scene_validation.errors)
        )
    medium = str(program.get("medium_family") or "")
    compiler = _COMPILERS.get(medium)
    if compiler is None:
        raise ValueError(f"Unsupported visual-world medium: {medium!r}.")
    fragment = compiler(program, scene_program)
    fragment = fragment.replace(
        "</section>",
        _relation_telemetry(scene_program) + "</section>",
        1,
    )
    safety_source = fragment.replace(
        "__VEX_SOURCE_ASSET__",
        '<div class="vw-source-placeholder"></div>',
    )
    safety = validate_authored_html_safety(safety_source)
    if not safety.safe:
        raise ValueError(
            "Compiled Visual World HTML failed safety validation: "
            + "; ".join(safety.errors)
        )
    embedded_source = bool(
        source_asset_data_uri
        and "__VEX_SOURCE_ASSET__" in fragment
    )
    if embedded_source:
        fragment = fragment.replace(
            "__VEX_SOURCE_ASSET__",
            (
                '<img class="vw-source-image" '
                f'src="{html.escape(source_asset_data_uri, quote=True)}" '
                'alt="Grounded source frame">'
            ),
        )
    else:
        fragment = fragment.replace(
            "__VEX_SOURCE_ASSET__",
            '<div class="vw-source-placeholder"></div>',
        )
    fingerprint = dict(program.get("fingerprint") or {})
    return CompiledVisualWorldStage(
        html=fragment,
        metadata={
            "world_id": str(program.get("world_id") or ""),
            "world_signature": str(program.get("world_signature") or ""),
            "medium_family": medium,
            "canvas_system": str(program.get("canvas_system") or ""),
            "shape_language": str(program.get("shape_language") or ""),
            "material_system": str(program.get("material_system") or ""),
            "typography_system": str(program.get("typography_system") or ""),
            "camera_depth": str(program.get("camera_depth") or ""),
            "motion_choreography": str(
                program.get("motion_choreography") or ""
            ),
            "background_mode": str(program.get("background_mode") or ""),
            "card_policy": str(program.get("card_policy") or ""),
            "panel_ratio_target": float(
                fingerprint.get("panel_ratio_target") or 0.0
            ),
            "fingerprint": fingerprint,
            "source_asset_embedded": embedded_source,
            "object_coverage": scene_validation.object_coverage,
            "relation_coverage": scene_validation.relation_coverage,
            "grounded_copy_ratio": scene_validation.grounded_copy_ratio,
            "safety": safety.to_dict(),
        },
    )


def _kinetic_typography(
    program: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    elements = _elements(scene_program)
    hero_index = max(
        range(len(elements)),
        key=lambda index: float(elements[index].get("emphasis") or 0.5),
    )
    hero = elements[hero_index]
    supporting = [
        item for index, item in enumerate(elements) if index != hero_index
    ]
    fragments = "\n".join(
        (
            f'<div class="vw-type-fragment fragment-{index + 1}" '
            f'{_element_attrs(item, scene_program)} '
            f'{_anim_attrs(item, scene_program, index + 1, mode="slide-right")}>'
            f'<i>{index + 1:02d}</i>'
            f'<strong>{_escape(item.get("text"))}</strong>'
            "</div>"
        )
        for index, item in enumerate(supporting[:5])
    )
    relations = _relation_svg(scene_program, class_name="vw-type-relations")
    return f"""
      <style>
        .vw-kinetic {{ position:absolute; inset:0; overflow:hidden; }}
        .vw-kinetic .vw-type-kicker {{ position:absolute; left:6%; top:7%; color:var(--accent); font-size:18px; font-weight:900; text-transform:uppercase; }}
        .vw-kinetic .vw-type-hero {{ position:absolute; left:6%; right:7%; top:16%; z-index:4; display:grid; align-content:center; min-height:43%; }}
        .vw-kinetic .vw-type-hero em {{ color:var(--accent-2); font-size:17px; font-style:normal; font-weight:900; text-transform:uppercase; }}
        .vw-kinetic .vw-type-hero strong {{ max-width:94%; margin-top:14px; color:var(--text); font-family:"Arial Narrow","Roboto Condensed","Segoe UI",sans-serif; font-size:clamp(76px,9.4vw,156px); font-weight:950; line-height:.83; text-transform:uppercase; overflow-wrap:anywhere; }}
        .vw-kinetic .vw-type-hero::after {{ content:""; width:42%; height:14px; margin-top:26px; background:var(--accent); transform-origin:left; transform:scaleX(var(--route-progress,0)); }}
        .vw-kinetic .vw-type-fragments {{ position:absolute; left:7%; right:6%; bottom:7%; z-index:5; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px 28px; }}
        .vw-kinetic .vw-type-fragment {{ display:grid; grid-template-columns:42px 1fr; gap:12px; align-items:start; min-width:0; color:var(--text); }}
        .vw-kinetic .vw-type-fragment i {{ color:var(--accent); font-size:15px; font-style:normal; font-weight:950; }}
        .vw-kinetic .vw-type-fragment strong {{ font-size:clamp(18px,2.15vw,34px); font-weight:850; line-height:.98; overflow-wrap:anywhere; }}
        .vw-kinetic .vw-type-relations {{ position:absolute; inset:0; z-index:2; width:100%; height:100%; opacity:.5; }}
        .vw-kinetic .vw-relation {{ fill:none; stroke:var(--accent-2); stroke-width:.7; stroke-dasharray:1; stroke-dashoffset:calc(1 - var(--line-progress,0)); vector-effect:non-scaling-stroke; }}
      </style>
      <section id="{_safe_id(program.get("world_id"))}" class="visual-world-canvas vw-kinetic"
        data-medium-family="kinetic_typography"
        data-world-signature="{_escape(program.get("world_signature"))}">
        <div class="vw-type-kicker">{_escape(program.get("scene_type")).replace("_", " ")}</div>
        {relations}
        <div class="vw-type-hero" {_element_attrs(hero, scene_program)}
          {_anim_attrs(hero, scene_program, 0, mode="scale")}>
          <em>{_escape(hero.get("role")).replace("_", " ")}</em>
          <strong>{_escape(hero.get("text"))}</strong>
        </div>
        <div class="vw-type-fragments">{fragments}</div>
      </section>
    """


def _editorial_collage(
    program: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    pieces = "\n".join(
        (
            f'<article class="vw-collage-piece piece-{index + 1}" '
            f'{_element_attrs(item, scene_program)} '
            f'{_anim_attrs(item, scene_program, index, mode="rise")}>'
            f'<small>{_escape(item.get("role")).replace("_", " ")}</small>'
            f'<strong>{_escape(item.get("text"))}</strong>'
            "</article>"
        )
        for index, item in enumerate(_elements(scene_program)[:7])
    )
    return f"""
      <style>
        .vw-collage {{ position:absolute; inset:0; overflow:hidden; color:#111; }}
        .vw-collage .vw-collage-masthead {{ position:absolute; left:5.5%; top:6%; z-index:5; width:48%; color:var(--text); }}
        .vw-collage .vw-collage-masthead span {{ display:block; color:var(--accent); font-size:17px; font-weight:950; text-transform:uppercase; }}
        .vw-collage .vw-collage-masthead b {{ display:block; margin-top:10px; font-family:Georgia,"Times New Roman",serif; font-size:clamp(42px,5.3vw,84px); line-height:.9; overflow-wrap:anywhere; }}
        .vw-collage .vw-collage-rule {{ position:absolute; left:5.5%; right:5.5%; top:27%; z-index:2; height:3px; background:var(--text); transform-origin:left; transform:scaleX(var(--route-progress,0)); }}
        .vw-collage .vw-collage-pieces {{ position:absolute; inset:31% 5.5% 6% 5.5%; }}
        .vw-collage .vw-collage-piece {{ position:absolute; display:grid; align-content:center; padding:20px 24px; background:var(--panel); color:var(--text); box-shadow:10px 12px 0 color-mix(in srgb,var(--text) 16%,transparent); clip-path:polygon(2% 3%,98% 0,100% 94%,4% 100%,0 16%); }}
        .vw-collage .vw-collage-piece small {{ color:var(--accent); font-size:13px; font-weight:950; text-transform:uppercase; }}
        .vw-collage .vw-collage-piece strong {{ margin-top:8px; font-family:Georgia,"Times New Roman",serif; font-size:clamp(20px,2.4vw,38px); line-height:1; overflow-wrap:anywhere; }}
        .vw-collage .piece-1 {{ left:0; top:4%; width:42%; height:39%; rotate:-2deg; }}
        .vw-collage .piece-2 {{ right:1%; top:0; width:48%; height:31%; rotate:1.5deg; background:var(--accent); color:#fff; }}
        .vw-collage .piece-2 small {{ color:#fff; }}
        .vw-collage .piece-3 {{ left:7%; bottom:2%; width:34%; height:42%; rotate:1deg; background:var(--accent-2); color:#fff; }}
        .vw-collage .piece-3 small {{ color:#fff; }}
        .vw-collage .piece-4 {{ right:4%; bottom:5%; width:49%; height:48%; rotate:-1deg; }}
        .vw-collage .piece-5 {{ left:41%; top:37%; width:24%; height:25%; rotate:3deg; background:var(--text); color:var(--bg); }}
        .vw-collage .piece-5 small {{ color:var(--accent); }}
        .vw-collage .piece-6 {{ right:28%; bottom:0; width:22%; height:24%; rotate:-4deg; }}
        .vw-collage .piece-7 {{ left:0; top:47%; width:24%; height:27%; rotate:-1deg; }}
      </style>
      <section id="{_safe_id(program.get("world_id"))}" class="visual-world-canvas vw-collage"
        data-medium-family="editorial_collage"
        data-world-signature="{_escape(program.get("world_signature"))}">
        <header class="vw-collage-masthead" data-anim="rise" data-delay=".03" data-span=".42" data-y="22" data-scale=".98">
          <span>{_escape(program.get("scene_type")).replace("_", " ")}</span>
          <b>{_escape(_thesis(scene_program))}</b>
        </header>
        <div class="vw-collage-rule" data-line data-delay=".08"></div>
        <div class="vw-collage-pieces">{pieces}</div>
      </section>
    """


def _data_sculpture(
    program: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    masses = "\n".join(
        (
            f'<div class="vw-mass mass-{index + 1}" '
            f'{_element_attrs(item, scene_program)} '
            f'{_anim_attrs(item, scene_program, index, mode="pop")}>'
            f'<small>{_escape(item.get("role")).replace("_", " ")}</small>'
            f'<strong>{_escape(item.get("text"))}</strong>'
            "</div>"
        )
        for index, item in enumerate(_elements(scene_program)[:7])
    )
    particles = "".join(
        f'<i style="--i:{index};--x:{(index * 37) % 97}%;--y:{(index * 61) % 89}%"></i>'
        for index in range(28)
    )
    return f"""
      <style>
        .vw-data-sculpture {{ position:absolute; inset:0; overflow:hidden; }}
        .vw-data-sculpture .vw-data-title {{ position:absolute; left:5.5%; top:6%; z-index:7; max-width:62%; color:var(--text); }}
        .vw-data-sculpture .vw-data-title span {{ color:var(--accent); font-size:16px; font-weight:950; text-transform:uppercase; }}
        .vw-data-sculpture .vw-data-title b {{ display:block; margin-top:9px; font-size:clamp(42px,5.8vw,92px); line-height:.92; overflow-wrap:anywhere; }}
        .vw-data-sculpture .vw-particle-field {{ position:absolute; inset:0; opacity:.7; }}
        .vw-data-sculpture .vw-particle-field i {{ position:absolute; left:var(--x); top:var(--y); width:calc(3px + (var(--i) % 4) * 2px); aspect-ratio:1; border-radius:50%; background:var(--accent-2); box-shadow:0 0 18px var(--accent-2); transform:translate3d(calc((var(--p,0) - .5) * 42px),calc((.5 - var(--p,0)) * 30px),0); }}
        .vw-data-sculpture .vw-masses {{ position:absolute; inset:27% 5% 5% 5%; }}
        .vw-data-sculpture .vw-mass {{ position:absolute; z-index:4; display:grid; place-content:center; text-align:center; aspect-ratio:1; border-radius:50%; color:var(--text); background:radial-gradient(circle at 34% 28%,color-mix(in srgb,var(--accent-2) 52%,white),color-mix(in srgb,var(--accent) 42%,var(--bg)) 42%,color-mix(in srgb,var(--bg) 88%,black)); box-shadow:inset -24px -28px 60px color-mix(in srgb,black 34%,transparent),0 30px 80px color-mix(in srgb,var(--glow) 38%,transparent); }}
        .vw-data-sculpture .vw-mass::after {{ content:""; position:absolute; inset:-8%; border:2px solid color-mix(in srgb,var(--accent-2) 55%,transparent); border-radius:50%; rotate:calc(var(--p,0) * 120deg); }}
        .vw-data-sculpture .vw-mass small {{ font-size:12px; font-weight:950; text-transform:uppercase; opacity:.78; }}
        .vw-data-sculpture .vw-mass strong {{ max-width:80%; margin:8px auto 0; font-size:clamp(17px,2.2vw,34px); line-height:.94; overflow-wrap:anywhere; }}
        .vw-data-sculpture .mass-1 {{ left:2%; top:13%; width:27%; }}
        .vw-data-sculpture .mass-2 {{ left:34%; top:0; width:20%; }}
        .vw-data-sculpture .mass-3 {{ right:4%; top:9%; width:31%; }}
        .vw-data-sculpture .mass-4 {{ left:24%; bottom:0; width:24%; }}
        .vw-data-sculpture .mass-5 {{ right:26%; bottom:3%; width:18%; }}
        .vw-data-sculpture .mass-6 {{ left:0; bottom:1%; width:16%; }}
        .vw-data-sculpture .mass-7 {{ right:0; bottom:0; width:15%; }}
        .vw-data-sculpture .vw-data-relations {{ position:absolute; inset:27% 5% 5% 5%; z-index:2; width:90%; height:68%; overflow:visible; }}
        .vw-data-sculpture .vw-relation {{ fill:none; stroke:var(--accent); stroke-width:1.2; stroke-dasharray:1; stroke-dashoffset:calc(1 - var(--line-progress,0)); vector-effect:non-scaling-stroke; filter:drop-shadow(0 0 8px var(--accent)); }}
      </style>
      <section id="{_safe_id(program.get("world_id"))}" class="visual-world-canvas vw-data-sculpture"
        data-medium-family="data_sculpture"
        data-world-signature="{_escape(program.get("world_signature"))}">
        <header class="vw-data-title" data-anim="rise" data-delay=".03" data-span=".42" data-y="20" data-scale=".98">
          <span>{_escape(program.get("scene_type")).replace("_", " ")}</span>
          <b>{_escape(_thesis(scene_program))}</b>
        </header>
        <div class="vw-particle-field" aria-hidden="true">{particles}</div>
        {_relation_svg(scene_program, class_name="vw-data-relations")}
        <div class="vw-masses">{masses}</div>
      </section>
    """


def _spatial_metaphor(
    program: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    objects = "\n".join(
        (
            f'<article class="vw-spatial-object object-{index + 1}" '
            f'{_element_attrs(item, scene_program)} '
            f'{_anim_attrs(item, scene_program, index, mode="slide-right")}>'
            f'<small>{_escape(item.get("role")).replace("_", " ")}</small>'
            f'<strong>{_escape(item.get("text"))}</strong>'
            "</article>"
        )
        for index, item in enumerate(_elements(scene_program)[:7])
    )
    return f"""
      <style>
        .vw-spatial {{ position:absolute; inset:0; overflow:hidden; perspective:1100px; }}
        .vw-spatial .vw-spatial-title {{ position:absolute; left:5.5%; top:6%; z-index:8; max-width:58%; color:var(--text); }}
        .vw-spatial .vw-spatial-title span {{ color:var(--accent); font-size:16px; font-weight:950; text-transform:uppercase; }}
        .vw-spatial .vw-spatial-title b {{ display:block; margin-top:8px; font-size:clamp(40px,5.4vw,86px); line-height:.92; overflow-wrap:anywhere; }}
        .vw-spatial .vw-floor {{ position:absolute; left:-12%; right:-12%; top:45%; bottom:-38%; background:linear-gradient(180deg,color-mix(in srgb,var(--accent-2) 15%,transparent),transparent 72%),repeating-linear-gradient(90deg,color-mix(in srgb,var(--stroke) 28%,transparent) 0 2px,transparent 2px 120px); transform:rotateX(67deg); transform-origin:top center; border-top:3px solid color-mix(in srgb,var(--accent) 68%,transparent); }}
        .vw-spatial .vw-track {{ position:absolute; left:7%; right:7%; top:57%; z-index:2; height:16px; border-radius:50%; background:linear-gradient(90deg,var(--accent-2),var(--accent)); box-shadow:0 0 36px color-mix(in srgb,var(--accent) 56%,transparent); transform-origin:left; transform:scaleX(var(--route-progress,0)) skewY(-4deg); }}
        .vw-spatial .vw-spatial-objects {{ position:absolute; inset:28% 5% 7% 5%; z-index:5; }}
        .vw-spatial .vw-spatial-object {{ position:absolute; display:grid; align-content:center; min-width:0; padding:18px 22px; color:var(--text); background:linear-gradient(145deg,color-mix(in srgb,var(--panel) 82%,white 4%),color-mix(in srgb,var(--bg) 72%,black)); border:2px solid color-mix(in srgb,var(--stroke) 62%,transparent); border-radius:50% 50% 18% 18% / 28% 28% 16% 16%; box-shadow:inset 0 5px 0 color-mix(in srgb,white 18%,transparent),0 28px 50px color-mix(in srgb,black 30%,transparent); transform-style:preserve-3d; }}
        .vw-spatial .vw-spatial-object::before {{ content:""; position:absolute; left:12%; right:12%; top:-11px; height:20px; border-radius:50%; background:color-mix(in srgb,var(--accent) 68%,var(--panel)); border:2px solid var(--accent); }}
        .vw-spatial .vw-spatial-object small {{ color:var(--accent-2); font-size:12px; font-weight:950; text-transform:uppercase; }}
        .vw-spatial .vw-spatial-object strong {{ margin-top:8px; font-size:clamp(17px,2vw,30px); line-height:1; overflow-wrap:anywhere; }}
        .vw-spatial .object-1 {{ left:0; top:13%; width:24%; height:31%; }}
        .vw-spatial .object-2 {{ left:26%; top:38%; width:21%; height:28%; }}
        .vw-spatial .object-3 {{ left:48%; top:5%; width:22%; height:34%; }}
        .vw-spatial .object-4 {{ right:1%; top:32%; width:25%; height:33%; }}
        .vw-spatial .object-5 {{ left:14%; bottom:0; width:20%; height:25%; }}
        .vw-spatial .object-6 {{ left:55%; bottom:0; width:19%; height:24%; }}
        .vw-spatial .object-7 {{ right:0; bottom:0; width:17%; height:22%; }}
      </style>
      <section id="{_safe_id(program.get("world_id"))}" class="visual-world-canvas vw-spatial"
        data-medium-family="spatial_metaphor"
        data-world-signature="{_escape(program.get("world_signature"))}">
        <header class="vw-spatial-title" data-anim="rise" data-delay=".03" data-span=".42" data-y="20" data-scale=".98">
          <span>{_escape(program.get("scene_type")).replace("_", " ")}</span>
          <b>{_escape(_thesis(scene_program))}</b>
        </header>
        <div class="vw-floor" aria-hidden="true"></div>
        <div class="vw-track" data-line data-delay=".12" aria-hidden="true"></div>
        <div class="vw-spatial-objects">{objects}</div>
      </section>
    """


def _diagrammatic_system(
    program: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    nodes = "\n".join(
        (
            f'<article class="vw-system-node role-{_safe_id(item.get("role"))}" '
            f'{_position_style(item)} {_element_attrs(item, scene_program)} '
            f'{_anim_attrs(item, scene_program, index, mode="pop")}>'
            f'<i>{index + 1:02d}</i>'
            f'<small>{_escape(item.get("role")).replace("_", " ")}</small>'
            f'<strong>{_escape(item.get("text"))}</strong>'
            "</article>"
        )
        for index, item in enumerate(_elements(scene_program)[:8])
    )
    return f"""
      <style>
        .vw-system {{ position:absolute; inset:0; overflow:hidden; }}
        .vw-system .vw-system-title {{ position:absolute; left:5%; top:5%; z-index:6; max-width:48%; color:var(--text); }}
        .vw-system .vw-system-title span {{ color:var(--accent); font-size:15px; font-weight:950; text-transform:uppercase; }}
        .vw-system .vw-system-title b {{ display:block; margin-top:8px; font-size:clamp(38px,4.8vw,76px); line-height:.94; overflow-wrap:anywhere; }}
        .vw-system .vw-system-map {{ position:absolute; inset:23% 4% 5% 4%; }}
        .vw-system .vw-system-relations {{ position:absolute; inset:0; z-index:1; width:100%; height:100%; overflow:visible; }}
        .vw-system .vw-relation {{ fill:none; stroke:var(--accent-2); stroke-width:.7; stroke-dasharray:1; stroke-dashoffset:calc(1 - var(--line-progress,0)); vector-effect:non-scaling-stroke; }}
        .vw-system .vw-system-node {{ position:absolute; z-index:3; display:grid; grid-template-columns:36px 1fr; align-content:center; column-gap:10px; min-width:0; color:var(--text); }}
        .vw-system .vw-system-node::before {{ content:""; position:absolute; left:-10px; top:50%; width:18px; aspect-ratio:1; translate:0 -50%; rotate:45deg; background:var(--accent); box-shadow:0 0 24px color-mix(in srgb,var(--accent) 65%,transparent); }}
        .vw-system .vw-system-node i {{ grid-row:1 / span 2; align-self:center; color:var(--accent); font-size:14px; font-style:normal; font-weight:950; }}
        .vw-system .vw-system-node small {{ color:var(--accent-2); font-size:11px; font-weight:900; text-transform:uppercase; }}
        .vw-system .vw-system-node strong {{ margin-top:5px; font-size:clamp(17px,1.8vw,28px); line-height:1.02; overflow-wrap:anywhere; }}
        .vw-system .vw-system-node.role-result, .vw-system .vw-system-node.role-metric {{ color:var(--accent); }}
      </style>
      <section id="{_safe_id(program.get("world_id"))}" class="visual-world-canvas vw-system"
        data-medium-family="diagrammatic_system"
        data-world-signature="{_escape(program.get("world_signature"))}">
        <header class="vw-system-title" data-anim="rise" data-delay=".03" data-span=".42" data-y="20" data-scale=".98">
          <span>{_escape(program.get("scene_type")).replace("_", " ")}</span>
          <b>{_escape(_thesis(scene_program))}</b>
        </header>
        <div class="vw-system-map">
          {_relation_svg(scene_program, class_name="vw-system-relations")}
          {nodes}
        </div>
      </section>
    """


def _product_interface(
    program: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    rows = "\n".join(
        (
            f'<article class="vw-ui-row role-{_safe_id(item.get("role"))}" '
            f'{_element_attrs(item, scene_program)} '
            f'{_anim_attrs(item, scene_program, index, mode="slide-right")}>'
            '<i aria-hidden="true"></i>'
            f'<span>{_escape(item.get("role")).replace("_", " ")}</span>'
            f'<strong>{_escape(item.get("text"))}</strong>'
            "</article>"
        )
        for index, item in enumerate(_elements(scene_program)[:7])
    )
    return f"""
      <style>
        .vw-product {{ position:absolute; inset:0; overflow:hidden; color:var(--text); }}
        .vw-product .vw-ui-shell {{ position:absolute; inset:6% 5%; display:grid; grid-template-columns:25% 1fr; overflow:hidden; background:var(--panel); border:1px solid color-mix(in srgb,var(--stroke) 48%,transparent); box-shadow:0 40px 100px color-mix(in srgb,black 28%,transparent); }}
        .vw-product .vw-ui-sidebar {{ padding:34px 28px; background:color-mix(in srgb,var(--bg) 82%,var(--panel)); border-right:1px solid color-mix(in srgb,var(--stroke) 28%,transparent); }}
        .vw-product .vw-ui-brand {{ font-size:24px; font-weight:950; }}
        .vw-product .vw-ui-nav {{ display:grid; gap:13px; margin-top:38px; }}
        .vw-product .vw-ui-nav i {{ height:12px; background:color-mix(in srgb,var(--stroke) 24%,transparent); }}
        .vw-product .vw-ui-nav i:nth-child(2) {{ width:72%; background:var(--accent); }}
        .vw-product .vw-ui-main {{ position:relative; padding:34px 38px; overflow:hidden; }}
        .vw-product .vw-ui-main header span {{ color:var(--accent); font-size:14px; font-weight:950; text-transform:uppercase; }}
        .vw-product .vw-ui-main header b {{ display:block; max-width:78%; margin-top:8px; font-size:clamp(32px,4vw,64px); line-height:.96; overflow-wrap:anywhere; }}
        .vw-product .vw-ui-rows {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:30px; }}
        .vw-product .vw-ui-row {{ display:grid; grid-template-columns:12px 1fr; align-content:center; column-gap:14px; min-height:116px; padding:20px 22px; background:color-mix(in srgb,var(--bg) 38%,var(--panel)); border:1px solid color-mix(in srgb,var(--stroke) 30%,transparent); }}
        .vw-product .vw-ui-row i {{ grid-row:1 / span 2; width:8px; height:70%; align-self:center; background:var(--accent-2); }}
        .vw-product .vw-ui-row span {{ color:var(--accent-2); font-size:11px; font-weight:950; text-transform:uppercase; }}
        .vw-product .vw-ui-row strong {{ margin-top:6px; font-size:clamp(17px,1.8vw,27px); line-height:1.04; overflow-wrap:anywhere; }}
        .vw-product .vw-ui-row.role-result, .vw-product .vw-ui-row.role-metric {{ background:color-mix(in srgb,var(--accent) 13%,var(--panel)); }}
      </style>
      <section id="{_safe_id(program.get("world_id"))}" class="visual-world-canvas vw-product"
        data-medium-family="product_interface"
        data-world-signature="{_escape(program.get("world_signature"))}">
        <div class="vw-ui-shell" data-anim="scale" data-delay=".02" data-span=".48" data-y="8" data-scale=".97">
          <aside class="vw-ui-sidebar">
            <div class="vw-ui-brand">VEX / PROOF</div>
            <div class="vw-ui-nav" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
          </aside>
          <main class="vw-ui-main">
            <header>
              <span>{_escape(program.get("scene_type")).replace("_", " ")}</span>
              <b>{_escape(_thesis(scene_program))}</b>
            </header>
            <div class="vw-ui-rows">{rows}</div>
          </main>
        </div>
      </section>
    """


def _source_media_composite(
    program: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    annotations = "\n".join(
        (
            f'<article class="vw-source-note note-{index + 1}" '
            f'{_element_attrs(item, scene_program)} '
            f'{_anim_attrs(item, scene_program, index, mode="slide-left")}>'
            f'<span>{index + 1:02d}</span>'
            f'<strong>{_escape(item.get("text"))}</strong>'
            "</article>"
        )
        for index, item in enumerate(_elements(scene_program)[:6])
    )
    return f"""
      <style>
        .vw-source {{ position:absolute; inset:0; overflow:hidden; background:#050505; }}
        .vw-source .vw-source-frame {{ position:absolute; inset:0 31% 0 0; overflow:hidden; }}
        .vw-source .vw-source-image {{ width:100%; height:100%; display:block; object-fit:cover; }}
        .vw-source .vw-source-placeholder {{ position:absolute; inset:0; background:linear-gradient(135deg,var(--bg),var(--panel)); }}
        .vw-source .vw-source-frame::after {{ content:""; position:absolute; inset:0; background:linear-gradient(90deg,transparent 55%,color-mix(in srgb,var(--bg) 92%,transparent)); }}
        .vw-source .vw-source-panel {{ position:absolute; top:0; right:0; bottom:0; width:36%; padding:5% 4%; background:color-mix(in srgb,var(--bg) 94%,transparent); color:var(--text); }}
        .vw-source .vw-source-panel header span {{ color:var(--accent); font-size:14px; font-weight:950; text-transform:uppercase; }}
        .vw-source .vw-source-panel header b {{ display:block; margin-top:10px; font-size:clamp(30px,3.5vw,56px); line-height:.94; overflow-wrap:anywhere; }}
        .vw-source .vw-source-notes {{ display:grid; gap:14px; margin-top:28px; }}
        .vw-source .vw-source-note {{ display:grid; grid-template-columns:38px 1fr; gap:10px; align-items:start; padding:13px 0; border-top:1px solid color-mix(in srgb,var(--stroke) 38%,transparent); }}
        .vw-source .vw-source-note span {{ color:var(--accent); font-size:13px; font-weight:950; }}
        .vw-source .vw-source-note strong {{ font-size:clamp(16px,1.6vw,25px); line-height:1.03; overflow-wrap:anywhere; }}
      </style>
      <section id="{_safe_id(program.get("world_id"))}" class="visual-world-canvas vw-source"
        data-medium-family="source_media_composite"
        data-world-signature="{_escape(program.get("world_signature"))}">
        <div class="vw-source-frame">__VEX_SOURCE_ASSET__</div>
        <aside class="vw-source-panel">
          <header data-anim="rise" data-delay=".03" data-span=".42" data-y="20" data-scale=".98">
            <span>{_escape(program.get("scene_type")).replace("_", " ")}</span>
            <b>{_escape(_thesis(scene_program))}</b>
          </header>
          <div class="vw-source-notes">{annotations}</div>
        </aside>
      </section>
    """


def _relation_svg(
    scene_program: dict[str, Any],
    *,
    class_name: str,
) -> str:
    elements = {
        str(item.get("element_id") or ""): item
        for item in _elements(scene_program)
    }
    paths: list[str] = []
    for relation in scene_program.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        source = elements.get(str(relation.get("source_element_id") or ""))
        target = elements.get(str(relation.get("target_element_id") or ""))
        if not source or not target:
            continue
        x1 = _bounded(source.get("x"), 0.25) * 100
        y1 = _bounded(source.get("y"), 0.5) * 100
        x2 = _bounded(target.get("x"), 0.75) * 100
        y2 = _bounded(target.get("y"), 0.5) * 100
        bend = max(8.0, abs(x2 - x1) * 0.35)
        path = (
            f"M {x1:.2f} {y1:.2f} "
            f"C {x1 + bend:.2f} {y1:.2f}, "
            f"{x2 - bend:.2f} {y2:.2f}, {x2:.2f} {y2:.2f}"
        )
        paths.append(
            (
                '<path class="vw-relation" data-line '
                f'data-delay="{_bounded(relation.get("reveal_fraction"), 0.24):.3f}" '
                f'data-relation-id="{_escape(relation.get("relation_id"))}" '
                f'data-evidence-ids="{_escape(",".join(_strings(relation.get("evidence_ids"))))}" '
                f'd="{path}" pathLength="1" />'
            )
        )
    return (
        f'<svg class="{class_name}" viewBox="0 0 100 100" '
        'preserveAspectRatio="none" aria-hidden="true">'
        + "".join(paths)
        + "</svg>"
    )


def _relation_telemetry(scene_program: dict[str, Any]) -> str:
    entries = "".join(
        (
            '<span '
            f'data-relation-id="{_escape(item.get("relation_id"))}" '
            f'data-source-element-id="{_escape(item.get("source_element_id"))}" '
            f'data-target-element-id="{_escape(item.get("target_element_id"))}" '
            f'data-relation-type="{_escape(item.get("relation_type"))}" '
            f'data-evidence-ids="{_escape(",".join(_strings(item.get("evidence_ids"))))}">'
            "</span>"
        )
        for item in scene_program.get("relations") or []
        if isinstance(item, dict)
    )
    return (
        '<div class="vw-relation-telemetry" aria-hidden="true" '
        'style="position:absolute;width:0;height:0;overflow:hidden;">'
        f"{entries}</div>"
    )


def _element_attrs(
    item: dict[str, Any],
    scene_program: dict[str, Any],
) -> str:
    return " ".join(
        [
            f'data-element-id="{_escape(item.get("element_id"))}"',
            f'data-object-id="{_escape(item.get("object_id"))}"',
            f'data-role="{_escape(item.get("role"))}"',
            f'data-evidence-ids="{_escape(",".join(_strings(item.get("evidence_ids"))))}"',
            f'data-fact-ids="{_escape(",".join(_strings(item.get("fact_ids"))))}"',
            f'data-scene-program-signature="{_escape(scene_program.get("program_signature"))}"',
        ]
    )


def _anim_attrs(
    item: dict[str, Any],
    scene_program: dict[str, Any],
    index: int,
    *,
    mode: str,
) -> str:
    motion = next(
        (
            entry
            for entry in scene_program.get("motions") or []
            if isinstance(entry, dict)
            and entry.get("target_type") == "element"
            and entry.get("target_id") == item.get("element_id")
        ),
        {},
    )
    start = _bounded(
        motion.get("start_fraction"),
        0.08 + index * 0.07,
    )
    end = _bounded(motion.get("end_fraction"), min(start + 0.38, 0.92))
    span = max(0.12, end - start)
    return (
        f'data-anim="{mode}" data-delay="{start:.3f}" '
        f'data-span="{span:.3f}" data-y="24" data-scale=".94"'
    )


def _position_style(item: dict[str, Any]) -> str:
    x = _bounded(item.get("x"), 0.5)
    y = _bounded(item.get("y"), 0.5)
    width = min(max(_bounded(item.get("width"), 0.22), 0.16), 0.42)
    height = min(max(_bounded(item.get("height"), 0.16), 0.1), 0.3)
    return (
        f'style="left:{(x - width / 2) * 100:.3f}%;'
        f'top:{(y - height / 2) * 100:.3f}%;'
        f'width:{width * 100:.3f}%;height:{height * 100:.3f}%;"'
    )


def _elements(scene_program: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in scene_program.get("elements") or []
        if isinstance(item, dict)
    ]


def _thesis(scene_program: dict[str, Any]) -> str:
    elements = _elements(scene_program)
    if not elements:
        return "Grounded visual proof"
    strongest = max(
        elements,
        key=lambda item: float(item.get("emphasis") or 0.5),
    )
    return str(strongest.get("text") or "Grounded visual proof")


def _escape(value: Any) -> str:
    return html.escape(re.sub(r"\s+", " ", str(value or "")).strip(), quote=True)


def _safe_id(value: Any) -> str:
    return (
        re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "visual-world"))
        .strip("-_")
        .lower()
        or "visual-world"
    )


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, min(number, 1.0))


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value or [] if str(item).strip()]


_COMPILERS: dict[
    str,
    Callable[[dict[str, Any], dict[str, Any]], str],
] = {
    "data_sculpture": _data_sculpture,
    "diagrammatic_system": _diagrammatic_system,
    "editorial_collage": _editorial_collage,
    "kinetic_typography": _kinetic_typography,
    "product_interface": _product_interface,
    "source_media_composite": _source_media_composite,
    "spatial_metaphor": _spatial_metaphor,
}


__all__ = [
    "CompiledVisualWorldStage",
    "compile_visual_world_stage",
]
