from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError, safe_render_job_dir


def _theme_defaults(spec: dict[str, Any]) -> dict[str, str]:
    theme = dict(spec.get("theme") or {})
    defaults = {
        "background": "#060916",
        "panel_fill": "#11203B",
        "panel_stroke": "#5CC8FF",
        "accent": "#F59E0B",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
    }
    defaults.update({key: str(value) for key, value in theme.items() if value})
    return defaults


def _safe_scene_name(spec_id: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in spec_id).strip("_")
    return cleaned or "auto_visual"


def _scene_script(scene_name: str, spec: dict[str, Any], output_path: str, width: int, height: int, fps: float) -> str:
    payload = dict(spec)
    payload["theme"] = _theme_defaults(spec)
    payload["render_width"] = width
    payload["render_height"] = height
    payload["render_fps"] = fps
    payload["output_path"] = output_path
    spec_json = json.dumps(payload, ensure_ascii=True)
    return f"""from __future__ import annotations

import json
import math
import os

import bpy

SPEC = json.loads(r'''{spec_json}''')


def hex_to_rgb(value: str) -> tuple[float, float, float, float]:
    cleaned = str(value or "#FFFFFF").strip().lstrip("#")
    if len(cleaned) != 6:
        cleaned = "FFFFFF"
    red = int(cleaned[0:2], 16) / 255.0
    green = int(cleaned[2:4], 16) / 255.0
    blue = int(cleaned[4:6], 16) / 255.0
    return (red, green, blue, 1.0)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)


def make_material(name: str, base_color: str, emission_strength: float = 0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = hex_to_rgb(base_color)
    principled.inputs["Roughness"].default_value = 0.28
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.45
    if emission_strength > 0:
        principled.inputs["Emission Color"].default_value = hex_to_rgb(base_color)
        principled.inputs["Emission Strength"].default_value = emission_strength
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return mat


def add_text(name: str, body: str, *, size: float, location: tuple[float, float, float], extrude: float = 0.0, bevel: float = 0.0, color: str = "#FFFFFF", emission: float = 0.0):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = str(body or " ").strip() or " "
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = extrude
    obj.data.bevel_depth = bevel
    mat = make_material(f"mat_{{name}}", color, emission)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def add_panel(name: str, *, scale: tuple[float, float, float], location: tuple[float, float, float], fill: str, stroke: str):
    bpy.ops.mesh.primitive_plane_add(location=location)
    panel = bpy.context.object
    panel.name = name
    panel.scale = scale
    panel_mat = make_material(f"mat_{{name}}", fill, 0.0)
    if panel.data.materials:
        panel.data.materials[0] = panel_mat
    else:
        panel.data.materials.append(panel_mat)
    bpy.ops.mesh.primitive_plane_add(location=(location[0], location[1] - 0.01, location[2]))
    stroke_panel = bpy.context.object
    stroke_panel.name = f"{{name}}_stroke"
    stroke_panel.scale = (scale[0] * 1.04, scale[1] * 1.08, scale[2])
    stroke_mat = make_material(f"mat_{{name}}_stroke", stroke, 0.7)
    if stroke_panel.data.materials:
        stroke_panel.data.materials[0] = stroke_mat
    else:
        stroke_panel.data.materials.append(stroke_mat)
    return panel, stroke_panel


def add_ring(radius: float, *, location: tuple[float, float, float], color: str):
    bpy.ops.mesh.primitive_torus_add(location=location, major_radius=radius, minor_radius=max(radius * 0.05, 0.04))
    ring = bpy.context.object
    ring_mat = make_material("mat_ring", color, 1.4)
    if ring.data.materials:
        ring.data.materials[0] = ring_mat
    else:
        ring.data.materials.append(ring_mat)
    return ring


def insert_location_keys(obj, start_frame: int, end_frame: int, start_loc, end_loc) -> None:
    obj.location = start_loc
    obj.keyframe_insert(data_path="location", frame=start_frame)
    obj.location = end_loc
    obj.keyframe_insert(data_path="location", frame=end_frame)


def insert_rotation_keys(obj, start_frame: int, end_frame: int, start_rot, end_rot) -> None:
    obj.rotation_euler = start_rot
    obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)
    obj.rotation_euler = end_rot
    obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)


def configure_scene() -> tuple[int, int]:
    scene = bpy.context.scene
    scene.render.resolution_x = int(SPEC.get("render_width") or 1920)
    scene.render.resolution_y = int(SPEC.get("render_height") or 1080)
    scene.render.fps = max(int(round(float(SPEC.get("render_fps") or 30.0))), 15)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(SPEC.get("output_path") or "")
    duration = max(float(SPEC.get("duration") or 2.0), 1.0)
    frame_count = max(int(round(duration * scene.render.fps)), 24)
    scene.frame_start = 1
    scene.frame_end = frame_count
    engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items else "BLENDER_EEVEE"
    scene.render.engine = engine
    if engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = 32
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs[0].default_value = hex_to_rgb(SPEC.get("theme", {{}}).get("background", "#060916"))
    background.inputs[1].default_value = 0.95
    return scene.frame_start, scene.frame_end


reset_scene()
start_frame, end_frame = configure_scene()
theme = SPEC.get("theme", {{}})
duration = max(float(SPEC.get("duration") or 2.0), 1.0)
template = str(SPEC.get("template") or "quote_focus")

bpy.ops.mesh.primitive_plane_add(location=(0.0, 2.8, 0.0))
backdrop = bpy.context.object
backdrop.scale = (10.0, 5.8, 1.0)
if backdrop.data.materials:
    backdrop.data.materials[0] = make_material("mat_backdrop", theme.get("panel_fill", "#11203B"), 0.0)
else:
    backdrop.data.materials.append(make_material("mat_backdrop", theme.get("panel_fill", "#11203B"), 0.0))

bpy.ops.object.camera_add(location=(0.0, -8.2, 0.4), rotation=(math.radians(90), 0.0, 0.0))
camera = bpy.context.object
bpy.context.scene.camera = camera
insert_location_keys(camera, start_frame, end_frame, (0.0, -8.4, 0.55), (0.0, -7.7, 0.2))

bpy.ops.object.light_add(type="AREA", location=(0.0, -3.6, 5.2))
key_light = bpy.context.object
key_light.data.energy = 2800
key_light.data.color = hex_to_rgb(theme.get("accent", "#F59E0B"))[:3]
key_light.scale = (5.0, 5.0, 5.0)

bpy.ops.object.light_add(type="POINT", location=(4.2, -5.5, 2.0))
fill_light = bpy.context.object
fill_light.data.energy = 900
fill_light.data.color = hex_to_rgb(theme.get("panel_stroke", "#5CC8FF"))[:3]

headline = add_text(
    "headline",
    SPEC.get("headline", ""),
    size=0.42,
    location=(0.0, 0.0, 2.12),
    extrude=0.0,
    bevel=0.0,
    color=theme.get("text_secondary", "#CBD5E1"),
    emission=0.3,
)
insert_location_keys(headline, start_frame, max(start_frame + 10, int(start_frame + (end_frame - start_frame) * 0.3)), (0.0, 0.0, 2.4), (0.0, 0.0, 2.12))

ring = add_ring(2.15, location=(0.0, 1.3, 0.2), color=theme.get("panel_stroke", "#5CC8FF"))
insert_rotation_keys(ring, start_frame, end_frame, (math.radians(90), 0.0, 0.0), (math.radians(90), 0.0, math.radians(150)))

if template == "keyword_stack":
    keywords = [str(item).strip() for item in (SPEC.get("keywords") or []) if str(item).strip()][:3]
    if not keywords:
        keywords = [str(SPEC.get("emphasis_text") or "Key idea")]
    objects = []
    base_z = 0.8
    for index, keyword in enumerate(keywords, start=1):
        panel, stroke_panel = add_panel(
            f"keyword_panel_{{index}}",
            scale=(2.6, 0.52, 1.0),
            location=(0.0, 1.1, base_z - (index - 1) * 0.95),
            fill=theme.get("panel_fill", "#11203B"),
            stroke=theme.get("panel_stroke", "#5CC8FF"),
        )
        label = add_text(
            f"keyword_{{index}}",
            keyword,
            size=0.46,
            location=(0.0, 0.86, base_z - (index - 1) * 0.95),
            extrude=0.03,
            bevel=0.01,
            color=theme.get("text_primary", "#F8FAFC"),
            emission=0.65,
        )
        objects.extend([panel, stroke_panel, label])
        insert_location_keys(label, start_frame, min(end_frame, start_frame + 12 + index * 4), (0.0, 1.5, label.location[2] + 0.18), tuple(label.location))
    footer = add_text(
        "footer",
        SPEC.get("footer_text", ""),
        size=0.28,
        location=(0.0, 0.0, -2.0),
        color=theme.get("text_secondary", "#CBD5E1"),
        emission=0.15,
    )
    insert_location_keys(footer, start_frame, min(end_frame, start_frame + 18), (0.0, 0.0, -2.25), (0.0, 0.0, -2.0))
else:
    main_text = SPEC.get("emphasis_text") or SPEC.get("quote_text") or SPEC.get("headline") or "Key point"
    main = add_text(
        "main",
        main_text,
        size=0.92 if template == "metric_callout" else 0.62,
        location=(0.0, 0.7, 0.3),
        extrude=0.06 if template == "metric_callout" else 0.03,
        bevel=0.015,
        color=theme.get("text_primary", "#F8FAFC"),
        emission=0.95 if template == "metric_callout" else 0.75,
    )
    insert_location_keys(main, start_frame, min(end_frame, start_frame + 14), (0.0, 1.3, 0.5), (0.0, 0.7, 0.3))
    insert_rotation_keys(main, start_frame, end_frame, (0.0, 0.0, math.radians(-4)), (0.0, 0.0, math.radians(4)))

    footer_text = SPEC.get("footer_text", "")
    if template == "metric_callout":
        support = "\\n".join(SPEC.get("supporting_lines") or [])
        footer_text = support or footer_text
    footer = add_text(
        "footer",
        footer_text,
        size=0.28,
        location=(0.0, 0.4, -1.55),
        color=theme.get("text_secondary", "#CBD5E1"),
        emission=0.12,
    )
    insert_location_keys(footer, start_frame, min(end_frame, start_frame + 18), (0.0, 0.8, -1.8), (0.0, 0.4, -1.55))

bpy.ops.render.render(animation=True)
"""


class BlenderRenderer(VisualRenderer):
    name = "blender"
    supported_templates = {"quote_focus", "keyword_stack", "metric_callout"}

    def availability(self) -> RendererStatus:
        blender_path = getattr(config, "BLENDER_PATH", "blender")
        if shutil.which(blender_path) is None:
            return RendererStatus(False, f"Blender executable was not found: {blender_path}")
        return RendererStatus(True, "")

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        template = str(spec.get("template") or "")
        visual_hint = str(spec.get("visual_type_hint") or "")
        composition = str(spec.get("composition_mode") or "")
        score = 0.7
        if composition == "replace":
            score += 0.08
        if template == "quote_focus":
            score += 0.12
        if visual_hint == "abstract_motion":
            score += 0.14
        return round(score, 3)

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
        if not self.supports(spec):
            raise VisualRendererError(f"Blender renderer does not support template {spec.get('template')!r}.")

        spec_id = str(spec.get("visual_id") or spec.get("id") or "visual")
        scene_name = _safe_scene_name(spec_id)
        job_dir = safe_render_job_dir(render_root, spec_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / f"{scene_name}.mp4"
        script_path = job_dir / "scene.py"
        script_path.write_text(
            _scene_script(scene_name, spec, str(output_path), width, height, fps),
            encoding="utf-8",
        )
        blender_path = getattr(config, "BLENDER_PATH", "blender")
        command = [
            blender_path,
            "-b",
            "-P",
            str(script_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.is_file():
            stderr = (result.stderr or result.stdout or "").strip()
            raise VisualRendererError(f"Blender renderer failed for {spec_id}: {stderr}")
        metadata = probe_video(str(output_path))
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(metadata.get("width") or width),
            height=int(metadata.get("height") or height),
            duration_sec=float(metadata.get("duration_sec") or float(spec.get("duration") or 0.0)),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(script_path),
        )
