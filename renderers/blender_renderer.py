from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config
from engine import probe_video
from renderers.base import RenderedAsset, RendererStatus, VisualRenderer, VisualRendererError, safe_render_job_dir
from renderers.blender_spec import BlenderVisualSpec, SUPPORTED_BLENDER_TEMPLATES, THREE_D_BLENDER_TEMPLATES


BLENDER_SCRIPT = r'''
from __future__ import annotations

import json
import base64
import math
import random
from pathlib import Path

import bpy

SPEC = json.loads(base64.b64decode("__SPEC_JSON_B64__").decode("utf-8"))
random.seed(1337)


def hex_to_rgba(value, alpha=1.0):
    cleaned = str(value or "#FFFFFF").strip().lstrip("#")
    if len(cleaned) != 6:
        cleaned = "FFFFFF"
    return (
        int(cleaned[0:2], 16) / 255.0,
        int(cleaned[2:4], 16) / 255.0,
        int(cleaned[4:6], 16) / 255.0,
        alpha,
    )


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for block in list(collection):
            if getattr(block, "users", 0) == 0:
                collection.remove(block)


def make_material(name, color, *, emission=0.0, roughness=0.34, alpha=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    rgba = hex_to_rgba(color, alpha)
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.08
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.55
    if emission > 0:
        bsdf.inputs["Emission Color"].default_value = hex_to_rgba(color)
        bsdf.inputs["Emission Strength"].default_value = emission
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def assign_material(obj, mat):
    if hasattr(obj.data, "materials"):
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    return obj


def text_obj(name, body, *, size, loc, color=None, extrude=0.0, bevel=0.0, emission=0.0):
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(90), 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = str(body or " ").strip() or " "
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = extrude
    obj.data.bevel_depth = bevel
    assign_material(obj, make_material(f"mat_{name}", color or SPEC["text_color"], emission=emission))
    return obj


def plane_obj(name, *, loc, scale, color, alpha=1.0, emission=0.0):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (scale[0], 0.035, scale[1])
    assign_material(obj, make_material(f"mat_{name}", color, alpha=alpha, emission=emission))
    return obj


def add_torus(name, *, loc, radius, color, emission=0.4):
    bpy.ops.mesh.primitive_torus_add(location=loc, major_radius=radius, minor_radius=max(radius * 0.025, 0.018))
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler[0] = math.radians(90)
    assign_material(obj, make_material(f"mat_{name}", color, emission=emission))
    return obj


def key_loc(obj, start_loc, end_loc):
    obj.location = start_loc
    obj.keyframe_insert(data_path="location", frame=START_FRAME)
    obj.location = end_loc
    obj.keyframe_insert(data_path="location", frame=END_FRAME)


def key_rot(obj, start_rot, end_rot):
    obj.rotation_euler = start_rot
    obj.keyframe_insert(data_path="rotation_euler", frame=START_FRAME)
    obj.rotation_euler = end_rot
    obj.keyframe_insert(data_path="rotation_euler", frame=END_FRAME)


def key_scale(obj, start_scale, end_scale):
    obj.scale = start_scale
    obj.keyframe_insert(data_path="scale", frame=START_FRAME)
    obj.scale = end_scale
    obj.keyframe_insert(data_path="scale", frame=max(START_FRAME + 8, int(END_FRAME * 0.55)))
    obj.scale = start_scale
    obj.keyframe_insert(data_path="scale", frame=END_FRAME)


def position_xy(position):
    safe_x = 2.2 if SPEC.get("safe_area", True) else 2.7
    safe_z = 1.35 if SPEC.get("safe_area", True) else 1.7
    mapping = {
        "center": (0.0, 0.0),
        "center_left": (-safe_x, 0.0),
        "center_right": (safe_x, 0.0),
        "top_left": (-safe_x, safe_z),
        "top_right": (safe_x, safe_z),
        "bottom_left": (-safe_x, -safe_z),
        "bottom_right": (safe_x, -safe_z),
    }
    return mapping.get(str(position), (0.0, 0.0))


def configure_scene():
    scene = bpy.context.scene
    scene.render.resolution_x = int(SPEC["width"])
    scene.render.resolution_y = int(SPEC["height"])
    scene.render.fps = max(12, int(round(float(SPEC["fps"]))))
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = max(12, int(round(float(SPEC["duration"]) * scene.render.fps)))
    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 32
    scene.render.film_transparent = bool(SPEC.get("transparent_background"))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if SPEC.get("alpha") else "RGB"
    scene.render.filepath = str(Path(SPEC["frame_dir"]) / "frame_")
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = hex_to_rgba(SPEC["background_color"], 1.0)
    bg.inputs[1].default_value = 0.0 if SPEC.get("transparent_background") else 0.75
    return scene.frame_start, scene.frame_end


def setup_camera():
    bpy.ops.object.camera_add(location=(0.0, -7.6, 1.15), rotation=(math.radians(82), 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    if SPEC["camera_motion"] == "slow_push":
        key_loc(camera, (0.0, -8.2, 1.25), (0.0, -6.8, 0.92))
    elif SPEC["camera_motion"] == "orbit":
        key_loc(camera, (-1.2, -8.0, 1.28), (1.2, -7.7, 1.0))
        key_rot(camera, (math.radians(82), 0, math.radians(-4)), (math.radians(82), 0, math.radians(4)))
    elif SPEC["camera_motion"] == "handheld_subtle":
        key_loc(camera, (-0.08, -7.8, 1.16), (0.08, -7.65, 1.08))
    return camera


def setup_lights():
    bpy.ops.object.light_add(type="AREA", location=(0, -4.3, 5.0))
    key = bpy.context.object
    key.name = "Key_Light"
    key.data.energy = 2900
    key.data.color = hex_to_rgba(SPEC["accent_color"])[:3]
    key.scale = (4.8, 4.8, 4.8)
    bpy.ops.object.light_add(type="POINT", location=(3.7, -4.8, 2.3))
    fill = bpy.context.object
    fill.name = "Fill_Light"
    fill.data.energy = 650
    fill.data.color = hex_to_rgba(SPEC["text_color"])[:3]


def backdrop():
    if SPEC.get("transparent_background"):
        return None
    obj = plane_obj("Backdrop", loc=(0, 1.2, 0), scale=(5.4, 3.1), color=SPEC["background_color"], alpha=1.0)
    obj.location.y = 1.9
    return obj


def apply_object_motion(objects):
    motion = SPEC["object_motion"]
    for obj in objects:
        if motion == "spin_y":
            key_rot(obj, tuple(obj.rotation_euler), (obj.rotation_euler[0], obj.rotation_euler[1] + math.radians(360), obj.rotation_euler[2]))
        elif motion == "float":
            start = tuple(obj.location)
            key_loc(obj, (start[0], start[1], start[2] - 0.08), (start[0], start[1], start[2] + 0.08))
        elif motion == "drop_in":
            end = tuple(obj.location)
            key_loc(obj, (end[0], end[1], end[2] + 1.1), end)
        elif motion == "pulse":
            key_scale(obj, tuple(obj.scale), (obj.scale[0] * 1.08, obj.scale[1] * 1.08, obj.scale[2] * 1.08))


def import_model_or_placeholder():
    asset_path = SPEC.get("asset_path")
    imported = []
    if asset_path:
        suffix = Path(asset_path).suffix.lower()
        try:
            if suffix in {".glb", ".gltf"}:
                bpy.ops.import_scene.gltf(filepath=asset_path)
                imported = list(bpy.context.selected_objects)
            elif suffix == ".obj":
                if hasattr(bpy.ops.wm, "obj_import"):
                    bpy.ops.wm.obj_import(filepath=asset_path)
                else:
                    bpy.ops.import_scene.obj(filepath=asset_path)
                imported = list(bpy.context.selected_objects)
            elif suffix == ".blend":
                with bpy.data.libraries.load(asset_path, link=False) as (data_from, data_to):
                    data_to.objects = list(data_from.objects[:8])
                for obj in data_to.objects:
                    if obj:
                        bpy.context.collection.objects.link(obj)
                        imported.append(obj)
        except Exception:
            imported = []
    if imported:
        for obj in imported:
            obj.location = (0, 0.35, 0)
            obj.scale = (0.9, 0.9, 0.9)
        return imported
    bpy.ops.mesh.primitive_cube_add(location=(0, 0.35, 0.0))
    cube = bpy.context.object
    cube.name = "Placeholder_Product"
    cube.scale = (0.78, 0.78, 0.78)
    assign_material(cube, make_material("mat_placeholder", SPEC["accent_color"], emission=0.1))
    bevel = cube.modifiers.new("Soft bevel", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 3
    cube.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return [cube]


def arrow_object(name, loc, rot_z=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.045, depth=1.2, location=loc, rotation=(0, math.radians(90), rot_z))
    shaft = bpy.context.object
    shaft.name = f"{name}_shaft"
    assign_material(shaft, make_material(f"mat_{name}", SPEC["accent_color"], emission=0.9))
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.16, depth=0.38, location=(loc[0] + math.cos(rot_z) * 0.72, loc[1], loc[2] + math.sin(rot_z) * 0.72), rotation=(0, math.radians(90), rot_z))
    head = bpy.context.object
    head.name = f"{name}_head"
    assign_material(head, make_material(f"mat_{name}_head", SPEC["accent_color"], emission=1.2))
    return [shaft, head]


def template_three_d_title():
    main = text_obj("Three_D_Title", SPEC["headline"], size=0.72, loc=(0, 0.0, 0.38), extrude=0.12, bevel=0.018, emission=0.45)
    sub = text_obj("Three_D_Subtitle", SPEC["subtext"], size=0.24, loc=(0, -0.03, -0.62), color=SPEC["text_color"], emission=0.18) if SPEC["subtext"] else None
    ring = add_torus("Title_Ring", loc=(0, 0.62, 0.08), radius=1.85, color=SPEC["accent_color"], emission=1.0)
    key_loc(main, (0, 0.35, 0.62), (0, 0.0, 0.38))
    key_rot(ring, (math.radians(90), 0, 0), (math.radians(90), 0, math.radians(135)))
    apply_object_motion([obj for obj in [main, sub] if obj])


def template_floating_3d_label():
    x, z = position_xy(SPEC["position"])
    panel = plane_obj("Floating_Label_Card", loc=(x, 0.15, z), scale=(1.45, 0.42), color=SPEC["background_color"], alpha=0.64 if SPEC.get("alpha") else 0.88, emission=0.05)
    label = text_obj("Floating_Label", SPEC["label"], size=0.22, loc=(x, -0.02, z + 0.03), color=SPEC["text_color"], extrude=0.015, bevel=0.004, emission=0.45)
    accent = add_torus("Floating_Label_Accent", loc=(x - 1.3, 0.0, z), radius=0.16, color=SPEC["accent_color"], emission=1.2)
    apply_object_motion([panel, label, accent])


def template_object_orbit():
    objects = import_model_or_placeholder()
    add_torus("Orbit_Ring_A", loc=(0, 0.35, 0), radius=1.32, color=SPEC["accent_color"], emission=0.7)
    ring = add_torus("Orbit_Ring_B", loc=(0, 0.35, 0), radius=1.72, color=SPEC["text_color"], emission=0.25)
    ring.rotation_euler[1] = math.radians(62)
    key_rot(ring, tuple(ring.rotation_euler), (ring.rotation_euler[0], ring.rotation_euler[1], ring.rotation_euler[2] + math.radians(220)))
    apply_object_motion(objects)
    if SPEC["headline"]:
        text_obj("Object_Label", SPEC["headline"], size=0.22, loc=(0, -0.18, -1.35), emission=0.2)


def template_logo_reveal():
    logo = text_obj("Logo_Reveal", SPEC["text"], size=0.78, loc=(0, 0.0, 0.18), extrude=0.13, bevel=0.02, color=SPEC["text_color"], emission=0.7)
    plane_obj("Logo_Base", loc=(0, 0.35, -0.74), scale=(2.1, 0.08), color=SPEC["accent_color"], alpha=0.9, emission=0.55)
    key_loc(logo, (0, 0.0, 1.2), (0, 0.0, 0.18))
    key_rot(logo, (0, 0, math.radians(-8)), (0, 0, 0))


def template_screen_pointer_3d():
    x, z = position_xy(SPEC["position"])
    target_x = x * 0.35
    rot_z = math.atan2(z - target_x * 0.15, x - target_x) if abs(x) > 0.1 else math.radians(-28)
    arrow_parts = arrow_object("Screen_Pointer", (target_x, -0.08, z * 0.42), rot_z=rot_z)
    label = text_obj("Pointer_Label", SPEC["label"], size=0.18, loc=(x, -0.04, z), color=SPEC["text_color"], extrude=0.01, bevel=0.003, emission=0.55)
    apply_object_motion([*arrow_parts, label])


def template_data_tunnel():
    for index in range(14):
        z = -2.1 + index * 0.32
        ring = add_torus(f"Data_Ring_{index:02d}", loc=(0, 0.8 + index * 0.07, z), radius=0.55 + index * 0.075, color=SPEC["accent_color"] if index % 2 else SPEC["text_color"], emission=0.45)
        key_rot(ring, tuple(ring.rotation_euler), (ring.rotation_euler[0], ring.rotation_euler[1], ring.rotation_euler[2] + math.radians(90 + index * 8)))
        if index % 3 == 0:
            bpy.ops.mesh.primitive_cube_add(location=(math.sin(index) * 1.2, 0.25, z))
            cube = bpy.context.object
            cube.name = f"Data_Node_{index:02d}"
            cube.scale = (0.055, 0.055, 0.055)
            assign_material(cube, make_material(f"mat_node_{index}", SPEC["accent_color"], emission=1.3))
    text_obj("Data_Tunnel_Label", SPEC["headline"], size=0.34, loc=(0, -0.05, -1.75), color=SPEC["text_color"], extrude=0.02, bevel=0.004, emission=0.5)


def template_product_model_spin():
    objects = import_model_or_placeholder()
    plane_obj("Product_Stage", loc=(0, 0.55, -0.88), scale=(1.6, 0.08), color=SPEC["accent_color"], alpha=0.72, emission=0.22)
    if SPEC["headline"]:
        text_obj("Product_Label", SPEC["headline"], size=0.22, loc=(0, -0.06, -1.45), color=SPEC["text_color"], emission=0.25)
    apply_object_motion(objects)


def template_quote_focus():
    quote = text_obj("Quote_Focus", SPEC["text"], size=0.42, loc=(0, 0.0, 0.2), extrude=0.025, bevel=0.006, color=SPEC["text_color"], emission=0.45)
    add_torus("Quote_Ring", loc=(0, 0.55, 0.1), radius=1.55, color=SPEC["accent_color"], emission=0.8)
    key_loc(quote, (0, 0.2, 0.38), (0, 0.0, 0.2))


def template_keyword_stack():
    values = SPEC.get("keywords") or SPEC.get("supporting_lines") or [SPEC["headline"]]
    for index, value in enumerate(values[:4]):
        z = 0.8 - index * 0.52
        panel = plane_obj(f"Keyword_Panel_{index}", loc=(0, 0.18, z), scale=(1.55, 0.18), color=SPEC["background_color"], alpha=0.82 if not SPEC.get("alpha") else 0.58, emission=0.06)
        label = text_obj(f"Keyword_{index}", value, size=0.18, loc=(0, -0.02, z + 0.015), color=SPEC["text_color"], extrude=0.012, bevel=0.002, emission=0.35)
        key_loc(panel, (-0.35, 0.18, z), (0, 0.18, z))
        key_loc(label, (-0.28, -0.02, z + 0.015), (0, -0.02, z + 0.015))


def template_metric_callout():
    metric = text_obj("Metric_Value", SPEC["text"], size=0.92, loc=(0, 0.0, 0.36), extrude=0.1, bevel=0.018, color=SPEC["accent_color"], emission=0.8)
    text_obj("Metric_Headline", SPEC["headline"], size=0.26, loc=(0, -0.04, -0.58), color=SPEC["text_color"], emission=0.28)
    if SPEC["subtext"]:
        text_obj("Metric_Subtext", SPEC["subtext"], size=0.18, loc=(0, -0.04, -1.0), color=SPEC["text_color"], emission=0.16)
    apply_object_motion([metric])


reset_scene()
START_FRAME, END_FRAME = configure_scene()
setup_camera()
setup_lights()
backdrop()

TEMPLATES = {
    "three_d_title": template_three_d_title,
    "floating_3d_label": template_floating_3d_label,
    "object_orbit": template_object_orbit,
    "logo_reveal": template_logo_reveal,
    "screen_pointer_3d": template_screen_pointer_3d,
    "data_tunnel": template_data_tunnel,
    "product_model_spin": template_product_model_spin,
    "quote_focus": template_quote_focus,
    "keyword_stack": template_keyword_stack,
    "metric_callout": template_metric_callout,
}
TEMPLATES.get(SPEC["template"], template_quote_focus)()
bpy.ops.render.render(animation=True)
'''


def _safe_scene_name(spec_id: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in spec_id).strip("_") or "auto_visual"


def _scene_script(spec: BlenderVisualSpec, output_path: Path, frame_dir: Path) -> str:
    payload = spec.to_payload()
    payload["output_path"] = str(output_path)
    payload["frame_dir"] = str(frame_dir)
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=True).encode("utf-8")).decode("ascii")
    return BLENDER_SCRIPT.replace("__SPEC_JSON_B64__", encoded)


def _configured_blender_path() -> str:
    raw = str(getattr(config, "BLENDER_PATH", "blender") or "blender").strip().strip("\"'")
    return raw or "blender"


def _resolve_blender_executable() -> str | None:
    configured = _configured_blender_path()
    found = shutil.which(configured)
    if found:
        return found

    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        executable = candidate / ("blender.exe" if os.name == "nt" else "blender")
        if executable.is_file():
            return str(executable)
    if os.name == "nt" and candidate.suffix.lower() != ".exe":
        exe_candidate = candidate.with_suffix(".exe")
        if exe_candidate.is_file():
            return str(exe_candidate)
    return None


def _blender_command(script_path: Path) -> list[str]:
    return [
        _resolve_blender_executable() or _configured_blender_path(),
        "-b",
        "-P",
        str(script_path),
    ]


def _blender_timeout_sec() -> int | None:
    try:
        timeout = int(getattr(config, "BLENDER_RENDER_TIMEOUT_SEC", 3600))
    except (TypeError, ValueError):
        timeout = 3600
    if timeout <= 0:
        return None
    return max(30, timeout)


def _encode_timeout_sec(duration: float) -> int:
    try:
        configured = int(getattr(config, "ENCODE_VALIDATION_TIMEOUT_SEC", 300))
    except (TypeError, ValueError):
        configured = 300
    return max(30, configured, int(max(duration, 1.0) * 12))


def _encode_alpha_frames(frame_dir: Path, output_path: Path, fps: float, duration: float) -> None:
    pattern = str(frame_dir / "frame_%04d.png")
    command = [
        config.FFMPEG_PATH,
        "-framerate",
        f"{fps:.3f}",
        "-i",
        pattern,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "qtrle",
        "-pix_fmt",
        "argb",
        "-y",
        str(output_path),
    ]
    timeout_sec = _encode_timeout_sec(duration)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise VisualRendererError(f"Timed out while encoding Blender alpha frames after {timeout_sec}s.") from exc
    if result.returncode != 0 or not output_path.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise VisualRendererError(f"Failed to encode Blender alpha frames: {detail}")


def _encode_color_frames(frame_dir: Path, output_path: Path, fps: float, duration: float) -> None:
    pattern = str(frame_dir / "frame_%04d.png")
    command = [
        config.FFMPEG_PATH,
        "-framerate",
        f"{fps:.3f}",
        "-i",
        pattern,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    timeout_sec = _encode_timeout_sec(duration)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise VisualRendererError(f"Timed out while encoding Blender frames after {timeout_sec}s.") from exc
    if result.returncode != 0 or not output_path.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise VisualRendererError(f"Failed to encode Blender frames: {detail}")


class BlenderRenderer(VisualRenderer):
    name = "blender"
    supported_templates = set(SUPPORTED_BLENDER_TEMPLATES)

    def availability(self) -> RendererStatus:
        blender_path = _configured_blender_path()
        if _resolve_blender_executable() is None:
            return RendererStatus(False, f"Blender executable was not found: {blender_path}")
        return RendererStatus(True, "")

    def score_spec(self, spec: dict[str, Any]) -> float:
        if not self.supports(spec):
            return -1.0
        template = str(spec.get("template") or "").strip().lower()
        visual_hint = str(spec.get("visual_type_hint") or "").strip().lower()
        composition = str(spec.get("composition_mode") or "").strip().lower()
        asset_path = str(spec.get("asset_path") or "").strip()
        score = 0.72
        if template in THREE_D_BLENDER_TEMPLATES:
            score += 0.52
        if template in {"object_orbit", "product_model_spin"} and asset_path:
            score += 0.2
        if template in {"three_d_title", "logo_reveal", "data_tunnel"} and composition == "replace":
            score += 0.14
        if template in {"floating_3d_label", "screen_pointer_3d", "product_model_spin"} and composition in {"overlay", "picture_in_picture"}:
            score += 0.16
        if visual_hint in {"abstract_motion", "product_ui"}:
            score += 0.1
        if str(spec.get("camera_motion") or "").strip().lower() in {"orbit", "slow_push"}:
            score += 0.05
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
        blender_spec = BlenderVisualSpec.from_raw(spec, render_root=render_root, width=width, height=height, fps=fps)

        spec_id = blender_spec.visual_id
        scene_name = _safe_scene_name(spec_id)
        job_dir = safe_render_job_dir(render_root, spec_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = job_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / (f"{scene_name}.mov" if blender_spec.alpha else f"{scene_name}.mp4")
        script_path = job_dir / "scene.py"
        script_path.write_text(
            _scene_script(blender_spec, output_path, frame_dir),
            encoding="utf-8",
        )

        timeout_sec = _blender_timeout_sec()
        try:
            result = subprocess.run(
                _blender_command(script_path),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise VisualRendererError(f"Blender renderer timed out after {timeout_sec}s for {spec_id}.") from exc
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise VisualRendererError(f"Blender renderer failed for {spec_id}: {stderr}")
        frame_count = sum(1 for _ in frame_dir.glob("frame_*.png"))
        if frame_count == 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise VisualRendererError(f"Blender renderer did not produce frames for {spec_id}: {stderr}")
        if blender_spec.alpha:
            _encode_alpha_frames(frame_dir, output_path, blender_spec.fps, blender_spec.duration)
        else:
            _encode_color_frames(frame_dir, output_path, blender_spec.fps, blender_spec.duration)
        if not output_path.is_file():
            raise VisualRendererError(f"Blender renderer completed but did not produce {output_path}.")

        cleanup_error = ""
        try:
            shutil.rmtree(frame_dir)
        except OSError as exc:
            cleanup_error = str(exc)

        metadata = probe_video(str(output_path))
        return RenderedAsset(
            asset_path=str(output_path),
            width=int(metadata.get("width") or blender_spec.width),
            height=int(metadata.get("height") or blender_spec.height),
            duration_sec=float(metadata.get("duration_sec") or blender_spec.duration),
            renderer=self.name,
            job_dir=str(job_dir),
            script_path=str(script_path),
            artifact_paths={
                "script_path": str(script_path),
                "frame_dir": str(frame_dir) if cleanup_error else None,
            },
            metadata={
                "renderer": self.name,
                "template": blender_spec.template,
                "has_alpha": bool(blender_spec.alpha),
                "composition_mode": blender_spec.composition_mode,
                "script_path": str(script_path),
                "job_dir": str(job_dir),
                "frame_dir": str(frame_dir) if cleanup_error else None,
                "intermediate_frame_count": frame_count,
                "intermediate_frames_removed": not bool(cleanup_error),
                "intermediate_cleanup_error": cleanup_error or None,
                "duration_sec": blender_spec.duration,
                "width": blender_spec.width,
                "height": blender_spec.height,
                "fps": blender_spec.fps,
                "asset_path": str(output_path),
                "asset_source_path": blender_spec.asset_path,
            },
        )
