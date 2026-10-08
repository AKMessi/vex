"""Concrete source-grounded keyframe sketches for a cold-start designer."""
from __future__ import annotations
import re
from pathlib import Path
from importlib.resources import files
from PIL import Image,ImageDraw,ImageFont
from vex_visuals.evidence import payload_digest


def render_reference_sketch(program: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/(payload_digest(program)+".png")
    if path.is_file():
        return path
    canvas=program["canvas"]
    ratio=min(960/canvas["width"],720/canvas["height"])
    width,height=round(canvas["width"]*ratio),round(canvas["height"]*ratio)
    palette=program["palette"]
    image=Image.new("RGB",(width,height),palette["background"])
    draw=ImageDraw.Draw(image)
    for element in program["elements"]:
        layout=element["layout"]
        x,y=layout["x"]*width,layout["y"]*height
        w,h=layout["width"]*width,layout["height"]*height
        anchor=layout.get("anchor")
        if anchor=="center":x-=w/2;y-=h/2
        elif anchor=="top_right":x-=w
        elif anchor=="bottom_left":y-=h
        elif anchor=="bottom_right":x-=w;y-=h
        style=element.get("style") or {}
        fill=palette.get(style.get("fill"),palette["surface"])
        ink=palette["ink"]
        if element["type"] not in {"text","path","connector"}:
            draw.rounded_rectangle((x,y,x+w,y+h),radius=min(12,w/4,h/4),fill=fill,outline=palette["accent"],width=2)
        text=str(element.get("text") or "")
        if not text:continue
        font=ImageFont.truetype(str(files("renderers").joinpath("fonts/Inter.ttf")),max(13,round(float(style.get("font_size") or 30)*ratio)))
        lines=[];line=""
        for word in text.split():
            candidate=(line+" "+word).strip()
            if line and draw.textlength(candidate,font=font)>max(w-16,20):lines.append(line);line=word
            else:line=candidate
        if line:lines.append(line)
        draw.multiline_text((x+8,y+8),"\n".join(lines),fill=ink,font=font,spacing=3)
    image.save(path)
    return path
