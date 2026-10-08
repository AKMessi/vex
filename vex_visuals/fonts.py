"""Bundled OFL-licensed typography; no render-time network font dependency."""
from base64 import b64encode
from functools import lru_cache
from importlib.resources import files
import json


@lru_cache(maxsize=1)
def font_css() -> str:
    data = b64encode(files("renderers").joinpath("fonts/Inter.ttf").read_bytes()).decode("ascii")
    return "@font-face{font-family:Inter;font-style:normal;font-weight:100 900;font-display:block;src:url(data:font/ttf;base64," + data + ") format('truetype');}"


def font_runtime_source() -> str:
    return "export const fontCss = " + json.dumps(font_css()) + ";\n"
