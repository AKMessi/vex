from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from vex_runtime.production_references import open_reference, references, remove_reference, save_reference


def test_sanitized_references_are_deduplicated_and_checked(tmp_path):
    class Project:
        working_dir = tmp_path
        artifacts = {}

        def save(self):
            pass

    state = Project()
    source = tmp_path / "private.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "private source label")
    Image.new("RGB", (1300, 800), (255, 128, 0)).save(source, pnginfo=metadata)
    [saved] = save_reference(state, source, "Warm orange for the main path")
    copy = open_reference(state, saved["id"])
    assert copy.parent == tmp_path / "production_references"
    assert b"private source label" not in copy.read_bytes()
    with Image.open(copy) as image:
        assert image.size == (1024, 630)
    assert save_reference(state, source, "Use orange only once") == [{**saved, "note": "Use orange only once"}]
    Path(copy).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        open_reference(state, saved["id"])
    assert remove_reference(state, saved["id"]) == references(state) == []
    with pytest.raises(ValueError, match="attached"):
        open_reference(state, saved["id"])
