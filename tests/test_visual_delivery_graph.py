from pathlib import Path
from fractions import Fraction
from vex_runtime.edit_graph import EditGraph
from vex_runtime.edit_graph import EditGraphError
import pytest
from vex_runtime.edit_graph_renderer import render_edit_graph


@pytest.mark.parametrize("layers", ["invalid", [None], [{"layer_id": "missing"}]])
def test_invalid_layer_payloads_raise_graph_error(tmp_path, layers):
    raw = EditGraph.from_source(tmp_path / "source.mp4", duration=10, fps=30).to_dict()
    raw["layers"] = layers
    with pytest.raises(EditGraphError):
        EditGraph.from_mapping(raw)


def test_visual_layers_round_trip_and_trim_preserves_asset_offset(tmp_path):
    asset = tmp_path / "visual.mov"
    asset.write_bytes(b"asset")
    graph = EditGraph.from_source(
        tmp_path / "source.mp4", duration=10, fps="30000/1001"
    ).with_visual_layers(
        [
            {
                "asset_path": str(asset),
                "start": 2,
                "end": 6,
                "transition_in": 0.2,
                "transition_out": 0.2,
            }
        ]
    )
    assert EditGraph.from_mapping(graph.to_dict()) == graph
    trimmed = graph.trim(3, 5)
    assert trimmed.layers[0].output_start == 0
    assert trimmed.layers[0].output_end == 2
    assert trimmed.layers[0].asset_start == 1
    assert trimmed.layers[0].transition_in_sec == 0
    assert trimmed.layers[0].transition_out_sec == 0
    assert trimmed.fps == Fraction(30000, 1001)


def test_source_graph_compiles_fractional_fps_and_all_layers_in_one_pass(
    tmp_path, monkeypatch
):
    import engine

    asset = tmp_path / "visual.mov"
    asset.write_bytes(b"asset")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    graph = (
        EditGraph.from_source(source, duration=10, fps="30000/1001")
        .trim(1, 9)
        .with_visual_layers([{"asset_path": str(asset), "start": 2, "end": 5}])
    )
    monkeypatch.setattr(engine, "probe_video", lambda *_: {"has_audio": True})
    calls = []
    monkeypatch.setattr(
        engine, "_run_command", lambda command, message: calls.append(command)
    )
    output = render_edit_graph(graph, tmp_path, width=1280, height=720)
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("-r") + 1] == "30000/1001"
    assert command[command.index("-crf") + 1] == "16"
    script = Path(command[command.index("-filter_complex_script") + 1]).read_text()
    assert "trim=start=1.000000000:end=9.000000000" in script
    assert "fps=30000/1001" in script
    assert "overlay=" in script
    assert "atrim=" in script
