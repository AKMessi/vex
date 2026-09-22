from __future__ import annotations

from importlib.resources import files

from vex_web.server import _parse_multipart_form


def test_web_static_bundle_is_packaged() -> None:
    static = files("vex_web").joinpath("static")
    assert static.joinpath("index.html").is_file()
    assert static.joinpath("styles.css").is_file()
    assert static.joinpath("app.js").is_file()


def test_multipart_upload_parser_preserves_binary_edges() -> None:
    boundary = "----vex-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"\r\n\r\n'
        "Test cut\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
        "\x00\x01video\xff\r\n"
        f"--{boundary}--\r\n"
    ).encode("latin-1")

    fields, uploads = _parse_multipart_form(body, f"multipart/form-data; boundary={boundary}")

    assert fields == {"name": "Test cut"}
    assert uploads["file"] == ("clip.mp4", b"\x00\x01video\xff")
