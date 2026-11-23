from __future__ import annotations

import http.client
import io
import json
import os
import shutil
import subprocess
import threading
import time
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import main
import vex_web.server as web_server
from state import ProjectState, utc_now_iso
from vex_web.server import (
    TaskManager,
    VexHTTPServer,
    _parse_byte_range,
    _project_state,
    _stream_multipart_form,
)


def test_web_static_bundle_is_packaged() -> None:
    static = files("vex_web").joinpath("static")
    assert static.joinpath("index.html").is_file()
    assert static.joinpath("styles.css").is_file()
    assert static.joinpath("app.js").is_file()
    assert static.joinpath("favicon.svg").is_file()


def test_web_bundle_has_no_inline_script_or_style_escape_hatches() -> None:
    static = files("vex_web").joinpath("static")
    app_source = static.joinpath("app.js").read_text(encoding="utf-8").lower()
    index_source = static.joinpath("index.html").read_text(encoding="utf-8").lower()

    assert "onclick=" not in app_source
    assert "style=" not in app_source
    assert "<script>" not in index_source


def test_streaming_multipart_parser_writes_media_to_private_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(web_server, "UPLOAD_DIR", upload_dir)
    boundary = "----vex-stream-test"
    media = b"\x00frame\r\n--" + boundary.encode() + b"X-not-a-boundary\xff"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"\r\n\r\n'
        "Streaming test\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="C:\\fakepath\\clip.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + media + f"\r\n--{boundary}--\r\n".encode()

    fields, uploaded = _stream_multipart_form(
        io.BytesIO(body),
        len(body),
        f"multipart/form-data; boundary={boundary}",
    )

    assert fields == {"name": "Streaming test"}
    assert uploaded is not None
    assert uploaded.filename == "clip.mp4"
    assert uploaded.path.read_bytes() == media
    assert uploaded.size == len(media)
    if os.name != "nt":
        assert upload_dir.stat().st_mode & 0o777 == 0o700
    uploaded.path.unlink()


def test_streaming_multipart_parser_rejects_unknown_fields() -> None:
    boundary = "----vex-unknown-field"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="surprise"\r\n\r\n'
        "unexpected\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    with pytest.raises(web_server.WebRequestError, match="Unexpected upload field"):
        _stream_multipart_form(
            io.BytesIO(body),
            len(body),
            f"multipart/form-data; boundary={boundary}",
        )


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=90-", 100, (90, 99)),
        ("bytes=-10", 100, (90, 99)),
        ("BYTES=0-999", 100, (0, 99)),
    ],
)
def test_parse_byte_range(header: str, size: int, expected: tuple[int, int]) -> None:
    assert _parse_byte_range(header, size) == expected


@pytest.mark.parametrize("header", ["items=0-2", "bytes=", "bytes=10-2", "bytes=100-", "bytes=0-1,4-5", "bytes=-0"])
def test_parse_byte_range_rejects_invalid_or_unsupported_ranges(header: str) -> None:
    with pytest.raises(ValueError, match="not satisfiable"):
        _parse_byte_range(header, 100)


def test_project_lookup_requires_an_exact_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_scan(*args):
        pytest.fail("Exact project lookup scanned the whole library")

    monkeypatch.setattr(ProjectState, "load", classmethod(unexpected_scan))

    with pytest.raises(FileNotFoundError):
        _project_state("abc123")


def test_project_summary_does_not_expose_local_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_server, "_project_state", lambda _project_id: (_ for _ in ()).throw(FileNotFoundError()))

    summary = web_server._project_summary(
        {
            "project_id": "project-1",
            "project_name": "Private paths",
            "source_file": "/private/media/source.mp4",
            "working_dir": "/private/projects/project-1",
        }
    )

    assert "source_file" not in summary
    assert "working_dir" not in summary
    assert summary["source_name"] == "source.mp4"


@pytest.mark.parametrize("value", [None, "", "invalid", float("nan"), float("inf")])
def test_format_duration_marks_missing_or_invalid_values_as_unknown(value: object) -> None:
    assert web_server._format_duration(value) == "—"


def test_public_error_message_redacts_credentials_and_limits_length() -> None:
    message = web_server._public_error_message(
        RuntimeError(
            "Authorization: Bearer bearer-secret api_key='key-value' "
            "access_token=token-value sk-abcdefghijklmnopqrstuvwxyz " + "x" * 3_000
        )
    )

    assert "bearer-secret" not in message
    assert "key-value" not in message
    assert "token-value" not in message
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in message
    assert len(message) == 2_000


def test_task_manager_serializes_work_per_project_and_returns_snapshots() -> None:
    manager = TaskManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def work(_task):
        started.set()
        assert release.wait(timeout=2)
        return {"success": True, "message": "Done"}

    try:
        task = manager.submit("project-1", "chat", "Editing", work)
        assert started.wait(timeout=2)
        with pytest.raises(RuntimeError, match="already working"):
            manager.submit("project-1", "chat", "Editing again", work)
        snapshot = manager.snapshot(task.task_id)
        assert snapshot is not None
        assert snapshot["status"] == "running"
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snapshot = manager.snapshot(task.task_id)
            if snapshot and snapshot["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert snapshot is not None
        assert snapshot["message"] == "Done"
        assert manager.active_snapshot("project-1") is None
    finally:
        release.set()
        manager.shutdown()


def test_task_manager_enforces_global_pending_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_server, "MAX_PENDING_TASKS", 1)
    manager = TaskManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def work(_task):
        started.set()
        assert release.wait(timeout=2)
        return {"success": True}

    try:
        manager.submit("project-1", "chat", "First edit", work)
        assert started.wait(timeout=2)
        with pytest.raises(RuntimeError, match="at capacity"):
            manager.submit("project-2", "chat", "Second edit", work)
    finally:
        release.set()
        manager.shutdown()


@pytest.fixture
def running_web_server():
    manager = TaskManager(max_workers=1)
    server = VexHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        manager.shutdown()
        thread.join(timeout=2)


def _request(server: VexHTTPServer, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    response_headers = dict(response.getheaders())
    connection.close()
    return response.status, response_headers, payload


def test_studio_production_opens_verified_film_and_serves_playback(running_web_server, tmp_path, monkeypatch):
    from vex_runtime import studio_projects
    from vex_web import production

    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path))
    monkeypatch.setattr(studio_projects, "probe_video", lambda _: {"duration_sec": 12, "fps": 24, "width": 960, "height": 540})
    finished = threading.Event()

    def generate(params, *, on_progress):
        assert params["aspect"] == "portrait"
        assert params["generate_audio"] is False
        assert "resolution" not in params
        on_progress("Rendering the film", .62)
        root = Path(params["output_dir"])
        assert root.is_relative_to(tmp_path)
        root.mkdir(parents=True)
        output = root / "film.mp4"
        output.write_bytes(b"film-data")
        manifest = root / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        finished.set()
        return SimpleNamespace(
            output_path=str(output), manifest_path=str(manifest), renderer="hybrid", duration_sec=12,
            qa_passed=True, qa_issues=[], warnings=[], title="Light and focus",
        )

    monkeypatch.setattr(production, "generate_video", generate)
    payload = json.dumps({
        "prompt": "Animate light moving through the lens", "aspect": "portrait",
        "renderer": "hybrid", "generate_audio": False, "output_dir": str(tmp_path.parent),
        "resolution": "7680x7680",
    }).encode()
    status, _, data = _request(running_web_server, "POST", "/api/production", payload, {"Content-Type": "application/json"})
    assert status == 202
    assert finished.wait(3)
    task_id = json.loads(data)["task_id"]
    for _ in range(100):
        status, _, data = _request(running_web_server, "GET", f"/api/tasks/{task_id}")
        task = json.loads(data)
        if task["status"] not in {"queued", "running"}:
            break
        time.sleep(.01)
    assert task["status"] == "succeeded", task
    assert task["events"][0]["title"] == "Rendering the film"
    state = ProjectState.load_exact(task["result"]["project_id"])
    assert "generated_source" in state.artifacts
    status, _, body = _request(running_web_server, "GET", task["result"]["media_url"], headers={"Range": "bytes=0-3"})
    assert (status, body) == (206, b"film")


def test_studio_replaces_a_generated_shot_in_a_new_project(running_web_server, tmp_path, monkeypatch):
    from vex_runtime import studio_projects
    from vex_web import production

    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(web_server, "UPLOAD_DIR", upload_root)
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(studio_projects, "probe_video", lambda _: {
        "duration_sec": 2, "fps": 24, "width": 640, "height": 360,
    })
    composition = tmp_path / "film" / "remotion"
    composition.mkdir(parents=True)
    props = {"timeline": {
        "version": "vex-remotion-video-v1", "width": 640, "height": 360, "fps": 24,
        "duration_frames": 48, "audio": {}, "captions": [],
        "scenes": [
            {"id": f"beat_{i + 1:02d}", "start_frame": i * 24, "duration_frames": 24, "standfirst": "Scene"}
            for i in range(2)
        ],
    }}
    timeline = composition / "input_props.json"
    timeline.write_text(json.dumps(props), encoding="utf-8")
    film = composition / "final.mp4"
    film.write_bytes(b"production")
    manifest = composition.parent / "manifest.json"
    manifest.write_text(json.dumps({
        "version": "generated-video-manifest-v1", "qa": {"passed": True},
        "artifacts": {"timeline_path": str(timeline), "output_path": str(film)},
    }), encoding="utf-8")
    state = studio_projects.import_verified_film(film, manifest, "Optics", "offline", "")
    endpoint = f"/api/projects/{state.project_id}"

    def revise(source, *, replacements, output_dir):
        assert source == composition
        asset = replacements["beat_02"]
        assert asset.suffix == ".png" and asset.read_bytes() == b"replacement"
        output_dir.mkdir(parents=True)
        output = output_dir / "final.mp4"
        output.write_bytes(b"new-candidate")
        report = output_dir / "revision.json"
        report.write_text(json.dumps({
            "version": "vex-remotion-revision-v1", "status": "verified", "output": output.name,
        }), encoding="utf-8")
        (output_dir / "input_props.json").write_text(json.dumps(props), encoding="utf-8")
        return output, report, {"passed": True, "rendering": {"rendered_frames": 24, "reused_frames": 24}}

    monkeypatch.setattr(production, "revise_remotion_project", revise)
    status, _, data = _request(running_web_server, "GET", endpoint)
    assert status == 200 and len(json.loads(data)["production_shots"]) == 2
    boundary = "vex-replace-shot-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="shot_id"\r\n\r\n'
        "beat_02\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="panorama.png"\r\n'
        "Content-Type: image/png\r\n\r\nreplacement\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    status, _, _ = _request(running_web_server, "POST", f"{endpoint}/replace-shot",
                            body.replace(b"beat_02", b"beat_99"), headers)
    assert status == 400
    assert not list(upload_root.glob("upload_*"))

    status, _, data = _request(running_web_server, "POST", f"{endpoint}/replace-shot", body, headers)
    assert status == 202, data
    task = json.loads(data)
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and (task["status"] in {"queued", "running"} or list(upload_root.glob("upload_*"))):
        _, _, data = _request(running_web_server, "GET", f"/api/tasks/{task['task_id']}")
        task = json.loads(data)
        time.sleep(.01)
    assert task["status"] == "succeeded", task
    assert not list(upload_root.glob("upload_*"))
    revision_id = task["result"]["project_id"]
    assert revision_id != state.project_id and task["result"]["reused_frames"] == 24
    revision = ProjectState.load_exact(revision_id)
    assert Path(revision.working_file).read_bytes() == b"new-candidate"
    status, _, data = _request(running_web_server, "GET", f"/api/projects/{revision_id}")
    assert status == 200 and len(json.loads(data)["production_shots"]) == 2
    assert Path(state.working_file).read_bytes() == b"production"
    assert not ProjectState.load_exact(state.project_id).timeline


def test_stopping_a_production_before_the_next_stage_keeps_its_run_record():
    manager = TaskManager(max_workers=1)
    entered, release = threading.Event(), threading.Event()

    def work(task):
        manager.production_progress(task, "Writing a script", .08)
        entered.set()
        assert release.wait(2)
        manager.production_progress(task, "Rendering the film", .62)
        return {"success": True}

    try:
        task = manager.submit(web_server.PRODUCTION_SLOT, "production", "Creating", work)
        assert entered.wait(2)
        assert manager.stop_production(task.task_id)["stop_requested"]
        release.set()
        for _ in range(100):
            snapshot = manager.snapshot(task.task_id)
            if snapshot["status"] == "cancelled":
                break
            time.sleep(.01)
        assert snapshot["status"] == "cancelled"
        assert snapshot["stage"] == "Writing a script"
    finally:
        release.set()
        manager.shutdown()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required to prepare playback")
def test_studio_browser_preview_reuses_a_small_compatible_copy(tmp_path):
    from engine import probe_video
    from vex_web.media_preview import browser_preview

    source = tmp_path / "old-camera.avi"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=12:d=1",
        "-c:v", "mpeg4", "-y", str(source),
    ], check=True, timeout=30)
    initial = source.read_bytes()
    preview = browser_preview(tmp_path, source)
    assert probe_video(str(preview))["codec"] == "h264"
    assert probe_video(str(preview))["width"] == 320
    assert browser_preview(tmp_path, source, prepare=False) == preview
    assert source.read_bytes() == initial


def test_http_server_sets_security_headers_supports_head_and_uses_etags(running_web_server: VexHTTPServer) -> None:
    status, headers, body = _request(running_web_server, "GET", "/static/app.js")
    assert status == 200
    assert body.startswith(b"const ICONS")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert "unsafe-inline" not in headers["Content-Security-Policy"]
    assert headers["ETag"]

    head_status, head_headers, head_body = _request(running_web_server, "HEAD", "/static/app.js")
    assert head_status == 200
    assert head_body == b""
    assert head_headers["Content-Length"] == headers["Content-Length"]

    cached_status, _, cached_body = _request(
        running_web_server,
        "GET",
        "/static/app.js",
        headers={"If-None-Match": headers["ETag"]},
    )
    assert cached_status == 304
    assert cached_body == b""


def test_http_server_rejects_unknown_hosts_cross_site_posts_and_missing_assets(running_web_server: VexHTTPServer) -> None:
    status, _, _ = _request(running_web_server, "GET", "/api/health", headers={"Host": "example.com"})
    assert status == 421

    status, headers, _ = _request(
        running_web_server,
        "POST",
        "/api/projects",
        body=b"{}",
        headers={"Content-Type": "application/json", "Origin": "http://example.com"},
    )
    assert status == 403
    assert headers["Connection"] == "close"

    status, headers, payload = _request(running_web_server, "GET", "/static/missing.js")
    assert status == 404
    assert headers["Content-Type"].startswith("application/json")
    assert json.loads(payload)["error"] == "Not found."

    status, _, _ = _request(running_web_server, "GET", "/static/server.py")
    assert status == 404


def test_http_server_streams_media_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    running_web_server: VexHTTPServer,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"0123456789")
    state = SimpleNamespace(project_id="project-1", working_file=str(media), source_files=[str(media)])
    monkeypatch.setattr(web_server, "_project_state", lambda _project_id: state)

    status, headers, body = _request(
        running_web_server,
        "GET",
        "/api/projects/project-1/media/current",
        headers={"Range": "bytes=2-5"},
    )
    assert status == 206
    assert headers["Content-Range"] == "bytes 2-5/10"
    assert headers["Accept-Ranges"] == "bytes"
    assert body == b"2345"

    status, headers, body = _request(
        running_web_server,
        "GET",
        "/api/projects/project-1/media/current",
        headers={"Range": "bytes=20-30"},
    )
    assert status == 416
    assert headers["Content-Range"] == "bytes */10"
    assert json.loads(body)["error"] == "Media range is not satisfiable."


def test_uploaded_project_keeps_a_durable_source_inside_the_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    running_web_server: VexHTTPServer,
) -> None:
    upload_dir = tmp_path / "uploads"
    projects_dir = tmp_path / "projects"
    created: list[ProjectState] = []
    monkeypatch.setattr(web_server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(projects_dir))

    def fake_create_project(source: str, name: str | None, provider: str, model: str, *, managed_source: bool) -> ProjectState:
        assert managed_source
        project_id = "project-upload-test"
        working_dir = projects_dir / project_id
        working_dir.mkdir(parents=True)
        working_file = working_dir / "source_clip.mp4"
        shutil.copy2(source, working_file)
        state = ProjectState(
            project_id=project_id,
            project_name=name or "clip",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            source_files=[str(working_file)],
            working_file=str(working_file),
            working_dir=str(working_dir),
            output_dir=str(working_dir / "outputs"),
            metadata={"duration_sec": 1.0, "width": 1920, "height": 1080, "fps": 30},
            provider=provider,
            model=model,
        )
        state.save()
        created.append(state)
        return state

    monkeypatch.setattr(web_server, "create_project", fake_create_project)
    boundary = "----vex-http-upload"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"\r\n\r\n'
        "Durable upload\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + b"fake-video-data" + f"\r\n--{boundary}--\r\n".encode()

    status, _, payload = _request(
        running_web_server,
        "POST",
        "/api/projects",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    assert status == 201, payload
    assert created
    state = created[0]
    assert state.source_files == [state.working_file]
    assert Path(state.source_files[0]).read_bytes() == b"fake-video-data"
    assert Path(state.output_dir) == Path(state.working_dir) / "outputs"
    assert not list(upload_dir.glob("upload_*"))
    response = json.loads(payload)
    assert "source_path" not in response["project"]
    assert "working_file" not in response["project"]
    assert response["active_task"] is None


def test_create_project_cleans_partial_working_directory_on_probe_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"not-video")
    projects_dir = tmp_path / "projects"
    monkeypatch.setattr(config, "AGENT_PROJECTS_DIR", str(projects_dir))
    from vex_runtime import studio_projects

    monkeypatch.setattr(studio_projects, "probe_video", lambda _path: (_ for _ in ()).throw(RuntimeError("probe failed")))

    with pytest.raises(RuntimeError, match="probe failed"):
        main.create_project(str(source), None, "gemini", "test-model")

    assert not list(projects_dir.iterdir())
