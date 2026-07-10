from __future__ import annotations

from pathlib import Path

import pytest

import broll_intelligence
import sources


def test_stock_download_url_validation_rejects_non_https() -> None:
    with pytest.raises(RuntimeError, match="HTTPS"):
        broll_intelligence._validated_public_https_url("http://example.com/video.mp4")


def test_stock_download_url_validation_rejects_private_hosts() -> None:
    with pytest.raises(RuntimeError, match="non-public host"):
        broll_intelligence._validated_public_https_url("https://127.0.0.1/video.mp4")


def test_stock_download_url_validation_rejects_unexpected_pexels_api_host() -> None:
    with pytest.raises(RuntimeError, match="Unexpected stock media API host"):
        broll_intelligence._validated_public_https_url(
            "https://example.com/v1/videos/search",
            allowed_hosts={"api.pexels.com"},
        )


def test_stock_redirect_validation_rejects_private_redirect() -> None:
    handler = broll_intelligence._ValidatedHttpsRedirectHandler()

    with pytest.raises(RuntimeError, match="non-public host"):
        handler.redirect_request(
            object(),
            object(),
            302,
            "found",
            {},
            "https://127.0.0.1/internal.mp4",
        )


def test_stock_download_rejects_invalid_content_length(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    destination = tmp_path / "clip.mp4"

    class FakeResponse:
        headers = {"Content-Length": "not-an-int"}

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return b""

    monkeypatch.setattr(broll_intelligence, "_validated_public_https_url", lambda url, **_kwargs: url)
    monkeypatch.setattr(
        broll_intelligence,
        "_open_validated_https_request",
        lambda _request, **_kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="invalid Content-Length"):
        broll_intelligence.download_file("https://cdn.example/clip.mp4", destination)

    assert not destination.exists()
    assert not destination.with_name(".clip.mp4.part").exists()


def test_stock_download_rejects_truncated_content(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    destination = tmp_path / "clip.mp4"

    class FakeResponse:
        headers = {"Content-Length": "10"}

        def __init__(self) -> None:
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            self.read_count += 1
            return b"safe" if self.read_count == 1 else b""

    monkeypatch.setattr(broll_intelligence, "_validated_public_https_url", lambda url, **_kwargs: url)
    monkeypatch.setattr(
        broll_intelligence,
        "_open_validated_https_request",
        lambda _request, **_kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="Content-Length"):
        broll_intelligence.download_file("https://cdn.example/clip.mp4", destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.part"))


def test_stock_download_does_not_follow_predictable_part_symlink(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    destination = tmp_path / "clip.mp4"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"keep")
    predictable_part = destination.with_name(".clip.mp4.part")
    try:
        predictable_part.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    class FakeResponse:
        headers = {"Content-Length": "4"}

        def __init__(self) -> None:
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            self.read_count += 1
            return b"safe" if self.read_count == 1 else b""

    monkeypatch.setattr(broll_intelligence, "_validated_public_https_url", lambda url, **_kwargs: url)
    monkeypatch.setattr(
        broll_intelligence,
        "_open_validated_https_request",
        lambda _request, **_kwargs: FakeResponse(),
    )

    broll_intelligence.download_file("https://cdn.example/clip.mp4", destination)

    assert destination.read_bytes() == b"safe"
    assert outside.read_bytes() == b"keep"


def test_stock_json_response_must_be_an_object() -> None:
    class FakeResponse:
        headers: dict[str, str] = {}

        def read(self, _size: int) -> bytes:
            return b"[]"

    with pytest.raises(RuntimeError, match="non-object"):
        broll_intelligence._read_stock_json_response(FakeResponse(), "pexels")


def test_stock_search_rejects_malformed_result_collection(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        broll_intelligence,
        "pexels_get_json",
        lambda _url: ({"videos": "not-a-list"}, {}),
    )

    with pytest.raises(RuntimeError, match="invalid 'videos' collection"):
        broll_intelligence.search_pexels_videos("workflow", "landscape")


def test_youtube_download_resolver_ignores_candidates_outside_destination(tmp_path: Path) -> None:
    destination = tmp_path / "downloads"
    destination.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    inside = destination / "source_abc123.mp4"
    inside.write_bytes(b"inside")

    class FakeYdl:
        def prepare_filename(self, _info: dict) -> str:
            return str(outside)

    resolved = sources._resolve_downloaded_path(
        {"requested_downloads": [{"filepath": str(outside)}], "id": "abc123"},
        destination,
        FakeYdl(),
    )

    assert resolved == str(inside.resolve())


def test_youtube_download_resolver_raises_when_only_outside_candidate_exists(tmp_path: Path) -> None:
    destination = tmp_path / "downloads"
    destination.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")

    class FakeYdl:
        def prepare_filename(self, _info: dict) -> str:
            return str(outside)

    with pytest.raises(RuntimeError, match="no downloaded video file"):
        sources._resolve_downloaded_path(
            {"requested_downloads": [{"filepath": str(outside)}], "id": "abc123"},
            destination,
            FakeYdl(),
        )


def test_youtube_download_resolver_rejects_symlink_escape(tmp_path: Path) -> None:
    destination = tmp_path / "downloads"
    destination.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    linked = destination / "source_abc123.mp4"
    try:
        linked.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    class FakeYdl:
        def prepare_filename(self, _info: dict) -> str:
            return str(linked)

    with pytest.raises(RuntimeError, match="no downloaded video file"):
        sources._resolve_downloaded_path({"id": "abc123"}, destination, FakeYdl())


def test_youtube_download_resolver_excludes_preexisting_fallback(tmp_path: Path) -> None:
    destination = tmp_path / "downloads"
    destination.mkdir()
    stale = destination / "source_old.mp4"
    stale.write_bytes(b"stale")

    class FakeYdl:
        def prepare_filename(self, _info: dict) -> str:
            return str(destination / "missing.webm")

    with pytest.raises(RuntimeError, match="no downloaded video file"):
        sources._resolve_downloaded_path(
            {"id": "new"},
            destination,
            FakeYdl(),
            preexisting_paths={str(stale.resolve())},
        )


def test_youtube_download_rejects_non_youtube_url_before_network(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Only standard YouTube"):
        sources.download_youtube_video("https://example.com/video", str(tmp_path))
