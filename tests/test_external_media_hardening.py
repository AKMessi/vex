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
