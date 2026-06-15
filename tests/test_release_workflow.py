from __future__ import annotations

from pathlib import Path


def test_release_candidates_verify_the_attested_wheel_without_testpypi() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "name: Verify built wheel with pipx" in workflow
    assert "python -m pipx install --backend pip dist/*.whl" in workflow
    assert "vars.ENABLE_TESTPYPI_PUBLISH == 'true'" in workflow
    assert "needs.verify-artifact.result == 'success'" in workflow
    assert "needs.verify-testpypi.result == 'skipped'" in workflow
    assert (
        "actions/upload-artifact@"
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    ) in workflow
    assert (
        "actions/download-artifact@"
        "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
    ) in workflow
