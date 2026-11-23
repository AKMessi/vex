from concurrent.futures import ThreadPoolExecutor

import pytest

from vex_runtime.provider_meter import read_usage
from vex_runtime.visual_run import VisualRunError, model_budget, visual_run


def test_budget_reserves_before_dispatch_and_reconciles_split_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("VEX_PROVIDER_MAX_USD", ".0002")
    monkeypatch.setenv("VEX_PROVIDER_RATES", '{"provider/model":{"input":1,"output":2}}')
    with visual_run(tmp_path):
        with model_budget("x" * 100, 50, provider="provider", model="model") as usage:
            with pytest.raises(VisualRunError, match="spend limit"):
                with model_budget("x", 1, provider="provider", model="model"):
                    pytest.fail("Over-budget request was dispatched.")
            usage.update(prompt_tokens=10, completion_tokens=5)
        with pytest.raises(OSError, match="connection"):
            with model_budget("x" * 80, 50, provider="provider", model="model"):
                raise OSError("connection lost")
    with visual_run(tmp_path):
        with pytest.raises(VisualRunError, match="spend limit"):
            with model_budget("x", 1, provider="provider", model="model"):
                pytest.fail("Retry opened a new spend budget.")
        with pytest.raises(VisualRunError, match="Set an input"):
            with model_budget("x", 1, provider="unknown", model="unknown"):
                pytest.fail("Request with unknown pricing was dispatched.")

    receipt = read_usage(tmp_path)
    assert [row["status"] for row in receipt["requests"]] == ["reported_split", "estimated"]
    assert receipt["estimated_total_usd"] == .0002


def test_concurrent_requests_cannot_reserve_the_same_remaining_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("VEX_PROVIDER_MAX_USD", ".00015")
    monkeypatch.setenv("VEX_PROVIDER_RATES", '{"provider/model":{"input":1,"output":2}}')

    def attempt(_):
        try:
            with visual_run(tmp_path):
                with model_budget("x" * 30, 50, provider="provider", model="model"):
                    return True
        except VisualRunError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(4))) == 1
