from pathlib import Path
import pytest
from vex_runtime.visual_run import VisualRun, VisualRunError, visual_run, current_visual_run, model_budget


def test_completed_render_stage_survives_new_run_and_checks_output_hash(tmp_path):
    output = tmp_path / "asset.mp4"
    calls = []
    def work():
        calls.append(1)
        output.write_bytes(b"approved-video")
        return {"asset_path":str(output)}
    first = VisualRun(tmp_path)
    assert first.stage("render",{"program":"a"},work,output_paths=lambda result:[result["asset_path"]])[1] is False
    second = VisualRun(tmp_path)
    assert second.stage("render",{"program":"a"},work,output_paths=lambda result:[result["asset_path"]])[1] is True
    assert len(calls)==1
    output.write_bytes(b"corrupt")
    assert second.stage("render",{"program":"a"},work,output_paths=lambda result:[result["asset_path"]])[1] is False
    assert len(calls)==2


def test_failed_stage_can_resume_without_caching_failure(tmp_path):
    run=VisualRun(tmp_path)
    with pytest.raises(ValueError):
        run.stage("compile",{},lambda:(_ for _ in ()).throw(ValueError("invalid")))
    value,hit=run.stage("compile",{},lambda:{"compiled":True})
    assert value["compiled"] and not hit


def test_nested_models_share_budget_and_account_actual_usage(tmp_path):
    with visual_run(tmp_path,max_model_calls=1,max_tokens=10000) as run:
        with visual_run(tmp_path) as nested:
            assert nested is run
            with model_budget("prompt",100) as usage:
                usage["total_tokens"]=12
        assert run.snapshot()["tokens"]==12
        with pytest.raises(VisualRunError,match="calls budget"):
            with model_budget("second",100):
                pass
    assert current_visual_run() is None


def test_cancelled_run_cannot_publish_new_stage(tmp_path):
    run=VisualRun(tmp_path)
    run.cancel()
    with pytest.raises(VisualRunError,match="cancelled"):
        run.stage("publish",{},lambda:{"ok":True})
