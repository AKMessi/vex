from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from tests.test_visual_communication_contract import _ir
from tests.test_visual_verifier import _payload
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.harness import verify_generated_portfolio
from providers.context import compact_conversation


def _args(tmp_path):
    video=tmp_path/"final.mp4";video.write_bytes(b"video")
    beat=SimpleNamespace(beat_id="visual_001",start=0,end=4.8,duration=4.8,narration="Four tokens become one compressed KV entry.",title="Compression")
    return video,dict(project_dir=tmp_path,request=SimpleNamespace(width=1280,height=720,fps=30),beat_graph=SimpleNamespace(beats=[beat],duration_sec=4.8),cinematic_plan=SimpleNamespace(beat_compositions=[SimpleNamespace(beat_id=beat.beat_id,metadata={"visual_explanation_ir":_ir()})]),output_metadata={"fps":30,"duration_sec":4.8},local_quality={"passed":True,"score":.8},enable_repair=False)


def test_full_video_uses_same_independent_meaning_gate(tmp_path):
    video,args=_args(tmp_path)
    contract=build_communication_contract(_ir())
    colour=["white"]
    def extract(path,samples):
        for target,_ in samples:
            target.parent.mkdir(parents=True,exist_ok=True);Image.new("RGB",(64,36),colour[0]).save(target)
        return [target for target,_ in samples]
    result=verify_generated_portfolio(video,**args,vision_request=lambda *_:_payload(contract),frame_extractor=extract)
    assert result["passed"]
    assert result["beats"][0]["verification_receipt"]["asset_sha256"]
    wrong=_payload(contract);wrong["unsupported_claims"]=["Invented 90 percent reduction"]
    colour[0]="black"
    result=verify_generated_portfolio(video,**args,vision_request=lambda *_:wrong,frame_extractor=extract)
    assert not result["passed"]


def test_full_video_missing_frames_cannot_publish(tmp_path):
    video,args=_args(tmp_path)
    result=verify_generated_portfolio(video,**args,frame_extractor=lambda *_:[])
    assert not result["passed"]


def test_compaction_keeps_tool_results_and_user_preferences():
    messages=[{"role":"user","content":"Always preserve original audio"}]
    messages.extend({"role":"assistant","content":"old output "*1000} for _ in range(12))
    messages.extend([{"role":"user","content":"Add visuals"},{"role":"assistant","tool_calls":[{"id":"call1","name":"inspect","params":{}}]},{"role":"tool","tool_call_id":"call1","content":"metadata"}])
    result=compact_conversation(messages,max_chars=6000)
    assert "Always preserve original audio" in result[0]["content"]
    assert result[-2:]==messages[-2:]
    assert len(messages)==16
