from pathlib import Path
from types import SimpleNamespace
from tests.test_visual_communication_contract import _ir
from tests.test_visual_director import _spec, _asset, _LocalQA
from tests.test_visual_verifier import _payload
from vex_visuals.communication_contract import build_communication_contract
from vex_visuals.director import direct_rendered_visual
from vex_visuals.continuity import attach_continuity
from vex_visuals.experience import record_visual_experience, relevant_visual_experiences


def test_semantic_identity_is_stable_across_scene_ids():
    program=_spec()["open_visual_program"]
    first=attach_continuity(program,_ir())
    changed={**program,"program_id":"second-scene"}
    second=attach_continuity(changed,_ir())
    assert first["quality_contract"]["continuity"]["semantic_identities"]==second["quality_contract"]["continuity"]["semantic_identities"]


def test_only_verified_unchanged_frames_become_visual_references(tmp_path):
    frame=tmp_path/"frame.png"
    frame.write_bytes(b"evidence")
    contract=build_communication_contract(_ir())
    outcome=direct_rendered_visual(_spec(),_asset(),"initial",ir=_ir(),contract=contract.to_dict(),render_candidate=lambda *_:None,evaluate_local_quality=lambda *_:_LocalQA(True,.9,[]),extract_candidate_frames=lambda *_:[frame],strict=True,max_repair_rounds=0,provider_models=[("test","vision")],vision_request=lambda *_:_payload(contract))
    spec={**_spec(),"visual_explanation_ir":_ir()}
    record_visual_experience(tmp_path,spec,_asset(),outcome)
    assert relevant_visual_experiences(tmp_path,_ir())["reference_paths"]==[str(frame)]
    frame.write_bytes(b"altered")
    assert relevant_visual_experiences(tmp_path,_ir())["reference_paths"]==[]
