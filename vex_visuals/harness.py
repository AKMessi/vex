"""Workflow profiles over one renderer-independent publication boundary."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable
import config
from renderers.base import RenderedAsset
from vex_visuals.communication_contract import build_communication_contract, validate_communication_contract
from vex_visuals.director import direct_rendered_visual
from vex_visuals.evidence import build_verification_receipt
from vex_visuals.frame_sampling import extract_native_frames
from visual_explanation import build_visual_explanation_ir


@dataclass(frozen=True)
class WorkflowProfile:
    name: str
    strict: bool
    repair_rounds: int


@dataclass(frozen=True)
class WorkflowQuality:
    passed: bool
    score: float
    issues: list[str]
    def to_dict(self):
        return {"passed":self.passed,"score":self.score,"issues":self.issues}


def profile_for(name: str) -> WorkflowProfile:
    return WorkflowProfile(name,config.VISUAL_DIRECTOR_VERIFICATION_MODE=="strict",int(config.VISUAL_DIRECTOR_MAX_REPAIR_ROUNDS))


def verify_generated_portfolio(output_path: Path, *, project_dir: Path, request: Any, beat_graph: Any, cinematic_plan: Any, output_metadata: dict, local_quality: dict, vision_request=None, frame_extractor=None, enable_repair: bool = True, producer_specs: dict | None = None) -> dict:
    """Inspect actual beat windows; replace failed windows through the same harness.

    The initial producer is the native full-video scene program. A failed window
    may be regenerated as an Open Visual Program; its new renderer is then judged
    independently before compositing. Original scene data is never misrepresented
    as the program that produced a repaired window.
    """
    profile=profile_for("generated_video")
    if not config.VISUAL_DIRECTOR_ENABLED or config.VISUAL_DIRECTOR_VERIFICATION_MODE=="off":
        return {"enabled":False,"passed":bool(local_quality.get("passed")),"output_path":str(output_path),"beats":[]}
    records=[]
    overlays=[]
    overrides=dict(producer_specs or {})
    compositions={item.beat_id:item for item in cinematic_plan.beat_compositions}
    fps=float(output_metadata.get("fps") or request.fps)
    full_duration=float(output_metadata.get("duration_sec") or beat_graph.duration_sec)
    def frames_for(beat, path, directory, *, standalone=False):
        samples=[(directory/f"frame_{index:02d}.png",(0 if standalone else beat.start)+beat.duration*fraction) for index,fraction in enumerate([.03,.42,.68,.94],1)]
        if frame_extractor:
            return frame_extractor(path,samples)
        return extract_native_frames(path,samples,fps=fps,duration_sec=beat.duration if standalone else full_duration)[0]
    for beat in beat_graph.beats:
        item=compositions.get(beat.beat_id)
        metadata=dict(item.metadata or {}) if item else {}
        ir=dict(metadata.get("visual_explanation_ir") or {})
        if not ir:
            ir=build_visual_explanation_ir({"visual_id":beat.beat_id,"sentence_text":beat.narration,"context_text":beat.narration,"headline":beat.title,"duration":beat.duration}).to_dict()
        contract=build_communication_contract(ir).to_dict()
        errors=validate_communication_contract(contract,source_ir=ir)
        if errors:
            records.append({"beat_id":beat.beat_id,"start":beat.start,"end":beat.end,"passed":False,"issues":errors,"selected":{"selected_verifier_score":0}})
            continue
        spec={"visual_id":beat.beat_id,"visual_explanation_ir":ir,"visual_communication_contract":contract,"scene_program_v2":metadata.get("scene_program_v2") or {},"duration":beat.duration,"start":0.,"end":beat.duration}
        spec=overrides.get(beat.beat_id,spec)
        asset=RenderedAsset(str(output_path),request.width,request.height,beat.duration,"hyperframes",str(project_dir),str(project_dir/"index.html"),metadata={"fps":fps,"producer":"native_video_project"})
        frame_dir=project_dir/"shared_qa"/beat.beat_id
        initial_frames=frames_for(beat,output_path,frame_dir)
        initial=direct_rendered_visual(spec,asset,"native generated beat",ir=ir,contract=contract,render_candidate=lambda *_:(_ for _ in ()).throw(RuntimeError("Native producer requires a new encoding")),evaluate_local_quality=lambda *_:WorkflowQuality(bool(local_quality.get("passed")),float(local_quality.get("score") or .6),[]),extract_candidate_frames=lambda *_:initial_frames,strict=profile.strict,max_repair_rounds=0,vision_request=vision_request,cache_dir=project_dir/"visual_director_cache")
        selected=initial
        repair_report={}
        if not initial.passed and enable_repair and initial_frames and beat.duration<=16:
            try:
                from broll_intelligence import call_reasoning_model
                from vex_visuals.generative_authoring import compile_open_visual_program_for_spec
                from tools.auto_visuals import _render_generated_visual,_direct_rendered_visual_for_spec
                provider=config.VISUAL_AUTHORING_PROVIDER or config.PROVIDER
                model=config.VISUAL_AUTHORING_MODEL or (config.GROQ_MODEL if provider=="groq" else config.CLAUDE_MODEL if provider=="claude" else config.GEMINI_MODEL)
                repair_spec={**spec,"generation_provider":provider,"generation_model":model,"renderer_hint":"hyperframes","visual_reference_paths":[str(path) for path in initial_frames],"directed_visual_brief":{"observed_failure":initial.to_dict(),"instruction":"Regenerate this native beat's encoding from its source evidence. The supplied frames show what failed."},"auto_visuals_director":{"director_score":70,"copy_alignment":.8}}
                repair_spec,authored=compile_open_visual_program_for_spec(repair_spec,ir=ir,width=request.width,height=request.height,fps=fps,reasoning_call=call_reasoning_model,candidate_count=3)
                if not authored.passed:
                    raise ValueError("No grounded repair encoding")
                repair_root=project_dir/"shared_repairs"/beat.beat_id
                repaired,reason=_render_generated_visual(repair_spec,preferred_renderer="hyperframes",allowed_renderers={"hyperframes"},render_root=repair_root,width=request.width,height=request.height,fps=fps,renderer_strategy="first_success",tournament_size=1)
                repair_spec,repaired,reason,qa,repair_report=_direct_rendered_visual_for_spec(repair_spec,repaired,reason,render_root=repair_root,width=request.width,height=request.height,fps=fps)
                if qa.passed:
                    overrides[beat.beat_id]=repair_spec
                    overlays.append({"visual_id":beat.beat_id,"start":beat.start,"end":beat.end,"asset_path":repaired.asset_path,"compose_mode":"replace","force_fullscreen":True})
                    final_frames=frames_for(beat,Path(repaired.asset_path),repair_root/"final_evidence",standalone=True)
                    selected=direct_rendered_visual(repair_spec,repaired,reason,ir=ir,contract=contract,render_candidate=lambda *_:None,evaluate_local_quality=lambda *_:WorkflowQuality(qa.passed,qa.score,qa.issues),extract_candidate_frames=lambda *_:final_frames,strict=profile.strict,max_repair_rounds=0,vision_request=vision_request,cache_dir=project_dir/"visual_director_cache")
            except Exception as exc:
                repair_report={"passed":False,"error":f"{type(exc).__name__}: {exc}"}
        selected_asset=selected.selected.asset
        receipt=build_verification_receipt(selected_asset,selected.selected.spec,selected.selected.frame_paths,selected.to_dict())
        records.append({"beat_id":beat.beat_id,"start":beat.start,"end":beat.end,"initial":initial.to_dict(),"selected":selected.to_dict(),"repair":repair_report,"verification_receipt":receipt,"passed":selected.passed and receipt["passed"]})
    final_path=output_path
    if overlays:
        from engine import apply_visual_overlays,probe_video
        final_path=Path(apply_visual_overlays(str(output_path),str(project_dir),overlays))
        # The exported composite is new evidence. Inspect its actual beat windows.
        final=verify_generated_portfolio(final_path,project_dir=project_dir,request=request,beat_graph=beat_graph,cinematic_plan=cinematic_plan,output_metadata=probe_video(str(final_path)),local_quality={"passed":True,"score":.8},vision_request=vision_request,frame_extractor=frame_extractor,enable_repair=False,producer_specs=overrides)
        final["repaired_overlays"]=overlays
        final["repair_history"]=records
        (project_dir/"shared_visual_qa.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
        return final
    passed=bool(records) and all(record["passed"] for record in records)
    result={"version":"vex-shared-portfolio-harness-v1","enabled":True,"profile":profile.name,"passed":passed,"output_path":str(final_path),"beats":records,"score":sum(record["selected"]["selected_verifier_score"] for record in records)/max(len(records),1)}
    (project_dir/"shared_visual_qa.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
