from __future__ import annotations
import copy
import re
from vex_visuals.evidence import payload_digest
from vex_visuals.open_visual_program import sign_open_visual_program, validate_open_visual_program


def attach_continuity(program: dict, ir: dict, *, words: list[dict] | None = None, start_sec: float = 0, duration_sec: float | None = None) -> dict:
    result=copy.deepcopy(program)
    objects={str(item["object_id"]):item for item in ir.get("objects") or []}
    identities={}
    cues=[]
    for element in result.get("elements") or []:
        obj=objects.get(str((element.get("binding") or {}).get("id")))
        if not obj:
            continue
        meaning=re.sub(r"\b\d+\b","",str(obj.get("meaning") or obj.get("label") or "").casefold()).strip()
        identity=payload_digest({"meaning":meaning})[:16]
        identities[element["element_id"]]=identity
        if element.get("type")=="icon" and not element.get("geometry"):
            icon="database" if any(word in meaning for word in ("memory","cache","database")) else "search" if any(word in meaning for word in ("query","index","search")) else "gear" if any(word in meaning for word in ("tool","planner","agent")) else "document"
            element["geometry"]={"icon":icon}
        terms=set(re.findall(r"[a-z]+",str(obj.get("label") or "").lower()))-{"the","a","an","of","to","in"}
        for index,word in enumerate(words or []):
            window=(words or [])[index:index+max(len(terms)+2,3)]
            observed=set(re.findall(r"[a-z]+"," ".join(str(item.get("word") or item.get("text") or "") for item in window).lower()))
            if terms and len(terms&observed)/len(terms)>=.75:
                cue=float(word.get("start") or 0)
                duration=duration_sec or float(result["canvas"]["duration_sec"])
                if start_sec<=cue<start_sec+duration:
                    cues.append({"element_id":element["element_id"],"time_sec":cue,"fraction":round((cue-start_sec)/duration,4),"source":"transcript_word_alignment"})
                break
    duration=duration_sec or float(result["canvas"]["duration_sec"])
    readable=[str(element.get("text") or "") for element in result["elements"] if element.get("role") in {"takeaway","resolved_outcome","result"}]
    reading_hold=max(.6,min(duration*.35,max([len(text.split())/4 for text in readable] or [.6])))
    by_id={item["element_id"]:item for item in result["elements"]}
    for cue in cues:
        element=by_id[cue["element_id"]]
        if element.get("role")=="title":
            continue
        for track in result.get("tracks") or []:
            if track.get("target_id")!=cue["element_id"]:
                continue
            keys=track.get("keyframes") or []
            if len(keys)<2:
                continue
            begin,end=float(keys[0]["t"]),float(keys[-1]["t"])
            target=max(0,min(cue["fraction"]-.025,.62))
            span=min(max(end-begin,.08),.16)
            for key in keys:
                key["t"]=round(target+(float(key["t"])-begin)/max(end-begin,.001)*span,4)
    result["quality_contract"]["continuity"]={"semantic_identities":identities,"narration_cues":cues,"reading_hold_sec":round(reading_hold,3)}
    signed=sign_open_visual_program(result)
    return signed if validate_open_visual_program(signed,ir=ir).passed else program
