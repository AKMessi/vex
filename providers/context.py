"""Bound dispatch context without discarding the authoritative conversation log."""
from __future__ import annotations
import json
import re


def compact_conversation(messages: list[dict], *, max_chars: int = 60000) -> list[dict]:
    if len(json.dumps(messages,default=str))<=max_chars:
        return messages
    units=[]
    index=0
    while index<len(messages):
        unit=[messages[index]]
        if messages[index].get("role")=="assistant" and messages[index].get("tool_calls"):
            ids={str(call.get("id")) for call in messages[index]["tool_calls"]}
            while index+1<len(messages) and messages[index+1].get("role")=="tool":
                index+=1;unit.append(messages[index]);ids.discard(str(messages[index].get("tool_call_id")))
            # An open tool exchange belongs in the retained tail, intact.
        units.append(unit);index+=1
    kept=[];size=0;cut=len(units)
    for index in range(len(units)-1,-1,-1):
        cost=len(json.dumps(units[index],default=str))
        if kept and size+cost>max_chars*.7:
            break
        kept.insert(0,units[index]);size+=cost;cut=index
    old=[message for unit in units[:cut] for message in unit]
    constraints=[str(message.get("content") or "") for message in old if message.get("role")=="user" and re.search(r"\b(?:always|never|must|prefer|remember|don't|do not)\b",str(message.get("content") or ""),re.I)]
    requests=[str(message.get("content") or "")[:300] for message in old if message.get("role")=="user"]
    summary="Earlier user constraints (quoted):\n"+"\n".join(constraints)+"\nEarlier requests:\n"+"\n".join(requests[-12:])+"\nThe full conversation remains in project state. Current project facts come from the system's project summary."
    summary=summary[:max(2000,int(max_chars*.25))]
    return [{"role":"user","content":summary},*[message for unit in kept for message in unit]]
