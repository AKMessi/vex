from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from google import genai

import config
from engine import VideoEngineError, extract_segments, probe_video
from state import ProjectState
from tools.transcript import execute as transcribe
from tools.transcript_utils import parse_srt, transcript_artifact_path


def _format_transcript(segments: list[dict[str, float | str]]) -> str:
    return "\n".join(
        f"{segment['start']:.2f}-{segment['end']:.2f}: {segment['text']}"
        for segment in segments
    )


def _merge_overlapping_segments(segments: list[dict[str, float]]) -> list[dict[str, float]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: item["start"])
    merged = [ordered[0].copy()]
    for segment in ordered[1:]:
        current = merged[-1]
        if segment["start"] <= current["end"]:
            current["end"] = max(current["end"], segment["end"])
        else:
            merged.append(segment.copy())
    return merged


def _extract_json_array(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON array of segments.")
    return cleaned[start : end + 1]


def _select_segments_with_llm(
    provider_name: str,
    model_name: str,
    target_duration_sec: float,
    transcript_text: str,
    formatted_transcript: str,
) -> list[dict[str, float]]:
    system_prompt = (
        "You are a video editor. Given a transcript with timestamps, select the most engaging "
        "and informative segments that fit within the target duration. Return ONLY a JSON array "
        "of objects with 'start' and 'end' keys in seconds. No other text."
    )
    user_prompt = (
        f"Target duration: {target_duration_sec} seconds.\n\n"
        f"Transcript overview:\n{transcript_text}\n\n"
        f"Transcript with timestamps:\n{formatted_transcript}\n\n"
        "Select segments. Return JSON array only."
    )

    if provider_name == "claude":
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model_name or config.CLAUDE_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
    else:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_name or config.GEMINI_MODEL,
            contents=user_prompt,
            config=config.build_gemini_generation_config(
                system_prompt,
                model_name=model_name or config.GEMINI_MODEL,
            ),
        )
        raw_text = getattr(response, "text", "") or ""

    parsed = json.loads(_extract_json_array(raw_text))
    return [
        {"start": float(item["start"]), "end": float(item["end"])}
        for item in parsed
        if float(item["end"]) > float(item["start"])
    ]


def execute(params: dict, state: ProjectState) -> dict:
    transcript_path = transcript_artifact_path(state.working_dir, "transcript.txt")
    srt_path = transcript_artifact_path(state.working_dir, "transcript.srt")
    if transcript_path is None:
        transcribe_result = transcribe({}, state)
        state = transcribe_result["updated_state"]
        if not transcribe_result["success"]:
            return {
                "success": False,
                "message": transcribe_result["message"],
                "suggestion": None,
                "updated_state": state,
                "tool_name": "summarize_clip",
            }
        transcript_path = transcript_artifact_path(state.working_dir, "transcript.txt")
        srt_path = transcript_artifact_path(state.working_dir, "transcript.srt")
    if transcript_path is None or srt_path is None:
        return {
            "success": False,
            "message": "Transcript files are missing. Run transcribe_video first.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "summarize_clip",
        }

    transcript_text = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript_text:
        return {
            "success": False,
            "message": "Transcript is empty. Cannot summarize the clip.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "summarize_clip",
        }

    timestamped_segments = parse_srt(srt_path)
    if not timestamped_segments:
        return {
            "success": False,
            "message": "No timestamped transcript segments were found in transcript.srt.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "summarize_clip",
        }

    target_duration_sec = float(params.get("target_duration_sec", 60.0))
    formatted_transcript = _format_transcript(timestamped_segments)
    provider_name = (state.provider or config.PROVIDER or "gemini").strip().lower()
    if provider_name not in {"gemini", "claude"}:
        provider_name = "gemini"
    model_name = state.model or (
        config.CLAUDE_MODEL if provider_name == "claude" else config.GEMINI_MODEL
    )

    try:
        selected_segments = _select_segments_with_llm(
            provider_name=provider_name,
            model_name=model_name,
            target_duration_sec=target_duration_sec,
            transcript_text=transcript_text,
            formatted_transcript=formatted_transcript,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to summarize clip with {provider_name}: {exc}",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "summarize_clip",
        }

    merged_segments = _merge_overlapping_segments(selected_segments)
    if not merged_segments:
        return {
            "success": False,
            "message": f"No valid highlight segments were returned by {provider_name}.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "summarize_clip",
        }

    try:
        output_path = extract_segments(
            state.working_file,
            state.working_dir,
            [(segment["start"], segment["end"]) for segment in merged_segments],
        )
        state.working_file = output_path
        state.metadata = probe_video(output_path)
        description = f"Summarized clip to {state.metadata.get('duration_sec', 0.0):.2f}s using {len(merged_segments)} segments"
        op = {
            "op": "summarize_clip",
            "params": {
                "target_duration_sec": target_duration_sec,
                "segments": merged_segments,
            },
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result_file": output_path,
            "description": description,
        }
        state.apply_operation(op)
        return {
            "success": True,
            "message": f"Kept {len(merged_segments)} segments. New duration: {state.metadata.get('duration_sec', 0.0):.2f} seconds.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "summarize_clip",
        }
    except VideoEngineError as exc:
        return {
            "success": False,
            "message": str(exc),
            "suggestion": None,
            "updated_state": state,
            "tool_name": "summarize_clip",
        }
