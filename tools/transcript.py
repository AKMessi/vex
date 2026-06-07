from __future__ import annotations

import config
from state import ProjectState
from tools.transcript_utils import (
    build_sentence_segments,
    clean_transcript_text,
    format_srt_timestamp,
    transcript_artifact_path,
    write_json,
)


def _normalize_whisper_segments(raw_segments: list[dict]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized_segments: list[dict[str, object]] = []
    normalized_words: list[dict[str, object]] = []
    word_index = 1
    for segment_index, segment in enumerate(raw_segments, start=1):
        start_sec = float(segment.get("start") or 0.0)
        end_sec = float(segment.get("end") or 0.0)
        text = clean_transcript_text(str(segment.get("text") or ""))
        if end_sec <= start_sec or not text:
            continue
        segment_words: list[dict[str, object]] = []
        for raw_word in segment.get("words") or []:
            word_text = clean_transcript_text(str(raw_word.get("word") or raw_word.get("text") or ""))
            word_start = raw_word.get("start")
            word_end = raw_word.get("end")
            if word_start is None or word_end is None or not word_text:
                continue
            if float(word_end) <= float(word_start):
                continue
            payload = {
                "index": word_index,
                "start": round(float(word_start), 3),
                "end": round(float(word_end), 3),
                "text": word_text,
                "confidence": round(float(raw_word.get("probability", 0.0)), 4)
                if raw_word.get("probability") is not None
                else None,
            }
            segment_words.append(payload)
            normalized_words.append(payload)
            word_index += 1
        normalized_segments.append(
            {
                "index": segment_index,
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "text": text,
                "word_start_index": int(segment_words[0]["index"]) if segment_words else None,
                "word_end_index": int(segment_words[-1]["index"]) if segment_words else None,
                "words": segment_words,
            }
        )
    return normalized_segments, normalized_words


def execute(params: dict, state: ProjectState) -> dict:
    try:
        import whisper  # type: ignore
    except ImportError:
        return {
            "success": False,
            "message": "Whisper is not installed. Install it with `pip install openai-whisper` to enable transcription.",
            "suggestion": None,
            "updated_state": state,
            "tool_name": "transcribe_video",
        }

    model = whisper.load_model(config.WHISPER_MODEL)
    try:
        result = model.transcribe(state.working_file, word_timestamps=True, verbose=False)
    except TypeError:
        result = model.transcribe(state.working_file)
    txt_path = transcript_artifact_path(state.working_dir, "transcript.txt", for_write=True)
    srt_path = transcript_artifact_path(state.working_dir, "transcript.srt", for_write=True)
    segment_json_path = transcript_artifact_path(state.working_dir, "transcript.segments.json", for_write=True)
    words_json_path = transcript_artifact_path(state.working_dir, "transcript.words.json", for_write=True)
    sentences_json_path = transcript_artifact_path(state.working_dir, "transcript.sentences.json", for_write=True)
    transcript_text = clean_transcript_text(str(result.get("text") or ""))
    txt_path.write_text(transcript_text + "\n", encoding="utf-8")
    raw_segments = result.get("segments", [])
    segments, words = _normalize_whisper_segments(raw_segments if isinstance(raw_segments, list) else [])
    sentences = build_sentence_segments(words, fallback_segments=segments)
    srt_lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        srt_lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(segment['start'])} --> {format_srt_timestamp(segment['end'])}",
                segment["text"].strip(),
                "",
            ]
        )
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    write_json(segment_json_path, segments)
    write_json(words_json_path, words)
    write_json(sentences_json_path, sentences)
    state.artifacts["latest_transcript"] = {
        "txt_path": str(txt_path),
        "srt_path": str(srt_path),
        "segments_path": str(segment_json_path),
        "words_path": str(words_json_path),
        "sentences_path": str(sentences_json_path),
        "source": "audio_transcription",
        "segment_count": len(segments),
        "word_count": len(words),
        "sentence_count": len(sentences),
    }
    history = list(state.artifacts.get("transcript_history") or [])
    history.append(state.artifacts["latest_transcript"])
    state.artifacts["transcript_history"] = history[-10:]
    preview = "\n".join(
        f"{format_srt_timestamp(segment['start'])} {segment['text'].strip()}" for segment in segments[:10]
    )
    return {
        "success": True,
        "message": (
            f"Transcript saved to {txt_path}, {srt_path}, {segment_json_path.name}, {words_json_path.name}, "
            f"and {sentences_json_path.name}.\n{preview}"
        ),
        "suggestion": "[SUGGESTION]: Captions are ready. I can help turn them into timed overlays - reply 'yes' to apply or continue.",
        "updated_state": state,
        "tool_name": "transcribe_video",
    }
