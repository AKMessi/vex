# Add Song Music Director

`add_song` is the production song-mix path for Vex. It is intentionally separate
from `replace_audio`: `replace_audio` remains a low-level utility, while
`add_song` is a directed harness for background music, soundtracks, intro/outro
cues, and timed song sections.

## Why It Exists

Average language models are unreliable at audio engineering details. They can
identify that the user wants "background music" or "replace the soundtrack", but
they should not invent FFmpeg filter graphs, ducking thresholds, loop behavior,
or loudness gates.

Music Director v1 turns the model's job into bounded slot filling:

1. choose or pass a local `song_path`
2. optionally choose a mode such as `background`, `replace`, `intro`, `outro`,
   `intro_outro`, `segment`, or `highlight`
3. optionally pass timing, volume, ducking, looping, and fade overrides
4. let Vex build and verify the actual mix

## Skill Graph

The runtime lives in `tools/song_director.py`.

Current skills:

- `voiceover_bed`: song under existing speech/source audio with sidechain
  ducking
- `silent_video_soundtrack`: soundtrack for a source video with no audio
- `replace_soundtrack`: replace the source audio with the selected song
- `intro_sting`: short opening cue
- `outro_sting`: short closing cue
- `intro_outro_sting`: bookended opening and closing cues
- `highlight_montage`: more present music for high-energy sections
- `segment_music`: music only in a selected time range

Each skill defines preservation policy, default ducking, volume, fade length,
loudness target, QA floor, and hard gates. The final `SongMixPlan` records the
selected skill, placements, loop policy, loudness policy, and quality contract.

## Render Contract

`engine.add_song_to_video` owns FFmpeg execution. It builds one deterministic
`filter_complex` from the typed plan:

- resample and normalize the song input
- split the song for multi-placement plans
- trim, delay, fade, loop, pad, and volume-shape each placement
- sidechain-compress the music against source audio when ducking is enabled
- mix with `amix=duration=first` so output remains video-length
- apply limiting and optional final loudness normalization
- copy the video stream and encode only the final audio stream

The filter graph is written to `filtergraph.txt` before promotion.

## QA And Promotion

`add_song` writes a bundle under `song_mix_bundles/`:

- `manifest.json`
- `mix_plan.json`
- `filtergraph.txt`
- `audio_qa.json`
- `notes.md`

The project state is updated only when audio QA passes. QA checks:

- output exists and is non-empty
- duration stays within tolerance
- resolution is preserved
- output contains audio
- source-audio preservation contract is respected
- FFmpeg audio stats do not show obvious clipping

Rejected mixes remain in the bundle for inspection but are not promoted into the
timeline.
