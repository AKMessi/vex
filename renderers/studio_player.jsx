import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Player} from '@remotion/player';
import {GeneratedVideo} from './remotion_entry.jsx';

function timeLabel(frame, fps) {
  const seconds = Math.floor(frame / fps);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}.${String(frame % fps).padStart(2, '0')}`;
}

function Timeline({initial, assetBase, plan, preview}) {
  const player = useRef(null);
  const active = useRef(true);
  const dragged = useRef(null);
  const [timeline, setTimeline] = useState(initial);
  const [operations, setOperations] = useState([]);
  const [playhead, setPlayhead] = useState(0);
  const [selection, setSelection] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const current = player.current;
    const onFrame = ({detail}) => setPlayhead(detail.frame);
    current.addEventListener('frameupdate', onFrame);
    active.current = true;
    return () => {
      active.current = false;
      current.removeEventListener('frameupdate', onFrame);
    };
  }, []);

  const seek = (frame) => player.current?.seekTo(Math.max(0, Math.min(timeline.duration_frames - 1, frame)));
  const add = async (operation) => {
    if (busy) return;
    const next = [...operations, operation];
    setBusy(true);
    setError('');
    try {
      const planned = await plan(next);
      if (!active.current) return;
      player.current.pause();
      setTimeline(planned);
      setOperations(next);
      if (![...planned.scenes, ...(planned.overlays || [])].some(clip => clip.id === selection)) setSelection(null);
    } catch (failure) {
      if (active.current) setError(failure.message);
    } finally {
      if (active.current) setBusy(false);
    }
  };
  const move = (shot, target) => {
    if (!shot || shot === target) return;
    const ids = timeline.scenes.map(item => item.id);
    const destination = ids.indexOf(target);
    if (destination < 0 || !ids.includes(shot)) return;
    ids.splice(ids.indexOf(shot), 1);
    ids.splice(destination, 0, shot);
    add({op: 'reorder', shots: ids});
  };
  const selected = timeline.scenes.find(item => item.id === selection);
  const index = timeline.scenes.indexOf(selected);
  const full = [{start_frame: 0, duration_frames: timeline.duration_frames}];
  const voice = timeline.audio.voice_clips || (timeline.audio.narration ? full : []);
  const waveUrl = (src) => assetBase.replace(/production-asset\/$/, 'production-waveform/') + encodeURIComponent(src);
  const block = (clip) => ({
    left: `${100 * clip.start_frame / timeline.duration_frames}%`,
    width: `${100 * clip.duration_frames / timeline.duration_frames}%`,
  });
  const lane = (label, elements) => <div className="live-lane"><span>{label}</span><div className="live-track">
    {elements}
  </div></div>;
  const audioLane = (name, clips, source) => lane(name, clips.map((clip) => {
    const path = clip.src || source;
    return <button type="button" key={clip.start_frame} className="live-segment sound" style={block(clip)}
      onClick={() => seek(clip.start_frame)} title={name} aria-label={`${name} at ${timeLabel(clip.start_frame, timeline.fps)}`}>
      {path && <img alt="" loading="lazy" src={waveUrl(path)} />}
    </button>;
  }));

  return <>
    <div className="live-picture">
      <Player ref={player} component={GeneratedVideo} inputProps={{timeline, assetBase}}
        durationInFrames={timeline.duration_frames} fps={timeline.fps}
        compositionWidth={timeline.width} compositionHeight={timeline.height}
        controls playsInline acknowledgeRemotionLicense
        style={{width: '100%', maxHeight: 460}}
        errorFallback={({error: failure}) => <p role="alert">Live playback failed. {failure.message}</p>} />
    </div>
    <div className="live-clock">
      <output aria-label="Playhead" data-saved-playhead={operations.length ? undefined : playhead}>{timeLabel(playhead, timeline.fps)}</output>
      <input type="range" min="0" max={timeline.duration_frames - 1}
        value={Math.min(playhead, timeline.duration_frames - 1)}
        onChange={event => seek(Number(event.target.value))} aria-label="Scrub timeline" />
      <span>{timeLabel(timeline.duration_frames, timeline.fps)}</span>
    </div>
    {lane('Picture', timeline.scenes.map((shot, number) => <button type="button" key={shot.id}
      className="live-segment picture" aria-pressed={shot.id === selection} style={block(shot)} draggable={!busy}
      onDragStart={() => { dragged.current = shot.id; }}
      onDragOver={event => event.preventDefault()}
      onDrop={event => { event.preventDefault(); move(dragged.current, shot.id); dragged.current = null; }}
      onClick={() => { setSelection(shot.id); seek(shot.start_frame); }} title={shot.producer || 'Shot'}>
      <span>{number + 1}</span>
    </button>))}
    {audioLane('Voice', voice, timeline.audio.narration)}
    {audioLane('Music', timeline.audio.music ? full : [], timeline.audio.music)}
    {(timeline.audio.tracks || []).map(track => <div key={track.id}>
      {audioLane(track.name, track.clips)}
      <button type="button" className="secondary-btn" disabled={busy || !track.clips.length}
        onClick={() => add({op: 'audio_track_gain', id: track.id, gain: track.gain === 0 ? 1 : 0})}>
        {track.gain === 0 ? `Restore ${track.name}` : `Mute ${track.name}`}
      </button>
    </div>)}
    {[...new Set((timeline.overlays || []).map(clip => clip.track))].map(track => <React.Fragment key={track}>
      {lane(`Overlay ${track}`, timeline.overlays.filter(clip => clip.track === track).map(clip => <button key={clip.id}
        type="button" className="live-segment picture" aria-pressed={selection === clip.id} style={block(clip)}
        onClick={() => { setSelection(clip.id); seek(clip.start_frame); }} title={clip.mode || 'Overlay'}>Overlay</button>))}
    </React.Fragment>)}
    {lane('Captions', timeline.captions.map(cue => <button type="button" key={cue.id}
      className="live-segment caption" style={block(cue)} title={cue.text} onClick={() => seek(cue.start_frame)}>{cue.text}</button>))}
    <div className="live-edit-controls">
      {selected && <>
        <button type="button" className="secondary-btn" disabled={busy || index < 1}
          onClick={() => move(selected.id, timeline.scenes[index - 1].id)}>Move earlier</button>
        <button type="button" className="secondary-btn" disabled={busy || index + 1 >= timeline.scenes.length}
          onClick={() => move(selected.id, timeline.scenes[index + 1].id)}>Move later</button>
        <button type="button" className="secondary-btn"
          disabled={busy || playhead <= selected.start_frame || playhead >= selected.start_frame + selected.duration_frames}
          onClick={() => add({op: 'split', id: selected.id, at: `${playhead}/${timeline.fps}`})}>Split at playhead</button>
      </>}
      {timeline.overlays?.some(clip => clip.id === selection) && <button type="button" className="secondary-btn" disabled={busy}
        onClick={() => add({op: 'overlay_remove', id: selection})}>Remove overlay</button>}
      <button type="button" className="secondary-btn" disabled={busy || !voice.length}
        onClick={() => add({op: 'voice_gain', gain: timeline.audio.voice_gain === 0 ? 1 : 0})}>
        {timeline.audio.voice_gain === 0 ? 'Restore voice' : 'Mute voice'}</button>
      <button type="button" className="primary-btn" disabled={busy || !operations.length} onClick={() => preview(operations)}>
        Review and save
      </button>
    </div>
    {busy && <p role="status">Updating the draft…</p>}
    {error && <p role="alert">{error}</p>}
  </>;
}

export function mount(element, options) {
  const root = createRoot(element);
  root.render(<Timeline {...options} />);
  return () => root.unmount();
}
