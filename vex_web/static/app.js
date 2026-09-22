const ICONS = {
  grid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="4" width="6" height="6"/><rect x="14" y="4" width="6" height="6"/><rect x="4" y="14" width="6" height="6"/><rect x="14" y="14" width="6" height="6"/></svg>',
  film: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="5" width="18" height="14" rx="1"/><path d="m8 5 2 14M14 5l2 14M3 9h18M3 15h18"/></svg>',
  activity: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 16V8M8 19V5M12 14V10M16 18V6M20 12v-2"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Z"/><path d="m19.4 15 .1.1a1.6 1.6 0 0 1-2.3 2.3l-.1-.1a1.6 1.6 0 0 0-2.7 1.1v.2a1.6 1.6 0 0 1-3.2 0v-.2a1.6 1.6 0 0 0-2.7-1.1l-.1.1a1.6 1.6 0 1 1-2.3-2.3l.1-.1A1.6 1.6 0 0 0 5.1 12a1.6 1.6 0 0 0-1.1-2.7h-.2a1.6 1.6 0 0 1 0-3.2H4A1.6 1.6 0 0 0 5.1 3.4L5 3.3A1.6 1.6 0 1 1 7.3 1l.1.1A1.6 1.6 0 0 0 10.1 0h.2a1.6 1.6 0 0 1 3.2 0h.2a1.6 1.6 0 0 0 2.7 1.1l.1-.1A1.6 1.6 0 1 1 18.8 3l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.2a1.6 1.6 0 0 1 0 3.2h-.2a1.6 1.6 0 0 0-1.1 2.7l.1.1a1.6 1.6 0 0 1 .5 1.2" transform="translate(2 2) scale(.83)"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 5v14M5 12h14"/></svg>',
  arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 12h13M13 6l6 6-6 6"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m9 6 9 6-9 6V6Z"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m6 6 12 12M18 6 6 18"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M20 11a8 8 0 0 0-14.7-4L4 9M4 5v4h4M4 13a8 8 0 0 0 14.7 4L20 15m0 4v-4h-4"/></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m12 3 1.2 5.8L19 10l-5.8 1.2L12 17l-1.2-5.8L5 10l5.8-1.2L12 3ZM19 16l.6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z"/></svg>',
  edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m4 16.5-.8 4.3 4.3-.8L19 8.5 15.5 5 4 16.5Z"/><path d="m13.8 6.7 3.5 3.5M18 4.2l1.8-1.8 1.8 1.8-1.8 1.8"/></svg>',
};

const state = {
  view: 'studio',
  projects: [],
  detail: null,
  health: null,
  selectedId: localStorage.getItem('vex-studio-project') || '',
  modal: null,
  draft: '',
  taskId: '',
  task: null,
  error: '',
};

const app = document.getElementById('app');

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function icon(name) { return ICONS[name] || ''; }

function project() { return state.detail?.project || null; }

function showError(message) {
  state.error = message;
  render();
  window.clearTimeout(showError.timer);
  showError.timer = window.setTimeout(() => { state.error = ''; render(); }, 5000);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function refreshProjects() {
  const payload = await api('/api/projects');
  state.projects = payload.projects || [];
  if (!state.selectedId && state.projects.length) state.selectedId = state.projects[0].project_id;
  if (state.selectedId && !state.projects.some((item) => item.project_id === state.selectedId)) state.selectedId = state.projects[0]?.project_id || '';
  if (state.selectedId) localStorage.setItem('vex-studio-project', state.selectedId);
}

async function refreshDetail() {
  if (!state.selectedId) { state.detail = null; return; }
  state.detail = await api(`/api/projects/${encodeURIComponent(state.selectedId)}`);
}

async function boot() {
  try {
    [state.health] = await Promise.all([api('/api/health'), refreshProjects()]);
    await refreshDetail();
  } catch (error) {
    state.error = error.message;
  }
  render();
}

function selectProject(projectId) {
  state.selectedId = projectId;
  state.view = 'studio';
  state.taskId = '';
  state.task = null;
  localStorage.setItem('vex-studio-project', projectId);
  const url = new URL(window.location.href);
  url.searchParams.set('project', projectId);
  window.history.replaceState({}, '', url);
  refreshDetail().then(render).catch((error) => showError(error.message));
  render();
}

function navItem(view, label, iconName) {
  return `<button class="nav-item ${state.view === view ? 'active' : ''}" data-nav="${view}">${icon(iconName)}<span>${label}</span></button>`;
}

function renderSidebar() {
  const health = state.health || {};
  return `<aside class="sidebar">
    <div class="brand"><div class="brand-symbol">VX</div><div class="brand-copy"><span class="brand-name">Vex Studio</span><span class="brand-sub">Local video intelligence</span></div></div>
    <button class="new-project" data-action="new-project">${icon('plus')}<span>New project</span></button>
    <div class="nav-label">Workspace</div>
    <nav class="nav-list">
      ${navItem('studio', 'Studio', 'film')}
      ${navItem('projects', 'Projects', 'grid')}
      ${navItem('activity', 'Activity', 'activity')}
      ${navItem('settings', 'Settings', 'settings')}
    </nav>
    <div class="sidebar-bottom">
      <div class="local-status"><span class="status-dot"></span><div class="status-copy"><span class="status-title">Running locally</span><span class="status-detail">${esc(health.provider || 'Vex runtime')} · ${esc(health.version || 'dev')}</span></div></div>
      <div class="sidebar-meta">Your footage and project state stay on this machine.<br />No account. No upload queue.</div>
    </div>
  </aside>`;
}

function renderTopbar() {
  const current = project();
  return `<header class="topbar"><div class="breadcrumb"><strong>${state.view === 'studio' ? 'Studio' : state.view[0].toUpperCase() + state.view.slice(1)}</strong>${current && state.view === 'studio' ? `<span class="breadcrumb-sep">/</span><span>${esc(current.project_name)}</span>` : ''}</div><div class="top-actions"><button class="icon-btn" data-action="refresh" title="Refresh">${icon('refresh')}</button><button class="icon-btn" data-action="new-project" title="New project">${icon('plus')}</button></div></header>`;
}

function renderHeading(eyebrow, title, copy = '') {
  return `<div class="view-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1></div>${copy ? `<p class="heading-copy">${copy}</p>` : ''}</div>`;
}

function renderVideoCard() {
  const current = project();
  const media = state.detail?.media;
  return `<section class="card video-card"><div class="video-frame">${media?.available ? `<div class="video-overlay-label"><span class="status-dot"></span>Current working cut</div><video controls preload="metadata" src="${esc(media.current)}"></video>` : `<div class="video-empty"><div class="video-empty-icon">${icon('film')}</div><p>Your working cut will appear here once you load a video into Vex.</p></div>`}</div>${current ? `<div class="media-meta"><div class="media-meta-item"><span class="meta-label">Duration</span><span class="meta-value">${esc(current.duration)}</span></div><div class="media-meta-item"><span class="meta-label">Frame</span><span class="meta-value">${esc(current.resolution)}</span></div><div class="media-meta-item"><span class="meta-label">Frame rate</span><span class="meta-value">${esc(current.fps)} fps</span></div><div class="media-meta-item"><span class="meta-label">Source</span><span class="meta-value" title="${esc(current.source_name)}">${esc(current.source_name)}</span></div></div>` : ''}</section>`;
}

function renderComposer() {
  const busy = Boolean(state.task && ['queued', 'running'].includes(state.task.status));
  const prompts = ['Clean up the pauses and filler words', 'Give this a natural, cinematic grade', 'Add clean, readable captions', 'Find the strongest moments for shorts'];
  return `<section class="card composer-card"><div class="composer-top"><h3>Tell Vex what to change</h3><span>${busy ? esc(state.task.message || 'Working…') : 'Plain English is the interface'}</span></div><form class="composer" data-form="chat"><textarea id="prompt" name="message" placeholder="Try: remove the awkward intro, add captions, and make it feel more cinematic…" ${busy ? 'disabled' : ''}>${esc(state.draft)}</textarea><button class="send-btn" type="submit" ${busy ? 'disabled' : ''} aria-label="Send instruction">${icon('arrow')}</button></form><div class="quick-actions">${prompts.map((prompt) => `<button class="quick-action" data-prompt="${esc(prompt)}" ${busy ? 'disabled' : ''}>${esc(prompt)}</button>`).join('')}</div></section>`;
}

function renderTimeline() {
  const timeline = state.detail?.timeline || [];
  const labels = timeline.length ? timeline.slice(0, 10).map((item) => item.op) : Array.from({ length: 10 }, () => '');
  return `<section class="card timeline-card"><div class="timeline-headline"><span>Timeline</span><span>${timeline.length ? `${timeline.length} operation${timeline.length === 1 ? '' : 's'}` : 'No edits yet'}</span></div><div class="timeline-strip">${labels.map((label, index) => `<div class="timeline-segment"><span>${esc(label || `scene ${String(index + 1).padStart(2, '0')}`)}</span></div>`).join('')}</div></section>`;
}

function renderTrace() {
  const taskEvents = state.task?.events || [];
  const traceEvents = state.detail?.latest_trace?.events || [];
  const events = taskEvents.length ? taskEvents : traceEvents;
  const recent = events.slice(-7).reverse();
  return `<section class="card inspector-card"><div class="trace-head"><h3>${state.task && ['queued', 'running'].includes(state.task.status) ? 'Live run' : 'Latest run'}</h3><span class="card-kicker">${recent.length ? `${recent.length} steps` : 'Quiet'}</span></div>${recent.length ? `<div class="trace-list">${recent.map((event) => `<div class="trace-item"><span class="trace-marker ${esc(event.status || '')}"></span><div class="trace-copy"><span class="trace-title">${esc(event.title || event.kind || 'Update')}</span><span class="trace-detail">${esc(event.detail || '')}</span></div></div>`).join('')}</div>` : `<p class="empty-state">Vex will show the plan, tools, and QA as soon as you run an instruction.</p>`}</section>`;
}

function renderInspector() {
  const current = project();
  if (!current) return `<aside class="inspector-column"><section class="card inspector-card"><div class="project-title-row"><div class="project-title"><div class="eyebrow">Start here</div><h2>Load a project</h2><p>Bring in a video to open the Studio.</p></div><div class="project-badge">VX</div></div><button class="primary-btn" style="width:100%;margin-top:22px" data-action="new-project">${icon('upload')}<span>Choose a video</span></button></section></aside>`;
  const artifacts = state.detail?.artifacts || [];
  return `<aside class="inspector-column"><section class="card inspector-card"><div class="project-title-row"><div class="project-title"><h2 title="${esc(current.project_name)}">${esc(current.project_name)}</h2><p title="${esc(current.source_name)}">${esc(current.source_name)}</p></div><div class="project-badge">VX</div></div><div class="stat-grid"><div class="stat"><span class="stat-value">${esc(current.timeline_ops)}</span><span class="stat-label">Edits</span></div><div class="stat"><span class="stat-value">${esc(current.duration)}</span><span class="stat-label">Runtime</span></div><div class="stat"><span class="stat-value">${esc(current.resolution)}</span><span class="stat-label">Frame</span></div><div class="stat"><span class="stat-value">${esc(current.provider)}</span><span class="stat-label">Provider</span></div></div>${artifacts.length ? `<div class="section-rule"></div><div class="card-kicker">Project outputs</div><div class="trace-list" style="margin-top:12px">${artifacts.slice(0, 5).map((item) => `<div class="trace-item"><span class="trace-marker success"></span><div class="trace-copy"><span class="trace-title">${esc(item.label)}</span><span class="trace-detail">${esc(item.summary)}</span></div></div>`).join('')}</div>` : ''}</section>${renderTrace()}</aside>`;
}

function renderProjectCards(limit = 3) {
  return state.projects.slice(0, limit).map((item) => `<button class="project-card ${item.project_id === state.selectedId ? 'active' : ''}" data-project="${esc(item.project_id)}"><div class="project-thumb"><span class="project-thumb-label">${esc(item.resolution || 'Video')}</span></div><div class="project-card-title" title="${esc(item.project_name || item.source_name)}">${esc(item.project_name || item.source_name)}</div><div class="project-card-meta"><span>${esc(item.duration || '—')}</span><span>${esc(item.updated_label || 'recently')}</span></div></button>`).join('');
}

function renderStudio() {
  const current = project();
  return `${renderHeading('Studio / edit intelligence', current ? `Make ${esc(current.project_name)}<br /><em>move.</em>` : 'Edit with<br /><em>intention.</em>', current ? 'Your media, timeline, and creative history in one quiet workspace. Ask for the cut you want; Vex handles the machinery.' : 'A local-first studio for the work between “I have footage” and “I have something worth watching.”')}
    ${current ? `<div class="studio-grid"><div class="primary-column">${renderVideoCard()}${renderComposer()}${renderTimeline()}</div>${renderInspector()}</div><div class="section-heading"><div><h2>Recent projects</h2><p>Pick up where you left off.</p></div><button class="secondary-btn" data-nav="projects">View all ${icon('arrow')}</button></div><div class="project-grid">${renderProjectCards()}</div>` : `<div class="empty-library"><div class="video-empty-icon">${icon('spark')}</div><h2>Start with a frame.</h2><p>Load a local video and Vex will give you a working cut, a safe project copy, and a place to direct the edit in plain English.</p><button class="primary-btn" data-action="new-project">${icon('upload')}<span>Load your first video</span></button></div>`}`;
}

function renderProjects() {
  return `${renderHeading('Library / projects', 'A home for<br /><em>the good cuts.</em>', 'Every Vex project stays local, resumable, and inspectable. Your source file is never touched.')}${state.projects.length ? `<div class="library-grid">${state.projects.map((item) => `<button class="project-card ${item.project_id === state.selectedId ? 'active' : ''}" data-project="${esc(item.project_id)}"><div class="project-thumb"><span class="project-thumb-label">${esc(item.resolution || 'Video')}</span></div><div class="project-card-title" title="${esc(item.project_name || item.source_name)}">${esc(item.project_name || item.source_name)}</div><div class="project-card-meta"><span>${esc(item.duration || '—')} · ${esc(item.timeline_ops || 0)} edits</span><span>${esc(item.updated_label || 'recently')}</span></div></button>`).join('')}</div>` : `<div class="empty-library"><h2>No projects yet.</h2><p>Load a video to create a safe, resumable Vex project.</p><button class="primary-btn" data-action="new-project">${icon('plus')}<span>New project</span></button></div>`}`;
}

function renderActivity() {
  const current = project();
  const timeline = state.detail?.timeline || [];
  const runs = state.detail?.creative_runs || [];
  return `${renderHeading('Activity / history', 'Everything Vex<br /><em>has done.</em>', 'A readable record of edits, creative runs, and the decisions behind your current working cut.')}${current ? `<div class="section-heading" style="margin-top:0"><div><h2>${esc(current.project_name)}</h2><p>${esc(current.timeline_ops)} timeline operation${current.timeline_ops === 1 ? '' : 's'} · ${esc(current.updated_label)}</p></div><button class="secondary-btn" data-nav="studio">Back to studio ${icon('arrow')}</button></div>` : ''}<section class="card" style="padding:0 20px"><div class="activity-list">${timeline.length ? timeline.map((item, index) => `<div class="activity-row"><div class="activity-index">${String(index + 1).padStart(2, '0')}</div><div class="activity-copy"><span class="activity-title">${esc(item.op)}</span><span class="activity-detail">${esc(item.detail)}</span></div><span class="activity-time">${esc(formatTimestamp(item.timestamp))}</span></div>`).join('') : '<p class="empty-state">No edit history for this project yet.</p>'}</div></section>${runs.length ? `<div class="section-heading"><div><h2>Creative runs</h2><p>Quality-gated automation recorded by Vex.</p></div></div><section class="card" style="padding:0 20px"><div class="activity-list">${runs.map((run) => `<div class="activity-row"><div class="activity-index">${icon('spark')}</div><div class="activity-copy"><span class="activity-title">${esc(run.feature || 'Creative run')}</span><span class="activity-detail">${esc(creativeSummary(run))}</span></div><span class="quality-pill">${esc(formatScore(run.quality_score))}</span></div>`).join('')}</div></section>` : ''}`;
}

function renderSettings() {
  const health = state.health || {};
  const mediaReady = health.media_stack_ready;
  return `${renderHeading('Settings / local runtime', 'Quiet control<br /><em>over the stack.</em>', 'Vex is designed to stay close to your machine. These are the active local runtime signals, not a cloud account page.')}
    <div class="settings-grid"><section class="card settings-panel"><h2>Runtime</h2><p>Configuration is read from your local Vex environment and project state directory.</p><div style="margin-top:18px"><div class="setting-row"><div class="setting-label"><span class="setting-name">Provider</span><span class="setting-note">Used when a request needs model planning.</span></div><span class="setting-value">${esc(health.provider || '—')}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Model</span><span class="setting-note">Configured model name.</span></div><span class="setting-value">${esc(health.model || '—')}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Media stack</span><span class="setting-note">FFmpeg and ffprobe availability.</span></div><span class="setting-value" style="color:${mediaReady ? 'var(--lime)' : 'var(--red)'}">${mediaReady ? 'ready' : 'needs setup'}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Projects</span><span class="setting-note">Where Vex keeps project working copies.</span></div><span class="setting-value" title="${esc(health.projects_dir)}">local</span></div></div></section><aside class="primary-column"><section class="card settings-panel"><div class="settings-note">The Studio is served by Vex itself at localhost. No footage leaves this machine unless a tool explicitly uses a provider or stock-media API you have configured.</div></section><section class="card settings-panel"><div class="card-kicker">Build</div><h2 style="margin-top:8px">Vex ${esc(health.version || '')}</h2><p>Terminal-first editing, now with a visual command center.</p></section></aside></div>`;
}

function renderModal() {
  if (state.modal !== 'new-project') return '';
  return `<div class="modal-backdrop" data-action="close-modal"><section class="modal" role="dialog" aria-modal="true" aria-labelledby="new-project-title" onclick="event.stopPropagation()"><div class="modal-header"><div><h2 id="new-project-title">New project</h2><p>Bring in a local video. Vex makes a safe working copy before anything changes.</p></div><button class="icon-btn" data-action="close-modal" aria-label="Close">${icon('close')}</button></div><form class="modal-body" data-form="new-project"><label class="drop-zone" id="drop-zone"><input id="video-file" type="file" name="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska,video/*" /><span class="video-empty-icon">${icon('upload')}</span><strong id="file-label">Drop a video here or choose a file</strong><span>MP4, MOV, WEBM, MKV, M4V</span></label><div class="field"><label for="project-name">Project name <span class="muted">(optional)</span></label><input id="project-name" name="name" placeholder="e.g. Product film / episode 04" maxlength="120" /></div><div class="field"><label for="source-path">Or use a local path</label><input id="source-path" name="source_path" placeholder="/Users/you/Videos/episode-04.mov" /></div>${state.error ? `<div class="error-note">${esc(state.error)}</div>` : ''}<div class="modal-actions"><button class="secondary-btn" type="button" data-action="close-modal">Cancel</button><button class="primary-btn" type="submit">${icon('arrow')}<span>Create project</span></button></div></form></section></div>`;
}

function renderToast() { return state.error && !state.modal ? `<div class="toast error">${esc(state.error)}</div>` : ''; }

function render() {
  const content = state.view === 'projects' ? renderProjects() : state.view === 'activity' ? renderActivity() : state.view === 'settings' ? renderSettings() : renderStudio();
  app.innerHTML = `${renderSidebar()}<main class="main">${renderTopbar()}<div class="main-content">${content}</div></main>${renderModal()}${renderToast()}`;
  bindFileInputs();
  const prompt = document.getElementById('prompt');
  if (prompt && state.view === 'studio') {
    prompt.addEventListener('input', (event) => { state.draft = event.target.value; });
    prompt.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.target.form.requestSubmit(); } });
  }
}

function formatTimestamp(timestamp) { return timestamp ? String(timestamp).slice(11, 16) : '—'; }
function formatScore(value) { const score = Number(value); return Number.isFinite(score) ? `${(score > 1 ? score : score * 100).toFixed(0)} / 100` : '—'; }
function creativeSummary(run) { const summary = run.summary || {}; return Object.entries(summary).slice(0, 3).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${value}`).join(' · ') || 'Creative run completed'; }

function bindFileInputs() {
  const input = document.getElementById('video-file');
  const zone = document.getElementById('drop-zone');
  const label = document.getElementById('file-label');
  if (!input || !zone) return;
  input.addEventListener('change', () => { if (input.files[0]) label.textContent = input.files[0].name; });
  ['dragenter', 'dragover'].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.remove('dragging'); }));
  zone.addEventListener('drop', (event) => { if (event.dataTransfer.files[0]) { input.files = event.dataTransfer.files; label.textContent = event.dataTransfer.files[0].name; } });
}

async function createProject(form) {
  state.error = '';
  const file = form.querySelector('#video-file')?.files[0];
  const name = form.querySelector('#project-name')?.value || '';
  const sourcePath = form.querySelector('#source-path')?.value || '';
  let options;
  if (file) {
    const body = new FormData();
    body.append('file', file);
    body.append('name', name);
    options = { method: 'POST', body };
  } else {
    options = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, source_path: sourcePath }) };
  }
  const button = form.querySelector('button[type="submit"]');
  if (button) { button.disabled = true; button.querySelector('span').textContent = 'Creating…'; }
  try {
    const detail = await api('/api/projects', options);
    state.projects = [detail.project, ...state.projects.filter((item) => item.project_id !== detail.project.project_id)];
    state.selectedId = detail.project.project_id;
    localStorage.setItem('vex-studio-project', state.selectedId);
    state.detail = detail;
    state.modal = null;
    state.view = 'studio';
    const url = new URL(window.location.href); url.searchParams.set('project', state.selectedId); window.history.replaceState({}, '', url);
    render();
  } catch (error) {
    state.error = error.message;
    render();
  }
}

async function sendChat(message) {
  if (!state.selectedId || state.taskId) return;
  state.draft = '';
  render();
  try {
    const task = await api(`/api/projects/${encodeURIComponent(state.selectedId)}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message }) });
    state.taskId = task.task_id;
    state.task = task;
    render();
    pollTask(task.task_id);
  } catch (error) { showError(error.message); }
}

async function pollTask(taskId) {
  try {
    const task = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
    if (state.taskId !== taskId) return;
    state.task = task;
    render();
    if (['queued', 'running'].includes(task.status)) {
      window.setTimeout(() => pollTask(taskId), 850);
      return;
    }
    await refreshProjects();
    await refreshDetail();
    state.taskId = '';
    render();
    if (task.status === 'failed') showError(task.error || task.message || 'The edit failed.');
  } catch (error) {
    if (state.taskId === taskId) { state.taskId = ''; state.task = null; showError(error.message); }
  }
}

document.addEventListener('click', (event) => {
  const nav = event.target.closest('[data-nav]');
  if (nav) { state.view = nav.dataset.nav; render(); return; }
  const projectButton = event.target.closest('[data-project]');
  if (projectButton) { selectProject(projectButton.dataset.project); return; }
  const promptButton = event.target.closest('[data-prompt]');
  if (promptButton) { state.draft = promptButton.dataset.prompt; render(); document.getElementById('prompt')?.focus(); return; }
  const action = event.target.closest('[data-action]');
  if (!action) return;
  if (action.dataset.action === 'new-project') { state.modal = 'new-project'; state.error = ''; render(); return; }
  if (action.dataset.action === 'close-modal') { state.modal = null; state.error = ''; render(); return; }
  if (action.dataset.action === 'refresh') { Promise.all([refreshProjects(), refreshDetail()]).then(render).catch((error) => showError(error.message)); }
});

document.addEventListener('submit', (event) => {
  event.preventDefault();
  if (event.target.matches('[data-form="new-project"]')) { createProject(event.target); return; }
  if (event.target.matches('[data-form="chat"]')) { const message = event.target.querySelector('textarea').value.trim(); if (message) sendChat(message); }
});

window.addEventListener('popstate', () => { const id = new URL(window.location.href).searchParams.get('project'); if (id) selectProject(id); });

const queryProject = new URL(window.location.href).searchParams.get('project');
if (queryProject) state.selectedId = queryProject;
boot();
