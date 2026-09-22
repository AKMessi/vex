const ICONS = {
  grid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="4" y="4" width="6" height="6"/><rect x="14" y="4" width="6" height="6"/><rect x="4" y="14" width="6" height="6"/><rect x="14" y="14" width="6" height="6"/></svg>',
  film: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="1"/><path d="m8 5 2 14M14 5l2 14M3 9h18M3 15h18"/></svg>',
  activity: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M4 16V8M8 19V5M12 14V10M16 18V6M20 12v-2"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Z"/><path d="m19.4 15 .1.1a1.6 1.6 0 0 1-2.3 2.3l-.1-.1a1.6 1.6 0 0 0-2.7 1.1v.2a1.6 1.6 0 0 1-3.2 0v-.2a1.6 1.6 0 0 0-2.7-1.1l-.1.1a1.6 1.6 0 1 1-2.3-2.3l.1-.1A1.6 1.6 0 0 0 5.1 12a1.6 1.6 0 0 0-1.1-2.7h-.2a1.6 1.6 0 0 1 0-3.2H4A1.6 1.6 0 0 0 5.1 3.4L5 3.3A1.6 1.6 0 1 1 7.3 1l.1.1A1.6 1.6 0 0 0 10.1 0h.2a1.6 1.6 0 0 1 3.2 0h.2a1.6 1.6 0 0 0 2.7 1.1l.1-.1A1.6 1.6 0 1 1 18.8 3l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.2a1.6 1.6 0 0 1 0 3.2h-.2a1.6 1.6 0 0 0-1.1 2.7l.1.1a1.6 1.6 0 0 1 .5 1.2" transform="translate(2 2) scale(.83)"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M5 12h13M13 6l6 6-6 6"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M20 11a8 8 0 0 0-14.7-4L4 9M4 5v4h4M4 13a8 8 0 0 0 14.7 4L20 15m0 4v-4h-4"/></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="m12 3 1.2 5.8L19 10l-5.8 1.2L12 17l-1.2-5.8L5 10l5.8-1.2L12 3ZM19 16l.6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z"/></svg>',
};

const STORAGE_KEY = 'vex-studio-project';
const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', 'flv']);
const ACTIVE_TASK_STATUSES = new Set(['queued', 'running']);

function storageGet(key) {
  try { return window.localStorage.getItem(key) || ''; } catch { return ''; }
}

function storageSet(key, value) {
  try {
    if (value) window.localStorage.setItem(key, value);
    else window.localStorage.removeItem(key);
  } catch { /* Storage may be disabled in locked-down browser contexts. */ }
}

const state = {
  view: 'studio',
  projects: [],
  detail: null,
  health: null,
  selectedId: storageGet(STORAGE_KEY),
  modal: null,
  draft: '',
  taskId: '',
  task: null,
  loading: true,
  loadingDetail: false,
  refreshing: false,
  submitting: false,
  creatingProject: false,
  error: '',
  notice: '',
};

const app = document.getElementById('app');
let detailSequence = 0;
let pollGeneration = 0;
let pollTimer = 0;
let toastTimer = 0;
let createController = null;
let modalReturnAction = '';

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function icon(name) { return ICONS[name] || ''; }
function project() { return state.detail?.project || null; }
function array(value) { return Array.isArray(value) ? value : []; }
function isBusy() { return state.submitting || ACTIVE_TASK_STATUSES.has(state.task?.status); }

function statusClass(value) {
  return ['success', 'running', 'error'].includes(value) ? value : '';
}

function renderToastRegion() {
  const region = document.getElementById('toast-region');
  if (!region) return;
  const message = state.error || state.notice;
  region.innerHTML = message
    ? `<div class="toast ${state.error ? 'error' : 'success'}" role="status">${esc(message)}</div>`
    : '';
}

function showTransient(kind, message, duration = 5500) {
  window.clearTimeout(toastTimer);
  state.error = kind === 'error' ? String(message || 'Something went wrong.') : '';
  state.notice = kind === 'notice' ? String(message || '') : '';
  renderToastRegion();
  toastTimer = window.setTimeout(() => {
    state.error = '';
    state.notice = '';
    renderToastRegion();
  }, duration);
}

function showError(message) { showTransient('error', message); }
function showNotice(message) { showTransient('notice', message, 3500); }

async function api(path, options = {}) {
  const { timeout = 30_000, signal: externalSignal, ...fetchOptions } = options;
  const controller = new AbortController();
  let timedOut = false;
  const abortFromCaller = () => controller.abort();
  externalSignal?.addEventListener('abort', abortFromCaller, { once: true });
  if (externalSignal?.aborted) controller.abort();
  const timeoutTimer = timeout > 0 ? window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeout) : 0;
  let response;
  let text;
  try {
    response = await fetch(path, {
      cache: 'no-store',
      credentials: 'same-origin',
      ...fetchOptions,
      signal: controller.signal,
    });
    text = await response.text();
  } catch (error) {
    if (error?.name === 'AbortError' && !timedOut) throw error;
    if (timedOut) throw new Error('The local Vex server took too long to respond.');
    throw new Error('Vex Studio is not reachable. Check that the local server is still running.');
  } finally {
    window.clearTimeout(timeoutTimer);
    externalSignal?.removeEventListener('abort', abortFromCaller);
  }
  let payload = {};
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = {}; }
  }
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setSelectedProject(projectId) {
  state.selectedId = projectId || '';
  storageSet(STORAGE_KEY, state.selectedId);
}

function updateProjectUrl(projectId, mode = 'replace') {
  const url = new URL(window.location.href);
  if (projectId) url.searchParams.set('project', projectId);
  else url.searchParams.delete('project');
  const method = mode === 'push' ? 'pushState' : 'replaceState';
  window.history[method]({}, '', url);
}

async function refreshProjects() {
  const payload = await api('/api/projects');
  state.projects = array(payload.projects);
  if (!state.selectedId && state.projects.length) setSelectedProject(state.projects[0].project_id);
  if (state.selectedId && !state.projects.some((item) => item.project_id === state.selectedId)) {
    setSelectedProject(state.projects[0]?.project_id || '');
  }
  updateProjectUrl(state.selectedId, 'replace');
}

async function refreshHealth() {
  state.health = await api('/api/health');
}

async function refreshDetail() {
  const projectId = state.selectedId;
  const sequence = ++detailSequence;
  if (!projectId) {
    state.detail = null;
    return null;
  }
  const detail = await api(`/api/projects/${encodeURIComponent(projectId)}`);
  if (sequence !== detailSequence || projectId !== state.selectedId) return null;
  state.detail = detail;
  if (!state.taskId) state.task = detail.active_task || detail.latest_task || null;
  return detail.active_task || null;
}

async function boot() {
  const failures = [];
  const [healthResult, projectsResult] = await Promise.allSettled([refreshHealth(), refreshProjects()]);
  if (healthResult.status === 'rejected') failures.push(healthResult.reason.message);
  if (projectsResult.status === 'rejected') failures.push(projectsResult.reason.message);
  let activeTask = null;
  if (projectsResult.status === 'fulfilled' && state.selectedId) {
    try { activeTask = await refreshDetail(); } catch (error) { failures.push(error.message); }
  }
  state.loading = false;
  render();
  if (activeTask) startPolling(activeTask);
  if (failures.length) showError([...new Set(failures)].join(' '));
}

function stopPolling() {
  pollGeneration += 1;
  window.clearTimeout(pollTimer);
  pollTimer = 0;
}

function startPolling(task) {
  if (!task?.task_id) return;
  stopPolling();
  state.taskId = task.task_id;
  state.task = task;
  renderTaskStatus();
  const generation = pollGeneration;
  pollTask(task.task_id, generation, 0);
}

function schedulePoll(taskId, generation, failures) {
  const delay = failures ? Math.min(10_000, 900 * (2 ** Math.min(failures, 4))) : 900;
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(() => pollTask(taskId, generation, failures), delay);
}

async function pollTask(taskId, generation, failures) {
  if (generation !== pollGeneration || state.taskId !== taskId) return;
  try {
    const task = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
    if (generation !== pollGeneration || state.taskId !== taskId) return;
    state.task = task;
    renderTaskStatus();
    if (ACTIVE_TASK_STATUSES.has(task.status)) {
      schedulePoll(taskId, generation, 0);
      return;
    }

    state.taskId = '';
    const projectId = state.selectedId;
    const results = [];
    try {
      await refreshProjects();
      results.push({ status: 'fulfilled' });
      await refreshDetail();
      results.push({ status: 'fulfilled' });
    } catch (error) {
      results.push({ status: 'rejected', reason: error });
    }
    if (generation !== pollGeneration || projectId !== state.selectedId) return;
    render();
    const refreshFailure = results.find((result) => result.status === 'rejected');
    if (refreshFailure) showError(refreshFailure.reason.message);
    if (task.status === 'failed') showError(task.error || task.message || 'The edit failed.');
    else showNotice(task.message || 'Edit completed.');
  } catch (error) {
    if (generation !== pollGeneration || state.taskId !== taskId) return;
    if (error.status === 404) {
      state.taskId = '';
      state.task = null;
      renderTaskStatus();
      showError('The task record is unavailable. Refresh the project before retrying.');
      return;
    }
    const nextFailures = failures + 1;
    if (nextFailures === 3) showError('Connection interrupted. Vex is still retrying the active edit.');
    schedulePoll(taskId, generation, nextFailures);
  }
}

async function selectProject(projectId, historyMode = 'push') {
  if (!projectId || (projectId === state.selectedId && state.detail)) {
    state.view = 'studio';
    render();
    return;
  }
  stopPolling();
  state.taskId = '';
  state.task = null;
  state.detail = null;
  state.view = 'studio';
  state.loadingDetail = true;
  setSelectedProject(projectId);
  if (historyMode !== 'none') updateProjectUrl(projectId, historyMode);
  render();
  try {
    const activeTask = await refreshDetail();
    state.loadingDetail = false;
    render();
    if (activeTask) startPolling(activeTask);
  } catch (error) {
    state.loadingDetail = false;
    render();
    showError(error.message);
  }
}

function navItem(view, label, iconName) {
  return `<button type="button" class="nav-item ${state.view === view ? 'active' : ''}" data-nav="${view}" ${state.view === view ? 'aria-current="page"' : ''}>${icon(iconName)}<span>${label}</span></button>`;
}

function renderSidebar() {
  const health = state.health || {};
  return `<aside class="sidebar">
    <div class="brand"><div class="brand-symbol">VX</div><div class="brand-copy"><span class="brand-name">Vex Studio</span><span class="brand-sub">Local video intelligence</span></div></div>
    <button type="button" class="new-project" data-action="new-project">${icon('plus')}<span>New project</span></button>
    <div class="nav-label">Workspace</div>
    <nav class="nav-list" aria-label="Workspace">
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
  return `<header class="topbar"><div class="breadcrumb"><strong>${state.view === 'studio' ? 'Studio' : state.view[0].toUpperCase() + state.view.slice(1)}</strong>${current && state.view === 'studio' ? `<span class="breadcrumb-sep">/</span><span>${esc(current.project_name)}</span>` : ''}</div><div class="top-actions"><button type="button" class="icon-btn ${state.refreshing ? 'spinning' : ''}" data-action="refresh" title="Refresh" aria-label="Refresh" ${state.refreshing ? 'disabled' : ''}>${icon('refresh')}</button><button type="button" class="icon-btn" data-action="new-project" title="New project" aria-label="New project">${icon('plus')}</button></div></header>`;
}

function renderHeading(eyebrow, title, copy = '') {
  return `<div class="view-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1></div>${copy ? `<p class="heading-copy">${copy}</p>` : ''}</div>`;
}

function renderVideoCard() {
  const current = project();
  const media = state.detail?.media;
  return `<section class="card video-card"><div class="video-frame">${media?.available ? `<div class="video-overlay-label"><span class="status-dot"></span>Current working cut</div><video id="working-video" controls preload="metadata" src="${esc(media.current)}" aria-label="Current working cut"></video>` : `<div class="video-empty"><div class="video-empty-icon">${icon('film')}</div><p>Your working cut will appear here once you load a video into Vex.</p></div>`}</div>${current ? `<div class="media-meta"><div class="media-meta-item"><span class="meta-label">Duration</span><span class="meta-value">${esc(current.duration)}</span></div><div class="media-meta-item"><span class="meta-label">Frame</span><span class="meta-value">${esc(current.resolution)}</span></div><div class="media-meta-item"><span class="meta-label">Frame rate</span><span class="meta-value">${esc(current.fps)} fps</span></div><div class="media-meta-item"><span class="meta-label">Source</span><span class="meta-value" title="${esc(current.source_name)}">${esc(current.source_name)}</span></div></div>` : ''}</section>`;
}

function renderComposer() {
  const busy = isBusy();
  const prompts = ['Clean up the pauses and filler words', 'Give this a natural, cinematic grade', 'Add clean, readable captions', 'Find the strongest moments for shorts'];
  const status = state.submitting ? 'Starting your edit…' : busy ? esc(state.task?.message || 'Working…') : 'Plain English is the interface';
  return `<section class="card composer-card"><div class="composer-top"><h3>Tell Vex what to change</h3><span id="composer-status" aria-live="polite">${status}</span></div><form class="composer" data-form="chat"><label class="sr-only" for="prompt">Editing instruction</label><textarea id="prompt" name="message" maxlength="12000" placeholder="Try: remove the awkward intro, add captions, and make it feel more cinematic…" ${busy ? 'disabled' : ''}>${esc(state.draft)}</textarea><button class="send-btn" type="submit" ${busy ? 'disabled' : ''} aria-label="Send instruction">${icon('arrow')}</button></form><div class="quick-actions">${prompts.map((prompt) => `<button type="button" class="quick-action" data-prompt="${esc(prompt)}" ${busy ? 'disabled' : ''}>${esc(prompt)}</button>`).join('')}</div></section>`;
}

function renderTimeline() {
  const timeline = array(state.detail?.timeline);
  const labels = timeline.length ? timeline.slice(0, 10).map((item) => item.op) : Array.from({ length: 10 }, () => '');
  return `<section class="card timeline-card"><div class="timeline-headline"><span>Timeline</span><span>${timeline.length ? `${timeline.length} operation${timeline.length === 1 ? '' : 's'}` : 'No edits yet'}</span></div><div class="timeline-strip">${labels.map((label, index) => `<div class="timeline-segment"><span>${esc(label || `scene ${String(index + 1).padStart(2, '0')}`)}</span></div>`).join('')}</div></section>`;
}

function runOutput() {
  const stream = String(state.task?.stream || '').trim();
  const message = String(state.task?.result?.message || '').trim();
  const output = stream || message;
  if (!output) return '';
  const clipped = output.length > 1200 ? `…${output.slice(-1200)}` : output;
  return `<div class="run-output"><div class="card-kicker">Agent response</div><p>${esc(clipped)}</p></div>`;
}

function renderTrace() {
  const taskEvents = array(state.task?.events);
  const traceEvents = array(state.detail?.latest_trace?.events);
  const events = taskEvents.length ? taskEvents : traceEvents;
  const recent = events.slice(-7).reverse();
  const active = ACTIVE_TASK_STATUSES.has(state.task?.status);
  return `<section class="card inspector-card" id="trace-card" aria-live="polite"><div class="trace-head"><h3>${active ? 'Live run' : 'Latest run'}</h3><span class="card-kicker">${recent.length ? `${recent.length} steps` : 'Quiet'}</span></div>${recent.length ? `<div class="trace-list">${recent.map((event) => `<div class="trace-item"><span class="trace-marker ${statusClass(event.status)}"></span><div class="trace-copy"><span class="trace-title">${esc(event.title || event.kind || 'Update')}</span><span class="trace-detail" title="${esc(event.detail || '')}">${esc(event.detail || '')}</span></div></div>`).join('')}</div>` : `<p class="empty-state">Vex will show the plan, tools, and QA as soon as you run an instruction.</p>`}${runOutput()}</section>`;
}

function renderInspector() {
  const current = project();
  if (!current) return `<aside class="inspector-column"><section class="card inspector-card"><div class="project-title-row"><div class="project-title"><div class="eyebrow">Start here</div><h2>Load a project</h2><p>Bring in a video to open the Studio.</p></div><div class="project-badge">VX</div></div><button type="button" class="primary-btn inspector-primary" data-action="new-project">${icon('upload')}<span>Choose a video</span></button></section></aside>`;
  const artifacts = array(state.detail?.artifacts);
  return `<aside class="inspector-column"><section class="card inspector-card"><div class="project-title-row"><div class="project-title"><h2 title="${esc(current.project_name)}">${esc(current.project_name)}</h2><p title="${esc(current.source_name)}">${esc(current.source_name)}</p></div><div class="project-badge">VX</div></div><div class="stat-grid"><div class="stat"><span class="stat-value">${esc(current.timeline_ops)}</span><span class="stat-label">Edits</span></div><div class="stat"><span class="stat-value">${esc(current.duration)}</span><span class="stat-label">Runtime</span></div><div class="stat"><span class="stat-value">${esc(current.resolution)}</span><span class="stat-label">Frame</span></div><div class="stat"><span class="stat-value">${esc(current.provider)}</span><span class="stat-label">Provider</span></div></div>${artifacts.length ? `<div class="section-rule"></div><div class="card-kicker">Project outputs</div><div class="trace-list artifact-list">${artifacts.slice(0, 5).map((item) => `<div class="trace-item"><span class="trace-marker success"></span><div class="trace-copy"><span class="trace-title">${esc(item.label)}</span><span class="trace-detail">${esc(item.summary)}</span></div></div>`).join('')}</div>` : ''}</section>${renderTrace()}</aside>`;
}

function projectCard(item) {
  return `<button type="button" class="project-card ${item.project_id === state.selectedId ? 'active' : ''}" data-project="${esc(item.project_id)}"><div class="project-thumb"><span class="project-thumb-label">${esc(item.resolution || 'Video')}</span></div><div class="project-card-title" title="${esc(item.project_name || item.source_name)}">${esc(item.project_name || item.source_name)}</div><div class="project-card-meta"><span>${esc(item.duration || '—')} · ${esc(item.timeline_ops || 0)} edits</span><span>${esc(item.updated_label || 'recently')}</span></div></button>`;
}

function renderProjectCards(limit = null) {
  const projects = limit ? state.projects.slice(0, limit) : state.projects;
  return projects.map(projectCard).join('');
}

function renderStudio() {
  const current = project();
  if (state.loadingDetail) return `${renderHeading('Studio / edit intelligence', 'Opening your<br /><em>working cut.</em>', 'Loading project state from this machine.')}<div class="loading-panel"><span class="loading-line"></span><span class="loading-line short"></span></div>`;
  return `${renderHeading('Studio / edit intelligence', current ? `Make ${esc(current.project_name)}<br /><em>move.</em>` : 'Edit with<br /><em>intention.</em>', current ? 'Your media, timeline, and creative history in one quiet workspace. Ask for the cut you want; Vex handles the machinery.' : 'A local-first studio for the work between “I have footage” and “I have something worth watching.”')}
    ${current ? `<div class="studio-grid"><div class="primary-column">${renderVideoCard()}${renderComposer()}${renderTimeline()}</div>${renderInspector()}</div><div class="section-heading"><div><h2>Recent projects</h2><p>Pick up where you left off.</p></div><button type="button" class="secondary-btn" data-nav="projects">View all ${icon('arrow')}</button></div><div class="project-grid">${renderProjectCards(3)}</div>` : `<div class="empty-library"><div class="video-empty-icon">${icon('spark')}</div><h2>Start with a frame.</h2><p>Load a local video and Vex will give you a working cut, a safe project copy, and a place to direct the edit in plain English.</p><button type="button" class="primary-btn" data-action="new-project">${icon('upload')}<span>Load your first video</span></button></div>`}`;
}

function renderProjects() {
  return `${renderHeading('Library / projects', 'A home for<br /><em>the good cuts.</em>', 'Every Vex project stays local, resumable, and inspectable. Your source file is never touched.')}${state.projects.length ? `<div class="library-grid">${renderProjectCards()}</div>` : `<div class="empty-library"><h2>No projects yet.</h2><p>Load a video to create a safe, resumable Vex project.</p><button type="button" class="primary-btn" data-action="new-project">${icon('plus')}<span>New project</span></button></div>`}`;
}

function renderActivity() {
  const current = project();
  const timeline = array(state.detail?.timeline);
  const runs = array(state.detail?.creative_runs);
  return `${renderHeading('Activity / history', 'Everything Vex<br /><em>has done.</em>', 'A readable record of edits, creative runs, and the decisions behind your current working cut.')}${current ? `<div class="section-heading section-heading-flush"><div><h2>${esc(current.project_name)}</h2><p>${esc(current.timeline_ops)} timeline operation${current.timeline_ops === 1 ? '' : 's'} · ${esc(current.updated_label)}</p></div><button type="button" class="secondary-btn" data-nav="studio">Back to studio ${icon('arrow')}</button></div>` : ''}<section class="card activity-card"><div class="activity-list">${timeline.length ? timeline.map((item, index) => `<div class="activity-row"><div class="activity-index">${String(index + 1).padStart(2, '0')}</div><div class="activity-copy"><span class="activity-title">${esc(item.op)}</span><span class="activity-detail">${esc(item.detail)}</span></div><span class="activity-time">${esc(formatTimestamp(item.timestamp))}</span></div>`).join('') : '<p class="empty-state">No edit history for this project yet.</p>'}</div></section>${runs.length ? `<div class="section-heading"><div><h2>Creative runs</h2><p>Quality-gated automation recorded by Vex.</p></div></div><section class="card activity-card"><div class="activity-list">${runs.map((run) => `<div class="activity-row"><div class="activity-index">${icon('spark')}</div><div class="activity-copy"><span class="activity-title">${esc(run.feature || 'Creative run')}</span><span class="activity-detail">${esc(creativeSummary(run))}</span></div><span class="quality-pill">${esc(formatScore(run.quality_score))}</span></div>`).join('')}</div></section>` : ''}`;
}

function renderSettings() {
  const health = state.health || {};
  const mediaReady = Boolean(health.media_stack_ready);
  return `${renderHeading('Settings / local runtime', 'Quiet control<br /><em>over the stack.</em>', 'Vex is designed to stay close to your machine. These are the active local runtime signals, not a cloud account page.')}
    <div class="settings-grid"><section class="card settings-panel"><h2>Runtime</h2><p>Configuration is read from your local Vex environment and project state directory.</p><div class="settings-list"><div class="setting-row"><div class="setting-label"><span class="setting-name">Provider</span><span class="setting-note">Used when a request needs model planning.</span></div><span class="setting-value">${esc(health.provider || '—')}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Model</span><span class="setting-note">Configured model name.</span></div><span class="setting-value">${esc(health.model || '—')}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Media stack</span><span class="setting-note">FFmpeg and ffprobe availability.</span></div><span class="setting-value ${mediaReady ? '' : 'unavailable'}">${mediaReady ? 'ready' : 'needs setup'}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Projects</span><span class="setting-note">Where Vex keeps project working copies.</span></div><span class="setting-value" title="${esc(health.projects_dir)}">local</span></div></div></section><aside class="primary-column"><section class="card settings-panel"><div class="settings-note">The Studio is served by Vex itself at localhost. No footage leaves this machine unless a tool explicitly uses a provider or stock-media API you have configured.</div></section><section class="card settings-panel"><div class="card-kicker">Build</div><h2 class="build-version">Vex ${esc(health.version || '')}</h2><p>Terminal-first editing, now with a visual command center.</p></section></aside></div>`;
}

function renderModal() {
  if (state.modal !== 'new-project') return '';
  return `<div class="modal-backdrop" data-action="close-modal"><section class="modal" data-modal-content role="dialog" aria-modal="true" aria-labelledby="new-project-title" tabindex="-1"><div class="modal-header"><div><h2 id="new-project-title">New project</h2><p>Bring in a local video. Vex makes a safe working copy before anything changes.</p></div><button type="button" class="icon-btn" data-action="close-modal" aria-label="Close">${icon('close')}</button></div><form class="modal-body" data-form="new-project"><label class="drop-zone" id="drop-zone"><input id="video-file" type="file" name="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,video/*" /><span class="video-empty-icon">${icon('upload')}</span><strong id="file-label">Drop a video here or choose a file</strong><span>MP4, MOV, AVI, WEBM, MKV, M4V, FLV</span></label><div class="field"><label for="project-name">Project name <span class="muted">(optional)</span></label><input id="project-name" name="name" placeholder="e.g. Product film / episode 04" maxlength="120" /></div><div class="field"><label for="source-path">Or use a local path</label><input id="source-path" name="source_path" placeholder="/Users/you/Videos/episode-04.mov" /></div><div id="modal-error" class="error-note" role="alert" hidden></div><div class="modal-actions"><button class="secondary-btn" type="button" data-action="close-modal">Cancel</button><button class="primary-btn" type="submit">${icon('arrow')}<span>Create project</span></button></div></form></section></div>`;
}

function renderTaskStatus() {
  const trace = document.getElementById('trace-card');
  if (trace) trace.outerHTML = renderTrace();
  const busy = isBusy();
  const status = document.getElementById('composer-status');
  if (status) status.textContent = state.submitting ? 'Starting your edit…' : busy ? state.task?.message || 'Working…' : 'Plain English is the interface';
  const textarea = document.getElementById('prompt');
  if (textarea) textarea.disabled = busy;
  document.querySelectorAll('.send-btn, .quick-action').forEach((button) => { button.disabled = busy; });
}

function render() {
  if (state.loading) return;
  const content = state.view === 'projects' ? renderProjects() : state.view === 'activity' ? renderActivity() : state.view === 'settings' ? renderSettings() : renderStudio();
  app.innerHTML = `${renderSidebar()}<main class="main">${renderTopbar()}<div class="main-content">${content}</div></main>${renderModal()}<div id="toast-region" aria-live="polite" aria-atomic="true"></div>`;
  bindFileInputs();
  bindPrompt();
  renderToastRegion();
  document.title = project() ? `${project().project_name} — Vex Studio` : 'Vex Studio';
}

function formatTimestamp(timestamp) {
  if (!timestamp) return '—';
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? '—' : new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit' }).format(date);
}

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? `${(score > 1 ? score : score * 100).toFixed(0)} / 100` : '—';
}

function displayValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function creativeSummary(run) {
  const summary = run?.summary && typeof run.summary === 'object' ? run.summary : {};
  return Object.entries(summary).slice(0, 3).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${displayValue(value)}`).join(' · ') || 'Creative run completed';
}

function bindPrompt() {
  const prompt = document.getElementById('prompt');
  if (!prompt || state.view !== 'studio') return;
  prompt.addEventListener('input', (event) => { state.draft = event.target.value; });
  prompt.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      event.target.form.requestSubmit();
    }
  });
}

function acceptedVideo(file) {
  const extension = String(file?.name || '').split('.').pop().toLowerCase();
  return VIDEO_EXTENSIONS.has(extension);
}

function setModalError(message) {
  const error = document.getElementById('modal-error');
  if (!error) return;
  error.textContent = String(message || 'Unable to create the project.');
  error.hidden = false;
}

function bindFileInputs() {
  const input = document.getElementById('video-file');
  const zone = document.getElementById('drop-zone');
  const label = document.getElementById('file-label');
  if (!input || !zone || !label) return;
  input.addEventListener('change', () => {
    if (!input.files[0]) return;
    if (!acceptedVideo(input.files[0])) {
      input.value = '';
      setModalError('Choose an MP4, MOV, AVI, WEBM, MKV, M4V, or FLV video.');
      return;
    }
    label.textContent = input.files[0].name;
    document.getElementById('modal-error').hidden = true;
  });
  ['dragenter', 'dragover'].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.remove('dragging');
  }));
  zone.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file || !acceptedVideo(file)) {
      setModalError('Choose an MP4, MOV, AVI, WEBM, MKV, M4V, or FLV video.');
      return;
    }
    try {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      label.textContent = file.name;
      document.getElementById('modal-error').hidden = true;
    } catch {
      setModalError('This browser could not attach the dropped file. Use the file picker instead.');
    }
  });
}

function openProjectModal() {
  modalReturnAction = document.activeElement?.dataset?.action || '';
  state.modal = 'new-project';
  render();
  window.requestAnimationFrame(() => document.querySelector('[data-modal-content]')?.focus());
}

function closeProjectModal() {
  const controller = createController;
  createController = null;
  if (controller) controller.abort();
  state.modal = null;
  state.creatingProject = false;
  render();
  window.requestAnimationFrame(() => {
    const selector = modalReturnAction ? `[data-action="${modalReturnAction}"]` : '[data-action="new-project"]';
    document.querySelector(selector)?.focus();
    modalReturnAction = '';
  });
}

async function createProject(form) {
  if (state.creatingProject) return;
  const file = form.querySelector('#video-file')?.files[0];
  const name = form.querySelector('#project-name')?.value.trim() || '';
  const sourcePath = form.querySelector('#source-path')?.value.trim() || '';
  if (!file && !sourcePath) {
    setModalError('Choose a video file or provide a local path.');
    return;
  }
  if (file && !acceptedVideo(file)) {
    setModalError('Choose an MP4, MOV, AVI, WEBM, MKV, M4V, or FLV video.');
    return;
  }

  const controller = new AbortController();
  createController = controller;
  state.creatingProject = true;
  form.setAttribute('aria-busy', 'true');
  const button = form.querySelector('button[type="submit"]');
  if (button) {
    button.disabled = true;
    button.querySelector('span').textContent = file ? 'Importing…' : 'Creating…';
  }
  let options;
  if (file) {
    const body = new FormData();
    body.append('file', file);
    body.append('name', name);
    options = { method: 'POST', body, signal: controller.signal, timeout: 0 };
  } else {
    options = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, source_path: sourcePath }), signal: controller.signal, timeout: 0 };
  }

  try {
    const detail = await api('/api/projects', options);
    state.detail = detail;
    setSelectedProject(detail.project.project_id);
    updateProjectUrl(state.selectedId, 'push');
    state.projects = [detail.project, ...state.projects.filter((item) => item.project_id !== detail.project.project_id)];
    state.modal = null;
    state.view = 'studio';
    try { await refreshProjects(); } catch { /* The newly created detail is still usable. */ }
    render();
    showNotice('Project created. Your source is safe in a local working copy.');
  } catch (error) {
    if (error?.name !== 'AbortError' && state.modal) setModalError(error.message);
  } finally {
    if (createController === controller) {
      state.creatingProject = false;
      createController = null;
      if (state.modal && form.isConnected) {
        form.removeAttribute('aria-busy');
        if (button) {
          button.disabled = false;
          button.querySelector('span').textContent = 'Create project';
        }
      }
    }
  }
}

async function sendChat(message) {
  if (!state.selectedId || isBusy()) return;
  state.submitting = true;
  renderTaskStatus();
  try {
    const task = await api(`/api/projects/${encodeURIComponent(state.selectedId)}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    state.draft = '';
    const prompt = document.getElementById('prompt');
    if (prompt) prompt.value = '';
    state.submitting = false;
    startPolling(task);
  } catch (error) {
    state.submitting = false;
    renderTaskStatus();
    showError(error.message);
  }
}

async function refreshAll() {
  if (state.refreshing) return;
  state.refreshing = true;
  render();
  const failures = [];
  const [healthResult, projectsResult] = await Promise.allSettled([refreshHealth(), refreshProjects()]);
  if (healthResult.status === 'rejected') failures.push(healthResult.reason.message);
  if (projectsResult.status === 'rejected') failures.push(projectsResult.reason.message);
  let activeTask = null;
  if (projectsResult.status === 'fulfilled') {
    try { activeTask = await refreshDetail(); } catch (error) { failures.push(error.message); }
  }
  state.refreshing = false;
  render();
  if (activeTask && activeTask.task_id !== state.taskId) startPolling(activeTask);
  if (failures.length) showError([...new Set(failures)].join(' '));
  else showNotice('Studio refreshed.');
}

document.addEventListener('click', (event) => {
  const nav = event.target.closest('[data-nav]');
  if (nav) {
    state.view = nav.dataset.nav;
    render();
    return;
  }
  const projectButton = event.target.closest('[data-project]');
  if (projectButton) {
    selectProject(projectButton.dataset.project);
    return;
  }
  const promptButton = event.target.closest('[data-prompt]');
  if (promptButton) {
    state.draft = promptButton.dataset.prompt;
    const prompt = document.getElementById('prompt');
    if (prompt) {
      prompt.value = state.draft;
      prompt.focus();
      prompt.setSelectionRange(prompt.value.length, prompt.value.length);
    }
    return;
  }
  const action = event.target.closest('[data-action]');
  if (!action) return;
  if (action.dataset.action === 'new-project') {
    openProjectModal();
    return;
  }
  if (action.dataset.action === 'close-modal') {
    if (action.classList.contains('modal-backdrop') && event.target !== action) return;
    closeProjectModal();
    return;
  }
  if (action.dataset.action === 'refresh') refreshAll();
});

document.addEventListener('submit', (event) => {
  event.preventDefault();
  if (event.target.matches('[data-form="new-project"]')) {
    createProject(event.target);
    return;
  }
  if (event.target.matches('[data-form="chat"]')) {
    const message = event.target.querySelector('textarea').value.trim();
    if (message) sendChat(message);
  }
});

document.addEventListener('keydown', (event) => {
  if (!state.modal) return;
  if (event.key === 'Escape') {
    event.preventDefault();
    closeProjectModal();
    return;
  }
  if (event.key !== 'Tab') return;
  const modal = document.querySelector('[data-modal-content]');
  const focusable = [...modal.querySelectorAll('button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])')];
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});

window.addEventListener('popstate', () => {
  const id = new URL(window.location.href).searchParams.get('project') || state.projects[0]?.project_id || '';
  if (id) selectProject(id, 'none');
});

window.addEventListener('online', () => {
  showNotice('Connection to the local Vex server restored.');
  if (state.taskId) schedulePoll(state.taskId, pollGeneration, 0);
});

window.addEventListener('offline', () => showError('The browser is offline. Vex will reconnect when the local connection returns.'));

document.addEventListener('visibilitychange', () => {
  if (!document.hidden && state.taskId) schedulePoll(state.taskId, pollGeneration, 0);
});

const queryProject = new URL(window.location.href).searchParams.get('project');
if (queryProject) setSelectedProject(queryProject);
boot();
