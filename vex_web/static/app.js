const ICONS = {
  grid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="4" y="4" width="6" height="6"/><rect x="14" y="4" width="6" height="6"/><rect x="4" y="14" width="6" height="6"/><rect x="14" y="14" width="6" height="6"/></svg>',
  film: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="1"/><path d="m8 5 2 14M14 5l2 14M3 9h18M3 15h18"/></svg>',
  activity: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M4 16V8M8 19V5M12 14V10M16 18V6M20 12v-2"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Z"/><path d="m19.4 15 .1.1a1.6 1.6 0 0 1-2.3 2.3l-.1-.1a1.6 1.6 0 0 0-2.7 1.1v.2a1.6 1.6 0 0 1-3.2 0v-.2a1.6 1.6 0 0 0-2.7-1.1l-.1.1a1.6 1.6 0 1 1-2.3-2.3l.1-.1A1.6 1.6 0 0 0 5.1 12a1.6 1.6 0 0 0-1.1-2.7h-.2a1.6 1.6 0 0 1 0-3.2H4A1.6 1.6 0 0 0 5.1 3.4L5 3.3A1.6 1.6 0 1 1 7.3 1l.1.1A1.6 1.6 0 0 0 10.1 0h.2a1.6 1.6 0 0 1 3.2 0h.2a1.6 1.6 0 0 0 2.7 1.1l.1-.1A1.6 1.6 0 1 1 18.8 3l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.2a1.6 1.6 0 0 1 0 3.2h-.2a1.6 1.6 0 0 0-1.1 2.7l.1.1a1.6 1.6 0 0 1 .5 1.2" transform="translate(2 2) scale(.83)"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M5 12h13M13 6l6 6-6 6"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><circle cx="10.5" cy="10.5" r="5.5"/><path d="m15 15 4 4"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M12 19V5m-6 6 6-6 6 6"/></svg>',
  monitor: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/></svg>',
  theme: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M20 15A8 8 0 0 1 9 4a8 8 0 1 0 11 11Z"/></svg>',
  undo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M8 6l-4 4 4 4m-4-4h10a5 5 0 0 1 0 10h-3"/></svg>',
  redo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M16 6l4 4-4 4m4-4H10a5 5 0 0 0 0 10h3"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M20 11a8 8 0 0 0-14.7-4L4 9M4 5v4h4M4 13a8 8 0 0 0 14.7 4L20 15m0 4v-4h-4"/></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="m12 3 1.2 5.8L19 10l-5.8 1.2L12 17l-1.2-5.8L5 10l5.8-1.2L12 3ZM19 16l.6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z"/></svg>',
};

const STORAGE_KEY = 'vex-studio-project';
const THEME_KEY = 'vex-studio-theme';
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

function setTheme(theme) {
  document.documentElement.dataset.theme = theme === 'dark' ? 'dark' : 'light';
}

setTheme(storageGet(THEME_KEY) || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));

const state = {
  view: 'create',
  projects: [],
  detail: null,
  health: null,
  selectedId: storageGet(STORAGE_KEY),
  modal: null,
  draft: '',
  drafts: new Map(),
  pendingEdit: '',
  mediaKind: 'current',
  browserPreview: false,
  production: null,
  productionBusy: false,
  productionDraft: {prompt: '', title: '', duration_sec: 24, aspect: 'landscape', renderer: 'auto', generate_audio: true},
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
let productionTimer = 0;
let productionSequence = 0;
let toastTimer = 0;
let createController = null;
let modalReturnAction = '';
let playerScript;
let unmountLive;

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
  if (mode === 'push') url.searchParams.delete('view');
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
  const initialId = state.selectedId;
  const view = new URL(window.location.href).searchParams.get('view');
  const [healthResult, projectsResult, productionResult] = await Promise.allSettled([
    refreshHealth(), refreshProjects(), api('/api/production'),
  ]);
  if (healthResult.status === 'rejected') failures.push(healthResult.reason.message);
  if (projectsResult.status === 'rejected') failures.push(projectsResult.reason.message);
  let activeTask = null;
  if (projectsResult.status === 'fulfilled' && state.selectedId) {
    try { activeTask = await refreshDetail(); } catch (error) { failures.push(error.message); }
  }
  if (productionResult.status === 'fulfilled') {
    state.production = productionResult.value.active || productionResult.value.latest;
    if (productionResult.value.active?.request) {
      state.productionDraft = {...state.productionDraft, ...productionResult.value.active.request};
    }
  }
  if (initialId && state.detail && view !== 'create') state.view = 'studio';
  if (state.production && ACTIVE_TASK_STATUSES.has(state.production.status) && !initialId) state.view = 'create';
  state.loading = false;
  render();
  if (activeTask) startPolling(activeTask);
  if (state.production && ACTIVE_TASK_STATUSES.has(state.production.status)) watchProduction(state.production.task_id);
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
    if (projectId && historyMode !== 'none') updateProjectUrl(projectId, historyMode);
    render();
    return;
  }
  stopPolling();
  state.submitting = false;
  state.pendingEdit = '';
  state.mediaKind = 'current';
  state.browserPreview = false;
  state.taskId = '';
  state.task = null;
  state.detail = null;
  state.view = 'studio';
  state.loadingDetail = true;
  setSelectedProject(projectId);
  state.draft = state.drafts.get(projectId) || '';
  if (historyMode !== 'none') updateProjectUrl(projectId, historyMode);
  render();
  try {
    const activeTask = await refreshDetail();
    if (state.selectedId !== projectId) return;
    state.loadingDetail = false;
    render();
    if (activeTask) startPolling(activeTask);
  } catch (error) {
    if (state.selectedId !== projectId) return;
    state.loadingDetail = false;
    render();
    showError(error.message);
  }
}

function watchProduction(taskId) {
  window.clearTimeout(productionTimer);
  const sequence = ++productionSequence;
  async function poll(failures = 0) {
    try {
      const task = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
      if (sequence !== productionSequence) return;
      state.production = task;
      renderProductionStatus();
      if (ACTIVE_TASK_STATUSES.has(task.status)) {
        productionTimer = window.setTimeout(poll, 900);
      } else if (task.result?.project_id) {
        await refreshProjects();
        render();
      }
    } catch (error) {
      if (sequence !== productionSequence) return;
      if (error.status === 404) {
        state.production = null;
        renderProductionStatus();
        showError('The production record is unavailable.');
        return;
      }
      if (failures === 2) showError('Connection interrupted. Studio will reconnect.');
      productionTimer = window.setTimeout(() => poll(failures + 1), Math.min(10_000, 900 * 2 ** Math.min(failures, 4)));
    }
  }
  poll();
}

async function sendProduction() {
  if (state.productionBusy || ACTIVE_TASK_STATUSES.has(state.production?.status)) return;
  state.productionBusy = true;
  renderProductionStatus();
  try {
    const task = await api('/api/production', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(state.productionDraft),
    });
    state.production = task;
    watchProduction(task.task_id);
  } catch (error) {
    if (error.status === 409) {
      const active = await api('/api/production').catch(() => null);
      if (active?.active) {
        state.production = active.active;
        watchProduction(active.active.task_id);
      }
    }
    showError(error.message);
  } finally {
    state.productionBusy = false;
    renderProductionStatus();
  }
}

async function stopProduction() {
  const task = state.production;
  if (!task?.task_id || task.stop_requested) return;
  try {
    const updated = await api(`/api/tasks/${encodeURIComponent(task.task_id)}/stop`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
    });
    if (state.production?.task_id === updated.task_id) state.production = updated;
    renderProductionStatus();
  } catch (error) {
    showError(error.message);
  }
}

async function retryProduction() {
  const task = state.production;
  if (!task || state.productionBusy) return;
  state.productionBusy = true;
  try {
    const run = await api(`/api/tasks/${encodeURIComponent(task.task_id)}/retry`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
    });
    state.production = run;
    watchProduction(run.task_id);
    render();
  } catch (error) {
    showError(error.message);
  } finally {
    state.productionBusy = false;
  }
}

function navItem(view, label, iconName) {
  return `<button type="button" class="nav-item ${state.view === view ? 'active' : ''}" data-nav="${view}" ${state.view === view ? 'aria-current="page"' : ''}>${icon(iconName)}<span>${label}</span></button>`;
}

function renderSidebar() {
  const health = state.health || {};
  return `<aside class="sidebar">
    <button type="button" class="brand" data-nav="create"><span class="brand-symbol">${icon('film')}</span><span class="brand-name">Vex <span>Studio</span></span></button>
    <button type="button" class="new-project" data-nav="create">${icon('plus')}<span>Create video</span></button>
    <button type="button" class="import-project" data-action="new-project">${icon('upload')}<span>Import footage</span></button>
    <div class="nav-label">Workspace</div>
    <nav class="nav-list" aria-label="Workspace">
      ${navItem('studio', 'Editor', 'monitor')}
      ${navItem('projects', 'Projects', 'grid')}
      ${navItem('activity', 'Activity', 'activity')}
      ${navItem('settings', 'Settings', 'settings')}
    </nav>
    ${state.projects.length ? `<div class="nav-label recent-label">Recent projects</div><div class="recent-list">${state.projects.slice(0, 6).map((item) => `<button type="button" class="recent-project ${state.selectedId === item.project_id && state.view === 'studio' ? 'active' : ''}" data-project="${esc(item.project_id)}" title="${esc(item.project_name)}">${icon('film')}<span>${esc(item.project_name)}</span></button>`).join('')}</div>` : ''}
    <div class="sidebar-bottom">
      <div class="local-status"><span class="status-dot ${health.media_stack_ready ? '' : 'offline'}"></span><div class="status-copy"><span class="status-title">Local workspace</span><span class="status-detail">${esc(health.model || health.provider || 'Vex runtime')}</span></div></div>
    </div>
  </aside>`;
}

function renderTopbar() {
  const current = project();
  const label = {create: 'Create', studio: 'Editor', projects: 'Projects', activity: 'Activity', settings: 'Settings'}[state.view];
  return `<header class="topbar"><div class="breadcrumb"><strong>${label}</strong>${current && state.view === 'studio' ? `<span class="breadcrumb-sep">/</span><span>${esc(current.project_name)}</span>` : ''}</div><div class="top-actions"><button type="button" class="command-button" data-action="commands" aria-label="Find a command or project">${icon('search')}<span>Find a command or project</span><kbd>Ctrl K</kbd></button><button type="button" class="icon-btn" data-action="theme" title="Switch color theme" aria-label="Switch color theme">${icon('theme')}</button><button type="button" class="icon-btn ${state.refreshing ? 'spinning' : ''}" data-action="refresh" title="Refresh" aria-label="Refresh" ${state.refreshing ? 'disabled' : ''}>${icon('refresh')}</button></div></header>`;
}

function renderHeading(title, copy = '') {
  return `<div class="view-heading"><h1>${esc(title)}</h1>${copy ? `<p class="heading-copy">${esc(copy)}</p>` : ''}</div>`;
}

function renderVideoCard() {
  const current = project();
  const media = state.detail?.media;
  const source = state.mediaKind === 'source';
  const cut = state.detail?.production;
  const live = state.mediaKind === 'live' && cut?.shots?.length > 0;
  const playback = media?.available ? new URL(source ? media.source : media.current, window.location.href) : null;
  if (state.browserPreview && playback) playback.pathname += '-preview';
  return `<section class="card video-card"><div class="card-header"><h3>Preview</h3><div class="segmented" aria-label="Select video version">
    <button type="button" data-media="current" aria-pressed="${!source && !live}">Current cut</button>
    <button type="button" data-media="source" aria-pressed="${source}">Source</button>
    ${cut?.shots?.length ? `<button type="button" data-media="live" aria-pressed="${live}">Live timeline</button>` : ''}
  </div></div>${live ? `<div id="vex-live-player" data-version="${esc(current.project_id)}:${cut.revision}"><p role="status">Opening live timeline…</p></div>`
      : `<div class="video-frame">${playback ? `<video id="working-video" controls playsinline preload="metadata" src="${esc(playback.pathname + playback.search)}" aria-label="${source ? 'Original footage' : 'Current working cut'}"></video>` : `<div class="video-empty">${icon('film')}<p>No working video available.</p></div>`}</div>`}
  <div id="media-feedback" class="media-feedback" hidden><span role="status"></span>${state.browserPreview ? '' : '<button type="button" class="secondary-btn" data-action="browser-preview">Prepare browser preview</button>'}</div>
  ${current ? `<div class="media-meta"><span>${esc(current.duration)}</span><span>${esc(current.resolution)}</span><span>${esc(current.fps)} fps</span></div>` : ''}</section>`;
}

function renderComposer() {
  const busy = isBusy();
  const prompts = ['Clean up the pauses and filler words', 'Give this a natural, cinematic grade', 'Add clean, readable captions', 'Find the strongest moments for shorts'];
  const status = state.submitting ? 'Starting your edit…' : busy ? esc(state.task?.message || 'Working…') : 'Plain English is the interface';
  return `<section class="composer-card"><form class="composer" data-form="chat"><label class="sr-only" for="prompt">Editing instruction</label><textarea id="prompt" name="message" maxlength="12000" placeholder="Describe the edit you want…" ${busy ? 'disabled' : ''}>${esc(state.draft)}</textarea><div class="composer-footer"><span id="composer-status" aria-live="polite">${busy ? status : 'Enter to send'}</span><button class="send-btn" type="submit" ${busy ? 'disabled' : ''} aria-label="Send instruction">${icon('send')}</button></div></form><div class="quick-actions">${prompts.slice(0, 2).map((prompt) => `<button type="button" class="quick-action" data-prompt="${esc(prompt)}" ${busy ? 'disabled' : ''}>${esc(prompt)}</button>`).join('')}</div></section>`;
}

const BRIEF_IDEAS = [
  ['Explain an idea', 'Create a short visual explainer showing how a camera lens brings light into focus.'],
  ['Introduce a product', 'Make a product launch teaser for a compact mechanical keyboard. Use simple type and deliberate reveals.'],
  ['Tell a story', 'Make a silent motion-design film about a tiny seed growing into a tree through the seasons.'],
];

function option(value, label, selected) {
  return `<option value="${esc(value)}" ${value === selected ? 'selected' : ''}>${esc(label)}</option>`;
}

function renderCreate() {
  const draft = state.productionDraft;
  return `<div class="creation-area">
    <div class="creation-heading"><div class="creation-icon">${icon('spark')}</div><h1>Create a video</h1><p>Describe the story, the style, and what you want people to see.</p></div>
    <form class="brief-form" data-form="production">
      <label class="sr-only" for="brief">Video brief</label>
      <textarea id="brief" name="prompt" placeholder="An animated explainer about how noise becomes a clear signal…" required maxlength="6000" rows="4">${esc(draft.prompt)}</textarea>
      <div class="brief-toolbar">
        <label>Format<select aria-label="Aspect ratio" name="aspect">${option('landscape', 'Landscape 16:9', draft.aspect)}${option('portrait', 'Portrait 9:16', draft.aspect)}${option('square', 'Square 1:1', draft.aspect)}</select></label>
        <label>Duration<select aria-label="Duration" name="duration_sec">${[12, 16, 24, 30, 45, 60, 90, 120, 180].map((seconds) => option(seconds, `${seconds} s`, draft.duration_sec)).join('')}</select></label>
        <label class="voice-option"><input type="checkbox" name="generate_audio" ${draft.generate_audio ? 'checked' : ''} /> Voiceover</label>
        <button class="primary-btn production-send" type="submit">${icon('send')}<span>Create video</span></button>
      </div>
      <details class="brief-details"><summary>Production settings</summary><div><label>Title<input name="title" placeholder="Optional" maxlength="90" value="${esc(draft.title)}" /></label><label>Renderer<select name="renderer">${option('auto', 'Auto', draft.renderer)}${option('hybrid', 'Hybrid', draft.renderer)}${option('remotion', 'Remotion', draft.renderer)}${option('hyperframes', 'HyperFrames', draft.renderer)}</select></label></div></details>
    </form>
    <div class="brief-ideas" aria-label="Example briefs">${BRIEF_IDEAS.map(([label, prompt]) => `<button type="button" class="quick-action" data-brief="${esc(prompt)}">${esc(label)}</button>`).join('')}</div>
    <div id="production-status">${renderProductionPanel()}</div>
  </div>`;
}

function renderProductionPanel() {
  const task = state.production;
  if (!task) return '';
  const active = ACTIVE_TASK_STATUSES.has(task.status);
  const result = task.result || {};
  const label = {queued: 'Queued', running: 'In progress', succeeded: 'Ready', failed: 'Needs attention', cancelled: 'Stopped'}[task.status];
  const issueLabels = array(result.issues).map((issue) => esc(String(issue).replaceAll('_', ' ')));
  return `<section class="card production-run">
    <header class="card-header"><div><h3>${esc(result.title || task.request?.title || 'Latest production')}</h3><p class="muted run-brief">${esc(task.request?.prompt || '')}</p></div>
      <span class="status-pill ${task.status}">${esc(label)}</span></header>
    ${active ? `<div class="run-stage"><p role="status">${esc(task.message || task.label)}</p><progress max="1" value="${esc(task.progress || 0)}" aria-label="Production progress"></progress>${array(task.events).length ? `<ul>${task.events.slice(-5).map((event) => `<li>${esc(event.title)}</li>`).join('')}</ul>` : ''}${task.progress < .98 ? `<button class="secondary-btn" type="button" data-action="stop-production" ${task.stop_requested ? 'disabled' : ''}>${task.stop_requested ? 'Stop requested' : 'Stop after current step'}</button>` : ''}</div>` : ''}
    ${result.media_url ? `<div class="video-frame production-video"><video id="production-video" controls playsinline preload="metadata" src="${esc(result.media_url)}" aria-label="Generated video"></video></div>` : ''}
    ${active ? '' : `<div class="run-stage completed"><p>${esc(task.message || label)}</p>${issueLabels.length ? `<ul>${issueLabels.map((issue) => `<li>${issue}</li>`).join('')}</ul>` : ''}<div class="run-actions">${result.project_id ? `<button class="primary-btn" type="button" data-project="${esc(result.project_id)}">Open in editor ${icon('arrow')}</button>` : ''}${result.qa_passed && result.media_url ? `<a class="secondary-btn" href="${esc(result.media_url)}" download="vex-film.mp4">Download video</a>` : ''}${['failed','cancelled'].includes(task.status) ? '<button class="secondary-btn" type="button" data-action="retry-production">Retry from saved stages</button>' : ''}<button class="secondary-btn" type="button" data-action="reuse-brief">Reuse brief</button></div></div>`}
  </section>`;
}

function renderProductionStatus() {
  const output = document.getElementById('production-status');
  if (output) {
    const template = document.createElement('template');
    template.innerHTML = renderProductionPanel();
    replaceWithMedia(output, template.content);
  }
  const busy = state.productionBusy || ACTIVE_TASK_STATUSES.has(state.production?.status);
  const button = document.querySelector('.production-send');
  if (button) {
    button.disabled = busy;
    button.querySelector('span').textContent = busy ? 'Creating…' : 'Create video';
  }
}

function renderTimeline() {
  const timeline = array(state.detail?.timeline);
  const graph = state.detail?.edit_graph;
  const spans = array(graph?.spans);
  if (spans.length) {
    const total = Math.max(Number(graph.duration_sec) || 0, 0.001);
    const mode = graph.provenance === 'source' ? 'Source-backed cut' : 'Rendered fallback';
    const segments = spans.map((span, index) => {
      const weight = Math.max(1, Math.min(24, Math.round((Number(span.duration_sec) || 0) / total * 24)));
      const title = `${span.source_name || 'Source'} · ${Number(span.source_start_sec || 0).toFixed(2)}–${Number(span.source_end_sec || 0).toFixed(2)}s`;
      return `<button type="button" class="timeline-segment span-weight-${weight}" data-seek="${esc(span.output_start_sec)}" title="${esc(title)}"><span>${esc(span.source_name || `Clip ${index + 1}`)}</span></button>`;
    }).join('');
    return `<section class="card timeline-card"><div class="timeline-headline"><span>Timeline · ${esc(mode)}</span><span>${esc(graph.span_count)} clip${graph.span_count === 1 ? '' : 's'}${graph.truncated ? ' · first 80 shown' : ''}</span></div><div class="timeline-strip graph-strip">${segments}</div></section>`;
  }
  return `<section class="card timeline-card"><div class="timeline-headline"><span>Timeline · legacy edit history</span><span>${timeline.length ? `${timeline.length} operation${timeline.length === 1 ? '' : 's'}` : 'No edits yet'}</span></div><div class="timeline-strip legacy-strip">${timeline.length ? timeline.slice(0, 10).map((item) => `<div class="timeline-segment"><span>${esc(item.op)}</span></div>`).join('') : '<span class="timeline-empty">No clips mapped yet.</span>'}</div></section>`;
}

function renderProductionTimeline() {
  const shots = array(state.detail?.production_shots);
  if (!shots.length) return '';
  const segments = shots.map((shot) => `<button type="button" class="timeline-segment" data-seek="${esc(shot.start_sec)}" title="${esc(shot.producer)} · ${esc(shot.start_sec.toFixed(2))}–${esc(shot.end_sec.toFixed(2))}s">Shot ${esc(shot.number)}</button>`).join('');
  const options = shots.map((shot) => `<option value="${esc(shot.id)}">Shot ${esc(shot.number)} · ${esc(shot.start_sec.toFixed(2))}–${esc(shot.end_sec.toFixed(2))}s</option>`).join('');
  const revision = state.task?.kind === 'shot_revision' && state.task.status === 'succeeded' ? state.task.result : null;
  const busy = isBusy() ? 'disabled' : '';
  return `<section class="card timeline-card"><div class="timeline-headline"><span>Film shots</span><span>${esc(shots.length)} ${shots.length === 1 ? 'shot' : 'shots'}</span></div>
    <div class="timeline-strip">${segments}</div>
    ${revision?.project_id ? `<div class="shot-result"><span>${revision.reused_frames ? `${esc(revision.rendered_frames)} frames rendered · ${esc(revision.reused_frames)} reused` : 'Full render verified'}</span><button type="button" class="secondary-btn" data-project="${esc(revision.project_id)}">Open revised cut</button></div>` : ''}
    <details class="shot-replacement"><summary>Replace a shot</summary>
      <form data-form="shot-revision" class="shot-form">
        <label class="field">Shot<select name="shot_id" ${busy}>${options}</select></label>
        <label class="field">Replacement image or video<input name="file" type="file" accept=".png,.jpg,.jpeg,.webp,.mp4,.mov,.webm" required ${busy}></label>
        <button type="submit" class="secondary-btn" ${busy}>Save as new cut</button>
      </form>
    </details></section>`;
}

function renderProductionEditing() {
  const cut = state.detail?.production;
  if (!cut?.shots?.length) return `<section class="card timeline-card"><button class="secondary-btn" type="button" data-action="attach-production" ${isBusy() ? 'disabled' : ''}>Open footage as an editable cut</button></section>`;
  const busy = isBusy() ? 'disabled' : '';
  const preview = state.task?.status === 'succeeded' && state.task.result?.revision === cut.revision
    ? state.task.result : null;
  const audio = preview?.audio;
  const clips = cut.shots.map(shot => `<option value="${esc(shot.id)}">Shot ${shot.number} · ${esc(shot.producer)}</option>`).join('');
  const cues = array(cut.captions);
  return `<section class="card timeline-card">
    <div class="timeline-headline"><span>Revise this cut</span><span>Revision ${cut.revision}</span></div>
    ${audio?.has_audio ? `<p class="muted">Delivered audio · ${esc(audio.integrated_lufs ?? '—')} LUFS · ${esc(audio.true_peak_dbtp ?? '—')} dBTP</p>` : ''}
    ${preview?.frames?.length ? `<div class="revision-preview">
      <div class="revision-frames">${preview.frames.map(row => `<figure><img alt="Proposed frame ${row.frame}" src="/api/projects/${encodeURIComponent(state.selectedId)}/production-preview/${encodeURIComponent(preview.cut)}/${row.frame}"><figcaption>${(row.frame / cut.fps).toFixed(2)}s</figcaption></figure>`).join('')}</div>
      ${array(preview.speech).map(name => `<audio controls preload="none" aria-label="Proposed narration" src="/api/projects/${encodeURIComponent(state.selectedId)}/production-preview/${encodeURIComponent(preview.cut)}/${encodeURIComponent(name)}"></audio>`).join('')}
      ${preview.review ? `<ul>${array(preview.review.observations).map(note => `<li>${esc(note)}</li>`).join('')}</ul>` : ''}
      <button class="secondary-btn" type="button" data-review-cut="${esc(preview.cut)}" ${busy}>Review frames with model</button>
      <button class="primary-btn" type="button" data-accept-cut="${esc(preview.cut)}" ${busy}>Save this revision</button>
    </div>` : ''}
    <details class="shot-replacement"><summary>Reorder shots or change format</summary>
      <form data-form="production-order" class="shot-form">
        <label class="field">Shot<select name="shot" ${busy}>${clips}</select></label>
        <label class="field">Move to position<input name="position" type="number" min="1" max="${cut.shots.length}" value="1" required ${busy}></label>
        <button class="secondary-btn" type="submit" ${busy}>Preview order</button>
      </form>
      <form data-form="production-canvas" class="shot-form">
        <label class="field">Canvas<select name="aspect" ${busy}><option value="1080x1920">Vertical · 1080 × 1920</option><option value="1920x1080">Landscape · 1920 × 1080</option><option value="1080x1080">Square · 1080 × 1080</option></select></label>
        <button class="secondary-btn" type="submit" ${busy}>Preview format</button>
      </form>
      <form data-form="production-split" class="shot-form">
        <label class="field">Shot<select name="shot" ${busy}>${clips}</select></label>
        <label class="field">Split at timeline time (seconds)<input name="at" type="number" min="0" step="any" required ${busy}></label>
        <button class="secondary-btn" type="submit" ${busy}>Preview split</button>
      </form>
    </details>
    ${cues.length ? `<details class="shot-replacement"><summary>Captions · ${cues.length}</summary>
      <form data-form="production-caption" class="shot-form">
        <label class="field">Caption<select name="cue" ${busy}>${cues.map(cue => `<option value="${esc(cue.id)}">${(cue.start_frame / cut.fps).toFixed(2)}s · ${esc(cue.text)}</option>`).join('')}</select></label>
        <label class="field">Text<input name="text" maxlength="280" value="${esc(cues[0].text)}" required ${busy}></label>
        <button class="secondary-btn" type="submit" ${busy}>Preview caption</button>
      </form>
    </details>` : ''}
    <details class="shot-replacement"><summary>Rewrite narration</summary>
      <form data-form="production-narration" class="shot-form">
        <label class="field">Shot<select name="shot" ${busy}>${clips}</select></label>
        <label class="field">Spoken line<input name="text" maxlength="280" placeholder="Enter a short replacement line" required ${busy}></label>
        <button class="secondary-btn" type="submit" ${busy}>Record and preview</button>
      </form>
    </details>
    ${cut.tracks?.music ? `<details class="shot-replacement"><summary>Music level</summary>
      <form data-form="production-music" class="shot-form">
        <label class="field">Gain<input name="gain" type="number" min="0" max="1" step="any" value="${cut.tracks.music_gain}" required ${busy}></label>
        <button class="secondary-btn" type="submit" ${busy}>Preview mix change</button>
      </form>
    </details>` : ''}
    <details class="shot-replacement"><summary>Direct a shot in plain English</summary>
      <form data-form="production-instruct" class="shot-form">
        <label class="field">Shot<select name="shot" ${busy}>${clips}</select></label>
        <label class="field">Edit instruction<input name="instruction" maxlength="2000" placeholder="Make the highlighted path amber" required ${busy}></label>
        ${array(cut.references).length ? `<label><input type="checkbox" name="references" ${busy}> Send attached reference images to the model</label>` : ''}
        <label><input type="checkbox" name="native" ${busy}> Allow native code in the configured sandbox</label>
        <button class="secondary-btn" type="submit" ${busy}>Preview proposal</button>
      </form>
    </details>
    <details class="shot-replacement"><summary>Project direction and locked shots</summary>
      <form data-form="production-direction" class="shot-form">
        <label class="field">Reusable direction<textarea name="notes" maxlength="3000" placeholder="Colors, typography, pacing, or copy to preserve" ${busy}>${esc(cut.direction?.notes)}</textarea></label>
        <fieldset><legend>Protect shots in production revisions</legend>${cut.shots.map(shot => `<label><input type="checkbox" name="locked" value="${esc(shot.id)}" ${array(cut.direction?.locked_shots).includes(shot.id) ? 'checked' : ''} ${busy}> Shot ${shot.number}</label>`).join(' ')}</fieldset>
        <button class="secondary-btn" type="submit" ${busy}>Save direction</button>
      </form>
    </details>
    <details class="shot-replacement"><summary>Visual references</summary>
      ${array(cut.references).map(row => `<form data-form="production-reference-remove" class="shot-form reference-card">
        <img src="/api/projects/${encodeURIComponent(state.selectedId)}/production-reference/${encodeURIComponent(row.id)}" alt="${esc(row.note)}" loading="lazy">
        <span>${esc(row.note)}</span><input type="hidden" name="id" value="${esc(row.id)}">
        <button type="submit" class="secondary-btn" ${busy}>Remove reference</button>
      </form>`).join('')}
      ${array(cut.references).length < 4 ? `<form data-form="production-reference" class="shot-form">
        <label class="field">Reference image<input name="file" type="file" accept="image/png,image/jpeg,image/webp" required ${busy}></label>
        <label class="field">What should the model use?<input name="note" maxlength="240" required ${busy}></label>
        <button type="submit" class="secondary-btn" ${busy}>Attach reference</button>
      </form>` : ''}
    </details>
  </section>`;
}

function renderMediaSearch() {
  return `<details class="card timeline-card media-search"><summary>Find footage or spoken words</summary>
    <form data-form="media-search" class="shot-form">
      <label class="field">Search this project<input name="query" maxlength="180" required></label>
      <button type="submit" class="secondary-btn">Search</button>
    </form><div data-search-results aria-live="polite"></div>
  </details>`;
}

function renderMessages() {
  const messages = [...array(state.detail?.chat)];
  const task = state.task;
  if (isBusy()) {
    const pending = state.pendingEdit || task?.request?.message;
    if (pending && messages.at(-1)?.content !== pending) messages.push({role: 'user', content: pending});
    if (task?.stream) messages.push({role: 'assistant', content: task.stream});
  } else if (task?.status === 'failed') {
    messages.push({role: 'assistant', content: task.error || task.message || 'This edit could not be completed.'});
  }
  return messages.length ? messages.map(({role, content}) => `<article class="message ${role === 'user' ? 'message-user' : 'message-assistant'}"><span class="message-author">${role === 'user' ? 'You' : 'Vex'}</span><p>${esc(content)}</p></article>`).join('')
    : `<div class="conversation-empty"><div class="thread-mark">${icon('spark')}</div><h3>Direct this edit</h3><p>Ask for a tighter cut, readable captions, or a new look.</p></div>`;
}

function updateMessages() {
  const thread = document.getElementById('conversation');
  if (!thread) return;
  const pinned = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 48;
  const markup = renderMessages();
  if (thread.innerHTML !== markup) thread.innerHTML = markup;
  if (pinned) thread.scrollTop = thread.scrollHeight;
}

function renderTrace() {
  const taskEvents = array(state.task?.events);
  const traceEvents = array(state.detail?.latest_trace?.events);
  const events = taskEvents.length ? taskEvents : traceEvents;
  const recent = events.slice(-7).reverse();
  const active = ACTIVE_TASK_STATUSES.has(state.task?.status);
  return `<section class="card inspector-card" id="trace-card"><div class="trace-head"><h3>${active ? 'Live run' : 'Latest run'}</h3></div>${recent.length ? `<div class="trace-list">${recent.map((event) => `<div class="trace-item"><span class="trace-marker ${statusClass(event.status)}"></span><div class="trace-copy"><span class="trace-title">${esc(event.title || event.kind || 'Update')}</span><span class="trace-detail" title="${esc(event.detail || '')}">${esc(event.detail || '')}</span></div></div>`).join('')}</div>` : `<p class="empty-state">No edit activity yet.</p>`}</section>`;
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
  if (state.loadingDetail) return '<div class="loading-panel" role="status">Opening project…</div>';
  if (!current) return `${renderHeading('Editor')}<div class="empty-library"><h2>Open a video to start editing</h2><button type="button" class="primary-btn" data-action="new-project">${icon('upload')} Import footage</button><button type="button" class="secondary-btn" data-nav="create">Create a video</button></div>`;
  return `<div class="editor-title"><div><h1>${esc(current.project_name)}</h1><p>${esc(current.source_name)}</p></div><div class="editor-actions">
    <button type="button" class="icon-btn" data-history="undo" title="Undo" aria-label="Undo last edit" ${isBusy() || !current.timeline_ops ? 'disabled' : ''}>${icon('undo')}</button>
    <button type="button" class="icon-btn" data-history="redo" title="Redo" aria-label="Redo last edit" ${isBusy() || !current.redo_count ? 'disabled' : ''}>${icon('redo')}</button>
    ${state.detail?.media?.available ? `<a class="secondary-btn" href="${esc(state.detail.media.current)}" download="${esc(state.detail.media.download_name)}">Download cut</a>` : ''}
  </div></div>
    <div class="editor-grid"><section class="card conversation-panel"><header class="card-header"><h3>Conversation</h3><span class="conversation-model">${esc(current.model || current.provider)}</span></header><div id="conversation" class="conversation" aria-label="Edit conversation">${renderMessages()}</div>${renderComposer()}</section>
    <div class="preview-column">${renderVideoCard()}${renderProductionTimeline() || renderTimeline()}${renderProductionEditing()}${renderMediaSearch()}${renderTrace()}</div></div>`;
}

function renderProjects() {
  return `${renderHeading('Projects', 'Open a working cut from your local library.')}${state.projects.length ? `<div class="library-grid">${renderProjectCards()}</div>` : `<div class="empty-library"><h2>No projects yet</h2><button type="button" class="primary-btn" data-action="new-project">${icon('plus')}<span>Import footage</span></button></div>`}`;
}

function renderActivity() {
  const current = project();
  const timeline = array(state.detail?.timeline);
  const runs = array(state.detail?.creative_runs);
  return `${renderHeading('Activity', 'Edit history, renders, and quality checks for this project.')}${current ? `<div class="section-heading section-heading-flush"><div><h2>${esc(current.project_name)}</h2><p>${esc(current.timeline_ops)} timeline operation${current.timeline_ops === 1 ? '' : 's'} · ${esc(current.updated_label)}</p></div><button type="button" class="secondary-btn" data-nav="studio">Back to studio ${icon('arrow')}</button></div>` : ''}<section class="card activity-card"><div class="activity-list">${timeline.length ? timeline.map((item, index) => `<div class="activity-row"><div class="activity-index">${String(index + 1).padStart(2, '0')}</div><div class="activity-copy"><span class="activity-title">${esc(item.op)}</span><span class="activity-detail">${esc(item.detail)}</span></div><span class="activity-time">${esc(formatTimestamp(item.timestamp))}</span></div>`).join('') : '<p class="empty-state">No edit history for this project yet.</p>'}</div></section>${renderJobs()}${renderMediaLineage()}${runs.length ? `<div class="section-heading"><div><h2>Creative runs</h2><p>Quality-gated automation recorded by Vex.</p></div></div><section class="card activity-card"><div class="activity-list">${runs.map((run) => `<div class="activity-row"><div class="activity-index">${icon('spark')}</div><div class="activity-copy"><span class="activity-title">${esc(run.feature || 'Creative run')}</span><span class="activity-detail">${esc(creativeSummary(run))}</span></div><span class="quality-pill">${esc(formatScore(run.quality_score))}</span></div>`).join('')}</div></section>` : ''}`;
}

function renderMediaLineage() {
  const assets = array(state.detail?.media_assets);
  if (!assets.length) return '';
  return `<div class="section-heading"><div><h2>Media lineage</h2><p>Immutable outputs registered with this project.</p></div></div><section class="card activity-card"><div class="activity-list">${assets.map((asset) => `<div class="activity-row"><div class="activity-index">${icon('film')}</div><div class="activity-copy"><span class="activity-title">${esc(asset.name || asset.kind || 'Media asset')}</span><span class="activity-detail">${esc(asset.role || asset.kind || 'Asset')}${asset.checksum ? ` · ${esc(asset.checksum)}` : ''}${asset.parent_count ? ` · ${esc(asset.parent_count)} parent${asset.parent_count === 1 ? '' : 's'}` : ''}</span></div><span class="activity-time">${esc(formatTimestamp(asset.created_at))}</span></div>`).join('')}</div></section>`;
}

function renderJobs() {
  const jobs = array(state.detail?.jobs);
  if (!jobs.length) return '';
  return `<div class="section-heading"><div><h2>Tool jobs</h2><p>Queued and completed work from the command line.</p></div></div><section class="card activity-card"><div class="activity-list">${jobs.map((job) => {
    const marker = job.status === 'succeeded' ? 'success' : job.status === 'running' ? 'running' : job.status === 'failed' ? 'error' : '';
    const progress = job.status === 'running' && job.progress > 0 ? ` · ${Math.round(job.progress * 100)}%` : '';
    const detail = `${job.status}${job.stage && job.stage !== job.status ? ` · ${job.stage}` : ''}${progress}${job.message ? ` · ${job.message}` : ''}`;
    return `<div class="activity-row"><div class="activity-index"><span class="trace-marker ${marker}"></span></div><div class="activity-copy"><span class="activity-title">${esc(job.tool_name || 'Tool job')}</span><span class="activity-detail">${esc(detail)}</span></div><span class="activity-time">${esc(formatTimestamp(job.updated_at))}</span></div>`;
  }).join('')}</div></section>`;
}

function renderSettings() {
  const health = state.health || {};
  const mediaReady = Boolean(health.media_stack_ready);
  return `${renderHeading('Settings', 'Provider and media tools for this workspace.')}
    <div class="settings-grid"><section class="card settings-panel"><h2>Runtime</h2><p>Configuration is read from your local Vex environment and project state directory.</p><div class="settings-list"><div class="setting-row"><div class="setting-label"><span class="setting-name">Provider</span><span class="setting-note">Used when a request needs model planning.</span></div><span class="setting-value">${esc(health.provider || '—')}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Model</span><span class="setting-note">Configured model name.</span></div><span class="setting-value">${esc(health.model || '—')}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Media stack</span><span class="setting-note">FFmpeg and ffprobe availability.</span></div><span class="setting-value ${mediaReady ? '' : 'unavailable'}">${mediaReady ? 'ready' : 'needs setup'}</span></div><div class="setting-row"><div class="setting-label"><span class="setting-name">Projects</span><span class="setting-note">Where Vex keeps project working copies.</span></div><span class="setting-value" title="${esc(health.projects_dir)}">local</span></div></div></section><aside class="primary-column"><section class="card settings-panel"><div class="settings-note">The Studio is served by Vex itself at localhost. No footage leaves this machine unless a tool explicitly uses a provider or stock-media API you have configured.</div></section><section class="card settings-panel"><div class="card-kicker">Build</div><h2 class="build-version">Vex ${esc(health.version || '')}</h2><p>Terminal-first editing, now with a visual command center.</p></section></aside></div>`;
}

function renderModal() {
  if (state.modal === 'commands') {
    return `<div class="modal-backdrop" data-action="close-modal"><section class="modal command-palette" data-modal-content data-modal-key="commands" role="dialog" aria-modal="true" aria-label="Find a command or project"><label class="command-search">${icon('search')}<input id="command-query" type="search" placeholder="Find a command or project" aria-label="Find a command or project" /><kbd>Esc</kbd></label><div id="command-results" class="command-results">${commandResults('')}</div></section></div>`;
  }
  if (state.modal !== 'new-project') return '';
  return `<div class="modal-backdrop" data-action="close-modal"><section class="modal" data-modal-content data-modal-key="new-project" role="dialog" aria-modal="true" aria-labelledby="new-project-title" tabindex="-1"><div class="modal-header"><div><h2 id="new-project-title">New project</h2><p>Bring in a local video. Vex makes a safe working copy before anything changes.</p></div><button type="button" class="icon-btn" data-action="close-modal" aria-label="Close">${icon('close')}</button></div><form class="modal-body" data-form="new-project"><label class="drop-zone" id="drop-zone"><input id="video-file" type="file" name="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,video/*" /><span class="video-empty-icon">${icon('upload')}</span><strong id="file-label">Drop a video here or choose a file</strong><span>MP4, MOV, AVI, WEBM, MKV, M4V, FLV</span></label><div class="field"><label for="project-name">Project name <span class="muted">(optional)</span></label><input id="project-name" name="name" placeholder="e.g. Product film / episode 04" maxlength="120" /></div><div class="field"><label for="source-path">Or use a local path</label><input id="source-path" name="source_path" placeholder="/Users/you/Videos/episode-04.mov" /></div><div id="modal-error" class="error-note" role="alert" hidden></div><div class="modal-actions"><button class="secondary-btn" type="button" data-action="close-modal">Cancel</button><button class="primary-btn" type="submit">${icon('arrow')}<span>Create project</span></button></div></form></section></div>`;
}

function commandResults(query) {
  const commands = [
    ['Create video', 'nav', 'create', 'spark'],
    ['Import footage', 'action', 'new-project', 'upload'],
    ['Open editor', 'nav', 'studio', 'monitor'],
    ['Browse projects', 'nav', 'projects', 'grid'],
    ['Open settings', 'nav', 'settings', 'settings'],
    ...state.projects.map((item) => [item.project_name, 'project', item.project_id, 'film']),
  ];
  const needle = query.trim().toLowerCase();
  const matches = commands.filter(([title]) => title.toLowerCase().includes(needle));
  return matches.map(([title, kind, value, glyph]) => `<button type="button" data-${kind}="${esc(value)}">${icon(glyph)}<span>${esc(title)}</span></button>`).join('') || '<p class="empty-state">No matching commands or projects.</p>';
}

function openCommands() {
  modalReturnAction = 'commands';
  state.modal = 'commands';
  render();
  document.getElementById('command-query')?.focus();
}

function renderTaskStatus() {
  const trace = document.getElementById('trace-card');
  if (trace) trace.outerHTML = renderTrace();
  const busy = isBusy();
  const status = document.getElementById('composer-status');
  if (status) status.textContent = state.submitting ? 'Starting your edit…' : busy ? state.task?.message || 'Working…' : 'Enter to send';
  const textarea = document.getElementById('prompt');
  if (textarea) textarea.disabled = busy;
  document.querySelectorAll('.composer-card .send-btn, .composer-card .quick-action').forEach((button) => { button.disabled = busy; });
  document.querySelectorAll('[data-form="shot-revision"] select, [data-form="shot-revision"] input, [data-form="shot-revision"] button').forEach((control) => { control.disabled = busy; });
  document.querySelectorAll('[data-history]').forEach((button) => {
    button.disabled = busy || !project()?.[button.dataset.history === 'undo' ? 'timeline_ops' : 'redo_count'];
  });
  updateMessages();
}

function replaceWithMedia(scope, fragment) {
  const video = scope.querySelector('#working-video, #vex-live-player');
  const next = video && fragment.querySelector(`#${video.id}`);
  if (!next || next.getAttribute('src') !== video.getAttribute('src') || next.dataset.version !== video.dataset.version) {
    unmountLive?.();
    unmountLive = undefined;
    scope.replaceChildren(fragment);
    return;
  }
  // Keep the playing element connected while replacing the surrounding views.
  function patchAround(parent, draft) {
    const kept = [...parent.children].find((child) => child.contains(video));
    const replacement = [...draft.children].find((child) => child.contains(next));
    for (const child of [...parent.childNodes]) if (child !== kept) child.remove();
    let before = true;
    for (const child of [...draft.childNodes]) {
      if (child === replacement) before = false;
      else if (before) kept.before(child);
      else parent.append(child);
    }
    if (kept !== video) patchAround(kept, replacement);
  }
  patchAround(scope, fragment);
}

function render() {
  if (state.loading) return;
  const views = {create: renderCreate, studio: renderStudio, projects: renderProjects, activity: renderActivity, settings: renderSettings};
  const template = document.createElement('template');
  const focused = document.activeElement?.id;
  const selection = focused ? [document.activeElement.selectionStart, document.activeElement.selectionEnd] : [];
  const oldConversation = document.getElementById('conversation');
  const oldModal = app.querySelector('.modal-backdrop');
  const retainedModal = oldModal && oldModal.firstElementChild?.dataset.modalKey === state.modal;
  template.innerHTML = `${renderSidebar()}<main class="main">${renderTopbar()}<div class="main-content" data-view="${state.view}">${views[state.view]()}</div></main>${renderModal()}<div id="toast-region" aria-live="polite" aria-atomic="true"></div>`;
  if (retainedModal) template.content.querySelector('.modal-backdrop').replaceWith(oldModal);
  replaceWithMedia(app, template.content);
  if (!retainedModal) bindFileInputs();
  bindPrompt();
  bindWorkingVideo();
  bindLivePlayer();
  renderProductionStatus();
  const thread = document.getElementById('conversation');
  if (thread) thread.scrollTop = oldConversation?.scrollTop || thread.scrollHeight;
  if (focused && !state.modal) {
    const field = document.getElementById(focused);
    field?.focus({preventScroll: true});
    if (typeof selection[0] === 'number') field?.setSelectionRange?.(...selection);
  }
  renderToastRegion();
  document.title = project() ? `${project().project_name} — Vex Studio` : 'Vex Studio';
}

function loadLivePlayer() {
  if (window.VexStudioPlayer?.mount) return Promise.resolve(window.VexStudioPlayer);
  if (!playerScript) {
    playerScript = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = '/studio-player.js';
      script.onload = () => window.VexStudioPlayer?.mount ? resolve(window.VexStudioPlayer) : reject(new Error('Live player could not start.'));
      script.onerror = () => {
        script.remove();
        reject(new Error('Install the current Remotion runtime and try again.'));
      };
      document.head.append(script);
    }).catch(error => {
      playerScript = undefined;
      throw error;
    });
  }
  return playerScript;
}

async function bindLivePlayer() {
  const root = document.getElementById('vex-live-player');
  if (!root || root.dataset.loading) return;
  root.dataset.loading = 'true';
  const projectId = state.selectedId;
  const base = `/api/projects/${encodeURIComponent(projectId)}/`;
  try {
    const [player, current] = await Promise.all([loadLivePlayer(), api(base + 'production-live')]);
    if (!root.isConnected) return;
    unmountLive = player.mount(root, {
      initial: current.timeline, assetBase: base + 'production-asset/',
      plan: async operations => (await api(base + 'production-live', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({revision: current.revision, operations}),
      })).timeline,
      preview: operations => submitProductionEdit('production-preview', {revision: current.revision, operations}),
    });
  } catch (error) {
    if (root.isConnected) root.querySelector('[role="status"]').textContent = error.message;
  }
}

function bindWorkingVideo() {
  const video = document.getElementById('working-video');
  if (!video) return;
  function update() {
    const feedback = document.getElementById('media-feedback');
    if (!feedback) return;
    const preparing = state.browserPreview && video.readyState === 0 && !video.error;
    feedback.hidden = !preparing && !video.error;
    feedback.querySelector('[role="status"]').textContent = preparing ? 'Preparing browser preview…'
      : state.browserPreview ? 'The browser preview could not be prepared.' : 'This browser could not play the file directly.';
  }
  if (!video.dataset.listenersBound) {
    video.dataset.listenersBound = '1';
    video.addEventListener('error', update);
    video.addEventListener('loadedmetadata', update);
  }
  update();
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
  prompt.addEventListener('input', (event) => {
    state.draft = event.target.value;
    state.drafts.set(state.selectedId, state.draft);
  });
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
    stopPolling();
    state.taskId = '';
    state.task = null;
    state.draft = '';
    state.pendingEdit = '';
    state.submitting = false;
    state.mediaKind = 'current';
    state.browserPreview = false;
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
  const projectId = state.selectedId;
  state.pendingEdit = message;
  state.submitting = true;
  renderTaskStatus();
  try {
    const task = await api(`/api/projects/${encodeURIComponent(projectId)}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    state.drafts.delete(projectId);
    if (projectId !== state.selectedId) return;
    state.draft = '';
    const prompt = document.getElementById('prompt');
    if (prompt) prompt.value = '';
    state.submitting = false;
    startPolling(task);
  } catch (error) {
    if (projectId !== state.selectedId) return;
    state.submitting = false;
    state.pendingEdit = '';
    renderTaskStatus();
    showError(error.message);
  }
}

async function changeHistory(action) {
  const projectId = state.selectedId;
  if (!projectId || isBusy()) return;
  state.submitting = true;
  renderTaskStatus();
  try {
    const task = await api(`/api/projects/${encodeURIComponent(projectId)}/${action}`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
    });
    if (state.selectedId !== projectId) return;
    state.submitting = false;
    startPolling(task);
  } catch (error) {
    if (state.selectedId !== projectId) return;
    state.submitting = false;
    renderTaskStatus();
    showError(error.message);
  }
}

async function replaceShot(form) {
  const projectId = state.selectedId;
  if (!projectId || isBusy()) return;
  const body = new FormData(form);
  if (!body.get('file')?.size) return;
  state.submitting = true;
  renderTaskStatus();
  try {
    const task = await api(`/api/projects/${encodeURIComponent(projectId)}/replace-shot`, {
      method: 'POST', body, timeout: 0,
    });
    if (state.selectedId !== projectId) return;
    form.reset();
    state.submitting = false;
    startPolling(task);
  } catch (error) {
    if (state.selectedId !== projectId) return;
    state.submitting = false;
    renderTaskStatus();
    showError(error.message);
  }
}

async function submitProductionEdit(action, payload) {
  const projectId = state.selectedId;
  if (!projectId || isBusy()) return;
  state.submitting = true;
  renderTaskStatus();
  try {
    const form = payload instanceof FormData;
    const task = await api(`/api/projects/${encodeURIComponent(projectId)}/${action}`, {
      method: 'POST', headers: form ? {} : {'Content-Type': 'application/json'}, body: form ? payload : JSON.stringify(payload),
    });
    if (state.selectedId !== projectId) return;
    state.submitting = false;
    startPolling(task);
  } catch (error) {
    if (state.selectedId !== projectId) return;
    state.submitting = false;
    renderTaskStatus();
    showError(error.message);
  }
}

function stageProductionEdit(form) {
  const cut = state.detail?.production;
  if (!cut) return;
  const formData = new FormData(form);
  const shot = formData.get('shot');
  const revision = cut.revision;
  let operation;
  switch (form.dataset.form) {
    case 'production-order': {
      const shots = cut.shots.map(item => item.id).filter(id => id !== shot);
      shots.splice(Number(formData.get('position')) - 1, 0, shot);
      operation = {op: 'reorder', shots};
      break;
    }
    case 'production-caption':
      operation = {op: 'caption', id: formData.get('cue'), text: formData.get('text')};
      break;
    case 'production-narration':
      operation = {op: 'narration', id: shot, text: formData.get('text')};
      break;
    case 'production-canvas': {
      const [width, height] = formData.get('aspect').split('x').map(Number);
      operation = {op: 'canvas', width, height};
      break;
    }
    case 'production-split':
      operation = {op: 'split', id: shot, at: formData.get('at')};
      break;
    case 'production-music':
      operation = {op: 'music_gain', gain: Number(formData.get('gain'))};
      break;
    case 'production-instruct':
      submitProductionEdit('production-propose', {revision, shot, instruction: formData.get('instruction'),
        allow_native_code: formData.has('native'), use_references: formData.has('references')});
      return;
    case 'production-direction':
      submitProductionEdit('production-direction', {revision, notes: formData.get('notes'), locked_shots: formData.getAll('locked')});
      return;
    case 'production-reference':
      if (formData.get('file').size > 8 * 1024 * 1024) {
        showError('Choose a reference image up to 8 MiB.');
        return;
      }
      submitProductionEdit('production-reference', formData);
      return;
    case 'production-reference-remove':
      submitProductionEdit('production-reference-remove', {id: formData.get('id')});
      return;
    default:
      return;
  }
  submitProductionEdit('production-preview', {revision, operations: [operation]});
}

async function findMedia(form) {
  const results = form.parentElement.querySelector('[data-search-results]');
  const query = new FormData(form).get('query');
  const button = form.querySelector('button');
  button.disabled = true;
  results.textContent = 'Searching…';
  try {
    const report = await api(`/api/projects/${encodeURIComponent(state.selectedId)}/search-media`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query}),
    });
    const items = array(report.results);
    results.innerHTML = items.length ? `<ul>${items.map(row => `<li>${typeof row.start_sec === 'number'
      ? `<button class="secondary-btn" type="button" data-seek="${row.start_sec}">${row.start_sec.toFixed(2)}s · ${esc(row.text)}</button>
        ${state.detail?.production?.shots?.length && ['transcript','caption'].includes(row.kind)
          ? `<button class="secondary-btn" type="button" data-remove-start="${row.start_sec}" data-remove-end="${row.end_sec}">Preview cutting this range</button>` : ''}`
      : esc(row.name || row.text)}</li>`).join('')}</ul>` : 'No matching project media.';
  } catch (error) {
    results.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function navigate(view) {
  if (view === 'studio' && state.selectedId && !state.detail) {
    selectProject(state.selectedId);
    return;
  }
  state.view = view;
  const url = new URL(window.location.href);
  if (view === 'create') url.searchParams.set('view', 'create');
  else url.searchParams.delete('view');
  window.history.replaceState({}, '', url);
  render();
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
  const removal = event.target.closest('[data-remove-start]');
  if (removal) {
    submitProductionEdit('production-preview', {
      revision: state.detail.production.revision,
      operations: [{op: 'remove_range', start: removal.dataset.removeStart, end: removal.dataset.removeEnd}],
    });
    return;
  }
  const review = event.target.closest('[data-review-cut]');
  if (review) {
    submitProductionEdit('production-review', {cut: review.dataset.reviewCut});
    return;
  }
  const accept = event.target.closest('[data-accept-cut]');
  if (accept) {
    submitProductionEdit('production-commit', {cut: accept.dataset.acceptCut});
    return;
  }
  const nav = event.target.closest('[data-nav]');
  if (nav) {
    if (state.modal === 'commands') state.modal = null;
    navigate(nav.dataset.nav);
    return;
  }
  const projectButton = event.target.closest('[data-project]');
  if (projectButton) {
    if (state.modal === 'commands') state.modal = null;
    selectProject(projectButton.dataset.project);
    return;
  }
  const media = event.target.closest('[data-media]');
  if (media) {
    state.mediaKind = media.dataset.media;
    state.browserPreview = false;
    render();
    return;
  }
  const history = event.target.closest('[data-history]');
  if (history) {
    changeHistory(history.dataset.history);
    return;
  }
  const clip = event.target.closest('[data-seek]');
  if (clip) {
    if (state.mediaKind !== 'current') {
      state.mediaKind = 'current';
      state.browserPreview = false;
      render();
    }
    const video = document.getElementById('working-video');
    if (video) video.currentTime = Number(clip.dataset.seek) || 0;
    return;
  }
  const brief = event.target.closest('[data-brief]');
  if (brief) {
    state.productionDraft.prompt = brief.dataset.brief;
    const field = document.getElementById('brief');
    field.value = state.productionDraft.prompt;
    field.focus();
    return;
  }
  const promptButton = event.target.closest('[data-prompt]');
  if (promptButton) {
    state.draft = promptButton.dataset.prompt;
    state.drafts.set(state.selectedId, state.draft);
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
  if (action.dataset.action === 'commands') openCommands();
  if (action.dataset.action === 'browser-preview') {
    state.browserPreview = true;
    render();
  }
  if (action.dataset.action === 'theme') {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    storageSet(THEME_KEY, theme);
    setTheme(theme);
  }
  if (action.dataset.action === 'stop-production') stopProduction();
  if (action.dataset.action === 'retry-production') retryProduction();
  if (action.dataset.action === 'attach-production') submitProductionEdit('production-attach', {});
  if (action.dataset.action === 'reuse-brief') {
    state.productionDraft = {...state.productionDraft, ...state.production?.request};
    render();
    document.getElementById('brief')?.focus();
  }
  if (action.dataset.action === 'refresh') refreshAll();
});

document.addEventListener('input', (event) => {
  const input = event.target;
  if (input.id === 'command-query') document.getElementById('command-results').innerHTML = commandResults(input.value);
  if (!input.form?.matches('[data-form="production"]')) return;
  state.productionDraft[input.name] = input.type === 'checkbox' ? input.checked
    : input.name === 'duration_sec' ? Number(input.value) : input.value;
});

document.addEventListener('change', (event) => {
  const select = event.target;
  if (!select.matches('[data-form="production-caption"] select[name="cue"]')) return;
  const cut = state.detail?.production;
  const cue = cut?.captions.find(item => item.id === select.value);
  if (!cue) return;
  select.form.elements.text.value = cue.text;
  const player = document.getElementById('working-video');
  if (player) player.currentTime = cue.start_frame / cut.fps;
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
  if (event.target.matches('[data-form="production"]')) sendProduction();
  if (event.target.matches('[data-form="shot-revision"]')) replaceShot(event.target);
  if (event.target.matches('[data-form^="production-"]')) stageProductionEdit(event.target);
  if (event.target.matches('[data-form="media-search"]')) findMedia(event.target);
});

document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k' && state.modal !== 'new-project') {
    event.preventDefault();
    if (state.modal === 'commands') closeProjectModal();
    else openCommands();
    return;
  }
  if (event.target?.id === 'brief' && (event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    event.preventDefault();
    event.target.form.requestSubmit();
    return;
  }
  if (!state.modal) return;
  if (event.key === 'Escape') {
    event.preventDefault();
    closeProjectModal();
    return;
  }
  if (event.target.id === 'command-query' && event.key === 'Enter') {
    event.preventDefault();
    document.querySelector('#command-results button')?.click();
    return;
  }
  if (event.key !== 'Tab') return;
  const modal = document.querySelector('[data-modal-content]');
  const focusable = [...modal.querySelectorAll('button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])')];
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
  const url = new URL(window.location.href);
  if (url.searchParams.get('view') === 'create') {
    state.view = 'create';
    render();
    return;
  }
  const id = url.searchParams.get('project') || state.projects[0]?.project_id || '';
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
