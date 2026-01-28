console.log('[i2j_ui] script loaded');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const btnCapture = document.getElementById('capture');
const btnClearQueue = document.getElementById('clear-queue');
const btnProcessQueued = document.getElementById('process-queued');
const fileInput = document.getElementById('file');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const metaTable = document.getElementById('meta-table');
const actions = document.getElementById('actions');
const traceEl = document.getElementById('trace');
const traceTable = document.getElementById('trace-table');
const traceOcrTextEl = document.getElementById('trace-ocr-text');
const traceOcrJsonEl = document.getElementById('trace-ocr-json');
const tracePromptEl = document.getElementById('trace-prompt');
const traceVlmEl = document.getElementById('trace-vlm');
const traceStepsEl = document.getElementById('trace-steps');
const traceJsonEl = document.getElementById('trace-json');
const consoleLogEl = document.getElementById('console-log');
const btnAccept = document.getElementById('accept');
const btnReject = document.getElementById('reject');
const btnPricing = document.getElementById('pricing');
// Tabs
const tabs = Array.from(document.querySelectorAll('.tab'));
const panelScanner = document.getElementById('panel-scanner');
const panelPricing = document.getElementById('panel-pricing');
const pricingJson = document.getElementById('pricing-json');
const pricingProvidersWrap = document.getElementById('pricing-providers');
const btnPricingRun = document.getElementById('pricing-run');
const pricingOut = document.getElementById('pricing-out');
const pricingProcessedSel = document.getElementById('pricing-processed');
const btnPricingLoad = document.getElementById('pricing-load');
const pricingReqTable = document.getElementById('pricing-reqTable');
const pricingRespTable = document.getElementById('pricing-respTable');
const pricingMergeTable = document.getElementById('pricing-mergeTable');
const envPipeline = document.getElementById('env-pipeline');
const queueList = document.getElementById('queue-list');
const modelSel = document.getElementById('model');
const backendSel = document.getElementById('backend');
const ocrSel = document.getElementById('ocr');
const runOcrChk = document.getElementById('run-ocr');
const preprocChk = document.getElementById('preproc');
const examplesSel = document.getElementById('examples');
const btnRunExample = document.getElementById('run-example');
const btnLoadExampleOutput = document.getElementById('load-example-output');
const btnTestModel = document.getElementById('test-model');
const edgeCropRange = document.getElementById('edge-crop');
const edgeCropVal = document.getElementById('edge-crop-val');
const autoCropChk = document.getElementById('auto-crop');
const overlayBox = document.querySelector('.overlay-box');
let pollTimer = null;
let logPollTimer = null;
let traceEventSource = null;
let logEventSource = null;
let jobEventSource = null;
let examplesIndex = {};

let lastId = null;
let lastMetadata = null; // last completed job metadata (typed, not DOM-parsed)
let captureQueue = []; // Array of { blob: Blob, url: string }
let tableRows = [];
let ollamaModels = [];

function initTraceTable(count, previews = []) {
  if (!traceTable) return;
  const tbody = traceTable.querySelector('tbody');
  tbody.innerHTML = '';
  tableRows = [];
  for (let i = 0; i < count; i++) {
    const tr = document.createElement('tr');
    const tdIn = document.createElement('td');
    const tdOut = document.createElement('td');
    const tdOcr = document.createElement('td');

    const inImg = document.createElement('img');
    inImg.className = 'trace-thumb';
    const previewUrl = previews[i] || '';
    if (previewUrl) inImg.src = previewUrl;
    tdIn.appendChild(inImg);

    const outImg = document.createElement('img');
    outImg.className = 'trace-thumb';
    tdOut.appendChild(outImg);

    const pre = document.createElement('pre');
    pre.className = 'trace-ocr hscroll';
    tdOcr.appendChild(pre);

    tr.appendChild(tdIn);
    tr.appendChild(tdOut);
    tr.appendChild(tdOcr);
    tbody.appendChild(tr);
    tableRows.push({ tr, tdInImg: inImg, tdOutImg: outImg, tdOcrPre: pre });
  }
}

async function init() {
  try {
    const health = await fetch('/api/health').then(r => r.json());
    const parts = [];
    parts.push(`pipeline:${health.pipeline_imported ? 'ok' : 'error'}`);
    parts.push(`pricing:${health.pricing_available ? 'ok' : 'off'}`);
    parts.push(`sheets:${health.google_sheets_configured ? 'on' : 'off'}`);
    envPipeline.textContent = `${health.status} (${parts.join(', ')})`;
    if (!health.pipeline_imported && health.pipeline_import_error) {
      envPipeline.title = String(health.pipeline_import_error);
    } else {
      envPipeline.title = '';
    }
  } catch (e) {
    envPipeline.textContent = 'unavailable';
  }

  // Load pricing providers
  try {
    const resp = await fetch('/api/pricing/providers');
    if (resp.ok && pricingProvidersWrap) {
      const data = await resp.json();
      const providers = data.providers || [];
      pricingProvidersWrap.innerHTML = providers.map(p => `
        <label><input type="checkbox" value="${p}" ${p === 'abebooks' ? 'checked' : ''}> ${p}</label>
      `).join('');
    }
  } catch {}

  // Load processed list for pricing
  try {
    const res = await fetch('/api/pricing/processed/list').then(r => r.json());
    if (pricingProcessedSel) {
      pricingProcessedSel.innerHTML = '';
      const placeholder = document.createElement('option');
      placeholder.value = ''; placeholder.textContent = res.items?.length ? 'Choose…' : 'No processed JSONs found';
      pricingProcessedSel.appendChild(placeholder);
      for (const it of (res.items || [])) {
        const opt = document.createElement('option');
        opt.value = it.path; opt.textContent = it.label;
        pricingProcessedSel.appendChild(opt);
      }
    }
  } catch {}

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.srcObject = stream;
  } catch (e) {
    console.warn('getUserMedia failed, use file upload instead', e);
    document.querySelector('.video-panel').classList.add('disabled');
  }

  // Load initial backend+model defaults to Gemini 2.5 flash
  backendSel.value = 'gemini';
  setModelOptionsForBackend('gemini');

  // Enforce model selection based on backend choice
  function enforceModelByBackend() {
    const b = (backendSel.value || 'ollama').toLowerCase();
    if (b === 'gemini') {
      // Default to a callable Gemini id (1.5 deprecated -> removed)
      const preferred = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.5-pro', 'gemini-2.0-flash'];
      const current = Array.from(modelSel.options).map(o => o.value);
      const pick = preferred.find(p => current.includes(p)) || current[0];
      if (pick) modelSel.value = pick;
    } else if (b === 'openai' || b === 'gpt' || b.startsWith('gpt-')) {
      modelSel.value = 'gpt-4o';
    }
  }
  backendSel.addEventListener('change', enforceModelByBackend);
  enforceModelByBackend();

  // Also update model option list to match backend
  function setModelOptionsForBackend(backend) {
    const b = (backend || 'ollama').toLowerCase();
    modelSel.innerHTML = '';
    if (b === 'gemini') {
      // Only include supported Gemini ids (drop 1.5 variants)
      const gemModels = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.5-pro', 'gemini-2.0-flash'];
      for (const m of gemModels) {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        modelSel.appendChild(opt);
      }
      modelSel.value = 'gemini-2.5-flash';
      appendUiLog('[ui] backend=gemini → model options set to 2.5/pro/flash-latest/2.0');
      return;
    }
    if (b === 'openai' || b === 'gpt' || b.startsWith('gpt-')) {
      const oaModels = ['gpt-4o', 'gpt-4o-mini'];
      for (const m of oaModels) {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        modelSel.appendChild(opt);
      }
      modelSel.value = 'gpt-4o';
      appendUiLog('[ui] backend=openai → model options set to GPT-4o list');
      return;
    }
    // Default: Ollama list
    for (const m of (ollamaModels || [])) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      modelSel.appendChild(opt);
    }
    if (ollamaModels && ollamaModels.length) {
      modelSel.value = ollamaModels.find(m => m.startsWith('gemma3:4b')) || ollamaModels[0];
    }
    appendUiLog('[ui] backend=ollama → model options set to local list');
  }
  backendSel.addEventListener('change', () => setModelOptionsForBackend(backendSel.value || 'gemini'));
  setModelOptionsForBackend(backendSel.value || 'gemini');

  // Default toggles: run OCR off, auto-crop on
  runOcrChk.checked = false;
  autoCropChk.checked = true;

  // Load examples
  try {
    const ex = await fetch('/api/examples').then(r => r.json());
    examplesSel.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = ''; placeholder.textContent = ex.items.length ? 'Choose example…' : 'No examples found';
    examplesSel.appendChild(placeholder);
    for (const item of ex.items) {
      const opt = document.createElement('option');
      opt.value = item.id; opt.textContent = `${item.id} (${item.count})${item.has_output ? ' • has output' : ''}`;
      examplesSel.appendChild(opt);
      examplesIndex[item.id] = { count: item.count };
    }
  } catch {}
}
function updateOverlay() {
  const pct = Number(edgeCropRange.value || 0);
  overlayBox.style.top = pct + '%';
  overlayBox.style.left = pct + '%';
  overlayBox.style.right = pct + '%';
  overlayBox.style.bottom = pct + '%';
  edgeCropVal.textContent = pct + '%';
}

edgeCropRange.addEventListener('input', updateOverlay);
updateOverlay();

function drawFrameToBlob(mime = 'image/jpeg', quality = 0.92) {
  const w = video.videoWidth || 1280;
  const h = video.videoHeight || 720;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  return new Promise(resolve => canvas.toBlob(resolve, mime, quality));
}

function renderTable(obj) {
  if (!obj || typeof obj !== 'object') return '';
  const keys = Object.keys(obj).sort();
  const rows = keys.map(k => {
    const v = obj[k];
    let valStr;
    // Preserve the original typed value for Accept/save (arrays should NOT be flattened).
    // Avoid embedding huge values (like _trace) into HTML attributes.
    const shouldStore = (k !== '_trace');
    const jsonValue = shouldStore ? JSON.stringify(v === undefined ? null : v) : null;
    const dataAttr = (shouldStore && jsonValue !== null) ? ` data-json="${escapeHtml(jsonValue)}"` : '';
    if (Array.isArray(v)) {
      valStr = `<pre class="hscroll"${dataAttr}>${escapeHtml(JSON.stringify(v, null, 2))}</pre>`;
    }
    else if (v && typeof v === 'object') {
      const preClass = (k === '_trace') ? ' class="hscroll"' : '';
      valStr = `<pre${preClass}${dataAttr}>${escapeHtml(JSON.stringify(v, null, 2))}</pre>`;
    }
    else {
      const display = (v === null || v === undefined) ? '' : String(v);
      // Wrap primitives so we can attach data-json for accurate parsing on Accept
      valStr = `<span${dataAttr}>${escapeHtml(display)}</span>`;
    }
    return `<tr><td>${escapeHtml(k)}</td><td${dataAttr}>${valStr}</td></tr>`;
  }).join('');
  return `<table class="kv"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderTrace(metadata) {
  const t = (metadata && metadata._trace) ? metadata._trace : null;
  if (!t) {
    if (traceTable) traceTable.querySelector('tbody').innerHTML = '';
    traceOcrTextEl.textContent = '';
    traceOcrJsonEl.textContent = '';
    tracePromptEl.textContent = '';
    traceVlmEl.textContent = '';
    traceStepsEl.innerHTML = '';
    traceJsonEl.textContent = '';
    return;
  }
  const images = Array.isArray(t.images) ? t.images : [];
  if (!tableRows.length && images.length) initTraceTable(images.length, []);
  for (let i = 0; i < images.length; i++) {
    const img = images[i] || {};
    if (!tableRows[i]) continue;
    const row = tableRows[i];
    if (img.original_b64) {
      if (!row.tdInImg.src || row.tdInImg.src.startsWith('blob:')) {
        row.tdInImg.src = img.original_b64;
      }
    }
    const proc = img.auto_cropped_b64 || img.edge_cropped_b64 || img.preprocessed_b64;
    if (proc) row.tdOutImg.src = proc;
    if (img.ocr_text) row.tdOcrPre.textContent = img.ocr_text;
  }
  // OCR texts
  let ocrTexts = [];
  images.forEach(img => { if (img && img.ocr_text) ocrTexts.push(img.ocr_text); });
  traceOcrTextEl.textContent = ocrTexts.join('\n\n' + ('-'.repeat(40)) + '\n\n');
  // OCR JSON heuristic
  traceOcrJsonEl.textContent = t.ocr_json ? JSON.stringify(t.ocr_json, null, 2) : '';
  // prompt/raw
  tracePromptEl.textContent = t.enhanced_prompt || '';
  traceVlmEl.textContent = t.ollama_raw || '';
  // steps feed
  const steps = Array.isArray(t.steps) ? t.steps : [];
  traceStepsEl.innerHTML = steps.map((s, i) => {
    const info = s.info ? escapeHtml(JSON.stringify(s.info)) : '';
    return `<div>[${String(i+1).padStart(2,'0')}] ${escapeHtml(s.step || '')} ${info}</div>`;
  }).join('\n');
  // raw trace JSON (redact large base64 fields to keep UI light)
  try {
    const redacted = JSON.parse(JSON.stringify(t));
    const imgs = Array.isArray(redacted.images) ? redacted.images : [];
    for (const im of imgs) {
      if (im.original_b64) im.original_b64 = '[base64 omitted]';
      if (im.preprocessed_b64) im.preprocessed_b64 = '[base64 omitted]';
      if (im.edge_cropped_b64) im.edge_cropped_b64 = '[base64 omitted]';
      if (im.auto_cropped_b64) im.auto_cropped_b64 = '[base64 omitted]';
    }
    traceJsonEl.textContent = JSON.stringify(redacted, null, 2);
  } catch { traceJsonEl.textContent = ''; }
}

function startTracePolling(id) {
  let lastTs = 0;
  let lastSeq = -1;
  // Prefer SSE if available
  try {
    if (traceEventSource) { try { traceEventSource.close(); } catch (e) {} }
    traceEventSource = new EventSource(`/api/trace_stream?id=${encodeURIComponent(id)}&last_ts=${lastTs}&last_seq=${lastSeq}`);
    traceEventSource.onopen = () => {
      // when reconnecting after server restart, keep using SSE
    };
    traceEventSource.addEventListener('trace', (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        const it = payload;
        if (it && it.ts !== undefined) lastTs = Math.max(lastTs, it.ts || 0);
        if (it && it.seq !== undefined) lastSeq = Math.max(lastSeq, it.seq || -1);
        if (it && it.trace) renderTrace({ _trace: it.trace });
      } catch {}
    });
    traceEventSource.onerror = () => {
      // Delay fallback briefly to allow auto-retry to reconnect after server restart
      const prevLastTs = lastTs, prevLastSeq = lastSeq;
      setTimeout(() => {
        if (!traceEventSource || traceEventSource.readyState === EventSource.CLOSED) {
          try { if (traceEventSource) traceEventSource.close(); } catch (e) {}
          traceEventSource = null;
          startTraceInterval(id, prevLastTs, prevLastSeq);
        }
      }, 2200);
    };
    return;
  } catch (e) {
    // ignore and fallback
  }
  startTraceInterval(id, lastTs, lastSeq);
}

function startTraceInterval(id, lastTs, lastSeq) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/trace_poll?id=${encodeURIComponent(id)}&last_ts=${lastTs}&last_seq=${lastSeq}`);
      const data = await resp.json();
      const items = data.items || [];
      if (!items.length) return;
      lastTs = Math.max(lastTs, ...items.map(it => it.ts || 0));
      lastSeq = Math.max(lastSeq, ...items.map(it => it.seq || -1));
      const latest = items[items.length - 1].trace;
      renderTrace({ _trace: latest });
    } catch {}
  }, 3200);
}

function startLogPolling(id) {
  let lastTs = 0;
  let lastSeq = -1;
  // Prefer SSE
  try {
    if (logEventSource) { try { logEventSource.close(); } catch (e) {} }
    if (consoleLogEl) consoleLogEl.textContent = '';
    logEventSource = new EventSource(`/api/log_stream?id=${encodeURIComponent(id)}&last_ts=${lastTs}&last_seq=${lastSeq}`);
    logEventSource.onopen = () => {
      // keep SSE after restart
    };
    logEventSource.addEventListener('log', (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        const it = payload;
        if (it && it.ts !== undefined) lastTs = Math.max(lastTs, it.ts || 0);
        if (it && it.seq !== undefined) lastSeq = Math.max(lastSeq, it.seq || -1);
        if (consoleLogEl && it && it.line !== undefined) {
          const text = String(it.line || '');
          consoleLogEl.textContent += (consoleLogEl.textContent ? '\n' : '') + text;
          if (consoleLogEl.textContent.length > 10000) {
            consoleLogEl.textContent = consoleLogEl.textContent.slice(-10000);
          }
          consoleLogEl.scrollTop = consoleLogEl.scrollHeight;
        }
      } catch {}
    });
    logEventSource.onerror = () => {
      const prevLastTs = lastTs, prevLastSeq = lastSeq;
      setTimeout(() => {
        if (!logEventSource || logEventSource.readyState === EventSource.CLOSED) {
          try { if (logEventSource) logEventSource.close(); } catch (e) {}
          logEventSource = null;
          startLogInterval(id, prevLastTs, prevLastSeq);
        }
      }, 2200);
    };
    return;
  } catch (e) {}
  startLogInterval(id, lastTs, lastSeq);
}

function startLogInterval(id, lastTs, lastSeq) {
  if (logPollTimer) clearInterval(logPollTimer);
  if (consoleLogEl) consoleLogEl.textContent = '';
  logPollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/log_poll?id=${encodeURIComponent(id)}&last_ts=${lastTs}&last_seq=${lastSeq}`);
      const data = await resp.json();
      const items = data.items || [];
      if (!items.length) return;
      lastTs = Math.max(lastTs, ...items.map(it => it.ts || 0));
      lastSeq = Math.max(lastSeq, ...items.map(it => it.seq || -1));
      if (consoleLogEl) {
        const text = items.map(it => (it.line || '')).join('\n');
        consoleLogEl.textContent += (consoleLogEl.textContent ? '\n' : '') + text;
        if (consoleLogEl.textContent.length > 10000) {
          consoleLogEl.textContent = consoleLogEl.textContent.slice(-10000);
        }
        consoleLogEl.scrollTop = consoleLogEl.scrollHeight;
      }
    } catch {}
  }, 1200);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

// UI console logger (always available, even without a running pipeline)
function appendUiLog(line) {
  try {
    if (!consoleLogEl) return;
    const ts = new Date().toISOString();
    const text = `[${ts}] ${String(line || '')}`;
    consoleLogEl.textContent += (consoleLogEl.textContent ? '\n' : '') + text;
    if (consoleLogEl.textContent.length > 10000) {
      consoleLogEl.textContent = consoleLogEl.textContent.slice(-10000);
    }
    consoleLogEl.scrollTop = consoleLogEl.scrollHeight;
  } catch {}
}

function refreshQueueList() {
  queueList.innerHTML = '';
  captureQueue.forEach((item, i) => {
    const li = document.createElement('li');
    li.className = 'queue-item';

    const img = document.createElement('img');
    img.className = 'queue-thumb';
    img.alt = `capture ${i + 1}`;
    img.loading = 'lazy';
    if (item && item.url) img.src = item.url;

    const meta = document.createElement('div');
    meta.className = 'queue-meta';
    const kb = item && item.blob ? Math.round(item.blob.size / 1024) : 0;
    meta.textContent = `image ${i + 1} (${kb} KB)`;

    li.appendChild(img);
    li.appendChild(meta);
    queueList.appendChild(li);
  });
}

function readTableToObject() {
  const rows = Array.from(document.querySelectorAll('table.kv tbody tr'));
  const obj = {};
  for (const tr of rows) {
    const tds = tr.querySelectorAll('td');
    const k = tds[0].textContent.trim();
    const valTd = tds[1];
    let v;
    // Prefer the typed JSON stored by renderTable()
    const json = (valTd && valTd.dataset && typeof valTd.dataset.json === 'string') ? valTd.dataset.json : null;
    if (json) {
      try { v = JSON.parse(json); } catch { v = valTd.textContent.trim(); }
    } else {
      v = valTd.textContent.trim();
      try { v = JSON.parse(v); } catch {}
    }
    obj[k] = v;
  }
  return obj;
}

async function processSingle(blob, filename = 'capture.jpg') {
  statusEl.textContent = 'Uploading...';
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');
  lastMetadata = null;
  if (consoleLogEl) consoleLogEl.textContent = '';

  const fd = new FormData();
  fd.append('image', blob, filename);
  fd.append('model', modelSel.value || 'gemma3:4b');
  fd.append('ocr_engine', ocrSel.value || 'easyocr');
  fd.append('run_ocr', runOcrChk && runOcrChk.checked ? 'true' : 'false');
  fd.append('use_preprocessing', preprocChk.checked ? 'true' : 'false');
  fd.append('edge_crop', String(Number(edgeCropRange.value || 0)));
  fd.append('crop_ocr', autoCropChk.checked ? 'true' : 'false');
  fd.append('llm_backend', (backendSel.value || 'ollama'));

  // initialize table with 1 row and a local preview
  try {
    const preview = URL.createObjectURL(blob);
    initTraceTable(1, [preview]);
  } catch {}

  const resp = await fetch('/api/process_image', { method: 'POST', body: fd });
  const data = await resp.json();
  if (!resp.ok) {
    statusEl.textContent = 'Error';
    errorEl.textContent = data.error || 'Unknown error';
    errorEl.classList.remove('hidden');
    return;
  }

  lastId = data.id;
  statusEl.textContent = `Started: ${data.files.join(', ')}`;
  if (lastId) startTracePolling(lastId);
  if (lastId) startLogPolling(lastId);
  if (jobEventSource) { try { jobEventSource.close(); } catch (e) {} }
  jobEventSource = new EventSource(`/api/job_stream?id=${encodeURIComponent(lastId)}`);
  jobEventSource.addEventListener('job', (ev) => {
    try {
      const j = JSON.parse(ev.data);
      if (!j || !j.status) return;
      if (j.status === 'error') {
        statusEl.textContent = 'Error';
        errorEl.textContent = j.error || 'Unknown error';
        errorEl.classList.remove('hidden');
        cleanupStreams();
        return;
      }
      if (j.status === 'done') {
        statusEl.textContent = `Processed: ${(j.files || []).join(', ')}`;
        if (j.metadata) {
          lastMetadata = j.metadata;
          metaTable.innerHTML = renderTable(j.metadata);
          renderTrace(j.metadata);
        }
        actions.classList.remove('hidden');
        cleanupStreams();
      }
    } catch {}
  });
}

async function processBatch(blobs) {
  statusEl.textContent = 'Uploading batch...';
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');
  lastMetadata = null;
  if (consoleLogEl) consoleLogEl.textContent = '';

  const fd = new FormData();
  for (let i = 0; i < blobs.length; i++) {
    fd.append('images', blobs[i], `capture_${i}.jpg`);
  }
  fd.append('model', modelSel.value || 'gemma3:4b');
  fd.append('ocr_engine', ocrSel.value || 'easyocr');
  fd.append('run_ocr', runOcrChk && runOcrChk.checked ? 'true' : 'false');
  fd.append('use_preprocessing', preprocChk.checked ? 'true' : 'false');
  fd.append('edge_crop', String(Number(edgeCropRange.value || 0)));
  fd.append('crop_ocr', autoCropChk.checked ? 'true' : 'false');
  fd.append('llm_backend', (backendSel.value || 'ollama'));

  // initialize table rows with local previews
  try {
    const previews = blobs.map(b => URL.createObjectURL(b));
    initTraceTable(blobs.length, previews);
  } catch {}

  const resp = await fetch('/api/process_images', { method: 'POST', body: fd });
  const data = await resp.json();
  if (!resp.ok) {
    statusEl.textContent = 'Error';
    errorEl.textContent = data.error || 'Unknown error';
    errorEl.classList.remove('hidden');
    return;
  }

  lastId = data.id;
  statusEl.textContent = `Started: ${data.files.join(', ')}`;
  if (lastId) startTracePolling(lastId);
  if (lastId) startLogPolling(lastId);
  if (jobEventSource) { try { jobEventSource.close(); } catch (e) {} }
  jobEventSource = new EventSource(`/api/job_stream?id=${encodeURIComponent(lastId)}`);
  jobEventSource.addEventListener('job', (ev) => {
    try {
      const j = JSON.parse(ev.data);
      if (!j || !j.status) return;
      if (j.status === 'error') {
        statusEl.textContent = 'Error';
        errorEl.textContent = j.error || 'Unknown error';
        errorEl.classList.remove('hidden');
        cleanupStreams();
        return;
      }
      if (j.status === 'done') {
        statusEl.textContent = `Processed: ${(j.files || []).join(', ')}`;
        if (j.metadata) {
          lastMetadata = j.metadata;
          metaTable.innerHTML = renderTable(j.metadata);
          renderTrace(j.metadata);
        }
        actions.classList.remove('hidden');
        cleanupStreams();
      }
    } catch {}
  });
}

btnCapture.addEventListener('click', async () => {
  const blob = await drawFrameToBlob();
  if (!blob) {
    statusEl.textContent = 'Capture failed';
    return;
  }
  const url = URL.createObjectURL(blob);
  captureQueue.push({ blob, url });
  refreshQueueList();
});

btnClearQueue.addEventListener('click', () => {
  // Revoke preview URLs to avoid leaking memory
  try {
    for (const it of captureQueue) {
      if (it && it.url) URL.revokeObjectURL(it.url);
    }
  } catch {}
  captureQueue = [];
  refreshQueueList();
});

btnProcessQueued.addEventListener('click', async () => {
  if (!captureQueue.length) {
    statusEl.textContent = 'Queue empty';
    return;
  }
  // Clear previous results/logs/trace before starting a new batch
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');
  if (consoleLogEl) consoleLogEl.textContent = '';
  initTraceTable(0, []);
  await processBatch(captureQueue.map(it => it.blob));
});

fileInput.addEventListener('change', async (e) => {
  if (!e.target.files || !e.target.files.length) return;
  const blobs = Array.from(e.target.files);
  // Clear previous results/logs/trace when loading a new image set
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');
  if (consoleLogEl) consoleLogEl.textContent = '';
  initTraceTable(0, []);
  if (blobs.length === 1) await processSingle(blobs[0], blobs[0].name);
  else await processBatch(blobs);
});

btnAccept.addEventListener('click', async () => {
  if (!lastId) {
    statusEl.textContent = 'Nothing to accept yet';
    return;
  }
  // Prefer the real typed JSON from the backend; fall back to DOM parsing.
  const metadata = lastMetadata || readTableToObject();
  const notes = prompt('Add a note (optional):') || '';
  try {
    statusEl.textContent = 'Saving...';
    const resp = await fetch('/api/accept', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: lastId, metadata, notes }) });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = 'Save failed';
      errorEl.textContent = data.detail || data.error || 'Unknown error';
      errorEl.classList.remove('hidden');
      return;
    }
    statusEl.textContent = `Saved → ${data.path}`;
    // Switch to pricing tab and post the accepted metadata to the iframe
    try {
      switchTab('pricing');
      const iframe = document.querySelector('#panel-pricing iframe');
      const key = data && data.transfer_key;
      if (iframe && key) {
        // Navigate iframe with transfer key so it can fetch
        const url = new URL(iframe.src, window.location.origin);
        url.searchParams.set('key', key);
        iframe.src = url.toString();
        // Also send the metadata as a backup (no auto-run to avoid double runs)
        const send = () => {
          try {
            iframe.contentWindow && iframe.contentWindow.postMessage(
              { type: 'scannerAccepted', id: lastId, metadata, autoRun: false },
              '*'
            );
          } catch {}
        };
        iframe.addEventListener('load', send, { once: true });
      } else if (iframe) {
        // Fallback to postMessage if key missing
        const send = () => {
          try {
            iframe.contentWindow && iframe.contentWindow.postMessage(
              { type: 'scannerAccepted', id: lastId, metadata, autoRun: true },
              '*'
            );
          } catch {}
        };
        if (iframe.contentWindow && iframe.contentDocument && iframe.contentDocument.readyState === 'complete') send(); else iframe.addEventListener('load', send, { once: true });
      }
    } catch {}
  } catch (e) {
    statusEl.textContent = 'Save failed';
    errorEl.textContent = String(e);
    errorEl.classList.remove('hidden');
  }
});

btnReject.addEventListener('click', async () => {
  if (!lastId) return;
  const reason = prompt('Reason (optional):') || '';
  const resp = await fetch('/api/reject', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: lastId, reason }) });
  statusEl.textContent = resp.ok ? 'Rejected' : 'Reject failed';
  actions.classList.add('hidden');
});

btnRunExample.addEventListener('click', async () => {
  const id = examplesSel.value;
  if (!id) return;
  statusEl.textContent = `Running example ${id}...`;
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');
  lastMetadata = null;
  if (consoleLogEl) consoleLogEl.textContent = '';
  initTraceTable(0, []);

  // initialize table rows for this example using known count
  try {
    const count = (examplesIndex[id] && examplesIndex[id].count) || 0;
    if (count > 0) initTraceTable(count, []);
  } catch {}

  const resp = await fetch('/api/process_example', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ book_id: id, model: modelSel.value || 'gemma3:4b', ocr_engine: ocrSel.value || 'easyocr', run_ocr: runOcrChk && runOcrChk.checked, use_preprocessing: preprocChk.checked, edge_crop: Number(edgeCropRange.value || 0), crop_ocr: autoCropChk.checked, llm_backend: backendSel.value || 'ollama' }) });
  const data = await resp.json();
  if (!resp.ok) {
    statusEl.textContent = 'Error';
    errorEl.textContent = data.detail || data.error || 'Unknown error';
    errorEl.classList.remove('hidden');
    return;
  }

  lastId = data.id;
  statusEl.textContent = `Started: ${(data.files || []).join(', ') || id}`;
  if (lastId) startTracePolling(lastId);
  if (lastId) startLogPolling(lastId);
  if (jobEventSource) { try { jobEventSource.close(); } catch (e) {} }
  jobEventSource = new EventSource(`/api/job_stream?id=${encodeURIComponent(lastId)}`);
  jobEventSource.addEventListener('job', (ev) => {
    try {
      const j = JSON.parse(ev.data);
      if (!j || !j.status) return;
      if (j.status === 'error') {
        statusEl.textContent = 'Error';
        errorEl.textContent = j.error || 'Unknown error';
        errorEl.classList.remove('hidden');
        cleanupStreams();
        return;
      }
      if (j.status === 'done') {
        statusEl.textContent = `Processed: ${(j.files || []).join(', ') || id}`;
        if (j.metadata) {
          lastMetadata = j.metadata;
          metaTable.innerHTML = renderTable(j.metadata);
          renderTrace(j.metadata);
        }
        actions.classList.remove('hidden');
        cleanupStreams();
      }
    } catch {}
  });
});

btnLoadExampleOutput.addEventListener('click', async () => {
  const id = examplesSel.value;
  if (!id) return;
  statusEl.textContent = `Loading saved output for ${id}...`;
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');
  lastMetadata = null;

  const resp = await fetch(`/api/example_output?book_id=${encodeURIComponent(id)}`);
  const data = await resp.json();
  if (!resp.ok) {
    statusEl.textContent = 'Error';
    errorEl.textContent = data.detail || data.error || 'Unknown error';
    errorEl.classList.remove('hidden');
    return;
  }

  lastId = data.id;
  statusEl.textContent = `Loaded saved output: ${data.file}`;
  lastMetadata = data.metadata || null;
  metaTable.innerHTML = renderTable(data.metadata);
  renderTrace(data.metadata);
  actions.classList.remove('hidden');
});

btnPricing.addEventListener('click', async () => {
  switchTab('pricing');
  // Carry current scan JSON over even without Accept
  try {
    const iframe = document.querySelector('#panel-pricing iframe');
    const metadata = lastMetadata || null;
    if (!iframe || !metadata) return;
    const send = () => {
      try {
        iframe.contentWindow && iframe.contentWindow.postMessage(
          { type: 'scannerAccepted', id: lastId, metadata, autoRun: true },
          '*'
        );
      } catch {}
    };
    // Wait for the pricing iframe to load if needed
    iframe.addEventListener('load', send, { once: true });
    // Best-effort immediate send (works if already loaded)
    send();
  } catch {}
});

btnTestModel.addEventListener('click', async () => {
  const backend = (backendSel.value || 'ollama');
  const model = (modelSel.value || '');
  appendUiLog(`[test] starting: backend=${backend} model=${model}`);
  statusEl.textContent = 'Testing model...';
  errorEl.classList.add('hidden');
  try {
    const resp = await fetch('/api/test_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend, model })
    });
    appendUiLog(`[test] response status: ${resp.status} ${resp.statusText}`);
    const data = await resp.json();
    appendUiLog(`[test] response body: ${JSON.stringify({ ok: data.ok, backend: data.backend, model: data.model, status: data.status, error: data.error ? String(data.error).slice(0,200) : undefined })}`);
    if (!resp.ok || !data.ok) {
      statusEl.textContent = 'Model test failed';
      errorEl.textContent = data.error || data.detail || 'Unknown error';
      errorEl.classList.remove('hidden');
    } else {
      statusEl.textContent = `Model test ok (${data.backend}:${data.model})`;
      if (consoleLogEl && data.detail) {
        appendUiLog(`[test] detail: ${String(data.detail).slice(0, 300)}`);
      }
    }
  } catch (e) {
    statusEl.textContent = 'Model test error';
    errorEl.textContent = String(e);
    errorEl.classList.remove('hidden');
    appendUiLog(`[test] exception: ${String(e)}`);
  }
});

init();
console.log('[i2j_ui] init called');

function switchTab(name) {
  tabs.forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  if (name === 'pricing') {
    panelPricing.classList.remove('hidden');
    panelPricing.classList.add('active');
    panelScanner.classList.add('hidden');
    panelScanner.classList.remove('active');
  } else {
    panelScanner.classList.remove('hidden');
    panelScanner.classList.add('active');
    panelPricing.classList.add('hidden');
    panelPricing.classList.remove('active');
  }
}

tabs.forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));

window.addEventListener('beforeunload', () => {
  try { if (traceEventSource) traceEventSource.close(); } catch (e) {}
  try { if (logEventSource) logEventSource.close(); } catch (e) {}
  try { if (jobEventSource) jobEventSource.close(); } catch (e) {}
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  if (logPollTimer) { clearInterval(logPollTimer); logPollTimer = null; }
});

function cleanupStreams() {
  if (traceEventSource) { try { traceEventSource.close(); } catch (e) {} traceEventSource = null; }
  if (logEventSource) { try { logEventSource.close(); } catch (e) {} logEventSource = null; }
  if (jobEventSource) { try { jobEventSource.close(); } catch (e) {} jobEventSource = null; }
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  if (logPollTimer) { clearInterval(logPollTimer); logPollTimer = null; }
}


