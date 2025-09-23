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
const btnAccept = document.getElementById('accept');
const btnReject = document.getElementById('reject');
const btnPricing = document.getElementById('pricing');
const envPipeline = document.getElementById('env-pipeline');
const queueList = document.getElementById('queue-list');
const modelSel = document.getElementById('model');
const ocrSel = document.getElementById('ocr');
const preprocChk = document.getElementById('preproc');
const examplesSel = document.getElementById('examples');
const btnRunExample = document.getElementById('run-example');
const btnLoadExampleOutput = document.getElementById('load-example-output');
const edgeCropRange = document.getElementById('edge-crop');
const edgeCropVal = document.getElementById('edge-crop-val');
const autoCropChk = document.getElementById('auto-crop');
const overlayBox = document.querySelector('.overlay-box');
let pollTimer = null;
let examplesIndex = {};

let lastId = null;
let captureQueue = []; // Array of Blobs
let tableRows = [];

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
    envPipeline.textContent = `${health.status} (pipeline:${health.pipeline_imported ? 'ok' : 'error'})`;
  } catch (e) {
    envPipeline.textContent = 'unavailable';
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.srcObject = stream;
  } catch (e) {
    console.warn('getUserMedia failed, use file upload instead', e);
    document.querySelector('.video-panel').classList.add('disabled');
  }

  // Load models
  try {
    const data = await fetch('/api/models').then(r => r.json());
    const models = data.models || [];
    modelSel.innerHTML = '';
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m.startsWith('gemma3:4b')) opt.selected = true;
      modelSel.appendChild(opt);
    }
  } catch {}

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
    if (Array.isArray(v)) valStr = v.join(', ');
    else if (v && typeof v === 'object') valStr = `<pre>${escapeHtml(JSON.stringify(v, null, 2))}</pre>`;
    else valStr = (v === null || v === undefined) ? '' : String(v);
    return `<tr><td>${escapeHtml(k)}</td><td>${valStr}</td></tr>`;
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
  // raw trace JSON
  try { traceJsonEl.textContent = JSON.stringify(t, null, 2); } catch { traceJsonEl.textContent = ''; }
}

function startTracePolling(id) {
  let lastTs = 0;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/trace_poll?id=${encodeURIComponent(id)}&last_ts=${lastTs}`);
      const data = await resp.json();
      const items = data.items || [];
      if (!items.length) return;
      lastTs = Math.max(lastTs, ...items.map(it => it.ts || 0));
      const latest = items[items.length - 1].trace;
      renderTrace({ _trace: latest });
    } catch {}
  }, 500);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function refreshQueueList() {
  queueList.innerHTML = '';
  captureQueue.forEach((b, i) => {
    const li = document.createElement('li');
    li.textContent = `image ${i+1} (${Math.round(b.size/1024)} KB)`;
    queueList.appendChild(li);
  });
}

function readTableToObject() {
  const rows = Array.from(document.querySelectorAll('table.kv tbody tr'));
  const obj = {};
  for (const tr of rows) {
    const tds = tr.querySelectorAll('td');
    const k = tds[0].textContent.trim();
    let v = tds[1].textContent.trim();
    try { v = JSON.parse(v); } catch {}
    obj[k] = v;
  }
  return obj;
}

async function processSingle(blob, filename = 'capture.jpg') {
  statusEl.textContent = 'Uploading...';
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');

  const fd = new FormData();
  fd.append('image', blob, filename);
  fd.append('model', modelSel.value || 'gemma3:4b');
  fd.append('ocr_engine', ocrSel.value || 'easyocr');
  fd.append('use_preprocessing', preprocChk.checked ? 'true' : 'false');
  fd.append('edge_crop', String(Number(edgeCropRange.value || 0)));
  fd.append('crop_ocr', autoCropChk.checked ? 'true' : 'false');

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
  (async function pollResult() {
    try {
      const r = await fetch(`/api/job_result?id=${encodeURIComponent(lastId)}`);
      const j = await r.json();
      if (r.status === 202) {
        setTimeout(pollResult, 600);
        return;
      }
      if (j && j.status === 'error') {
        statusEl.textContent = 'Error';
        errorEl.textContent = j.error || 'Unknown error';
        errorEl.classList.remove('hidden');
        return;
      }
      if (r.ok && j && j.status === 'done') {
        statusEl.textContent = `Processed: ${j.files.join(', ')}`;
        metaTable.innerHTML = renderTable(j.metadata);
        renderTrace(j.metadata);
        actions.classList.remove('hidden');
      }
    } catch (e) {}
  })();
}

async function processBatch(blobs) {
  statusEl.textContent = 'Uploading batch...';
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');

  const fd = new FormData();
  for (let i = 0; i < blobs.length; i++) {
    fd.append('images', blobs[i], `capture_${i}.jpg`);
  }
  fd.append('model', modelSel.value || 'gemma3:4b');
  fd.append('ocr_engine', ocrSel.value || 'easyocr');
  fd.append('use_preprocessing', preprocChk.checked ? 'true' : 'false');
  fd.append('edge_crop', String(Number(edgeCropRange.value || 0)));
  fd.append('crop_ocr', autoCropChk.checked ? 'true' : 'false');

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
  (async function pollResult() {
    try {
      const r = await fetch(`/api/job_result?id=${encodeURIComponent(lastId)}`);
      const j = await r.json();
      if (r.status === 202) {
        setTimeout(pollResult, 600);
        return;
      }
      if (j && j.status === 'error') {
        statusEl.textContent = 'Error';
        errorEl.textContent = j.error || 'Unknown error';
        errorEl.classList.remove('hidden');
        return;
      }
      if (r.ok && j && j.status === 'done') {
        statusEl.textContent = `Processed: ${j.files.join(', ')}`;
        metaTable.innerHTML = renderTable(j.metadata);
        renderTrace(j.metadata);
        actions.classList.remove('hidden');
      }
    } catch (e) {}
  })();
}

btnCapture.addEventListener('click', async () => {
  const blob = await drawFrameToBlob();
  captureQueue.push(blob);
  refreshQueueList();
});

btnClearQueue.addEventListener('click', () => {
  captureQueue = [];
  refreshQueueList();
});

btnProcessQueued.addEventListener('click', async () => {
  if (!captureQueue.length) {
    statusEl.textContent = 'Queue empty';
    return;
  }
  await processBatch(captureQueue);
});

fileInput.addEventListener('change', async (e) => {
  if (!e.target.files || !e.target.files.length) return;
  const blobs = Array.from(e.target.files);
  if (blobs.length === 1) await processSingle(blobs[0], blobs[0].name);
  else await processBatch(blobs);
});

btnAccept.addEventListener('click', async () => {
  if (!lastId) return;
  const metadata = readTableToObject();
  const resp = await fetch('/api/accept', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: lastId, metadata }) });
  const data = await resp.json();
  statusEl.textContent = resp.ok ? `Saved → ${data.path}` : 'Save failed';
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

  // initialize table rows for this example using known count
  try {
    const count = (examplesIndex[id] && examplesIndex[id].count) || 0;
    if (count > 0) initTraceTable(count, []);
  } catch {}

  const resp = await fetch('/api/process_example', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ book_id: id, model: modelSel.value || 'gemma3:4b', ocr_engine: ocrSel.value || 'easyocr', use_preprocessing: preprocChk.checked, edge_crop: Number(edgeCropRange.value || 0), crop_ocr: autoCropChk.checked }) });
  const data = await resp.json();
  if (!resp.ok) {
    statusEl.textContent = 'Error';
    errorEl.textContent = data.detail || data.error || 'Unknown error';
    errorEl.classList.remove('hidden');
    return;
  }

  lastId = data.id;
  statusEl.textContent = `Processed example: ${id}`;
  metaTable.innerHTML = renderTable(data.metadata);
  renderTrace(data.metadata);
  actions.classList.remove('hidden');
  if (lastId) startTracePolling(lastId);
});

btnLoadExampleOutput.addEventListener('click', async () => {
  const id = examplesSel.value;
  if (!id) return;
  statusEl.textContent = `Loading saved output for ${id}...`;
  errorEl.classList.add('hidden');
  metaTable.innerHTML = '';
  actions.classList.add('hidden');

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
  metaTable.innerHTML = renderTable(data.metadata);
  renderTrace(data.metadata);
  actions.classList.remove('hidden');
});

btnPricing.addEventListener('click', async () => {
  const meta = readTableToObject();
  statusEl.textContent = 'Pricing lookup… (placeholder)';
  const resp = await fetch('/api/pricing_lookup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ isbn_13: meta['isbn_13'] || null, isbn_10: meta['isbn_10'] || null, title: meta['title'] || null, authors: meta['authors'] || [] }) });
  const data = await resp.json();
  if (!resp.ok) {
    statusEl.textContent = 'Pricing lookup failed';
    return;
  }
  statusEl.textContent = data.message || 'Pricing lookup placeholder';
});

init();


