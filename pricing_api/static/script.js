const envEl = document.getElementById('env');
const providersEl = document.getElementById('providers');
const jsonEl = document.getElementById('json');
const outEl = document.getElementById('out');
const btnRun = document.getElementById('btnRun');
const processedSel = document.getElementById('processed');
const btnLoad = document.getElementById('btnLoad');
const reqTable = document.getElementById('reqTable');
const respTable = document.getElementById('respTable');

async function init() {
  try {
    const root = await fetch('/').then(r => r.json());
    envEl.textContent = `status: ${root.status}`;
  } catch (e) {
    envEl.textContent = 'status: unavailable';
  }

  try {
    const list = await fetch('/providers').then(r => r.json());
    const provs = list.providers || [];
    providersEl.innerHTML = '';
    for (const p of provs) {
      const id = `prov-${p}`;
      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.value = p; cb.id = id; cb.checked = true;
      const span = document.createElement('span');
      span.textContent = p;
      label.appendChild(cb); label.appendChild(span);
      providersEl.appendChild(label);
    }
  } catch {}

  // Load processed list
  try {
    const res = await fetch('/processed/list').then(r => r.json());
    processedSel.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = ''; placeholder.textContent = res.items?.length ? 'Choose…' : 'No processed JSONs found';
    processedSel.appendChild(placeholder);
    for (const it of (res.items || [])) {
      const opt = document.createElement('option');
      opt.value = it.path; opt.textContent = it.label;
      processedSel.appendChild(opt);
    }
  } catch {}
}

btnLoad.addEventListener('click', async () => {
  const path = processedSel.value;
  if (!path) return;
  try {
    const data = await fetch(`/processed/load?path=${encodeURIComponent(path)}`).then(r => r.json());
    // Show the full raw JSON exactly as stored (including nulls/missing values)
    jsonEl.value = JSON.stringify(data.raw, null, 2);
  } catch (e) {
    alert('Failed to load processed JSON');
  }
});

function getSelectedProviders() {
  const cbs = providersEl.querySelectorAll('input[type="checkbox"]');
  const selected = [];
  for (const cb of cbs) if (cb.checked) selected.push(cb.value);
  return selected;
}

function toTable(obj) {
  if (!obj || typeof obj !== 'object') return '';
  const keys = Object.keys(obj);
  if (!keys.length) return '<div class="muted">(empty)</div>';
  const rows = keys.map(k => {
    let v = obj[k];
    if (Array.isArray(v)) v = v.join(', ');
    else if (v && typeof v === 'object') v = JSON.stringify(v);
    else if (v === null || v === undefined) v = '';
    return `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v))}</td></tr>`;
  }).join('');
  return `<table class="kv"><tbody>${rows}</tbody></table>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

btnRun.addEventListener('click', async () => {
  outEl.textContent = 'Running...';
  let payload;
  try {
    payload = JSON.parse(jsonEl.value || '{}');
  } catch (e) {
    outEl.textContent = 'Invalid JSON input';
    return;
  }
  const providers = getSelectedProviders();
  if (providers.length) payload.providers = providers;
  try {
    const resp = await fetch('/lookup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    outEl.textContent = JSON.stringify(data, null, 2);
    // Render request table
    reqTable.innerHTML = toTable(data.query || payload);
    // Choose best offer: prefer exact ISBN match, else same title (case-insens), else first
    const offers = data.offers || [];
    const q_isbn13 = (data.query?.isbn_13 || '').replace(/[-\s]/g, '');
    const q_isbn10 = (data.query?.isbn_10 || '').replace(/[-\s]/g, '');
    const q_title = (data.query?.title || '').trim().toLowerCase();
    let best = null;
    for (const o of offers) {
      const oi13 = (o.isbn_13 || '').replace(/[-\s]/g, '');
      const oi10 = (o.isbn_10 || '').replace(/[-\s]/g, '');
      if (q_isbn13 && oi13 === q_isbn13) { best = o; break; }
      if (q_isbn10 && oi10 === q_isbn10) { best = o; break; }
    }
    if (!best && q_title) {
      best = offers.find(o => (o.title || '').trim().toLowerCase() === q_title) || null;
    }
    if (!best) best = offers[0] || null;
    respTable.innerHTML = best ? toTable(best) : '<div class="muted">No offers</div>';
  } catch (e) {
    outEl.textContent = String(e);
  }
});

init();


