const envEl = document.getElementById('env');
const providersEl = document.getElementById('providers');
const jsonEl = document.getElementById('json');
const outEl = document.getElementById('out');
const btnRun = document.getElementById('btnRun');
const processedSel = document.getElementById('processed');
const btnLoad = document.getElementById('btnLoad');
const reqTable = document.getElementById('reqTable');
const respTable = document.getElementById('respTable');
const mergeTable = document.getElementById('mergeTable');

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

    // Build merged view: start with original input fields (ensure keys exist), then fill missing from best
    const src = data.query || payload || {};
    // Start merged as a deep-ish copy of the original full JSON (not only request fields)
    // We use the current textarea value as the authoritative source so it includes all fields.
    let merged;
    try { merged = JSON.parse(jsonEl.value || '{}'); } catch { merged = { ...src }; }
    if (merged && typeof merged !== 'object') merged = { ...src };
    // Ensure commonly expected keys exist even if not present
    const ensureKeys = [
      'title','subtitle','authors','publisher','publication_date','isbn_13','isbn_10',
      'asin','edition','binding_type','language','page_count','categories','description','condition_keywords','price'
    ];
    for (const k of ensureKeys) if (!(k in merged)) merged[k] = null;
    if (merged.price === null || typeof merged.price !== 'object') merged.price = { currency: null, amount: null };
    if (!Array.isArray(merged.authors) && merged.authors !== null) merged.authors = [String(merged.authors)];
    if (!Array.isArray(merged.categories) && merged.categories !== null) merged.categories = [String(merged.categories)];
    if (!Array.isArray(merged.condition_keywords) && merged.condition_keywords !== null) merged.condition_keywords = [String(merged.condition_keywords)];
    // add traceability fields
    merged.info_url = merged.info_url ?? null;
    merged.source_provider = merged.source_provider ?? null;
    if (best) {
      const pick = (a, b) => (a === null || a === undefined || (Array.isArray(a) && !a.length) || (typeof a === 'string' && !a.trim())) ? b : a;
      merged.title = pick(merged.title, best.title ?? null);
      merged.subtitle = pick(merged.subtitle, best.subtitle ?? null);
      merged.authors = pick(merged.authors, Array.isArray(best.authors) ? best.authors : null);
      merged.publisher = pick(merged.publisher, best.publisher ?? null);
      merged.publication_date = pick(merged.publication_date, best.publication_date ?? null);
      merged.isbn_13 = pick(merged.isbn_13, best.isbn_13 ?? null);
      merged.isbn_10 = pick(merged.isbn_10, best.isbn_10 ?? null);
      merged.description = pick(merged.description, best.description ?? null);
      merged.page_count = pick(merged.page_count, best.page_count ?? null);
      merged.categories = pick(merged.categories, Array.isArray(best.categories) ? best.categories : null);
      merged.language = pick(merged.language, best.language ?? null);
      merged.info_url = best.url ?? null;
      merged.source_provider = best.provider ?? null;
    }
    mergeTable.innerHTML = toTable(merged);
  } catch (e) {
    outEl.textContent = String(e);
  }
});

init();


