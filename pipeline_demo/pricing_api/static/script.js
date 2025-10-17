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
const reviewCommentEl = document.getElementById('review-comment');
const approveBtn = document.getElementById('approve-final');
const rejectBtn = document.getElementById('reject-final');

let currentReviewId = null;
let lastBestOffer = null;
let lastMerged = null;

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
      cb.type = 'checkbox'; cb.value = p; cb.id = id; cb.checked = (p === 'abebooks');
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
    const val = obj[k];
    let cellHtml = '';
    if (Array.isArray(val)) {
      cellHtml = escapeHtml(val.join(', '));
    } else if (val && typeof val === 'object') {
      // pretty JSON inside <pre> for readability
      cellHtml = `<pre>${escapeHtml(JSON.stringify(val, null, 2))}</pre>`;
    } else if (val === null || val === undefined) {
      cellHtml = '';
    } else {
      cellHtml = escapeHtml(String(val));
    }
    return `<tr><td>${escapeHtml(k)}</td><td>${cellHtml}</td></tr>`;
  }).join('');
  return `<table class="kv"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function normalizeTitle(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function titleTokens(s) {
  const stop = new Set(['the','of','and','for','a','an','to','in','on','by','from','with','at','as','is','are','be','or','not','but']);
  return normalizeTitle(s)
    .split(' ')
    .filter(t => t && t.length > 1 && !stop.has(t));
}

function jaccardSim(a, b) {
  const sa = new Set(a);
  const sb = new Set(b);
  let inter = 0;
  for (const t of sa) if (sb.has(t)) inter++;
  const union = sa.size + sb.size - inter;
  return union ? inter / union : 0;
}

function fuzzyTitleMatch(qTitle, oTitle) {
  const qn = normalizeTitle(qTitle);
  const on = normalizeTitle(oTitle);
  if (!qn || !on) return false;
  if (on.includes(qn) || qn.includes(on)) return true;
  const sim = jaccardSim(titleTokens(qn), titleTokens(on));
  return sim >= 0.5;
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
    // Choose best offer: ONLY consider offers matching the query's publication year.
    // Within those: prefer exact ISBN match, else same title (case-insens), else first.
    const offers = data.offers || [];
    const q_isbn13 = (data.query?.isbn_13 || '').replace(/[-\s]/g, '');
    const q_isbn10 = (data.query?.isbn_10 || '').replace(/[-\s]/g, '');
    const q_title = (data.query?.title || '').trim().toLowerCase();
    const extractYear = (v) => {
      const m = String(v ?? '').match(/(18|19|20)\d{2}/);
      return m ? m[0] : null;
    };
    const q_year = extractYear(data.query?.publication_date ?? null);
    console.log('[pricing-ui] Query year:', q_year);
    console.log('[pricing-ui] Offers received:', offers.length, offers.map(o => ({ provider: o.provider, amount: o.amount, currency: o.currency, pub: o.publication_date, title: o.title })));
    let candidates = offers;
    if (q_year) {
      candidates = offers.filter(o => extractYear(o.publication_date) === q_year);
    }
    console.log('[pricing-ui] Candidates after year filter:', candidates.length);
    let best = null;
    for (const o of candidates) {
      const oi13 = (o.isbn_13 || '').replace(/[-\s]/g, '');
      const oi10 = (o.isbn_10 || '').replace(/[-\s]/g, '');
      if (q_isbn13 && oi13 === q_isbn13) { best = o; break; }
      if (q_isbn10 && oi10 === q_isbn10) { best = o; break; }
    }
    if (!best && q_title) {
      best = candidates.find(o => normalizeTitle(o.title) === normalizeTitle(q_title)) || null;
    }
    // Fuzzy title match within year-matched candidates
    if (!best && q_title) {
      const fuzzy = candidates.filter(o => fuzzyTitleMatch(q_title, o.title || ''));
      if (fuzzy.length) {
        const numeric = fuzzy.filter(o => typeof o.amount === 'number' && !Number.isNaN(o.amount));
        best = numeric.length ? numeric.reduce((a, c) => (c.amount < a.amount ? c : a)) : fuzzy[0];
      }
    }
    // Fallback within year-matched candidates: choose the cheapest numeric amount, else first
    if (!best && candidates.length) {
      const numeric = candidates.filter(o => typeof o.amount === 'number' && !Number.isNaN(o.amount));
      if (numeric.length) {
        best = numeric.reduce((acc, cur) => (cur.amount < acc.amount ? cur : acc));
      } else {
        best = candidates[0];
      }
    }
    console.log('[pricing-ui] Best offer chosen:', best);
    // Do NOT fall back to non-matching-year offers
    respTable.innerHTML = best ? toTable(best) : '<div class="muted">No offers</div>';

    // Build merged view
    const src = data.query || payload || {};
    let merged;
    try { merged = JSON.parse(jsonEl.value || '{}'); } catch { merged = { ...src }; }
    if (merged && typeof merged !== 'object') merged = { ...src };
    const ensureKeys = ['title','subtitle','authors','publisher','publication_date','isbn_13','isbn_10','asin','edition','binding_type','language','page_count','categories','description','condition_keywords','price'];
    for (const k of ensureKeys) if (!(k in merged)) merged[k] = null;
    if (merged.price === null || typeof merged.price !== 'object') merged.price = { currency: null, amount: null };
    if (!Array.isArray(merged.authors) && merged.authors !== null) merged.authors = [String(merged.authors)];
    if (!Array.isArray(merged.categories) && merged.categories !== null) merged.categories = [String(merged.categories)];
    if (!Array.isArray(merged.condition_keywords) && merged.condition_keywords !== null) merged.condition_keywords = [String(merged.condition_keywords)];
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

      // Populate price from best offer in all cases (overwrite any existing)
      const bAmt = best.amount;
      const parsedAmt = (typeof bAmt === 'number') ? bAmt : (bAmt !== null && bAmt !== undefined && !Number.isNaN(Number(bAmt)) ? Number(bAmt) : null);
      console.log('[pricing-ui] Price merge - raw amount:', bAmt, 'type:', typeof bAmt, 'parsed:', parsedAmt, 'currency:', best.currency);
      merged.price = { currency: (best.currency ?? null), amount: parsedAmt };
      console.log('[pricing-ui] Merged price now:', merged.price);
    }
    mergeTable.innerHTML = toTable(merged);
    lastBestOffer = best || null;
    lastMerged = merged || null;
  } catch (e) {
    outEl.textContent = String(e);
  }
});

// Listen for messages from parent (scanner UI)
window.addEventListener('message', (ev) => {
  try {
    const msg = ev.data || {};
    if (msg && msg.type === 'scannerAccepted' && msg.metadata) {
      currentReviewId = msg.id || null;
      jsonEl.value = JSON.stringify(msg.metadata, null, 2);
      // Optionally: auto-run lookup using current provider selections
      btnRun.click();
    }
  } catch {}
});

async function finalize(decision) {
  const comment = (reviewCommentEl && reviewCommentEl.value) || '';
  const payload = { id: currentReviewId, decision, comment };
  if (decision === 'approved') {
    // Ensure we have merged; if not, try to parse the JSON box
    if (!lastMerged) {
      try { lastMerged = JSON.parse(jsonEl.value || '{}'); } catch { lastMerged = {}; }
    }
    payload.merged = lastMerged;
    if (lastBestOffer) payload.best_offer = lastBestOffer;
  }
  try {
    const resp = await fetch('/api/pricing/finalize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (resp.ok) {
      alert(`Finalized: ${decision} → ${data.path || ''}`);
    } else {
      alert(`Finalize failed: ${data.detail || data.error || 'unknown error'}`);
    }
  } catch (e) {
    alert(String(e));
  }
}

approveBtn.addEventListener('click', () => finalize('approved'));
rejectBtn.addEventListener('click', () => finalize('rejected'));

init();



