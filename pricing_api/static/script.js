const envEl = document.getElementById('env');
const providersEl = document.getElementById('providers');
const jsonEl = document.getElementById('json');
const outEl = document.getElementById('out');
const btnRun = document.getElementById('btnRun');

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
}

function getSelectedProviders() {
  const cbs = providersEl.querySelectorAll('input[type="checkbox"]');
  const selected = [];
  for (const cb of cbs) if (cb.checked) selected.push(cb.value);
  return selected;
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
  } catch (e) {
    outEl.textContent = String(e);
  }
});

init();


