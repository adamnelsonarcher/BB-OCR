const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const btnCapture = document.getElementById('capture');
const fileInput = document.getElementById('file');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const metaTable = document.getElementById('meta-table');
const actions = document.getElementById('actions');
const btnAccept = document.getElementById('accept');
const btnReject = document.getElementById('reject');
const envPipeline = document.getElementById('env-pipeline');

let lastId = null;

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
}

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
	const rows = Object.entries(obj).map(([k, v]) => {
		const val = (v === null || v === undefined) ? '' : (typeof v === 'object' ? JSON.stringify(v) : String(v));
		return `<tr><td>${k}</td><td>${val}</td></tr>`;
	}).join('');
	return `<table class="kv"><tbody>${rows}</tbody></table>`;
}

async function processBlob(blob, filename = 'capture.jpg') {
	statusEl.textContent = 'Uploading...';
	errorEl.classList.add('hidden');
	metaTable.innerHTML = '';
	actions.classList.add('hidden');

	const fd = new FormData();
	fd.append('image', blob, filename);
	fd.append('model', document.getElementById('model').value || 'gemma3:4b');
	fd.append('ocr_engine', document.getElementById('ocr').value || 'easyocr');
	fd.append('use_preprocessing', document.getElementById('preproc').checked ? 'true' : 'false');

	const resp = await fetch('/api/process_image', { method: 'POST', body: fd });
	const data = await resp.json();
	if (!resp.ok) {
		statusEl.textContent = 'Error';
		errorEl.textContent = data.error || 'Unknown error';
		errorEl.classList.remove('hidden');
		return;
	}

	lastId = data.id;
	statusEl.textContent = `Processed: ${data.file}`;
	metaTable.innerHTML = renderTable(data.metadata);
	actions.classList.remove('hidden');
}

btnCapture.addEventListener('click', async () => {
	const blob = await drawFrameToBlob();
	await processBlob(blob);
});

fileInput.addEventListener('change', async (e) => {
	if (!e.target.files || !e.target.files[0]) return;
	await processBlob(e.target.files[0], e.target.files[0].name);
});

btnAccept.addEventListener('click', async () => {
	if (!lastId) return;
	const rows = Array.from(document.querySelectorAll('table.kv tr'));
	const metadata = {};
	for (const tr of rows) {
		const tds = tr.querySelectorAll('td');
		const k = tds[0].textContent;
		let v = tds[1].textContent;
		try { v = JSON.parse(v); } catch {}
		metadata[k] = v;
	}
	const resp = await fetch('/api/accept', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ id: lastId, metadata })
	});
	const data = await resp.json();
	statusEl.textContent = resp.ok ? `Saved → ${data.path}` : 'Save failed';
});

btnReject.addEventListener('click', async () => {
	if (!lastId) return;
	const reason = prompt('Reason (optional):') || '';
	const resp = await fetch('/api/reject', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ id: lastId, reason })
	});
	statusEl.textContent = resp.ok ? 'Rejected' : 'Reject failed';
	actions.classList.add('hidden');
});

init();
