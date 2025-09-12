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
const btnAccept = document.getElementById('accept');
const btnReject = document.getElementById('reject');
const envPipeline = document.getElementById('env-pipeline');
const queueList = document.getElementById('queue-list');
const modelSel = document.getElementById('model');
const ocrSel = document.getElementById('ocr');
const preprocChk = document.getElementById('preproc');
const examplesSel = document.getElementById('examples');
const btnRunExample = document.getElementById('run-example');

let lastId = null;
let captureQueue = []; // Array of Blobs

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
			opt.value = item.id; opt.textContent = `${item.id} (${item.count})`;
			examplesSel.appendChild(opt);
		}
	} catch {}
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

	const resp = await fetch('/api/process_image', { method: 'POST', body: fd });
	const data = await resp.json();
	if (!resp.ok) {
		statusEl.textContent = 'Error';
		errorEl.textContent = data.error || 'Unknown error';
		errorEl.classList.remove('hidden');
		return;
	}

	lastId = data.id;
	statusEl.textContent = `Processed: ${data.files.join(', ')}`;
	metaTable.innerHTML = renderTable(data.metadata);
	actions.classList.remove('hidden');
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

	const resp = await fetch('/api/process_images', { method: 'POST', body: fd });
	const data = await resp.json();
	if (!resp.ok) {
		statusEl.textContent = 'Error';
		errorEl.textContent = data.error || 'Unknown error';
		errorEl.classList.remove('hidden');
		return;
	}

	lastId = data.id;
	statusEl.textContent = `Processed: ${data.files.join(', ')}`;
	metaTable.innerHTML = renderTable(data.metadata);
	actions.classList.remove('hidden');
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
	// Read table values back
	const rows = Array.from(document.querySelectorAll('table.kv tbody tr'));
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

btnRunExample.addEventListener('click', async () => {
	const id = examplesSel.value;
	if (!id) return;
	statusEl.textContent = `Running example ${id}...`;
	errorEl.classList.add('hidden');
	metaTable.innerHTML = '';
	actions.classList.add('hidden');

	const resp = await fetch('/api/process_example', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			book_id: id,
			model: modelSel.value || 'gemma3:4b',
			ocr_engine: ocrSel.value || 'easyocr',
			use_preprocessing: preprocChk.checked
		})
	});
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
	actions.classList.remove('hidden');
});

init();
