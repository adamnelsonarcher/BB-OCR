import os
import sys
import time
import json
from typing import Any, Dict, Optional, List
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import tempfile
import threading
from typing import Callable

# Resolve path to the OCR/LLM pipeline (pipeline_demo/extractor)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "extractor")
if PIPELINE_DIR not in sys.path:
	sys.path.insert(0, PIPELINE_DIR)
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)
# Ensure the inner pricing package root is importable (project_root/pricing_api)
PRICING_PKG_ROOT = os.path.join(PROJECT_ROOT, "pricing_api")
if PRICING_PKG_ROOT not in sys.path:
	sys.path.insert(0, PRICING_PKG_ROOT)

# Pricing aggregator (shared with pricing_api app)
try:
	from pricing_api.core.aggregator import aggregate_offers, DEFAULT_PROVIDERS  # type: ignore
except Exception:
	aggregate_offers = None  # type: ignore
	DEFAULT_PROVIDERS = []  # type: ignore

# Import the existing extractor
try:
	from enhanced_extractor import EnhancedBookMetadataExtractor  # type: ignore
except Exception as import_error:
	EnhancedBookMetadataExtractor = None  # type: ignore
	IMPORT_ERROR = import_error
else:
	IMPORT_ERROR = None

# Directories for uploads and results
DATA_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, "..")), "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
ACCEPTED_DIR = os.path.join(DATA_DIR, "accepted")
REJECTED_DIR = os.path.join(DATA_DIR, "rejected")
STATIC_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, "..")), "static")
PRICING_STATIC_DIR = os.path.join(PROJECT_ROOT, "pricing_api", "static")

for d in [DATA_DIR, UPLOADS_DIR, ACCEPTED_DIR, REJECTED_DIR]:
	os.makedirs(d, exist_ok=True)

app = FastAPI(title="Image-to-JSON Book Scanner UI", version="0.2.3")
# Best-effort: prevent proxies from hijacking local calls to Ollama
try:
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
except Exception:
    pass
# In-memory trace streams per job id
_TRACE_LOCK = threading.Lock()
_TRACE_STREAMS: Dict[str, list] = {}
_TRACE_SEQ: Dict[str, int] = {}
_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_SEM = threading.BoundedSemaphore(1)

# In-memory console log streams per job id
_LOG_LOCK = threading.Lock()
_LOG_STREAMS: Dict[str, list] = {}
_LOG_SEQ: Dict[str, int] = {}

# In-memory job status/result streams per job id (for SSE)
_STATUS_LOCK = threading.Lock()
_STATUS_STREAMS: Dict[str, list] = {}
_STATUS_SEQ: Dict[str, int] = {}

# One-time Ollama warm-check state
_OLLAMA_WARMED = False

def _make_trace_sink(job_id: str) -> Callable[[dict], None]:
    # Track which heavy image fields we've already sent to clients per image index
    heavy_fields = ("original_b64", "preprocessed_b64", "edge_cropped_b64", "auto_cropped_b64")
    seen_by_image: Dict[int, Dict[str, bool]] = {}
    seen_prompt = False
    seen_vlm = False

    def _sink(trace: dict) -> None:
        # Build a lightweight snapshot to stream (omit heavy data repeatedly)
        lite: Dict[str, Any] = {}
        # Images
        images = trace.get("images") or []
        out_images: List[Dict[str, Any]] = []
        for idx, img in enumerate(images):
            seen = seen_by_image.setdefault(idx, {k: False for k in heavy_fields})
            out: Dict[str, Any] = {}
            # small fields
            if isinstance(img, dict):
                if "ocr_text" in img:
                    out["ocr_text"] = img["ocr_text"]
                if "preprocessing_steps" in img:
                    out["preprocessing_steps"] = img["preprocessing_steps"]
                # heavy fields only once per field
                for k in heavy_fields:
                    v = img.get(k)
                    if v and not seen.get(k, False):
                        out[k] = v
                        seen[k] = True
            out_images.append(out)
        if out_images:
            lite["images"] = out_images
        # steps (limit)
        steps = trace.get("steps") or []
        if steps:
            lite["steps"] = steps[-50:]
        # ocr json (small)
        if trace.get("ocr_json") is not None:
            lite["ocr_json"] = trace.get("ocr_json")
        # enhanced prompt (send once)
        if trace.get("enhanced_prompt") and not seen_prompt:
            lite["enhanced_prompt"] = trace.get("enhanced_prompt")
            seen_prompt = True
        # vlm raw (send once)
        if trace.get("ollama_raw") and not seen_vlm:
            lite["ollama_raw"] = trace.get("ollama_raw")
            seen_vlm = True

        with _TRACE_LOCK:
            seq = _TRACE_SEQ.get(job_id, 0) + 1
            _TRACE_SEQ[job_id] = seq
            _TRACE_STREAMS.setdefault(job_id, []).append({
                "ts": int(time.time() * 1000),
                "seq": seq,
                "trace": lite,
            })
    return _sink

def _sse_format(event: str, data: str) -> str:
    # Minimal SSE encoding: event and data lines terminated by two newlines
    return f"event: {event}\n" + "\n".join([f"data: {line}" for line in data.splitlines() or [""]]) + "\n\n"

def _status_push(job_id: str, payload: Dict[str, Any]) -> None:
    with _STATUS_LOCK:
        seq = _STATUS_SEQ.get(job_id, 0) + 1
        _STATUS_SEQ[job_id] = seq
        _STATUS_STREAMS.setdefault(job_id, []).append({
            "ts": int(time.time() * 1000),
            "seq": seq,
            **payload,
        })

class _JobLogTee:
    def __init__(self, original, job_id: str):
        self._original = original
        self._job_id = job_id
        self._buffer = ""
        self._last_line = None
        self._repeat_count = 0
    def _append_line(self, line: str) -> None:
        # Coalesce duplicate consecutive lines to reduce spam
        if line == self._last_line:
            self._repeat_count += 1
            return
        # Flush previous repeated line
        if self._last_line is not None:
            out = self._last_line
            if self._repeat_count > 0:
                out = f"{out} (x{self._repeat_count + 1})"
            with _LOG_LOCK:
                seq = _LOG_SEQ.get(self._job_id, 0) + 1
                _LOG_SEQ[self._job_id] = seq
                _LOG_STREAMS.setdefault(self._job_id, []).append({
                    "ts": int(time.time() * 1000),
                    "seq": seq,
                    "line": out,
                })
        # Start tracking new line
        self._last_line = line
        self._repeat_count = 0
    def write(self, s: str) -> int:
        # Do not forward to original console; keep logs in per-job stream only
        self._buffer += s
        # Flush complete lines to the log stream
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._append_line(line)
        return len(s)
    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

@app.get("/api/trace_poll")
async def trace_poll(id: str, last_ts: int = 0, last_seq: int = -1):
    with _TRACE_LOCK:
        items = _TRACE_STREAMS.get(id, [])
        if last_seq >= 0:
            new_items = [it for it in items if it.get("seq", -1) > last_seq]
        else:
            new_items = [it for it in items if it.get("ts", 0) > last_ts]
        # cap per response to avoid flooding
        if len(new_items) > 100:
            new_items = new_items[-100:]
    return {"items": new_items, "count": len(new_items)}

@app.get("/api/trace_stream")
async def trace_stream(id: str, last_ts: int = 0, last_seq: int = -1):
    """Server-Sent Events stream for trace updates.
    Sends only new items since last_seq/last_ts and keeps streaming.
    """
    async def event_generator():
        nonlocal last_ts, last_seq
        # Suggest client retry interval and send initial ping to flush headers
        yield "retry: 2000\n\n"
        yield _sse_format("ping", "{}")
        last_heartbeat = time.monotonic()
        # Initial burst of any buffered items
        while True:
            with _TRACE_LOCK:
                items = _TRACE_STREAMS.get(id, [])
                if last_seq >= 0:
                    new_items = [it for it in items if it.get("seq", -1) > last_seq]
                else:
                    new_items = [it for it in items if it.get("ts", 0) > last_ts]
            if new_items:
                for it in new_items:
                    last_ts = max(last_ts, it.get("ts", 0))
                    last_seq = max(last_seq, it.get("seq", -1))
                    yield _sse_format("trace", json.dumps(it))
                last_heartbeat = time.monotonic()
            # heartbeat every 10s to keep connection alive through proxies
            now = time.monotonic()
            if now - last_heartbeat > 10.0:
                yield _sse_format("ping", "{}")
                last_heartbeat = now
            # brief sleep to avoid busy loop
            await asyncio.sleep(0.35)
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })

@app.get("/api/log_poll")
async def log_poll(id: str, last_ts: int = 0, last_seq: int = -1):
    with _LOG_LOCK:
        items = _LOG_STREAMS.get(id, [])
        if last_seq >= 0:
            new_items = [it for it in items if it.get("seq", -1) > last_seq]
        else:
            new_items = [it for it in items if it.get("ts", 0) > last_ts]
        # cap per response to avoid flooding
        if len(new_items) > 100:
            new_items = new_items[-100:]
    return {"items": new_items, "count": len(new_items)}

@app.get("/api/log_stream")
async def log_stream(id: str, last_ts: int = 0, last_seq: int = -1):
    """Server-Sent Events stream for console log lines."""
    async def event_generator():
        nonlocal last_ts, last_seq
        yield "retry: 2000\n\n"
        yield _sse_format("ping", "{}")
        last_heartbeat = time.monotonic()
        while True:
            with _LOG_LOCK:
                items = _LOG_STREAMS.get(id, [])
                if last_seq >= 0:
                    new_items = [it for it in items if it.get("seq", -1) > last_seq]
                else:
                    new_items = [it for it in items if it.get("ts", 0) > last_ts]
            if new_items:
                for it in new_items:
                    last_ts = max(last_ts, it.get("ts", 0))
                    last_seq = max(last_seq, it.get("seq", -1))
                    yield _sse_format("log", json.dumps(it))
                last_heartbeat = time.monotonic()
            now = time.monotonic()
            if now - last_heartbeat > 10.0:
                yield _sse_format("ping", "{}")
                last_heartbeat = now
            await asyncio.sleep(0.25)
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })

@app.get("/api/job_stream")
async def job_stream(id: str, last_ts: int = 0, last_seq: int = -1):
    """Server-Sent Events stream for job status/result updates."""
    async def event_generator():
        nonlocal last_ts, last_seq
        # Suggest retry and send initial ping to keep connection and flush headers
        yield "retry: 2000\n\n"
        yield _sse_format("ping", "{}")
        last_heartbeat = time.monotonic()
        # Emit any buffered items immediately, then keep streaming
        while True:
            with _STATUS_LOCK:
                items = _STATUS_STREAMS.get(id, [])
                if last_seq >= 0:
                    new_items = [it for it in items if it.get("seq", -1) > last_seq]
                else:
                    new_items = [it for it in items if it.get("ts", 0) > last_ts]
            if new_items:
                for it in new_items:
                    last_ts = max(last_ts, it.get("ts", 0))
                    last_seq = max(last_seq, it.get("seq", -1))
                    yield _sse_format("job", json.dumps(it))
                # If the last item is terminal, we can exit to let client close
                last_item = new_items[-1]
                if last_item.get("status") in ("done", "error"):
                    return
                last_heartbeat = time.monotonic()
            now = time.monotonic()
            if now - last_heartbeat > 10.0:
                yield _sse_format("ping", "{}")
                last_heartbeat = now
            await asyncio.sleep(0.25)
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })


def _ollama_quick_ping() -> None:
    global _OLLAMA_WARMED
    if _OLLAMA_WARMED:
        return
    try:
        # Check tags to confirm server is reachable and pick a model
        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        model: Optional[str] = None
        if tags.status_code == 200:
            models = [m.get("name") for m in tags.json().get("models", []) if m.get("name")]
            if models:
                model = next((m for m in models if str(m).startswith("gemma3:4b")), models[0])
        # Send a tiny generate to verify model hotness without blocking long
        if model:
            try:
                requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": model, "prompt": "ping", "stream": False},
                    timeout=3,
                )
            except Exception:
                pass
        _OLLAMA_WARMED = True
    except Exception:
        # Ignore errors; this is a best-effort one-time check
        pass


@app.on_event("startup")
async def _on_startup():
    # Run the warm-check in a background thread to avoid blocking app startup
    try:
        t = threading.Thread(target=_ollama_quick_ping, daemon=True)
        t.start()
    except Exception:
        pass

@app.get("/api/job_status")
async def job_status(id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(id)
        if not job:
            return {"id": id, "status": "unknown"}
        return {"id": id, "status": job.get("status"), "error": job.get("error")}

@app.get("/api/job_result")
async def job_result(id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "job not found"})
        status = job.get("status")
        if status == "error":
            return {"id": id, "status": status, "error": job.get("error"), "files": job.get("files", [])}
        if status != "done":
            return JSONResponse(status_code=202, content={"status": status})
        return {"id": id, "status": status, "metadata": job.get("metadata"), "files": job.get("files", [])}

def _run_extractor_job(job_id: str, image_paths: List[str], *, model: str, ocr_engine: str, use_preprocessing: bool, edge_crop: float, crop_ocr: bool) -> None:
	_JOB_SEM.acquire()
	try:
		with _JOBS_LOCK:
			_JOBS[job_id] = {"status": "running", "files": [os.path.basename(p) for p in image_paths]}
		# Broadcast running status
		_status_push(job_id, {"status": "running", "files": [os.path.basename(p) for p in image_paths]})
		# Prepare sinks
		trace_sink = _make_trace_sink(job_id)
		with _LOG_LOCK:
			_LOG_STREAMS.setdefault(job_id, [])
		# Tee stdout/stderr to per-job log stream
		_orig_out, _orig_err = sys.stdout, sys.stderr
		sys.stdout = _JobLogTee(_orig_out, job_id)
		sys.stderr = _JobLogTee(_orig_err, job_id)
		try:
			extractor = _build_extractor(model=model, ocr_engine=ocr_engine, use_preprocessing=use_preprocessing, edge_crop=edge_crop, auto_crop=crop_ocr)
			ocr_indices = _compute_default_ocr_indices(len(image_paths))
			metadata = extractor.extract_metadata_from_images(image_paths, ocr_image_indices=ocr_indices, capture_trace=True, trace_sink=trace_sink)
		finally:
			sys.stdout = _orig_out
			sys.stderr = _orig_err
		with _JOBS_LOCK:
			_JOBS[job_id].update({"status": "done", "metadata": metadata})
		# Broadcast completion with metadata and files
		_status_push(job_id, {"status": "done", "files": [os.path.basename(p) for p in image_paths], "metadata": metadata})
	except Exception as e:
		with _JOBS_LOCK:
			_JOBS[job_id].update({"status": "error", "error": str(e)})
		_status_push(job_id, {"status": "error", "error": str(e)})
	finally:
		# Trim stored streams to avoid unbounded growth and release semaphore
		with _TRACE_LOCK:
			items = _TRACE_STREAMS.get(job_id, [])
			if len(items) > 200:
				_TRACE_STREAMS[job_id] = items[-200:]
		with _LOG_LOCK:
			logs = _LOG_STREAMS.get(job_id, [])
			if len(logs) > 1000:
				_LOG_STREAMS[job_id] = logs[-1000:]
		try:
			_JOB_SEM.release()
		except Exception:
			pass


# CORS for local use
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.isdir(PRICING_STATIC_DIR):
	app.mount("/pricing_static", StaticFiles(directory=PRICING_STATIC_DIR), name="pricing_static")

class AcceptPayload(BaseModel):
	id: str
	metadata: Dict[str, Any]
	notes: Optional[str] = None


@app.get("/")
async def root_index():
	index_path = os.path.join(STATIC_DIR, "index.html")
	if not os.path.exists(index_path):
		raise HTTPException(status_code=404, detail="index.html not found")
	return FileResponse(index_path)


@app.get("/api/health")
async def health():
	return {
		"status": "ok",
		"pipeline_imported": IMPORT_ERROR is None,
		"pipeline_dir": PIPELINE_DIR,
	}


@app.get("/api/models")
async def list_models():
	try:
		resp = requests.get("http://localhost:11434/api/tags", timeout=3)
		if resp.status_code == 200:
			models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
			return {"models": models}
		return JSONResponse(status_code=502, content={"error": f"ollama tags status {resp.status_code}"})
	except Exception:
		# Provide a conservative default list
		return {"models": [
			"gemma3:4b",
			"llava:13b",
			"llava:7b",
			"llava-phi3",
			"moondream",
		]}


def _build_extractor(model: str, ocr_engine: str, use_preprocessing: bool, *, edge_crop: float = 0.0, auto_crop: bool = False) -> "EnhancedBookMetadataExtractor":
	if EnhancedBookMetadataExtractor is None:
		raise RuntimeError(f"Failed to import pipeline: {IMPORT_ERROR}")
	try:
		return EnhancedBookMetadataExtractor(
			model=model,
			ocr_engine=ocr_engine,
			use_preprocessing=use_preprocessing,
			edge_crop_percent=float(max(0.0, min(45.0, edge_crop))),
			crop_for_ocr=bool(auto_crop),
			warm_model=False,
			ollama_timeout_seconds=180.0,
		)
	except Exception:
		# Fallback to tesseract if easyocr fails to init
		if ocr_engine.lower() == "easyocr":
			return EnhancedBookMetadataExtractor(
				model=model,
				ocr_engine="tesseract",
				use_preprocessing=use_preprocessing,
				edge_crop_percent=float(max(0.0, min(45.0, edge_crop))),
				crop_for_ocr=bool(auto_crop),
				warm_model=False,
				ollama_timeout_seconds=180.0,
			)
		raise


def _find_books_dir() -> Optional[str]:
	candidates = [
		os.path.join(PIPELINE_DIR, "books"),
		os.path.join(PROJECT_ROOT, "books"),
	]
	for p in candidates:
		if os.path.isdir(p):
			return p
	return None


def _find_output_dirs() -> List[str]:
	# Look in extractor output locations
	return [p for p in [
		os.path.join(PIPELINE_DIR, "output"),
		os.path.join(PIPELINE_DIR, "batch_output"),
	] if os.path.isdir(p)]


def _compute_default_ocr_indices(n: int) -> List[int]:
    # Skip the first image (cover) for OCR by default
    if n >= 3:
        return [1, 2]
    if n == 2:
        return [1]
    if n == 1:
        return [0]
    return []


@app.post("/api/process_image")
async def process_image(
	image: UploadFile = File(...),
	model: str = Form("gemma3:4b"),
	ocr_engine: str = Form("easyocr"),
	use_preprocessing: bool = Form(True),
	edge_crop: float = Form(0.0),
	crop_ocr: bool = Form(True),
):
	if image.content_type is None or not image.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="Uploaded file must be an image")

	# Save upload (write to system temp to avoid dev server reloads on file change)
	timestamp = int(time.time() * 1000)
	extension = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
	item_id = f"capture_{timestamp}"
	tmp_dir = os.path.join(tempfile.gettempdir(), "bb_ocr_ui_uploads")
	os.makedirs(tmp_dir, exist_ok=True)
	saved_path = os.path.join(tmp_dir, f"{item_id}{extension}")
	with open(saved_path, "wb") as f:
		f.write(await image.read())

	# Reset any prior streams for this id (unlikely for unique ids)
	with _TRACE_LOCK:
		_TRACE_STREAMS[item_id] = []
		_TRACE_SEQ[item_id] = 0
	with _LOG_LOCK:
		_LOG_STREAMS[item_id] = []
		_LOG_SEQ[item_id] = 0
	# Start background job and return immediately
	t = threading.Thread(target=_run_extractor_job, args=(item_id, [saved_path]), kwargs={
		'model': model, 'ocr_engine': ocr_engine, 'use_preprocessing': use_preprocessing, 'edge_crop': float(edge_crop), 'crop_ocr': bool(crop_ocr)
	}, daemon=True)
	t.start()
	with _JOBS_LOCK:
		_JOBS[item_id] = {"status": "queued", "files": [os.path.basename(saved_path)]}
	# Broadcast queued status
	_status_push(item_id, {"status": "queued", "files": [os.path.basename(saved_path)]})
	return {"id": item_id, "files": [os.path.basename(saved_path)], "status": "started"}


@app.post("/api/process_images")
async def process_images(
	images: List[UploadFile] = File(...),
	model: str = Form("gemma3:4b"),
	ocr_engine: str = Form("easyocr"),
	use_preprocessing: bool = Form(True),
	edge_crop: float = Form(0.0),
	crop_ocr: bool = Form(True),
):
	if not images:
		raise HTTPException(status_code=400, detail="No images uploaded")

	item_id = f"batch_{int(time.time() * 1000)}"
	saved_paths: List[str] = []
	for idx, uf in enumerate(images):
		if uf.content_type is None or not uf.content_type.startswith("image/"):
			raise HTTPException(status_code=400, detail=f"File {uf.filename} is not an image")
		ext = os.path.splitext(uf.filename or f"capture_{idx}.jpg")[1] or ".jpg"
		# Write to system temp to avoid uvicorn reloads
		tmp_dir = os.path.join(tempfile.gettempdir(), "bb_ocr_ui_uploads")
		os.makedirs(tmp_dir, exist_ok=True)
		path = os.path.join(tmp_dir, f"{item_id}_{idx}{ext}")
		with open(path, "wb") as f:
			f.write(await uf.read())
		saved_paths.append(path)

	# Reset any prior streams for this id (unlikely for unique ids)
	with _TRACE_LOCK:
		_TRACE_STREAMS[item_id] = []
		_TRACE_SEQ[item_id] = 0
	with _LOG_LOCK:
		_LOG_STREAMS[item_id] = []
		_LOG_SEQ[item_id] = 0
	# Start background job and return immediately
	t = threading.Thread(target=_run_extractor_job, args=(item_id, saved_paths), kwargs={
		'model': model, 'ocr_engine': ocr_engine, 'use_preprocessing': use_preprocessing, 'edge_crop': float(edge_crop), 'crop_ocr': bool(crop_ocr)
	}, daemon=True)
	t.start()
	with _JOBS_LOCK:
		_JOBS[item_id] = {"status": "queued", "files": [os.path.basename(p) for p in saved_paths]}
	_status_push(item_id, {"status": "queued", "files": [os.path.basename(p) for p in saved_paths]})
	return {"id": item_id, "files": [os.path.basename(p) for p in saved_paths], "status": "started"}


@app.get("/api/examples")
async def examples():
	books_dir = _find_books_dir()
	if not books_dir:
		return {"books_dir": None, "items": []}
	items: List[Dict[str, Any]] = []
	for entry in sorted(os.listdir(books_dir)):
		full = os.path.join(books_dir, entry)
		if not os.path.isdir(full):
			continue
		images = [f for f in sorted(os.listdir(full)) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
		# mark if an extracted output exists in extractor output dirs
		has_output = False
		for outd in _find_output_dirs():
			cand = os.path.join(outd, f"book_{entry}_enhanced.json")
			if os.path.isfile(cand):
				has_output = True
				break
		if images:
			items.append({"id": entry, "count": len(images), "has_output": has_output})
	return {"books_dir": books_dir, "items": items}


class ExamplePayload(BaseModel):
	book_id: str
	model: Optional[str] = "gemma3:4b"
	ocr_engine: Optional[str] = "easyocr"
	use_preprocessing: Optional[bool] = True
	edge_crop: Optional[float] = 0.0
	crop_ocr: Optional[bool] = True


@app.post("/api/process_example")
async def process_example(payload: ExamplePayload):
	books_dir = _find_books_dir()
	if not books_dir:
		raise HTTPException(status_code=404, detail="Books directory not found")
	book_dir = os.path.join(books_dir, payload.book_id)
	if not os.path.isdir(book_dir):
		raise HTTPException(status_code=404, detail="Example book not found")
	image_paths = [
		os.path.join(book_dir, f) for f in sorted(os.listdir(book_dir))
		if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
	]
	if not image_paths:
		raise HTTPException(status_code=400, detail="No images in example directory")

	# Background job for example
	job_id = f"example_{payload.book_id}"
	# If a job with same id is already running, don't start again
	with _JOBS_LOCK:
		existing = _JOBS.get(job_id)
		if existing and existing.get("status") in ("running", "queued"):
			return {"id": job_id, "files": existing.get("files", []), "status": existing.get("status")}
	# Reset streams for new run
	with _TRACE_LOCK:
		_TRACE_STREAMS[job_id] = []
		_TRACE_SEQ[job_id] = 0
	with _LOG_LOCK:
		_LOG_STREAMS[job_id] = []
		_LOG_SEQ[job_id] = 0
	with _STATUS_LOCK:
		_STATUS_STREAMS[job_id] = []
		_STATUS_SEQ[job_id] = 0
	t = threading.Thread(target=_run_extractor_job, args=(job_id, image_paths), kwargs={
		'model': payload.model or 'gemma3:4b', 'ocr_engine': payload.ocr_engine or 'easyocr', 'use_preprocessing': bool(payload.use_preprocessing), 'edge_crop': float(payload.edge_crop or 0.0), 'crop_ocr': bool(payload.crop_ocr or False)
	}, daemon=True)
	t.start()
	with _JOBS_LOCK:
		_JOBS[job_id] = {"status": "queued", "files": [os.path.basename(p) for p in image_paths]}
	_status_push(job_id, {"status": "queued", "files": [os.path.basename(p) for p in image_paths]})
	return {"id": job_id, "files": [os.path.basename(p) for p in image_paths], "status": "started"}


@app.get("/api/example_output")
async def example_output(book_id: str):
	# search in extractor outputs
	for outd in _find_output_dirs():
		candidate = os.path.join(outd, f"book_{book_id}_enhanced.json")
		if os.path.isfile(candidate):
			try:
				with open(candidate, 'r', encoding='utf-8') as f:
					metadata = json.load(f)
			except Exception as e:
				raise HTTPException(status_code=500, detail=str(e))
			return {"id": f"example_output_{book_id}", "file": os.path.basename(candidate), "metadata": metadata}
	raise HTTPException(status_code=404, detail="Saved output not found for example")


class PricingLookupPayload(BaseModel):
	# Accept flexible input types; we'll normalize inside the handler
	isbn_13: Optional[Any] = None
	isbn_10: Optional[Any] = None
	title: Optional[Any] = None
	authors: Optional[Any] = None
	publisher: Optional[Any] = None
	publication_date: Optional[Any] = None
	providers: Optional[Any] = None


@app.get("/api/pricing/providers")
async def pricing_providers():
	return {"providers": [name for name, _ in (DEFAULT_PROVIDERS or [])]}


@app.post("/api/pricing_lookup")
async def pricing_lookup(payload: PricingLookupPayload):
	if aggregate_offers is None:
		return JSONResponse(status_code=500, content={"error": "pricing aggregator unavailable"})
	# Defensive normalization to tolerate strings/nulls from various JSONs
	def _to_str(x: Any) -> Optional[str]:
		return None if x is None else str(x)
	def _to_str_list(x: Any) -> List[str]:
		if x is None:
			return []
		if isinstance(x, list):
			return [str(i) for i in x if i is not None]
		if isinstance(x, str):
			return [x]
		return [str(x)]
	safe_title = _to_str(payload.title)
	safe_authors = _to_str_list(payload.authors)  # handles string -> [string]
	safe_isbn13 = _to_str(payload.isbn_13)
	safe_isbn10 = _to_str(payload.isbn_10)
	safe_publisher = _to_str(payload.publisher)
	safe_pubdate = _to_str(payload.publication_date)
	safe_providers = None
	if payload.providers is not None:
		safe_providers = [str(p) for p in payload.providers if p is not None]
	offers, errors = await aggregate_offers(
		title=safe_title,
		authors=safe_authors,
		isbn_13=safe_isbn13,
		isbn_10=safe_isbn10,
		publisher=safe_publisher,
		publication_date=safe_pubdate,
		providers=safe_providers,
		timeout_seconds=8.0,
	)
	return {
		"query": {
			"title": safe_title,
			"authors": safe_authors,
			"isbn_13": safe_isbn13,
			"isbn_10": safe_isbn10,
			"publisher": safe_publisher,
			"publication_date": safe_pubdate,
			"providers": safe_providers,
		},
		"providers": safe_providers or [name for name, _ in (DEFAULT_PROVIDERS or [])],
		"offers": offers,
		"errors": errors,
	}

# Compatibility endpoints to support embedded pricing static UI (absolute paths)
@app.get("/providers")
async def providers_alias():
	return await pricing_providers()


@app.get("/processed/list")
async def processed_list_alias():
	return await pricing_processed_list()


@app.get("/processed/load")
async def processed_load_alias(path: str):
	return await pricing_processed_load(path)


@app.post("/lookup")
async def lookup_alias(payload: PricingLookupPayload):
	return await pricing_lookup(payload)


@app.get("/pricing_embed")
async def pricing_embed():
	# Serve pricing static index.html but rewrite absolute /static references to /pricing_static
	index_path = os.path.join(PRICING_STATIC_DIR, "index.html")
	if not os.path.isfile(index_path):
		raise HTTPException(status_code=404, detail="pricing index not found")
	try:
		with open(index_path, 'r', encoding='utf-8') as f:
			html = f.read()
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
	# Rewrite only href/src that start with /static/
	html = html.replace('href="/static/', 'href="/pricing_static/')
	html = html.replace("src=\"/static/", "src=\"/pricing_static/")
	return HTMLResponse(content=html, media_type="text/html")


# Pricing processed files listing/loading (mirror pricing_api UI)
def _pricing_processed_dirs() -> List[str]:
	# Look for outputs inside extractor and accepted i2j_ui data within this repo
	candidates = [
		os.path.join(PROJECT_ROOT, "extractor", "output"),
		os.path.join(PROJECT_ROOT, "extractor", "batch_output"),
		os.path.join(PROJECT_ROOT, "i2j_ui", "data", "accepted"),
	]
	return [d for d in candidates if os.path.isdir(d)]


def _pricing_allowed_path(p: str) -> bool:
	try:
		rp = os.path.abspath(p)
		for base in _pricing_processed_dirs():
			ab = os.path.abspath(base)
			if rp.startswith(ab + os.sep) or rp == ab:
				return True
		return False
	except Exception:
		return False


@app.get("/api/pricing/processed/list")
async def pricing_processed_list():
	items = []
	for base in _pricing_processed_dirs():
		for name in os.listdir(base):
			if not name.lower().endswith(".json"):
				continue
			path = os.path.join(base, name)
			try:
				st = os.stat(path)
			except Exception:
				continue
			label = name
			items.append({
				"label": label,
				"path": path,
				"size": st.st_size,
				"mtime": st.st_mtime,
				"dir": base,
			})
	items.sort(key=lambda x: x["mtime"], reverse=True)
	return {"items": items}


@app.get("/api/pricing/processed/load")
async def pricing_processed_load(path: str):
	if not _pricing_allowed_path(path):
		raise HTTPException(status_code=400, detail="Invalid path")
	try:
		with open(path, 'r', encoding='utf-8') as f:
			data = json.load(f)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
	payload = {
		"title": data.get("title"),
		"subtitle": data.get("subtitle"),
		"authors": data.get("authors"),
		"publisher": data.get("publisher"),
		"publication_date": data.get("publication_date"),
		"isbn_13": data.get("isbn_13"),
		"isbn_10": data.get("isbn_10"),
	}
	return {"path": path, "payload": payload, "raw": data}


@app.post("/api/accept")
async def accept(payload: AcceptPayload):
	output_path = os.path.join(ACCEPTED_DIR, f"{payload.id}.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(payload.metadata, f, indent=2, ensure_ascii=False)
	return {"status": "saved", "path": output_path}


class RejectPayload(BaseModel):
	id: str
	reason: Optional[str] = None


@app.post("/api/reject")
async def reject(payload: RejectPayload):
	log_path = os.path.join(REJECTED_DIR, f"{payload.id}.txt")
	with open(log_path, "w", encoding="utf-8") as f:
		f.write(payload.reason or "rejected")
	return {"status": "rejected", "path": log_path}


