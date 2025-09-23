import os
import sys
import time
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import tempfile
import threading
from typing import Callable
import logging

# Resolve path to the OCR/LLM pipeline (pipeline_demo/extractor)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "extractor")
if PIPELINE_DIR not in sys.path:
	sys.path.insert(0, PIPELINE_DIR)

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

for d in [DATA_DIR, UPLOADS_DIR, ACCEPTED_DIR, REJECTED_DIR]:
	os.makedirs(d, exist_ok=True)

app = FastAPI(title="Image-to-JSON Book Scanner UI", version="0.2.3")
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
        try:
            self._original.write(s)
        except Exception:
            pass
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
	except Exception as e:
		with _JOBS_LOCK:
			_JOBS[job_id].update({"status": "error", "error": str(e)})
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

# Quiet uvicorn access logs for noisy polling endpoints
class _AccessLogSilencer(logging.Filter):
    _SILENCE_PATHS = ("/api/trace_poll", "/api/log_poll", "/api/job_result")
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = getattr(record, 'args', {}) or {}
            request_line = args.get('request_line') or ''
            status_code = int(args.get('status_code') or 0)
            # Hide frequent success/accepted logs for polling
            if any(p in request_line for p in self._SILENCE_PATHS) and status_code in (200, 202):
                return False
        except Exception:
            pass
        return True

try:
    logging.getLogger("uvicorn.access").addFilter(_AccessLogSilencer())
except Exception:
    pass


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
	crop_ocr: bool = Form(False),
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
	return {"id": item_id, "files": [os.path.basename(saved_path)], "status": "started"}


@app.post("/api/process_images")
async def process_images(
	images: List[UploadFile] = File(...),
	model: str = Form("gemma3:4b"),
	ocr_engine: str = Form("easyocr"),
	use_preprocessing: bool = Form(True),
	edge_crop: float = Form(0.0),
	crop_ocr: bool = Form(False),
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
	crop_ocr: Optional[bool] = False


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
	t = threading.Thread(target=_run_extractor_job, args=(job_id, image_paths), kwargs={
		'model': payload.model or 'gemma3:4b', 'ocr_engine': payload.ocr_engine or 'easyocr', 'use_preprocessing': bool(payload.use_preprocessing), 'edge_crop': float(payload.edge_crop or 0.0), 'crop_ocr': bool(payload.crop_ocr or False)
	}, daemon=True)
	t.start()
	with _JOBS_LOCK:
		_JOBS[job_id] = {"status": "queued", "files": [os.path.basename(p) for p in image_paths]}
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
	isbn_13: Optional[str] = None
	isbn_10: Optional[str] = None
	title: Optional[str] = None
	authors: Optional[List[str]] = None


@app.post("/api/pricing_lookup")
async def pricing_lookup(payload: PricingLookupPayload):
	# Placeholder implementation; integrate separately with pricing_api if needed
	query = {
		"isbn_13": payload.isbn_13,
		"isbn_10": payload.isbn_10,
		"title": payload.title,
		"authors": payload.authors or [],
	}
	return {
		"query": query,
		"status": "not_implemented",
		"message": "Pricing lookup placeholder. Use the pricing API UI for aggregation.",
		"offers": [],
	}


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


