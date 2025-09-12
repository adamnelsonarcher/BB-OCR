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

# Resolve path to the OCR/LLM pipeline
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ollama+ocr_to_json")
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

app = FastAPI(title="Image-to-JSON Book Scanner UI", version="0.2.1")

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
	except Exception as e:
		# Provide a conservative default list
		return {"models": [
			"gemma3:4b",
			"llava:13b",
			"llava:7b",
			"llava-phi3",
			"moondream",
		]}


def _build_extractor(model: str, ocr_engine: str, use_preprocessing: bool) -> "EnhancedBookMetadataExtractor":
	if EnhancedBookMetadataExtractor is None:
		raise RuntimeError(f"Failed to import pipeline: {IMPORT_ERROR}")
	try:
		return EnhancedBookMetadataExtractor(
			model=model,
			ocr_engine=ocr_engine,
			use_preprocessing=use_preprocessing,
		)
	except Exception as e:
		# Fallback to tesseract if easyocr fails to init
		if ocr_engine.lower() == "easyocr":
			return EnhancedBookMetadataExtractor(
				model=model,
				ocr_engine="tesseract",
				use_preprocessing=use_preprocessing,
			)
		raise


def _find_books_dir() -> Optional[str]:
	candidates = [
		os.path.join(PIPELINE_DIR, "books"),
		os.path.join(os.path.dirname(PIPELINE_DIR), "books"),
		os.path.join(os.path.dirname(PIPELINE_DIR), "ollama_to_JSON", "books"),
		os.path.join(PROJECT_ROOT, "books"),
	]
	for p in candidates:
		if os.path.isdir(p):
			return p
	return None


def _compute_default_ocr_indices(n: int) -> List[int]:
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
):
	if image.content_type is None or not image.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="Uploaded file must be an image")

	# Save upload
	timestamp = int(time.time() * 1000)
	extension = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
	item_id = f"capture_{timestamp}"
	saved_path = os.path.join(UPLOADS_DIR, f"{item_id}{extension}")
	with open(saved_path, "wb") as f:
		f.write(await image.read())

	# Run pipeline on the single image, ensure OCR on that image
	try:
		extractor = _build_extractor(model=model, ocr_engine=ocr_engine, use_preprocessing=use_preprocessing)
		metadata = extractor.extract_metadata_from_images([saved_path], ocr_image_indices=[0])
	except Exception as e:
		return JSONResponse(status_code=500, content={
			"id": item_id,
			"error": str(e),
		})

	return {
		"id": item_id,
		"files": [os.path.basename(saved_path)],
		"metadata": metadata,
	}


@app.post("/api/process_images")
async def process_images(
	images: List[UploadFile] = File(...),
	model: str = Form("gemma3:4b"),
	ocr_engine: str = Form("easyocr"),
	use_preprocessing: bool = Form(True),
):
	if not images:
		raise HTTPException(status_code=400, detail="No images uploaded")

	item_id = f"batch_{int(time.time() * 1000)}"
	saved_paths: List[str] = []
	for idx, uf in enumerate(images):
		if uf.content_type is None or not uf.content_type.startswith("image/"):
			raise HTTPException(status_code=400, detail=f"File {uf.filename} is not an image")
		ext = os.path.splitext(uf.filename or f"capture_{idx}.jpg")[1] or ".jpg"
		path = os.path.join(UPLOADS_DIR, f"{item_id}_{idx}{ext}")
		with open(path, "wb") as f:
			f.write(await uf.read())
		saved_paths.append(path)

	try:
		extractor = _build_extractor(model=model, ocr_engine=ocr_engine, use_preprocessing=use_preprocessing)
		ocr_indices = _compute_default_ocr_indices(len(saved_paths))
		metadata = extractor.extract_metadata_from_images(saved_paths, ocr_image_indices=ocr_indices)
	except Exception as e:
		return JSONResponse(status_code=500, content={
			"id": item_id,
			"error": str(e),
		})

	return {
		"id": item_id,
		"files": [os.path.basename(p) for p in saved_paths],
		"metadata": metadata,
	}


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
		if images:
			items.append({"id": entry, "count": len(images)})
	return {"books_dir": books_dir, "items": items}


class ExamplePayload(BaseModel):
	book_id: str
	model: Optional[str] = "gemma3:4b"
	ocr_engine: Optional[str] = "easyocr"
	use_preprocessing: Optional[bool] = True


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

	try:
		extractor = _build_extractor(model=payload.model or "gemma3:4b", ocr_engine=payload.ocr_engine or "easyocr", use_preprocessing=bool(payload.use_preprocessing))
		ocr_indices = _compute_default_ocr_indices(len(image_paths))
		metadata = extractor.extract_metadata_from_images(image_paths, ocr_image_indices=ocr_indices)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

	return {
		"id": f"example_{payload.book_id}",
		"files": [os.path.basename(p) for p in image_paths],
		"metadata": metadata,
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
