import os
import sys
import time
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

app = FastAPI(title="Image-to-JSON Book Scanner UI", version="0.1.0")

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

	# Run pipeline on the single image, ensuring OCR runs on index 0
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
		"file": os.path.basename(saved_path),
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
