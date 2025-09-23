# Image→JSON Book Scanner UI (technical)

A minimal, technical web UI to capture images (webcam or file), run the existing OCR+LLM pipeline from `ollama+ocr_to_json`, and review metadata for accept/reject.

## Prerequisites
- Python 3.9+
- Ollama running locally with a suitable vision model (default: `gemma3:4b`)
- Tesseract installed if using `pytesseract` (or use `easyocr`)

## Install
```powershell
cd C:\Users\PC\Desktop\projects_and_code\Work\BB-OCR\img_to_json\i2j_ui
pip install -r requirements.txt
```

## Run backend
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open: `http://127.0.0.1:8000/`

## Features
- Two-column, wide-screen layout: controls/results on the left, live camera + queue on the right.
- Model dropdown populated from Ollama (`/api/models`), OCR engine dropdown, preprocessing toggle.
- Webcam capture queue: capture multiple frames, then "Process Queued" to process as a batch.
- File upload supports multiple images.
- Examples:
  - Populated from `books/` in the legacy paths; shows count and if a saved output exists.
  - "Run Example" processes the example images through the pipeline.
  - "Load Output" reads saved JSON from `ollama+ocr_to_json/test_output/book_{id}_enhanced.json` if present.
- Results displayed as a key/value table with enforced wrapping for long fields.
- Accept saves JSON to `i2j_ui/data/accepted/{id}.json`.
- Reject logs to `i2j_ui/data/rejected/{id}.txt`.
- Pricing Lookup (placeholder): sends current table fields to `/api/pricing_lookup` and displays a stub message.

## Endpoints
- GET `/` static UI
- GET `/api/health` backend health
- GET `/api/models` list Ollama models (fallback defaults if unavailable)
- GET `/api/examples` list example directories and whether a saved output exists
- GET `/api/example_output?book_id=...` return saved JSON for an example (if available)
- POST `/api/process_image` form-data: `image`, fields: `model`, `ocr_engine`, `use_preprocessing`
- POST `/api/process_images` same as above, multiple files under `images`
- POST `/api/process_example` json: `{ book_id, model?, ocr_engine?, use_preprocessing? }`
- POST `/api/accept` json: `{ id, metadata, notes? }`
- POST `/api/reject` json: `{ id, reason? }`
- POST `/api/pricing_lookup` json: `{ isbn_13?, isbn_10?, title?, authors? }` (placeholder)

## Notes
- Uploaded images are stored in `i2j_ui/data/uploads`. Cleanup as needed.
- The backend imports from `ollama+ocr_to_json` by adding it to `sys.path`.
- Default OCR indices use info pages when multiple images are provided (2nd/3rd).
