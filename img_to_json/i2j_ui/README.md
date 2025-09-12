# Image→JSON Book Scanner UI (technical)

A minimal, technical web UI to capture an image (webcam or file), run the existing OCR+LLM pipeline from `ollama+ocr_to_json`, and review metadata for accept/reject.

## Prerequisites
- Python 3.9+
- Ollama running locally with a suitable vision model (default: `gemma3:4b`)
- Tesseract installed if using `pytesseract` (or use `easyocr`)

## Install
```bash
cd i2j_ui
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If EasyOCR fails on your system, switch OCR engine in the UI to `tesseract`.

## Run backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open: `http://localhost:8000/`

## Endpoints
- GET `/` static UI
- GET `/api/health` backend health
- POST `/api/process_image` form-data: `image`, fields: `model`, `ocr_engine`, `use_preprocessing`
- POST `/api/accept` json: `{ id, metadata, notes? }` -> saves to `i2j_ui/data/accepted/{id}.json`
- POST `/api/reject` json: `{ id, reason? }` -> logs to `i2j_ui/data/rejected/{id}.txt`

## Notes
- Uploaded images are stored in `i2j_ui/data/uploads`. Cleanup as needed.
- The backend imports from `ollama+ocr_to_json` by adding it to `sys.path`.
- For multi-image flow, extend the UI to take multiple captures before processing.
