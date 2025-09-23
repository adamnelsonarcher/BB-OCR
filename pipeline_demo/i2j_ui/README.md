# Image→JSON Book Scanner UI (Extractor UI)

Minimal web UI to capture images, run the OCR+Ollama extractor, review fields, and accept/reject outputs.

## Run
```bash
cd pipeline_demo/i2j_ui
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open `http://127.0.0.1:8000/`.

## Features
- Webcam capture queue or multi-file upload
- Model and OCR engine selection, preprocessing toggle
- Edge crop slider and optional auto text crop
- Example runner against `pipeline_demo/extractor/books`
- Accept saves to `pipeline_demo/i2j_ui/data/accepted/{id}.json`

## Endpoints
- GET `/` static UI
- GET `/api/health`, `/api/models`, `/api/examples`, `/api/example_output`
- POST `/api/process_image`, `/api/process_images`
- POST `/api/process_example`, `/api/accept`, `/api/reject`
- POST `/api/pricing_lookup` (placeholder)
