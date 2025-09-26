# Image→JSON Book Scanner UI (with embedded Pricing UI)

Minimal web UI to capture images, run the OCR+Ollama extractor, review fields, accept/reject outputs, and run pricing/metadata aggregation – all in one app.

## Run
```bash
cd pipeline_demo/i2j_ui
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open `http://127.0.0.1:8000/`.

## Features
- Two tabs in the header: Scanner and Pricing
- Scanner
  - Webcam capture queue or multi-file upload
  - Model and OCR engine selection, preprocessing toggle
  - Edge crop slider and optional auto text crop
  - Example runner against `pipeline_demo/extractor/books`
  - Live trace stream: input/processed images, OCR text, prompt, VLM raw, steps, logs
  - Accept saves to `pipeline_demo/i2j_ui/data/accepted/{id}.json`
- Pricing (embedded)
  - The original pricing UI is embedded via an iframe
  - Uses the same provider list and aggregation logic as `pricing_api`
  - Can load processed extractor JSONs and run lookups; shows Raw/Request/Best/Merged tables

## Endpoints
- Core UI
  - GET `/` (static UI)
  - GET `/api/health`, `/api/models`, `/api/examples`, `/api/example_output`
  - POST `/api/process_image`, `/api/process_images`
  - POST `/api/process_example`, `/api/accept`, `/api/reject`
- Pricing integration (no second server needed)
  - Static assets mounted at `/pricing_static/*` (from `pipeline_demo/pricing_api/static`)
  - GET `/pricing_embed` (serves pricing UI HTML, rewritten to use `/pricing_static`)
  - GET `/api/pricing/providers` (provider list)
  - POST `/api/pricing_lookup` (runs aggregator)
  - GET `/api/pricing/processed/list`, `/api/pricing/processed/load` (helper endpoints to browse/load processed JSON)
  - Compatibility aliases used by the embedded pricing UI: `/providers`, `/lookup`, `/processed/list`, `/processed/load`

Notes:
- The pricing aggregator is imported from `pricing_api.core.aggregator`. You do not need to run a separate pricing server.
- If you prefer the standalone pricing app, you can still run `uvicorn pricing_api.app.main:app` separately; the embedded UI will continue to work.
