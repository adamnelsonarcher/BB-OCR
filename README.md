# BB-OCR (Book Image → JSON Metadata)

BB-OCR takes photos of a book (cover/title page/info pages), runs OCR + an LLM-assisted extractor, and outputs **structured JSON metadata** (title, authors, ISBN, publisher, publication date, etc.). That JSON can then be used to drive **online lookup / pricing aggregation** and downstream inventory workflows.

The most complete “demo app” lives in `pipeline_demo/i2j_ui`.

## Quickstart (recommended): run the Image→JSON UI

```bash
cd pipeline_demo/i2j_ui
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/`.

- **Scanner tab**: webcam capture queue or multi-file upload → process → review → accept/reject
- **Pricing tab**: run lookup/aggregation using the extracted metadata (embedded; no second server)

Accepted JSON outputs are saved under `pipeline_demo/i2j_ui/data/accepted/`.

## What’s in this repo

For the working pipeline + demo UI, see:

- `pipeline_demo/extractor/`: OCR + extraction pipeline (trace artifacts, prompts, JSON validation)
- `pipeline_demo/i2j_ui/`: main Image→JSON web UI (Scanner + embedded Pricing)
- `pipeline_demo/pricing_api/`: pricing/lookup providers + aggregation logic (also embedded into the UI)

There are also legacy/experimental folders from earlier exploration.

## Architecture doc

For a visual breakdown (diagram + swimlane), see:
[`BBOCR_Formal_7-2-25.pdf`](./BBOCR_Formal_7-2-25.pdf)

## Integration targets (downstream)

The JSON output is designed to feed into inventory/cataloging systems via:

- **CSV import pipelines**
- **RPA automation** (e.g., PyAutoGUI-driven workflows)
- **Label printing support** (optional downstream step)

## Technologies (current)

- **OCR**: EasyOCR, Tesseract
- **Preprocessing**: OpenCV
- **Extraction/structuring**: Python + heuristics + LLM prompting
- **Lookup/providers**: Google Books + other sources (see `pipeline_demo/pricing_api/providers/`)
