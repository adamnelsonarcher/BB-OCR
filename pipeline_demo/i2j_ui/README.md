# Image→JSON Book Scanner UI (with embedded Pricing)

`i2j_ui` is a small web app that:

- Takes photos of a book (webcam capture queue or multi-file upload)
- Runs OCR + an LLM to extract **structured book metadata** (title, authors, ISBN, publisher, etc.)
- Outputs a **formatted JSON** result you can review and accept/reject
- Lets you run an **online lookup / pricing aggregation** using that JSON (embedded Pricing tab)

This is designed to be the “main demo app” for the Image→JSON pipeline.

## Requirements

- Python 3.10+ recommended
- If using **Ollama (local)**: install and run Ollama, and have at least one vision-capable model pulled
- If using **Gemini/OpenAI** backends: set the appropriate environment variables for `pipeline_demo/llm_providers`

## Run the UI

```bash
cd pipeline_demo/i2j_ui
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/`.

## How to use (Scanner)

- **Capture**: uses your webcam to add one image to the queue
- **Process Queued**: runs the pipeline on everything in the capture queue
- **Upload**: choose one or many images to process (multi-file upload)
- **Examples**: run example image sets from `pipeline_demo/extractor/books`
- **Review**: the Result table shows extracted fields; the Processing Trace shows images, OCR text, prompt, and raw model output
- **Accept / Reject**:
  - **Accept** saves JSON to `pipeline_demo/i2j_ui/data/accepted/{id}.json`
  - **Reject** logs a reason to `pipeline_demo/i2j_ui/data/rejected/{id}.txt`

## How to use (Pricing)

- Click the **Pricing** tab (or use **Pricing Lookup** after Accept)
- Select providers and run a lookup using the extracted metadata
- The embedded Pricing UI can also load previously processed JSON files

## Notes

- Pricing is embedded — you **do not** need to run a second server.
- The embedded pricing UI reuses the same aggregation logic as `pipeline_demo/pricing_api`.

## API (optional)

Most users can ignore this, but these endpoints drive the UI:

- **Core UI**: `GET /`, `GET /api/health`, `POST /api/process_image`, `POST /api/process_images`, `POST /api/process_example`, `POST /api/accept`, `POST /api/reject`
- **Pricing**: `GET /pricing_embed`, `GET /api/pricing/providers`, `POST /api/pricing_lookup`, `GET /api/pricing/processed/list`, `GET /api/pricing/processed/load`
