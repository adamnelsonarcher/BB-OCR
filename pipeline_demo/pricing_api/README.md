# Pricing API + UI

Aggregates metadata/prices from providers and displays a minimal UI to select/merge best results.

## Run
```bash
cd pipeline_demo/pricing_api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8099
```
Open `http://127.0.0.1:8099/ui`.

## Providers
- Google Books (metadata only)
- AbeBooks (HTML scrape to price)
- Amazon/Biblio stubs (placeholders)

## Inputs
- The UI can load any JSON from:
  - `pipeline_demo/extractor/output`
  - `pipeline_demo/extractor/batch_output`
  - `pipeline_demo/i2j_ui/data/accepted`

## API
- `GET /providers`
- `GET /processed/list`, `GET /processed/load?path=...`
- `POST /lookup` body includes `title`, `authors`, `isbn_13`, `isbn_10`, etc.

## Notes
- Timeouts and simple de-dupe are applied; scraping can be brittle if markup changes.
