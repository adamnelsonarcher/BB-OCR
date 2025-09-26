# Pricing API + UI

Aggregates metadata and prices from external providers (Google Books, AbeBooks) and exposes a small UI to run lookups and review results.

## Run (standalone)
```bash
cd pipeline_demo/pricing_api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8099
```
Open `http://127.0.0.1:8099/ui`.

You can also use the embedded UI inside `i2j_ui` (no separate server required). The embedded app mounts these endpoints under the main FastAPI app.

## Endpoints
- `GET /` → health + available providers
- `GET /ui` → static UI
- `GET /providers` → list of provider names
- `GET /processed/list` → recent processed JSON files (from extractor and accepted folders)
- `GET /processed/load?path=...` → load a JSON and return both `payload` and full `raw`
- `POST /lookup` → run aggregation against selected providers

## Request payload (flexible types)
The API is tolerant to input types and normalizes internally:
- `title`, `subtitle`, `publisher`, `publication_date`, `isbn_13`, `isbn_10`: any → string or null
- `authors`: string|list|null → List[string]
- `providers`: string|list|null → List[string] or null (null ⇒ default providers)

Example:
```json
{
  "title": "The Hoosier Schoolmaster",
  "authors": "Edward Eggleston",
  "publication_date": 1892,
  "isbn_13": null,
  "isbn_10": null,
  "providers": ["abebooks", "google_books"]
}
```

## Response shape
```json
{
  "query": { "title": "...", "authors": ["..."], "isbn_13": null, "isbn_10": null, "publisher": null, "publication_date": "...", "providers": ["..."] },
  "providers": ["google_books", "abebooks", "amazon", "biblio"],
  "offers": [ { /* provider offer objects */ } ],
  "errors": { "provider": "error message if any" }
}
```

Each offer can include (provider-dependent):
- `provider`, `listing_id`, `title`, `authors`, `publisher`, `publication_date`
- `isbn_13`, `isbn_10`
- `currency`, `amount` (numeric when available)
- `url`, `source`

## UI flow
1) Load a processed extractor JSON from the left dropdown (or paste JSON in the textarea).
2) Pick providers and click Run.
3) The right panel shows:
   - Request table (normalized `query`)
   - Best offer table (simple heuristic: ISBN match > exact title > first)
   - Merged table (starts with your input JSON, then fills missing fields with the best offer’s values). The `price` field in your JSON is an object: `{ "currency": null|"USD"|..., "amount": number|null }`. The offers list contains `currency` and `amount` per provider.

Notes:
- Google Books returns metadata only (no price).
- AbeBooks is HTML-scraped; selectors and formatting can change. We handle several common patterns for currency/amount.
- Inputs can come from:
  - `pipeline_demo/extractor/output`
  - `pipeline_demo/extractor/batch_output`
  - `pipeline_demo/i2j_ui/data/accepted`

## Provider details
- Google Books (`pricing_api/pricing_api/providers/google_books.py`): REST API, returns metadata fields; we map identifiers into `isbn_13`/`isbn_10`.
- AbeBooks (`pricing_api/pricing_api/providers/abebooks_html.py`): HTML scraping; attempts multiple selectors and currency formats. Outputs `currency` and numeric `amount` when parsed.

## Troubleshooting
- If you see 422 errors (Unprocessable Entity): you can send strings or lists for `authors/providers`, and numbers or strings for `publication_date`; the server normalizes them.
- If prices are missing: open the provider `url` to confirm there’s a visible price; HTML may differ from our selectors.
- Network timeouts: the aggregator applies per-provider timeouts (6–8s typical).
