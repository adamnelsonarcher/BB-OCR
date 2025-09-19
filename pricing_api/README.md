Pricing API (experimental)

A small FastAPI service and CLI for testing book pricing/metadata providers.

Implemented providers:
- Google Books (metadata only, no prices) — useful for validation/enrichment

Stubs (ready to implement):
- Amazon Product Advertising API
- AbeBooks
- Biblio

Run the API:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8099
```

Example request:
```bash
curl -X POST http://localhost:8099/lookup \
  -H "Content-Type: application/json" \
  -d '{"title":"The Hobbit", "authors":["J. R. R. Tolkien"], "isbn_13":null, "isbn_10":null}'
```

Run the CLI:
```bash
python -m pricing_api.cli --title "The Hobbit" --author "Tolkien"
```

Configuration:
- Google Books API key (optional): set env var GOOGLE_BOOKS_API_KEY
- Timeouts are conservative to keep tests fast

Notes:
- Google Books offers no pricing, so the provider returns normalized offers with amount=null and currency=null, plus a metadata URL when available.
- Parallel querying is supported; non-responsive providers are bounded by timeouts.


