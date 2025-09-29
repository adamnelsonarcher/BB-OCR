# Book Pricing Pipeline (modular demo)

This folder mirrors your pipeline across multiple files, keeping OCR, extraction, and pricing API/UI in one place.

## What it does
- OCR + preprocessing (EasyOCR/Tesseract) for book images
- Enhanced metadata extraction with Ollama VLM + OCR context
- Pricing/metadata aggregation (Google Books + AbeBooks) with a small UI

## Structure
```
pipeline_demo/
├── extractor/
│   ├── enhanced_extractor.py
│   ├── process_book_enhanced.py
│   ├── batch_processor_enhanced.py
│   └── prompts/
│       └── enhanced_book_metadata_prompt.txt
├── ocr_testing/
│   └── preprocessing/
│       └── image_preprocessor.py
│   
├── pricing_api/
│   ├── app/
│   │   └── main.py
│   ├── pricing_api/
│   │   ├── core/aggregator.py
│   │   └── providers/
│   │       ├── google_books.py
│   │       ├── abebooks_html.py
│   │       ├── amazon_stub.py
│   │       └── biblio_stub.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
├── requirements.txt
└── README.md (this file)
```

## Installation
1) Python 3.9+ recommended
2) Install system OCR engine(s):
   - Windows Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
3) Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4) Run Ollama and pull a vision-capable model:
   ```bash
   ollama serve
   ollama pull gemma3:4b
   ```
5) Optional: set Google Books API key
   ```bash
   # Powers better metadata coverage
   # Windows PowerShell:
   $env:GOOGLE_BOOKS_API_KEY="YOUR_KEY"
   # bash/zsh:
   export GOOGLE_BOOKS_API_KEY=YOUR_KEY
   ```

## Usage
### 1) Enhanced extraction (single book)
```bash
cd pipeline_demo/extractor
python process_book_enhanced.py 1 -o output
```
Common options: `--model`, `--ocr-engine`, `--ocr-indices`, `--edge-crop`, `--crop-ocr`, `--no-preprocessing`, `--books-dir`.

### 2) Batch extraction
```bash
cd pipeline_demo/extractor
python batch_processor_enhanced.py --max-workers 2
```

### 3) Extractor UI (overall process)
```bash
cd pipeline_demo/i2j_ui
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open `http://127.0.0.1:8000/`.

### 4) Pricing API + UI
```bash
cd pipeline_demo/pricing_api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8099
```
Open the UI at `/ui`. Load any JSON from `pipeline_demo/extractor/output` or `pipeline_demo/extractor/batch_output`, then run lookups.

## Output
- Extraction produces structured JSON similar to:
```json
{
  "title": "The Hobbit",
  "subtitle": null,
  "authors": ["J. R. R. Tolkien"],
  "publisher": "George Allen & Unwin",
  "publication_date": "1937",
  "isbn_10": null,
  "isbn_13": "9780261103344",
  "price": {"currency": null, "amount": null},
  "_processing_info": {
    "ocr_engine": "easyocr",
    "preprocessing_used": true,
    "ocr_images_processed": 2,
    "total_images": 3
  }
}
```
- API returns:
```json
{
  "query": { },
  "offers": [ ],
  "errors": { },
  "best": { },
  "merged": { }
}
```

## Notes
- AbeBooks uses light HTML parsing in this demo to keep a single-file footprint. For production, switch to BeautifulSoup and add resilience (selector variants, backoff, retries).
- Google Books provides metadata only; pricing depends mostly on AbeBooks until Amazon/Biblio are integrated.
- If ISBNs are missing, best-offer selection falls back to title match; adding publisher/binding to scoring can help.

## Troubleshooting
- Ollama connection errors: ensure `ollama serve` is running and the model is pulled.
- No text found: try `--ocr-engine tesseract`, enable preprocessing, or tweak cropping (`--edge-crop 8 --crop-ocr`).
- Windows path quoting: wrap paths with spaces in quotes.

## Extending
- Add more providers by mimicking the async `lookup(...)` signature and append to `DEFAULT_PROVIDERS`.
- Swap the simple AbeBooks regex with a robust BeautifulSoup parser.
- Emit a confidence score by combining OCR quality and provider match scoring.

## License
Part of the BB-OCR system. For internal experimentation and demo purposes.


