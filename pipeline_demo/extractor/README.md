# Extractor (OCR + Ollama)

Runs OCR + preprocessing, builds an enhanced prompt with OCR context, and calls Ollama to produce structured JSON.

## Requirements
- Python 3.9+
- Ollama running locally (`ollama serve`) and a vision model (`ollama pull gemma3:4b`)
- Tesseract installed if using `pytesseract` (or use `easyocr`)

## Single book
```bash
cd pipeline_demo/extractor
python process_book_enhanced.py 1 -o output
```
Common options:
- `--model`, `--ocr-engine easyocr|tesseract`
- `--ocr-indices` (default: 1 2)
- `--edge-crop <0-45>`, `--crop-ocr`, `--crop-margin <px>`
- `--no-preprocessing`, `--no-warm-model`
- `--books-dir` to point at your images

## Batch
```bash
cd pipeline_demo/extractor
python batch_processor_enhanced.py --max-workers 2
```

## Outputs
- `output/book_<id>_enhanced.json`
- `batch_output/book_<id>_enhanced.json`

## Prompt
- `prompts/enhanced_book_metadata_prompt.txt`
