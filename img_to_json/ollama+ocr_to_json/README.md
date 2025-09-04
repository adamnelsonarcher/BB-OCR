# Enhanced Book Metadata Extraction Pipeline

A sophisticated system that combines OCR (Optical Character Recognition) with Ollama vision-language models to extract accurate structured metadata from book images. This pipeline significantly improves accuracy by providing OCR context from information pages to the vision model.

## Overview

This enhanced pipeline addresses the common issue of vision models hallucinating publication years, publishers, and other critical metadata by:

1. **Running OCR** on information pages (typically pages 2 and 3 containing copyright/imprint information)
2. **Preprocessing images** to improve OCR accuracy using advanced computer vision techniques
3. **Extracting heuristic metadata** from OCR text using pattern matching
4. **Providing OCR context** to the Ollama vision model alongside the images
5. **Generating structured JSON** with evidence snippets and validation

## Key Features

- **Dual-mode processing**: Combines OCR text analysis with vision model capabilities
- **Advanced preprocessing**: Applies CLAHE, denoising, sharpening, and other CV techniques
- **Multiple OCR engines**: Supports both EasyOCR and Tesseract
- **Intelligent fallback**: Uses heuristic metadata extraction if vision model fails
- **Comprehensive validation**: JSON schema validation and quality checks
- **Batch processing**: Parallel processing of multiple books with progress tracking
- **Detailed reporting**: Processing statistics and error tracking

## Architecture

```
Book Images → OCR Processing → Heuristic Extraction → Enhanced Prompt → Ollama Model → Structured JSON
     ↓              ↓                    ↓                    ↓              ↓
Image Files    Text Extraction    Pattern Matching    Context Integration   Metadata
```

## Installation

1. **Clone the repository and navigate to the enhanced pipeline directory:**
   ```bash
   cd ollama+ocr_to_json
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies:**

   **For Tesseract OCR (Ubuntu/Debian):**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

   **For Tesseract OCR (macOS):**
   ```bash
   brew install tesseract
   ```

   **For Tesseract OCR (Windows):**
   Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

4. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```

5. **Pull the required model:**
   ```bash
   ollama pull gemma3:4b
   ```

## Directory Structure

```
ollama+ocr_to_json/
├── enhanced_extractor.py          # Core extraction class
├── process_book_enhanced.py       # Single book processor
├── batch_processor_enhanced.py    # Batch processing script
├── prompts/
│   └── enhanced_book_metadata_prompt.txt  # Enhanced prompt template
├── requirements.txt               # Python dependencies
├── README.md                     # This documentation
└── output/                       # Generated JSON files (created automatically)
```

## Usage

### Single Book Processing

Process a single book by its directory ID:

```bash
python process_book_enhanced.py BOOK_ID
```

**Examples:**
```bash
# Basic usage
python process_book_enhanced.py 1

# With custom options
python process_book_enhanced.py 2a --model llama3.2-vision --ocr-engine tesseract --no-preprocessing

# Specify OCR indices (process images 1 and 3, 0-based indexing)
python process_book_enhanced.py 5 --ocr-indices 0 2

# Custom output directory
python process_book_enhanced.py 7 --output-dir my_results
```

**Command Line Options:**
- `--model, -m`: Ollama model to use (default: gemma3:4b)
- `--ocr-engine`: OCR engine (easyocr/tesseract, default: easyocr)
- `--no-preprocessing`: Disable image preprocessing
- `--ocr-indices`: Which images to run OCR on (0-based, default: 1 2)
- `--output-dir, -o`: Output directory (default: output)
- `--no-raw`: Don't show processing information
- `--books-dir`: Custom books directory path

### Batch Processing

Process multiple books in parallel:

```bash
python batch_processor_enhanced.py
```

**Examples:**
```bash
# Process all books with default settings
python batch_processor_enhanced.py

# Process specific books
python batch_processor_enhanced.py --book-ids 1 2a 5 7

# Use 4 parallel workers
python batch_processor_enhanced.py --max-workers 4

# Custom configuration
python batch_processor_enhanced.py --model llama3.2-vision --ocr-engine tesseract --output-dir batch_results
```

**Batch Command Line Options:**
- `--books-dir`: Path to books directory
- `--output-dir, -o`: Output directory (default: batch_output)
- `--model, -m`: Ollama model to use
- `--ocr-engine`: OCR engine to use
- `--no-preprocessing`: Disable preprocessing
- `--ocr-indices`: Images to run OCR on
- `--max-workers`: Parallel processing workers (default: 2)
- `--book-ids`: Specific books to process
- `--no-progress`: Disable progress bar

### Direct Library Usage

```python
from enhanced_extractor import EnhancedBookMetadataExtractor

# Initialize extractor
extractor = EnhancedBookMetadataExtractor(
    model="gemma3:4b",
    ocr_engine="easyocr",
    use_preprocessing=True
)

# Process a book directory
metadata = extractor.process_book_directory("/path/to/book/images")

# Process specific images
image_paths = ["cover.jpg", "copyright.jpg", "back.jpg"]
metadata = extractor.extract_metadata_from_images(image_paths, ocr_image_indices=[1])

print(json.dumps(metadata, indent=2))
```

## Configuration

### OCR Image Selection

By default, the pipeline runs OCR on images at indices 1 and 2 (2nd and 3rd images), which typically contain:
- Index 1: Copyright/imprint page
- Index 2: Additional information page or back cover

You can customize this with `--ocr-indices`:
```bash
# Run OCR only on the 2nd image (index 1)
python process_book_enhanced.py 1 --ocr-indices 1

# Run OCR on 1st and 3rd images (indices 0 and 2)
python process_book_enhanced.py 1 --ocr-indices 0 2
```

### OCR Engines

**EasyOCR (Default)**:
- Better for general text recognition
- Handles multiple languages
- More robust with varied fonts

**Tesseract**:
- Faster processing
- Better for high-quality scanned text
- More configurable

### Image Preprocessing

The pipeline applies several preprocessing steps to improve OCR accuracy:

1. **Grayscale conversion**
2. **Image resizing** (1.5x scale factor)
3. **Denoising** (Gaussian blur)
4. **Contrast enhancement**
5. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
6. **Gentle sharpening**

Disable with `--no-preprocessing` if your images are already optimized.

## Output Format

The pipeline generates structured JSON with the following fields:

```json
{
  "title": "Book Title",
  "subtitle": "Book Subtitle",
  "authors": ["Author Name"],
  "publisher": "Publisher Name",
  "publication_date": "2023",
  "isbn_10": "1234567890",
  "isbn_13": "9781234567890",
  "asin": "B08XXXXXXX",
  "edition": "First Edition",
  "binding_type": "Hardcover",
  "language": "English",
  "page_count": 256,
  "categories": ["Fiction", "Mystery"],
  "description": "Book description...",
  "condition_keywords": ["good", "minor wear"],
  "price": {
    "currency": "USD",
    "amount": 15.99
  },
  "_processing_info": {
    "ocr_engine": "easyocr",
    "preprocessing_used": true,
    "ocr_images_processed": 2,
    "total_images": 3,
    "heuristic_metadata_found": true
  }
}
```

### Processing Information

The `_processing_info` object contains metadata about how the book was processed, useful for debugging and quality assessment.

## Error Handling

The pipeline includes robust error handling:

1. **OCR Failures**: Falls back to image-only processing
2. **Preprocessing Errors**: Uses original images
3. **JSON Parsing Errors**: Uses heuristic metadata as fallback
4. **Validation Errors**: Saves metadata with error flags
5. **Network Issues**: Provides clear error messages

## Performance

**Typical processing times** (per book):
- **Single image**: 5-15 seconds
- **With OCR**: 10-25 seconds
- **With preprocessing**: 15-30 seconds

**Batch processing** with parallel workers can significantly reduce total processing time.

## Troubleshooting

### Common Issues

1. **"Could not connect to Ollama server"**
   ```bash
   # Ensure Ollama is running
   ollama serve
   ```

2. **"No books found to process"**
   - Check that books directory exists and contains subdirectories
   - Verify book directory structure matches expected format

3. **OCR errors**
   ```bash
   # Try different OCR engine
   python process_book_enhanced.py 1 --ocr-engine tesseract
   
   # Disable preprocessing if images are high quality
   python process_book_enhanced.py 1 --no-preprocessing
   ```

4. **Memory issues with batch processing**
   ```bash
   # Reduce parallel workers
   python batch_processor_enhanced.py --max-workers 1
   ```

### Debugging

Enable verbose output to see detailed processing information:
```bash
python process_book_enhanced.py 1  # Shows processing info by default
```

Check the generated JSON files for `_processing_info` and `evidence` fields to understand how metadata was extracted.

## Comparison with Original Pipeline

| Feature | Original Pipeline | Enhanced Pipeline |
|---------|------------------|-------------------|
| **Input** | Images only | Images + OCR context |
| **Accuracy** | Moderate (hallucination issues) | High (OCR validation) |
| **Processing** | Vision model only | OCR + Vision model |
| **Fallback** | None | Heuristic extraction |
| **Evidence** | Limited | Comprehensive snippets |
| **Preprocessing** | None | Advanced CV techniques |
| **Validation** | Basic | Multi-level validation |

## Future Enhancements

- **Multi-language support**: Extend OCR to support more languages
- **Custom model fine-tuning**: Train models on book-specific datasets
- **Advanced heuristics**: Improve pattern matching for metadata extraction
- **Cloud integration**: Support for cloud-based OCR services
- **Quality scoring**: Confidence metrics for extracted metadata

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is part of the BB-OCR book metadata extraction system. Please refer to the main project license.

---

For questions or issues, please refer to the main project documentation or create an issue in the repository.
