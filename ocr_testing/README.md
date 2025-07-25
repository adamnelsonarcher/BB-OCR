# Book Cover OCR Testing

This project compares different OCR engines (EasyOCR and Tesseract) for extracting text and metadata from book covers.

## Project Structure

```
ocr_testing/
├── hueristics/              # Text extraction and metadata analysis
│   ├── book_extractor.py    # Book-specific metadata extraction
│   └── extractor.py         # General metadata extraction
├── ocr_engines/             # OCR engine implementations
│   ├── compare_ocr_engines.py  # Main comparison script
│   ├── test_easyocr.py      # Standalone EasyOCR test
│   ├── test_tesseract.py    # Standalone Tesseract test
│   └── requirements.txt     # Dependencies for OCR engines
├── preprocessing/           # Image preprocessing techniques
│   ├── image_preprocessor.py  # Preprocessing implementation
│   ├── test_preprocessing.py  # Test script for preprocessing
│   └── requirements.txt     # Dependencies for preprocessing
├── results/                 # Output directory
│   ├── images/              # Preprocessed images
│   └── json/                # JSON results from OCR comparison
└── run_all_tests.py         # Script to run tests on all images
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- EasyOCR
- Tesseract OCR
- Pillow
- NLTK
- Matplotlib

## Installation

1. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

2. Install Python dependencies:
   ```
   pip install -r ocr_engines/requirements.txt
   pip install -r preprocessing/requirements.txt
   ```

3. Download NLTK data (for book metadata extraction):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   ```

## Usage

### Running OCR Comparison on a Single Image

```bash
python ocr_engines/compare_ocr_engines.py --image path/to/image.png [--preprocess]
```

Options:
- `--image`, `-i`: Path to the image file (required)
- `--preprocess`, `-p`: Apply preprocessing to the image before OCR (optional)

### Running OCR Comparison on All Images in Dataset

```bash
python run_all_tests.py [--preprocess]
```

Options:
- `--preprocess`, `-p`: Apply preprocessing to all images before OCR (optional)

### Testing Image Preprocessing

```bash
python preprocessing/test_preprocessing.py --image path/to/image.png --output results/images [--mode steps]
```

Options:
- `--image`, `-i`: Path to the image file (required)
- `--output`, `-o`: Directory to save preprocessed images (optional)
- `--mode`, `-m`: Test mode: "steps" to show all preprocessing steps, "book_cover" to test the book cover preprocessing function (default: "book_cover")

## Preprocessing Techniques

The preprocessing module includes the following techniques:
- Grayscale conversion
- Contrast enhancement
- Noise removal
- Sharpening
- Adaptive thresholding
- Deskewing (rotation correction)
- Border removal

## Book Metadata Extraction

The book metadata extractor attempts to identify:
- Book title
- Author
- Publisher
- Publication year
- ISBN
- Price
- Genre
- Series

## Results

Results are saved in JSON format in the `results/json/` directory with the following structure:

```json
{
  "image_path": "path/to/image.png",
  "preprocessing_used": true,
  "easyocr": {
    "processing_time": 5.67,
    "text_length": 150,
    "text": "Extracted text...",
    "preprocessed_image": "path/to/preprocessed/image.png",
    "book_metadata": {
      "title": "Book Title",
      "author": "Author Name",
      "publisher": "Publisher",
      "year": "2023",
      "isbn": "1234567890",
      "price": null,
      "genre": "fiction",
      "series": null
    }
  },
  "tesseract": {
    "processing_time": 1.23,
    "text_length": 145,
    "text": "Extracted text...",
    "preprocessed_image": "path/to/preprocessed/image.png",
    "book_metadata": {
      "title": "Book Title",
      "author": "Author Name",
      "publisher": "Publisher",
      "year": "2023",
      "isbn": "1234567890",
      "price": null,
      "genre": "fiction",
      "series": null
    }
  }
} 