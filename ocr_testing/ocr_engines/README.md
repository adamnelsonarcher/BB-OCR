# OCR Engines Test Scripts

This repository contains scripts to test different OCR (Optical Character Recognition) engines for text extraction from images.

## Available Scripts

### 1. EasyOCR Test Script

`test_easyocr.py` - Uses the EasyOCR library for text extraction.

Usage:
```bash
python test_easyocr.py --image path/to/image.jpg
```

### 2. Tesseract Test Script

`test_tesseract.py` - Uses Tesseract OCR via pytesseract for text extraction.

Usage:
```bash
python test_tesseract.py --image path/to/image.jpg
```


### 4. OCR Engine Comparison Script

`compare_ocr_engines.py` - Compares the performance and results of EasyOCR and Tesseract OCR.

Usage:
```bash
python compare_ocr_engines.py --image path/to/image.jpg
```

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. For the Tesseract script, you need to install Tesseract OCR engine:
   - Windows: Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

## Notes

- The EasyOCR script will download model files on first run
- The Tesseract script requires the Tesseract executable to be installed and in your PATH
- The comparison script saves detailed results to a JSON file for further analysis 