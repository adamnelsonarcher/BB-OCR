# OCR Testing Utils

Utilities used by the extractor for image preprocessing and basic heuristic metadata extraction.

## Preprocessing
- Module: `preprocessing/image_preprocessor.py`
- Provides `preprocess_for_book_cover(image_path, output_path=None)`
- Techniques: grayscale, resize, denoise, contrast, CLAHE, sharpen

## Usage (library)
```python
from preprocessing.image_preprocessor import preprocess_for_book_cover
img, out_path, steps = preprocess_for_book_cover('cover.jpg', 'preprocessed.png')
```
