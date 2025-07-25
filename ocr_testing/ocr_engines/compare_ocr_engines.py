#!/usr/bin/env python
import argparse
import os
import time
import json
from PIL import Image
import easyocr
import pytesseract
import sys

# Fix the import to use relative path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from hueristics.book_extractor import extract_book_metadata_from_text

# Import the preprocessing module
try:
    from preprocessing.image_preprocessor import preprocess_for_book_cover
    PREPROCESSING_AVAILABLE = True
except ImportError:
    print("Warning: Preprocessing module not available. Running without preprocessing.")
    PREPROCESSING_AVAILABLE = False

def process_with_easyocr(image_path, use_preprocessing=False):
    """
    Process an image with EasyOCR and measure the time taken.
    
    Args:
        image_path (str): Path to the image file
        use_preprocessing (bool): Whether to apply preprocessing to the image
        
    Returns:
        tuple: (text, metadata, processing_time, preprocessed_image_path)
    """
    start_time = time.time()
    
    # Apply preprocessing if enabled
    preprocessed_image_path = None
    if use_preprocessing and PREPROCESSING_AVAILABLE:
        # Create directory for preprocessed images
        os.makedirs("results/images", exist_ok=True)
        
        # Generate output path for preprocessed image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        preprocessed_image_path = os.path.join("results/images", f"{base_name}_preprocessed.png")
        
        # Preprocess the image
        print(f"Applying preprocessing to {image_path}...")
        _, preprocessed_image_path, _ = preprocess_for_book_cover(image_path, preprocessed_image_path)
        
        # Use the preprocessed image for OCR
        ocr_image_path = preprocessed_image_path
    else:
        ocr_image_path = image_path
    
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR
    results = reader.readtext(ocr_image_path)
    
    # Extract text from results
    text = " ".join([result[1] for result in results])
    
    # Extract book metadata
    metadata = extract_book_metadata_from_text(text)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return text, metadata, processing_time, preprocessed_image_path

def process_with_tesseract(image_path, use_preprocessing=False):
    """
    Process an image with Tesseract OCR and measure the time taken.
    
    Args:
        image_path (str): Path to the image file
        use_preprocessing (bool): Whether to apply preprocessing to the image
        
    Returns:
        tuple: (text, metadata, processing_time, preprocessed_image_path)
    """
    start_time = time.time()
    
    # Apply preprocessing if enabled
    preprocessed_image_path = None
    if use_preprocessing and PREPROCESSING_AVAILABLE:
        # Create directory for preprocessed images
        os.makedirs("results/images", exist_ok=True)
        
        # Generate output path for preprocessed image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        preprocessed_image_path = os.path.join("results/images", f"{base_name}_tesseract_preprocessed.png")
        
        # Preprocess the image
        print(f"Applying preprocessing to {image_path} for Tesseract...")
        _, preprocessed_image_path, _ = preprocess_for_book_cover(image_path, preprocessed_image_path)
        
        # Use the preprocessed image for OCR
        ocr_image_path = preprocessed_image_path
    else:
        ocr_image_path = image_path
    
    # Open the image
    image = Image.open(ocr_image_path)
    
    # Perform OCR with Tesseract
    text = pytesseract.image_to_string(image)
    
    # Extract book metadata
    metadata = extract_book_metadata_from_text(text)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return text, metadata, processing_time, preprocessed_image_path

def compare_ocr_engines(image_path, use_preprocessing=False):
    """
    Compare the performance of EasyOCR and Tesseract OCR on the given image.
    
    Args:
        image_path (str): Path to the image file
        use_preprocessing (bool): Whether to apply preprocessing to the image
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Comparing OCR engines on image: {image_path}")
    print("=" * 80)
    
    # Process with EasyOCR
    print("\nProcessing with EasyOCR...")
    try:
        easyocr_text, easyocr_metadata, easyocr_time, easyocr_preprocessed = process_with_easyocr(
            image_path, use_preprocessing
        )
        print(f"EasyOCR processing time: {easyocr_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing with EasyOCR: {e}")
        easyocr_text = ""
        easyocr_metadata = {}
        easyocr_time = 0
        easyocr_preprocessed = None
    
    # Process with Tesseract
    print("\nProcessing with Tesseract OCR...")
    try:
        tesseract_text, tesseract_metadata, tesseract_time, tesseract_preprocessed = process_with_tesseract(
            image_path, use_preprocessing
        )
        print(f"Tesseract processing time: {tesseract_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing with Tesseract OCR: {e}")
        tesseract_text = ""
        tesseract_metadata = {}
        tesseract_time = 0
        tesseract_preprocessed = None
    
    # Print comparison results
    print("\nResults Comparison:")
    print("=" * 80)
    
    print("\n1. Processing Time Comparison:")
    print(f"   EasyOCR:    {easyocr_time:.2f} seconds")
    print(f"   Tesseract:  {tesseract_time:.2f} seconds")
    
    if easyocr_time > 0 and tesseract_time > 0:
        time_diff = abs(easyocr_time - tesseract_time)
        faster = "EasyOCR" if tesseract_time > easyocr_time else "Tesseract"
        print(f"   {faster} was faster by {time_diff:.2f} seconds")
    
    print("\n2. Text Length Comparison:")
    print(f"   EasyOCR:    {len(easyocr_text)} characters")
    print(f"   Tesseract:  {len(tesseract_text)} characters")
    
    print("\n3. Extracted Book Metadata Comparison:")
    print("\nEasyOCR Metadata:")
    print(json.dumps(easyocr_metadata, indent=2))
    
    print("\nTesseract Metadata:")
    print(json.dumps(tesseract_metadata, indent=2))
    
    # Save results to file
    results = {
        "image_path": image_path,
        "preprocessing_used": use_preprocessing,
        "easyocr": {
            "processing_time": easyocr_time,
            "text_length": len(easyocr_text),
            "text": easyocr_text[:500] + "..." if len(easyocr_text) > 500 else easyocr_text,
            "preprocessed_image": easyocr_preprocessed,
            "book_metadata": easyocr_metadata
        },
        "tesseract": {
            "processing_time": tesseract_time,
            "text_length": len(tesseract_text),
            "text": tesseract_text[:500] + "..." if len(tesseract_text) > 500 else tesseract_text,
            "preprocessed_image": tesseract_preprocessed,
            "book_metadata": tesseract_metadata
        }
    }
    
    # Create results directory if it doesn't exist
    os.makedirs("results/json", exist_ok=True)
    
    # Save results to JSON file
    base_name = os.path.basename(image_path).split('.')[0]
    output_file = f"results/json/ocr_comparison_{base_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare OCR engines (EasyOCR and Tesseract)')
    parser.add_argument('--image', '-i', type=str, required=True, 
                        help='Path to the image file')
    parser.add_argument('--preprocess', '-p', action='store_true',
                        help='Apply preprocessing to the image before OCR')
    
    args = parser.parse_args()
    compare_ocr_engines(args.image, args.preprocess)

if __name__ == "__main__":
    main() 