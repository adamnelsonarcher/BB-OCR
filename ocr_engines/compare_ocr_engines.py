#!/usr/bin/env python
import argparse
import os
import time
import json
from PIL import Image
import easyocr
import pytesseract
from extractor import extract_metadata_from_text

def process_with_easyocr(image_path):
    """
    Process an image with EasyOCR and measure the time taken.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (text, metadata, processing_time)
    """
    start_time = time.time()
    
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR
    results = reader.readtext(image_path)
    
    # Extract text from results
    text = " ".join([result[1] for result in results])
    
    # Extract metadata
    metadata = extract_metadata_from_text(text)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return text, metadata, processing_time

def process_with_tesseract(image_path):
    """
    Process an image with Tesseract OCR and measure the time taken.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (text, metadata, processing_time)
    """
    start_time = time.time()
    
    # Open the image
    image = Image.open(image_path)
    
    # Perform OCR
    text = pytesseract.image_to_string(image)
    
    # Extract metadata
    metadata = extract_metadata_from_text(text)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return text, metadata, processing_time

def compare_ocr_engines(image_path):
    """
    Compare the performance of EasyOCR and Tesseract OCR on the given image.
    
    Args:
        image_path (str): Path to the image file
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Comparing OCR engines on image: {image_path}")
    print("=" * 80)
    
    # Process with EasyOCR
    print("\nProcessing with EasyOCR...")
    try:
        easyocr_text, easyocr_metadata, easyocr_time = process_with_easyocr(image_path)
        print(f"EasyOCR processing time: {easyocr_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing with EasyOCR: {e}")
        easyocr_text = ""
        easyocr_metadata = {}
        easyocr_time = 0
    
    # Process with Tesseract
    print("\nProcessing with Tesseract OCR...")
    try:
        tesseract_text, tesseract_metadata, tesseract_time = process_with_tesseract(image_path)
        print(f"Tesseract processing time: {tesseract_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing with Tesseract OCR: {e}")
        tesseract_text = ""
        tesseract_metadata = {}
        tesseract_time = 0
    
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
    
    print("\n3. Extracted Metadata Comparison:")
    print("\nEasyOCR Metadata:")
    print(json.dumps(easyocr_metadata, indent=2))
    
    print("\nTesseract Metadata:")
    print(json.dumps(tesseract_metadata, indent=2))
    
    # Save results to file
    results = {
        "image_path": image_path,
        "easyocr": {
            "processing_time": easyocr_time,
            "text_length": len(easyocr_text),
            "text": easyocr_text[:500] + "..." if len(easyocr_text) > 500 else easyocr_text,
            "metadata": easyocr_metadata
        },
        "tesseract": {
            "processing_time": tesseract_time,
            "text_length": len(tesseract_text),
            "text": tesseract_text[:500] + "..." if len(tesseract_text) > 500 else tesseract_text,
            "metadata": tesseract_metadata
        }
    }
    
    output_file = f"ocr_comparison_{os.path.basename(image_path).split('.')[0]}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare OCR engines (EasyOCR and Tesseract)')
    parser.add_argument('--image', '-i', type=str, required=True, 
                        help='Path to the image file')
    
    args = parser.parse_args()
    compare_ocr_engines(args.image)

if __name__ == "__main__":
    main() 