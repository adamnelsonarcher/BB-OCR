#!/usr/bin/env python
import argparse
import os
import sys
from PIL import Image
import pytesseract

def perform_ocr(image_path):
    """
    Perform OCR on the given image using Tesseract.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Open the image with PIL
        image = Image.open(image_path)
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image)
        
        # Get additional data like confidence scores and bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        return text, data
    except Exception as e:
        raise Exception(f"Error processing image with Tesseract: {str(e)}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Perform OCR on an image using Tesseract')
    parser.add_argument('--image', '-i', type=str, required=False, 
                        help='Path to the image file')
    args = parser.parse_args()
    
    # If no image path is provided via command line, ask for it
    image_path = args.image
    if not image_path:
        image_path = input("Enter the path to the image file: ")
    
    try:
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract is not installed or not in PATH.")
            print("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
            sys.exit(1)
            
        # Perform OCR
        text, data = perform_ocr(image_path)
        
        # Print the results
        print(f"\nOCR Results for {image_path}:")
        print("-" * 50)
        
        if not text.strip():
            print("No text detected in the image.")
        else:
            # Print the full text
            print("\nExtracted Text:")
            print("-" * 50)
            print(text)
            
            # Print detailed information about each detected word
            print("\nDetailed Word Information:")
            print("-" * 50)
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    confidence = data['conf'][i]
                    if confidence > 0:  # Only show items with confidence > 0
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        print(f"Word: {data['text'][i]}")
                        print(f"Confidence: {confidence}%")
                        print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")
                        print("-" * 30)
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 