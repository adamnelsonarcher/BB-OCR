#!/usr/bin/env python
import argparse
import easyocr
import os

def perform_ocr(image_path):
    """
    Perform OCR on the given image using EasyOCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        list: List of detected text
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Initialize the OCR reader (first time it will download the model files)
    reader = easyocr.Reader(['en'])  # Initialize for English
    
    # Perform OCR on the image
    results = reader.readtext(image_path)
    
    return results

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Perform OCR on an image using EasyOCR')
    parser.add_argument('--image', '-i', type=str, required=False, 
                        help='Path to the image file')
    args = parser.parse_args()
    
    # If no image path is provided via command line, ask for it
    image_path = args.image
    if not image_path:
        image_path = input("Enter the path to the image file: ")
    
    try:
        # Perform OCR
        results = perform_ocr(image_path)
        
        # Print the results
        print(f"\nOCR Results for {image_path}:")
        print("-" * 50)
        
        if not results:
            print("No text detected in the image.")
        else:
            for i, (bbox, text, prob) in enumerate(results, 1):
                print(f"Text {i}: {text} (Confidence: {prob:.2f})")
                print(f"Bounding Box: {bbox}")
                print("-" * 30)
            
            # Print just the text for easy copying
            print("\nExtracted Text:")
            print("-" * 50)
            for _, text, _ in results:
                print(text)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 