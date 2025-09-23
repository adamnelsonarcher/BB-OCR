#!/usr/bin/env python3
"""
Enhanced Book Processor

A comprehensive script to process books using the enhanced OCR + Ollama pipeline
and validate the extracted metadata with improved accuracy.
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
import jsonschema
from tqdm import tqdm

# Import the enhanced extractor module
from enhanced_extractor import EnhancedBookMetadataExtractor

# Define the JSON schema for validation (same as in enhanced_extractor)
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "subtitle": {"type": ["string", "null"]},
        "authors": {
            "type": "array",
            "items": {"type": "string"}
        },
        "publisher": {"type": ["string", "null"]},
        "publication_date": {"type": ["string", "null"]},
        "isbn_10": {"type": ["string", "null"]},
        "isbn_13": {"type": ["string", "null"]},
        "asin": {"type": ["string", "null"]},
        "edition": {"type": ["string", "null"]},
        "binding_type": {"type": ["string", "null"]},
        "language": {"type": ["string", "null"]},
        "page_count": {"type": ["integer", "null"]},
        "categories": {
            "type": "array",
            "items": {"type": "string"}
        },
        "description": {"type": ["string", "null"]},
        "condition_keywords": {
            "type": "array",
            "items": {"type": "string"}
        },
        "price": {
            "type": "object",
            "properties": {
                "currency": {"type": ["string", "null"]},
                "amount": {"type": ["number", "null"]}
            }
        }
    }
}

def validate_metadata(metadata):
    """Validate the metadata against the schema and check for quality."""
    # Validate against JSON schema
    try:
        jsonschema.validate(instance=metadata, schema=METADATA_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        return False, f"Schema validation failed: {e}"
    
    # Check for minimum required fields
    if metadata.get("title") is None:
        return False, "Missing title"
    
    # Check for empty arrays that should be empty lists instead of null
    for array_field in ["authors", "categories", "condition_keywords"]:
        if metadata.get(array_field) is None:
            metadata[array_field] = []
    
    # Validate ISBN formats if present
    isbn_10 = metadata.get("isbn_10")
    if isbn_10 and (not isinstance(isbn_10, str) or len(isbn_10.replace("-", "")) != 10):
        return False, f"Invalid ISBN-10 format: {isbn_10}"
    
    isbn_13 = metadata.get("isbn_13")
    if isbn_13 and (not isinstance(isbn_13, str) or len(isbn_13.replace("-", "")) != 13):
        return False, f"Invalid ISBN-13 format: {isbn_13}"
    
    return True, "Validation successful"

def process_book_enhanced(book_id, output_dir="output", model="gemma3:4b", ocr_engine="easyocr", 
                         use_preprocessing=True, ocr_indices=None, show_raw=True, books_dir=None,
                         crop_ocr=False, crop_margin=16, no_warm_model=False, extractor: EnhancedBookMetadataExtractor = None,
                         edge_crop=0.0):
    """Process a single book by ID using the enhanced pipeline."""
    
    # Determine the book directory path
    if books_dir is None:
        # Try to find books directory in parent directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Check common locations
        possible_books_dirs = [
            os.path.join(current_dir, "books"),
            os.path.join(parent_dir, "ollama_to_JSON", "books"),
            os.path.join(parent_dir, "books")
        ]
        
        books_dir = None
        for possible_dir in possible_books_dirs:
            if os.path.isdir(possible_dir):
                books_dir = possible_dir
                break
        
        if books_dir is None:
            print("Error: Could not find books directory. Please specify with --books-dir")
            return False
    
    book_dir = os.path.join(books_dir, str(book_id))
    
    if not os.path.isdir(book_dir):
        print(f"Error: Book directory {book_dir} not found")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"book_{book_id}_enhanced.json")
    
    print(f"\n🚀 ENHANCED BOOK PROCESSING STARTED")
    print(f"=" * 60)
    print(f"📖 Book ID: {book_id}")
    print(f"📁 Book directory: {book_dir}")
    print(f"🤖 Ollama model: {model}")
    print(f"👁️  OCR engine: {ocr_engine}")
    print(f"🔧 Image preprocessing: {'✅ enabled' if use_preprocessing else '❌ disabled'}")
    if ocr_indices:
        print(f"📋 OCR target indices: {ocr_indices}")
    else:
        print(f"📋 OCR target indices: [1, 2] (default)")
    print(f"💾 Output file: {output_file}")
    print(f"=" * 60)
    
    # Track processing time
    start_time = time.time()
    
    try:
        # Initialize or reuse the enhanced extractor
        if extractor is None:
            extractor = EnhancedBookMetadataExtractor(
                model=model,
                ocr_engine=ocr_engine,
                use_preprocessing=use_preprocessing,
                crop_for_ocr=crop_ocr,
                crop_margin=crop_margin,
                warm_model=not no_warm_model,
                edge_crop_percent=edge_crop
            )
        
        # Process the book directory
        metadata = extractor.process_book_directory(book_dir, ocr_indices)
        
        # Display processing information if requested and available
        if show_raw and "_processing_info" in metadata:
            print(f"\n📊 DETAILED PROCESSING INFORMATION")
            print(f"=" * 50)
            processing_info = metadata["_processing_info"]
            print(f"🤖 OCR Engine: {processing_info.get('ocr_engine', 'N/A')}")
            print(f"🔧 Preprocessing Used: {'✅' if processing_info.get('preprocessing_used') else '❌'}")
            print(f"📄 OCR Images Processed: {processing_info.get('ocr_images_processed', 0)}")
            print(f"📸 Total Images: {processing_info.get('total_images', 0)}")
            # Heuristic metadata removed; keep summary concise
            if processing_info.get('fallback_used'):
                print(f"⚠️  Fallback Used: ✅")
                print(f"❌ Ollama Error: {processing_info.get('ollama_error', 'N/A')}")
            print(f"=" * 50)
        
        # Validate the metadata
        is_valid, validation_msg = validate_metadata(metadata)
        
        if not is_valid:
            print(f"Validation error: {validation_msg}")
            # Save the invalid metadata with an error flag
            metadata["_validation_error"] = validation_msg
        else:
            print("Enhanced metadata validation successful!")
        
        # Save the metadata
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced metadata saved to {output_file}")
        
        # Print a comprehensive summary of the extracted metadata
        print(f"\n📋 ENHANCED METADATA EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"📖 Title: {metadata.get('title') or '❌ Not found'}")
        if metadata.get('subtitle'):
            print(f"📖 Subtitle: {metadata.get('subtitle')}")
        print(f"✍️  Author(s): {', '.join(metadata.get('authors', [])) or '❌ Not found'}")
        print(f"🏢 Publisher: {metadata.get('publisher') or '❌ Not found'}")
        print(f"📅 Publication Date: {metadata.get('publication_date') or '❌ Not found'}")
        print(f"📚 ISBN-10: {metadata.get('isbn_10') or '❌ Not found'}")
        print(f"📚 ISBN-13: {metadata.get('isbn_13') or '❌ Not found'}")
        if metadata.get('edition'):
            print(f"📖 Edition: {metadata.get('edition')}")
        if metadata.get('binding_type'):
            print(f"📘 Binding: {metadata.get('binding_type')}")
        if metadata.get('language'):
            print(f"🌐 Language: {metadata.get('language')}")
        if metadata.get('page_count'):
            print(f"📄 Page Count: {metadata.get('page_count')}")
        if metadata.get('categories'):
            print(f"🏷️  Categories: {', '.join(metadata.get('categories', []))}")
        if metadata.get('condition_keywords'):
            print(f"🔍 Condition Keywords: {', '.join(metadata.get('condition_keywords', []))}")
        
        price = metadata.get('price', {})
        if price.get('amount'):
            print(f"💰 Price: {price.get('currency', '')} {price.get('amount', '')}")
        
        # Calculate and display processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\n⏱️  PROCESSING COMPLETED")
        print(f"   Total time: {processing_time:.2f} seconds")
        print(f"   Status: {'✅ Success' if is_valid else '⚠️  Success with validation warnings'}")
        print("=" * 60)
        
        return is_valid
        
    except Exception as e:
        print(f"Error processing book {book_id}: {e}")
        
        # Calculate and display processing time even if there was an error
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nTotal processing time: {processing_time:.2f} seconds")
        
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process a book using enhanced OCR + Ollama pipeline")
    parser.add_argument("book_id", type=str, help="ID of the book to process (directory name in books/)")
    parser.add_argument("--output-dir", "-o", type=str, default="output", 
                        help="Directory to save output JSON (default: output)")
    parser.add_argument("--model", "-m", type=str, default="gemma3:4b", 
                        help="Ollama model to use (default: gemma3:4b)")
    parser.add_argument("--ocr-engine", type=str, choices=["easyocr", "tesseract"], default="easyocr",
                        help="OCR engine to use (default: easyocr)")
    parser.add_argument("--no-preprocessing", action="store_true", 
                        help="Disable image preprocessing for OCR")
    parser.add_argument("--ocr-indices", type=int, nargs="+",
                        help="Indices of images to run OCR on (0-based, default: 1 2)")
    parser.add_argument("--no-raw", action="store_true", 
                        help="Don't show processing information")
    parser.add_argument("--crop-ocr", action="store_true", 
                        help="Auto-crop text regions before OCR")
    parser.add_argument("--crop-margin", type=int, default=16, 
                        help="Margin pixels around detected text when cropping (default: 16)")
    parser.add_argument("--no-warm-model", action="store_true", 
                        help="Disable model warm-up on startup")
    parser.add_argument("--books-dir", type=str,
                        help="Path to books directory (auto-detected if not specified)")
    parser.add_argument("--edge-crop", type=float, default=0.0,
                        help="Centered edge crop percent [0-45] applied before OCR")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.model == "list":
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                print("\nAvailable Ollama models:")
                for model in models:
                    print(f"- {model.get('name')}")
                print("\nUse one of these model names with the --model option")
                sys.exit(0)
            else:
                print("Could not connect to Ollama server")
                sys.exit(1)
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            sys.exit(1)
    
    success = process_book_enhanced(
        args.book_id, 
        args.output_dir, 
        args.model, 
        args.ocr_engine,
        not args.no_preprocessing,
        args.ocr_indices,
        not args.no_raw,
        args.books_dir,
        args.crop_ocr,
        args.crop_margin,
        args.no_warm_model,
        None,
        args.edge_crop
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
