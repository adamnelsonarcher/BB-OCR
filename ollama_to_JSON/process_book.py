#!/usr/bin/env python3
"""
Simple Book Processor

A straightforward script to process a book and validate the extracted metadata.
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

# Import the extractor module
from extractor import BookMetadataExtractor

# Define the JSON schema for validation
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

def process_book(book_id, output_dir="output", model="gemma3:4b", show_raw=True):
    """Process a single book by ID."""
    # Determine the book directory path
    books_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books")
    book_dir = os.path.join(books_dir, str(book_id))
    
    if not os.path.isdir(book_dir):
        print(f"Error: Book directory {book_dir} not found")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"book_{book_id}.json")
    
    print(f"Processing book {book_id}...")
    print(f"Book directory: {book_dir}")
    print(f"Using model: {model}")
    
    # Track processing time
    start_time = time.time()
    
    try:
        # Initialize the extractor
        extractor = BookMetadataExtractor(model=model)
        
        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        
        for file in os.listdir(book_dir):
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_paths.append(os.path.join(book_dir, file))
        
        if not image_paths:
            print(f"Error: No image files found in {book_dir}")
            return False
        
        print(f"Found {len(image_paths)} images to process")
        
        # Encode all images
        images = [extractor.encode_image(img_path) for img_path in image_paths]
        
        # Create the request payload
        payload = {
            "model": model,
            "prompt": extractor.prompt,
            "stream": False,
            "images": images
        }
        
        print("\nSending request to Ollama...")
        
        # Send request to Ollama
        response = requests.post(extractor.ollama_url, json=payload)
        
        if response.status_code != 200:
            print(f"Error from Ollama API: {response.text}")
            return False
        
        # Extract the response
        result = response.json()
        response_text = result.get("response", "")
        
        # Display raw model output if requested
        if show_raw:
            print("\n--- Raw Model Output ---")
            print(response_text)
            print("------------------------\n")
        
        # Parse the JSON from the response
        try:
            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = response_text[json_start:json_end+1]
                metadata = json.loads(json_str)
            else:
                metadata = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response: {e}")
            return False
        
        # Validate the metadata
        is_valid, validation_msg = validate_metadata(metadata)
        
        if not is_valid:
            print(f"Validation error: {validation_msg}")
            # Save the invalid metadata with an error flag
            metadata["_validation_error"] = validation_msg
        else:
            print("Metadata validation successful!")
        
        # Save the metadata
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadata saved to {output_file}")
        
        # Print a summary of the extracted metadata
        print("\nExtracted Metadata Summary:")
        print(f"Title: {metadata.get('title')}")
        print(f"Author(s): {', '.join(metadata.get('authors', []))}")
        print(f"Publisher: {metadata.get('publisher')}")
        print(f"ISBN-10: {metadata.get('isbn_10')}")
        print(f"ISBN-13: {metadata.get('isbn_13')}")
        print(f"Publication Date: {metadata.get('publication_date')}")
        print(f"Binding: {metadata.get('binding_type')}")
        
        # Calculate and display processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nTotal processing time: {processing_time:.2f} seconds")
        
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
    parser = argparse.ArgumentParser(description="Process a book and validate metadata")
    parser.add_argument("book_id", type=str, help="ID of the book to process (directory name in books/)")
    parser.add_argument("--output-dir", "-o", type=str, default="output", 
                        help="Directory to save output JSON (default: output)")
    parser.add_argument("--model", "-m", type=str, default="gemma3:4b", 
                        help="Ollama model to use (default: gemma3:4b)")
    parser.add_argument("--no-raw", action="store_true", 
                        help="Don't show raw model output")
    
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
    
    success = process_book(args.book_id, args.output_dir, args.model, not args.no_raw)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
