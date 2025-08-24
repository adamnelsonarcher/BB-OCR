#!/usr/bin/env python3
"""
Simple Book Processor

A straightforward script to process a book and validate the extracted metadata.
"""

import os
import sys
import json
import argparse
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

def process_book(book_id, output_dir="output", model="gemma3:4b"):
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
    
    try:
        # Initialize the extractor
        extractor = BookMetadataExtractor(model=model)
        
        # Process the book
        metadata = extractor.process_book_directory(book_dir)
        
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
        
        return is_valid
        
    except Exception as e:
        print(f"Error processing book {book_id}: {e}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process a book and validate metadata")
    parser.add_argument("book_id", type=str, help="ID of the book to process (directory name in books/)")
    parser.add_argument("--output-dir", "-o", type=str, default="output", 
                        help="Directory to save output JSON (default: output)")
    parser.add_argument("--model", "-m", type=str, default="gemma3:4b", 
                        help="Ollama model to use (default: gemma3:4b)")
    
    args = parser.parse_args()
    
    success = process_book(args.book_id, args.output_dir, args.model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
