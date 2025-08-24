#!/usr/bin/env python3
"""
Example script to process a single book directory.
This script demonstrates how to use the extractor with a specific book directory.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractor import BookMetadataExtractor


def main():
    """Process a single book directory."""
    parser = argparse.ArgumentParser(description="Process a single book directory")
    parser.add_argument("--book-dir", type=str, required=True,
                        help="Path to the book directory")
    parser.add_argument("--output-file", type=str, default="book_metadata.json",
                        help="Path to save JSON results (default: book_metadata.json)")
    parser.add_argument("--model", type=str, default="gemma3:4b",
                        help="Ollama model to use (default: gemma3:4b)")
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = BookMetadataExtractor(model=args.model)
    
    try:
        print(f"Processing book from: {args.book_dir}")
        print(f"Using model: {args.model}")
        print(f"Saving results to: {args.output_file}")
        print()
        
        # Process the book
        metadata = extractor.process_book_directory(args.book_dir)
        
        # Save results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\nBook processed successfully.")
        print(f"Results saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
