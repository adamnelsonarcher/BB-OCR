#!/usr/bin/env python3
"""
Example script to process the books directory.
This script demonstrates how to use the batch processor with the specific books directory structure.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_processor import BatchBookProcessor


def main():
    """Process the books directory."""
    parser = argparse.ArgumentParser(description="Process the books directory")
    parser.add_argument("--books-dir", type=str, default="../books", 
                        help="Path to the books directory (default: ../books)")
    parser.add_argument("--output-dir", type=str, default="output", 
                        help="Directory to save individual JSON files (default: output)")
    parser.add_argument("--output-file", type=str, default="output/all_books.json", 
                        help="Path to save combined JSON results (default: output/all_books.json)")
    parser.add_argument("--model", type=str, default="gemma3:4b", 
                        help="Ollama model to use (default: gemma3:4b)")
    parser.add_argument("--workers", type=int, default=1, 
                        help="Number of worker threads (default: 1)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize the batch processor
    processor = BatchBookProcessor(
        model=args.model,
        max_workers=args.workers
    )
    
    try:
        print(f"Processing books from: {args.books_dir}")
        print(f"Using model: {args.model}")
        print(f"Using {args.workers} worker thread(s)")
        print(f"Saving individual results to: {args.output_dir}")
        print(f"Saving combined results to: {args.output_file}")
        print()
        
        # Process all books
        results = processor.process_books_directory(args.books_dir, args.output_dir)
        
        # Save combined results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessed {len(results)} books.")
        print(f"Combined results saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
