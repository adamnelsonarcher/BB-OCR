#!/usr/bin/env python3
"""
Batch Book Metadata Processor

This script processes multiple book directories and extracts metadata for each book.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from extractor import BookMetadataExtractor


class BatchBookProcessor:
    """Process multiple book directories and extract metadata."""
    
    def __init__(self, model: str = "gemma3:4b", prompt_file: str = None, max_workers: int = 1):
        """Initialize the batch processor."""
        self.extractor = BookMetadataExtractor(model=model, prompt_file=prompt_file)
        self.max_workers = max_workers
    
    def process_books_directory(self, books_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """Process all book directories within the main books directory."""
        # Get all subdirectories in the books directory
        book_dirs = []
        for item in os.listdir(books_dir):
            item_path = os.path.join(books_dir, item)
            if os.path.isdir(item_path):
                book_dirs.append(item_path)
        
        if not book_dirs:
            raise Exception(f"No book directories found in {books_dir}")
        
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process each book directory
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing if max_workers > 1
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_book = {
                    executor.submit(self.process_single_book, book_dir): book_dir 
                    for book_dir in book_dirs
                }
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_book), total=len(book_dirs), desc="Processing books"):
                    book_dir = future_to_book[future]
                    book_name = os.path.basename(book_dir)
                    try:
                        metadata = future.result()
                        results[book_name] = metadata
                        
                        # Save individual result if output_dir is specified
                        if output_dir:
                            output_file = os.path.join(output_dir, f"{book_name}.json")
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"Error processing {book_name}: {e}", file=sys.stderr)
                        results[book_name] = {"error": str(e)}
        else:
            # Process sequentially
            for book_dir in tqdm(book_dirs, desc="Processing books"):
                book_name = os.path.basename(book_dir)
                try:
                    metadata = self.process_single_book(book_dir)
                    results[book_name] = metadata
                    
                    # Save individual result if output_dir is specified
                    if output_dir:
                        output_file = os.path.join(output_dir, f"{book_name}.json")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error processing {book_name}: {e}", file=sys.stderr)
                    results[book_name] = {"error": str(e)}
        
        return results
    
    def process_single_book(self, book_dir: str) -> Dict[str, Any]:
        """Process a single book directory."""
        return self.extractor.process_book_directory(book_dir)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Batch process book directories for metadata extraction")
    parser.add_argument("books_dir", type=str, help="Directory containing book subdirectories")
    parser.add_argument("--output-dir", type=str, help="Directory to save individual JSON files")
    parser.add_argument("--output-file", type=str, help="Path to save combined JSON results")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model to use")
    parser.add_argument("--prompt-file", type=str, help="Custom prompt file path")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads (default: 1)")
    
    args = parser.parse_args()
    
    # Initialize the batch processor
    processor = BatchBookProcessor(
        model=args.model, 
        prompt_file=args.prompt_file,
        max_workers=args.workers
    )
    
    try:
        # Process all books
        results = processor.process_books_directory(args.books_dir, args.output_dir)
        
        # Save combined results if requested
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Combined results saved to {args.output_file}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
