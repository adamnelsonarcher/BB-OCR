#!/usr/bin/env python3
"""
Book Metadata Extraction - Simple Command Line Interface

Usage:
  python book.py process 1             # Process book #1
  python book.py process 1,2,3         # Process books #1, #2, and #3
  python book.py process all           # Process all books
  python book.py process 1-5           # Process books #1 through #5
  python book.py process 1,3-5,7       # Process books #1, #3, #4, #5, and #7
  python book.py list                  # List all available books
  python book.py info 1                # Show info about book #1
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Set

# Add the current directory to the path so we can import the modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
try:
    from extractor import BookMetadataExtractor
    from batch_processor import BatchBookProcessor
except ImportError:
    # Try with relative imports
    sys.path.insert(0, os.path.dirname(script_dir))
    from ollama_to_JSON.extractor import BookMetadataExtractor
    from ollama_to_JSON.batch_processor import BatchBookProcessor


def get_books_dir() -> str:
    """Get the books directory path."""
    # First check if there's a books directory in the current directory
    if os.path.isdir("books"):
        return "books"
    
    # Then check if there's a books directory one level up
    if os.path.isdir("../books"):
        return "../books"
    
    # Check if there's a books directory in the ollama_to_JSON directory
    if os.path.isdir("ollama_to_JSON/books"):
        return "ollama_to_JSON/books"
    
    # Finally, check if the script is in the ollama_to_JSON directory and books is a sibling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    if os.path.isdir(os.path.join(parent_dir, "books")):
        return os.path.join(parent_dir, "books")
    
    # If we can't find it, use the default
    return "books"


def list_available_books(books_dir: str) -> List[str]:
    """List all available book directories."""
    if not os.path.isdir(books_dir):
        print(f"Error: Books directory not found at {books_dir}")
        return []
    
    books = []
    for item in os.listdir(books_dir):
        item_path = os.path.join(books_dir, item)
        if os.path.isdir(item_path):
            books.append(item)
    
    return sorted(books, key=lambda x: int(x) if x.isdigit() else float('inf'))


def parse_book_selection(selection: str, available_books: List[str]) -> Set[str]:
    """Parse the book selection string and return a set of book IDs."""
    if selection.lower() == "all":
        return set(available_books)
    
    selected_books = set()
    parts = selection.split(",")
    
    for part in parts:
        if "-" in part:
            try:
                start, end = part.split("-")
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                for i in range(start_idx, end_idx + 1):
                    if str(i) in available_books:
                        selected_books.add(str(i))
            except ValueError:
                print(f"Warning: Invalid range '{part}'. Skipping.")
        else:
            try:
                book_id = part.strip()
                if book_id in available_books:
                    selected_books.add(book_id)
                else:
                    print(f"Warning: Book '{book_id}' not found. Skipping.")
            except ValueError:
                print(f"Warning: Invalid book ID '{part}'. Skipping.")
    
    return selected_books


def process_books(book_ids: Set[str], books_dir: str, output_dir: str, model: str) -> None:
    """Process the selected books."""
    if not book_ids:
        print("No books selected for processing.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if len(book_ids) == 1:
        # Process a single book
        book_id = next(iter(book_ids))
        book_dir = os.path.join(books_dir, book_id)
        output_file = os.path.join(output_dir, f"book_{book_id}.json")
        
        print(f"Processing book #{book_id}...")
        extractor = BookMetadataExtractor(model=model)
        
        try:
            metadata = extractor.process_book_directory(book_dir)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"✓ Book #{book_id} processed successfully.")
            print(f"  Results saved to {output_file}")
        except Exception as e:
            print(f"✗ Error processing book #{book_id}: {e}")
    else:
        # Process multiple books
        processor = BatchBookProcessor(model=model)
        book_dirs = {os.path.join(books_dir, book_id): book_id for book_id in book_ids}
        
        print(f"Processing {len(book_ids)} books...")
        
        for book_dir, book_id in book_dirs.items():
            output_file = os.path.join(output_dir, f"book_{book_id}.json")
            print(f"Processing book #{book_id}...")
            
            try:
                metadata = processor.process_single_book(book_dir)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                print(f"✓ Book #{book_id} processed successfully.")
            except Exception as e:
                print(f"✗ Error processing book #{book_id}: {e}")
        
        print(f"All results saved to {output_dir}")


def show_book_info(book_id: str, books_dir: str) -> None:
    """Show information about a book."""
    book_dir = os.path.join(books_dir, book_id)
    
    if not os.path.isdir(book_dir):
        print(f"Error: Book #{book_id} not found.")
        return
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for file in os.listdir(book_dir):
        if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Book #{book_id}")
    print(f"Directory: {book_dir}")
    print(f"Images: {len(image_files)}")
    
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Book Metadata Extraction - Simple Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process books")
    process_parser.add_argument("selection", help="Book selection (e.g., '1', '1,2,3', '1-5', 'all')")
    process_parser.add_argument("--books-dir", help="Books directory path")
    process_parser.add_argument("--output-dir", default="output", help="Output directory path")
    process_parser.add_argument("--model", default="gemma3:4b", help="Ollama model to use")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available books")
    list_parser.add_argument("--books-dir", help="Books directory path")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show book info")
    info_parser.add_argument("book_id", help="Book ID")
    info_parser.add_argument("--books-dir", help="Books directory path")
    
    args = parser.parse_args()
    
    # Determine the books directory
    books_dir = args.books_dir if hasattr(args, "books_dir") and args.books_dir else get_books_dir()
    
    if not os.path.isdir(books_dir):
        print(f"Error: Books directory not found at {books_dir}")
        return
    
    # Get available books
    available_books = list_available_books(books_dir)
    
    if args.command == "list":
        print(f"Available books in {books_dir}:")
        for book in available_books:
            print(f"  #{book}")
    
    elif args.command == "info":
        show_book_info(args.book_id, books_dir)
    
    elif args.command == "process":
        selected_books = parse_book_selection(args.selection, available_books)
        process_books(selected_books, books_dir, args.output_dir, args.model)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
