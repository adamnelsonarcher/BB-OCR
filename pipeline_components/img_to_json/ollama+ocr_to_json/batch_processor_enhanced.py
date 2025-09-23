#!/usr/bin/env python3
"""
Enhanced Batch Book Processor

A comprehensive script to process multiple books using the enhanced OCR + Ollama pipeline
with parallel processing capabilities and detailed progress tracking.
"""

import os
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from tqdm import tqdm

# Import the enhanced processor
from process_book_enhanced import process_book_enhanced, validate_metadata
from enhanced_extractor import EnhancedBookMetadataExtractor

class EnhancedBatchProcessor:
    """Process multiple books using the enhanced OCR + Ollama pipeline."""
    
    def __init__(self, books_dir=None, output_dir="batch_output", model="gemma3:4b", 
                 ocr_engine="easyocr", use_preprocessing=True, ocr_indices=None, 
                 max_workers=2, show_progress=True, crop_ocr=False, crop_margin=16, no_warm_model=False,
                 edge_crop=0.0):
        """Initialize the batch processor."""
        self.books_dir = books_dir
        self.output_dir = output_dir
        self.model = model
        self.ocr_engine = ocr_engine
        self.use_preprocessing = use_preprocessing
        self.ocr_indices = ocr_indices
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.crop_ocr = crop_ocr
        self.crop_margin = crop_margin
        self.no_warm_model = no_warm_model
        self.edge_crop = float(edge_crop)
        
        # Statistics
        self.stats = {
            'total_books': 0,
            'successful': 0,
            'failed': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Thread-safe lock for statistics
        self.stats_lock = threading.Lock()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find books directory if not specified
        if self.books_dir is None:
            self.books_dir = self._find_books_directory()
    
    def _find_books_directory(self):
        """Find the books directory in common locations."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Check common locations
        possible_books_dirs = [
            os.path.join(current_dir, "books"),
            os.path.join(parent_dir, "ollama_to_JSON", "books"),
            os.path.join(parent_dir, "books")
        ]
        
        for possible_dir in possible_books_dirs:
            if os.path.isdir(possible_dir):
                return possible_dir
        
        raise ValueError("Could not find books directory. Please specify with --books-dir")
    
    def _get_book_ids(self, book_ids=None):
        """Get list of book IDs to process."""
        if book_ids:
            return book_ids
        
        # Get all subdirectories in books directory
        book_ids = []
        for item in os.listdir(self.books_dir):
            book_path = os.path.join(self.books_dir, item)
            if os.path.isdir(book_path):
                book_ids.append(item)
        
        return sorted(book_ids)
    
    def _process_single_book(self, book_id):
        """Process a single book and return results."""
        result = {
            'book_id': book_id,
            'success': False,
            'validation_passed': False,
            'processing_time': 0,
            'error': None,
            'output_file': None
        }
        
        start_time = time.time()
        
        try:
            # Process the book
            # Reuse a single extractor per worker when running sequentially to keep model/ocr loaded
            success = process_book_enhanced(
                book_id=book_id,
                output_dir=self.output_dir,
                model=self.model,
                ocr_engine=self.ocr_engine,
                use_preprocessing=self.use_preprocessing,
                ocr_indices=self.ocr_indices,
                show_raw=False,  # Disable raw output in batch mode
                books_dir=self.books_dir,
                crop_ocr=self.crop_ocr,
                crop_margin=self.crop_margin,
                no_warm_model=self.no_warm_model,
                edge_crop=self.edge_crop
            )
            
            result['success'] = success
            result['output_file'] = os.path.join(self.output_dir, f"book_{book_id}_enhanced.json")
            
            # Check if output file exists and validate
            if os.path.exists(result['output_file']):
                try:
                    with open(result['output_file'], 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    is_valid, _ = validate_metadata(metadata)
                    result['validation_passed'] = is_valid
                except Exception as e:
                    result['error'] = f"Failed to validate output: {str(e)}"
            else:
                result['error'] = "Output file not created"
                
        except Exception as e:
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        
        # Update statistics thread-safely
        with self.stats_lock:
            if result['success']:
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
            
            if result['validation_passed']:
                self.stats['validation_passed'] += 1
            else:
                self.stats['validation_failed'] += 1
        
        return result
    
    def process_books(self, book_ids=None):
        """Process multiple books with optional parallel processing."""
        book_ids = self._get_book_ids(book_ids)
        self.stats['total_books'] = len(book_ids)
        self.stats['start_time'] = time.time()
        
        if not book_ids:
            print("No books found to process.")
            return []
        
        print(f"Processing {len(book_ids)} books using enhanced pipeline...")
        print(f"Books directory: {self.books_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Model: {self.model}")
        print(f"OCR engine: {self.ocr_engine}")
        print(f"Preprocessing: {'enabled' if self.use_preprocessing else 'disabled'}")
        print(f"Max workers: {self.max_workers}")
        print(f"OCR crop: {'enabled' if self.crop_ocr else 'disabled'} (margin: {self.crop_margin})")
        print(f"Edge crop: {self.edge_crop}%")
        if self.ocr_indices:
            print(f"OCR indices: {self.ocr_indices}")
        print("-" * 60)
        
        results = []
        
        if self.max_workers == 1:
            # Sequential processing
            if self.show_progress:
                book_ids = tqdm(book_ids, desc="Processing books")
            
            # Reuse a single extractor to keep OCR/model instances hot
            reused_extractor = EnhancedBookMetadataExtractor(
                model=self.model,
                ocr_engine=self.ocr_engine,
                use_preprocessing=self.use_preprocessing,
                crop_for_ocr=self.crop_ocr,
                crop_margin=self.crop_margin,
                warm_model=not self.no_warm_model,
                edge_crop_percent=self.edge_crop
            )
            
            for book_id in book_ids:
                # Use the reused extractor
                single_start = time.time()
                try:
                    success = process_book_enhanced(
                        book_id=book_id,
                        output_dir=self.output_dir,
                        model=self.model,
                        ocr_engine=self.ocr_engine,
                        use_preprocessing=self.use_preprocessing,
                        ocr_indices=self.ocr_indices,
                        show_raw=False,
                        books_dir=self.books_dir,
                        crop_ocr=self.crop_ocr,
                        crop_margin=self.crop_margin,
                        no_warm_model=self.no_warm_model,
                        extractor=reused_extractor,
                        edge_crop=self.edge_crop
                    )
                    output_file = os.path.join(self.output_dir, f"book_{book_id}_enhanced.json")
                    validation_passed = False
                    error_msg = None
                    if os.path.exists(output_file):
                        try:
                            with open(output_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            is_valid, _ = validate_metadata(metadata)
                            validation_passed = is_valid
                        except Exception as ve:
                            error_msg = f"Failed to validate output: {str(ve)}"
                    else:
                        error_msg = "Output file not created"
                    result = {
                        'book_id': book_id,
                        'success': success,
                        'validation_passed': validation_passed,
                        'processing_time': time.time() - single_start,
                        'error': error_msg,
                        'output_file': output_file if os.path.exists(output_file) else None
                    }
                    with self.stats_lock:
                        if result['success']:
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                        if result['validation_passed']:
                            self.stats['validation_passed'] += 1
                        else:
                            self.stats['validation_failed'] += 1
                except Exception as e:
                    result = {
                        'book_id': book_id,
                        'success': False,
                        'validation_passed': False,
                        'processing_time': time.time() - single_start,
                        'error': str(e),
                        'output_file': None
                    }
                    with self.stats_lock:
                        self.stats['failed'] += 1
                results.append(result)
                
                if self.show_progress and hasattr(book_ids, 'set_postfix'):
                    book_ids.set_postfix({
                        'Success': self.stats['successful'],
                        'Failed': self.stats['failed']
                    })
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_book = {
                    executor.submit(self._process_single_book, book_id): book_id 
                    for book_id in book_ids
                }
                
                # Process completed tasks
                if self.show_progress:
                    futures = tqdm(as_completed(future_to_book), total=len(book_ids), desc="Processing books")
                else:
                    futures = as_completed(future_to_book)
                
                for future in futures:
                    book_id = future_to_book[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if self.show_progress and hasattr(futures, 'set_postfix'):
                            futures.set_postfix({
                                'Success': self.stats['successful'],
                                'Failed': self.stats['failed']
                            })
                    except Exception as e:
                        results.append({
                            'book_id': book_id,
                            'success': False,
                            'validation_passed': False,
                            'processing_time': 0,
                            'error': f"Future exception: {str(e)}",
                            'output_file': None
                        })
                        
                        with self.stats_lock:
                            self.stats['failed'] += 1
        
        self.stats['end_time'] = time.time()
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate and save a comprehensive summary report."""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        # Create summary data
        summary = {
            'processing_summary': {
                'total_books': self.stats['total_books'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'validation_passed': self.stats['validation_passed'],
                'validation_failed': self.stats['validation_failed'],
                'success_rate': (self.stats['successful'] / self.stats['total_books']) * 100 if self.stats['total_books'] > 0 else 0,
                'validation_rate': (self.stats['validation_passed'] / self.stats['total_books']) * 100 if self.stats['total_books'] > 0 else 0
            },
            'timing': {
                'total_time_seconds': total_time,
                'average_time_per_book': total_time / self.stats['total_books'] if self.stats['total_books'] > 0 else 0,
                'books_per_hour': (self.stats['total_books'] / total_time) * 3600 if total_time > 0 else 0
            },
            'configuration': {
                'model': self.model,
                'ocr_engine': self.ocr_engine,
                'preprocessing_enabled': self.use_preprocessing,
                'ocr_indices': self.ocr_indices,
                'max_workers': self.max_workers,
                'books_directory': self.books_dir,
                'output_directory': self.output_dir
            },
            'detailed_results': results
        }
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, "batch_processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("ENHANCED BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total books processed: {self.stats['total_books']}")
        print(f"Successful: {self.stats['successful']} ({summary['processing_summary']['success_rate']:.1f}%)")
        print(f"Failed: {self.stats['failed']}")
        print(f"Validation passed: {self.stats['validation_passed']} ({summary['processing_summary']['validation_rate']:.1f}%)")
        print(f"Validation failed: {self.stats['validation_failed']}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per book: {summary['timing']['average_time_per_book']:.1f} seconds")
        print(f"Processing rate: {summary['timing']['books_per_hour']:.1f} books/hour")
        print(f"Summary saved to: {summary_file}")
        
        # Show failed books if any
        failed_books = [r for r in results if not r['success']]
        if failed_books:
            print(f"\nFailed books ({len(failed_books)}):")
            for result in failed_books:
                print(f"  - {result['book_id']}: {result['error']}")
        
        print("=" * 60)

def main():
    """Main entry point for the batch processor."""
    parser = argparse.ArgumentParser(description="Batch process books using enhanced OCR + Ollama pipeline")
    parser.add_argument("--books-dir", type=str,
                        help="Path to books directory (auto-detected if not specified)")
    parser.add_argument("--output-dir", "-o", type=str, default="batch_output", 
                        help="Directory to save output JSON files (default: batch_output)")
    parser.add_argument("--model", "-m", type=str, default="gemma3:4b", 
                        help="Ollama model to use (default: gemma3:4b)")
    parser.add_argument("--ocr-engine", type=str, choices=["easyocr", "tesseract"], default="easyocr",
                        help="OCR engine to use (default: easyocr)")
    parser.add_argument("--no-preprocessing", action="store_true", 
                        help="Disable image preprocessing for OCR")
    parser.add_argument("--ocr-indices", type=int, nargs="+",
                        help="Indices of images to run OCR on (0-based, default: 1 2)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Maximum number of parallel workers (default: 2)")
    parser.add_argument("--book-ids", type=str, nargs="+",
                        help="Specific book IDs to process (default: all books)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bar")
    parser.add_argument("--crop-ocr", action="store_true",
                        help="Auto-crop text regions before OCR")
    parser.add_argument("--crop-margin", type=int, default=16,
                        help="Margin pixels around detected text when cropping (default: 16)")
    parser.add_argument("--no-warm-model", action="store_true",
                        help="Disable model warm-up on startup")
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
    
    try:
        # Initialize the batch processor
        processor = EnhancedBatchProcessor(
            books_dir=args.books_dir,
            output_dir=args.output_dir,
            model=args.model,
            ocr_engine=args.ocr_engine,
            use_preprocessing=not args.no_preprocessing,
            ocr_indices=args.ocr_indices,
            max_workers=args.max_workers,
            show_progress=not args.no_progress,
            crop_ocr=args.crop_ocr,
            crop_margin=args.crop_margin,
            no_warm_model=args.no_warm_model,
            edge_crop=args.edge_crop
        )
        
        # Process the books
        results = processor.process_books(args.book_ids)
        
        # Exit with error code if any books failed
        failed_count = len([r for r in results if not r['success']])
        sys.exit(1 if failed_count > 0 else 0)
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
