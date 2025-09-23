#!/usr/bin/env python3
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

from process_book_enhanced import process_book_enhanced, validate_metadata
from enhanced_extractor import EnhancedBookMetadataExtractor

class EnhancedBatchProcessor:
    def __init__(self, books_dir=None, output_dir="batch_output", model="gemma3:4b", 
                 ocr_engine="easyocr", use_preprocessing=True, ocr_indices=None, 
                 max_workers=2, show_progress=True, crop_ocr=False, crop_margin=16, no_warm_model=False,
                 edge_crop=0.0):
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
        self.stats = {
            'total_books': 0,
            'successful': 0,
            'failed': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'start_time': None,
            'end_time': None
        }
        self.stats_lock = threading.Lock()
        os.makedirs(self.output_dir, exist_ok=True)
        if self.books_dir is None:
            self.books_dir = self._find_books_directory()

    def _find_books_directory(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        possible_books_dirs = [
            os.path.join(current_dir, "books"),
            os.path.join(parent_dir, "books")
        ]
        for possible_dir in possible_books_dirs:
            if os.path.isdir(possible_dir):
                return possible_dir
        raise ValueError("Could not find books directory. Please specify with --books-dir")

    def _get_book_ids(self, book_ids=None):
        if book_ids:
            return book_ids
        ids = []
        for item in os.listdir(self.books_dir):
            p = os.path.join(self.books_dir, item)
            if os.path.isdir(p):
                ids.append(item)
        return sorted(ids)

    def _process_single_book(self, book_id):
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
                edge_crop=self.edge_crop
            )
            result['success'] = success
            result['output_file'] = os.path.join(self.output_dir, f"book_{book_id}_enhanced.json")
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
            if self.show_progress:
                book_ids = tqdm(book_ids, desc="Processing books")
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
                    book_ids.set_postfix({'Success': self.stats['successful'], 'Failed': self.stats['failed']})
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_book = { executor.submit(self._process_single_book, book_id): book_id for book_id in book_ids }
                futures = tqdm(as_completed(future_to_book), total=len(book_ids), desc="Processing books") if self.show_progress else as_completed(future_to_book)
                for future in futures:
                    book_id = future_to_book[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if self.show_progress and hasattr(futures, 'set_postfix'):
                            futures.set_postfix({'Success': self.stats['successful'], 'Failed': self.stats['failed']})
                    except Exception as e:
                        results.append({'book_id': book_id, 'success': False, 'validation_passed': False, 'processing_time': 0, 'error': f"Future exception: {str(e)}", 'output_file': None})
                        with self.stats_lock:
                            self.stats['failed'] += 1
        self.stats['end_time'] = time.time()
        self._generate_summary_report(results)
        return results

    def _generate_summary_report(self, results):
        total_time = self.stats['end_time'] - self.stats['start_time']
        summary = {
            'processing_summary': {
                'total_books': self.stats['total_books'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'validation_passed': self.stats['validation_passed'],
                'validation_failed': self.stats['validation_failed'],
            },
            'timing': {
                'total_time_seconds': total_time,
                'average_time_per_book': total_time / self.stats['total_books'] if self.stats['total_books'] > 0 else 0,
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
        summary_file = os.path.join(self.output_dir, "batch_processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("\n" + "=" * 60)
        print("ENHANCED BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total books processed: {self.stats['total_books']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Validation passed: {self.stats['validation_passed']}")
        print(f"Validation failed: {self.stats['validation_failed']}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch process books using enhanced OCR + Ollama pipeline")
    parser.add_argument("--books-dir", type=str, help="Path to books directory (auto-detected if not specified)")
    parser.add_argument("--output-dir", "-o", type=str, default="batch_output", help="Directory to save output JSON files (default: batch_output)")
    parser.add_argument("--model", "-m", type=str, default="gemma3:4b", help="Ollama model to use (default: gemma3:4b)")
    parser.add_argument("--ocr-engine", type=str, choices=["easyocr", "tesseract"], default="easyocr", help="OCR engine to use (default: easyocr)")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable image preprocessing for OCR")
    parser.add_argument("--ocr-indices", type=int, nargs="+", help="Indices of images to run OCR on (0-based, default: 1 2)")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of parallel workers (default: 2)")
    parser.add_argument("--book-ids", type=str, nargs="+", help="Specific book IDs to process (default: all books)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--crop-ocr", action="store_true", help="Auto-crop text regions before OCR")
    parser.add_argument("--crop-margin", type=int, default=16, help="Margin pixels around detected text when cropping (default: 16)")
    parser.add_argument("--no-warm-model", action="store_true", help="Disable model warm-up on startup")
    parser.add_argument("--edge-crop", type=float, default=0.0, help="Centered edge crop percent [0-45] applied before OCR")
    args = parser.parse_args()
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
    results = processor.process_books(args.book_ids)
    failed_count = len([r for r in results if not r['success']])
    sys.exit(1 if failed_count > 0 else 0)

if __name__ == "__main__":
    main()


