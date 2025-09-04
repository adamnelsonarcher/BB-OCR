#!/usr/bin/env python3
"""
Quick test script for the enhanced pipeline.
Run this to test the enhanced book processing with detailed logging.
"""

import os
import sys
import subprocess

def main():
    """Run a test of the enhanced pipeline."""
    print("🧪 ENHANCED PIPELINE TEST")
    print("=" * 50)
    
    # Check if we have access to the parent books directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    books_dir = os.path.join(parent_dir, "ollama_to_JSON", "books")
    
    if not os.path.exists(books_dir):
        print(f"❌ Books directory not found at: {books_dir}")
        print("Please ensure the ollama_to_JSON/books directory exists with sample data.")
        return False
    
    # Find available books
    book_ids = []
    for item in os.listdir(books_dir):
        book_path = os.path.join(books_dir, item)
        if os.path.isdir(book_path):
            book_ids.append(item)
    
    if not book_ids:
        print(f"❌ No book directories found in {books_dir}")
        return False
    
    # Sort book IDs and pick the first one
    book_ids.sort()
    test_book_id = book_ids[0]
    
    print(f"📚 Available books: {', '.join(book_ids)}")
    print(f"🎯 Testing with book ID: {test_book_id}")
    print(f"📁 Books directory: {books_dir}")
    
    # Prepare the command
    cmd = [
        "python", "process_book_enhanced.py",
        test_book_id,
        "--books-dir", books_dir,
        "--output-dir", "test_output",
        "--ocr-engine", "easyocr",  # Default to easyocr
        # "--no-preprocessing"  # Uncomment to disable preprocessing
    ]
    
    print(f"\n🚀 Running command:")
    print(f"   {' '.join(cmd)}")
    print("\n" + "=" * 80)
    
    try:
        # Run the enhanced processor
        result = subprocess.run(cmd, cwd=current_dir, capture_output=False, text=True)
        
        print("\n" + "=" * 80)
        if result.returncode == 0:
            print(f"✅ TEST COMPLETED SUCCESSFULLY!")
            
            # Check if output file was created
            output_file = os.path.join(current_dir, "test_output", f"book_{test_book_id}_enhanced.json")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"📄 Output file created: {output_file} ({file_size} bytes)")
            else:
                print(f"⚠️  Output file not found: {output_file}")
        else:
            print(f"❌ TEST FAILED with exit code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print(f"❌ Could not find python or the script file")
        print(f"Make sure you're in the correct directory and have Python installed")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
