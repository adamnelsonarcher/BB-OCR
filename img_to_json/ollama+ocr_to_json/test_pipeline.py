#!/usr/bin/env python3
"""
Test script for the enhanced book metadata extraction pipeline.

This script runs basic tests to ensure the pipeline is working correctly.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from enhanced_extractor import EnhancedBookMetadataExtractor
        print("✓ Enhanced extractor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced extractor: {e}")
        return False
    
    try:
        from process_book_enhanced import process_book_enhanced, validate_metadata
        print("✓ Process book enhanced imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import process book enhanced: {e}")
        return False
    
    try:
        from batch_processor_enhanced import EnhancedBatchProcessor
        print("✓ Batch processor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import batch processor: {e}")
        return False
    
    return True

def test_extractor_initialization():
    """Test that the extractor can be initialized with different configurations."""
    print("\nTesting extractor initialization...")
    
    try:
        from enhanced_extractor import EnhancedBookMetadataExtractor
        
        # Test with default settings
        extractor1 = EnhancedBookMetadataExtractor()
        print("✓ Default extractor initialized")
        
        # Test with custom settings
        extractor2 = EnhancedBookMetadataExtractor(
            model="gemma3:4b",
            ocr_engine="easyocr",
            use_preprocessing=False
        )
        print("✓ Custom extractor initialized")
        
        # Test with tesseract
        extractor3 = EnhancedBookMetadataExtractor(ocr_engine="tesseract")
        print("✓ Tesseract extractor initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Extractor initialization failed: {e}")
        return False

def test_validation_schema():
    """Test the metadata validation function."""
    print("\nTesting metadata validation...")
    
    try:
        from process_book_enhanced import validate_metadata
        
        # Test valid metadata
        valid_metadata = {
            "title": "Test Book",
            "subtitle": None,
            "authors": ["Test Author"],
            "publisher": "Test Publisher",
            "publication_date": "2023",
            "isbn_10": None,
            "isbn_13": None,
            "asin": None,
            "edition": None,
            "binding_type": None,
            "language": None,
            "page_count": None,
            "categories": [],
            "description": None,
            "condition_keywords": [],
            "price": {"currency": None, "amount": None},
            "evidence": {
                "title_snippet": None,
                "publisher_snippet": None,
                "publication_year_snippet": None,
                "isbn_snippet": None,
                "notes": None
            }
        }
        
        is_valid, msg = validate_metadata(valid_metadata)
        if is_valid:
            print("✓ Valid metadata passed validation")
        else:
            print(f"✗ Valid metadata failed validation: {msg}")
            return False
        
        # Test invalid metadata (missing title)
        invalid_metadata = valid_metadata.copy()
        invalid_metadata["title"] = None
        
        is_valid, msg = validate_metadata(invalid_metadata)
        if not is_valid:
            print("✓ Invalid metadata correctly failed validation")
        else:
            print("✗ Invalid metadata incorrectly passed validation")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Validation testing failed: {e}")
        return False

def test_prompt_loading():
    """Test that the enhanced prompt can be loaded."""
    print("\nTesting prompt loading...")
    
    try:
        prompt_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "prompts", 
            "enhanced_book_metadata_prompt.txt"
        )
        
        if not os.path.exists(prompt_file):
            print(f"✗ Prompt file not found: {prompt_file}")
            return False
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        
        if len(prompt_content) < 100:
            print("✗ Prompt file seems too short")
            return False
        
        if "OCR CONTEXT" not in prompt_content:
            print("✗ Prompt doesn't contain OCR context instructions")
            return False
        
        print("✓ Enhanced prompt loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Prompt loading failed: {e}")
        return False

def test_directory_structure():
    """Test that the directory structure is correct."""
    print("\nTesting directory structure...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        "enhanced_extractor.py",
        "process_book_enhanced.py",
        "batch_processor_enhanced.py",
        "requirements.txt",
        "README.md",
        "prompts/enhanced_book_metadata_prompt.txt"
    ]
    
    all_found = True
    for file_path in required_files:
        full_path = os.path.join(current_dir, file_path)
        if os.path.exists(full_path):
            print(f"✓ Found {file_path}")
        else:
            print(f"✗ Missing {file_path}")
            all_found = False
    
    return all_found

def test_ocr_engines():
    """Test that OCR engines are available."""
    print("\nTesting OCR engines...")
    
    # Test EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        print("✓ EasyOCR is available")
        easyocr_available = True
    except Exception as e:
        print(f"✗ EasyOCR not available: {e}")
        easyocr_available = False
    
    # Test Tesseract
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 255)
        _ = pytesseract.image_to_string(test_image)
        print("✓ Tesseract is available")
        tesseract_available = True
    except Exception as e:
        print(f"✗ Tesseract not available: {e}")
        tesseract_available = False
    
    return easyocr_available or tesseract_available

def run_all_tests():
    """Run all tests and return overall success status."""
    print("=" * 60)
    print("ENHANCED PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Extractor Initialization", test_extractor_initialization),
        ("Validation Schema", test_validation_schema),
        ("Prompt Loading", test_prompt_loading),
        ("Directory Structure", test_directory_structure),
        ("OCR Engines", test_ocr_engines)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced pipeline is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
