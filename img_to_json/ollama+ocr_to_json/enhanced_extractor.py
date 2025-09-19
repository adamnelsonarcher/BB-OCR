#!/usr/bin/env python3
"""
Enhanced Book Metadata Extractor

This script combines OCR preprocessing with Ollama vision-language models
to extract more accurate structured book metadata from images.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import requests
from PIL import Image
import jsonschema
import easyocr
import pytesseract
import cv2
import numpy as np

# Add the parent directories to the path to import from other modules
parent_dir = os.path.dirname(os.path.abspath(__file__))
ocr_testing_dir = os.path.join(os.path.dirname(parent_dir), "ocr_testing")
sys.path.append(ocr_testing_dir)

# Import preprocessing and heuristic extraction modules
try:
    from preprocessing.image_preprocessor import preprocess_for_book_cover
    from hueristics.book_extractor import extract_book_metadata_from_text
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import OCR testing modules: {e}")
    print("Make sure the ocr_testing directory is available in the parent directory.")
    PREPROCESSING_AVAILABLE = False
    
    # Define fallback functions
    def preprocess_for_book_cover(image_path, output_path=None):
        """Fallback function when preprocessing is not available."""
        return image_path, output_path, ["original"]
    
    def extract_book_metadata_from_text(text):
        """Fallback function when heuristic extraction is not available."""
        return {}

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

class EnhancedBookMetadataExtractor:
    """Extract book metadata from images using OCR + Ollama vision-language model."""
    
    # Cache EasyOCR readers per language configuration to avoid reloading costs
    _easyocr_reader_cache: Dict[str, easyocr.Reader] = {}

    def __init__(self, model: str = "gemma3:4b", prompt_file: str = None, ocr_engine: str = "easyocr", use_preprocessing: bool = True,
                 crop_for_ocr: bool = False, crop_margin: int = 16, warm_model: bool = True):
        """Initialize the extractor with the specified model, OCR engine, and preprocessing options.

        Args:
            model: Ollama model name
            prompt_file: Path to enhanced prompt template
            ocr_engine: "easyocr" or "tesseract"
            use_preprocessing: Apply image preprocessing before OCR
            crop_for_ocr: Auto-crop text regions before OCR to reduce noise
            crop_margin: Margin (in pixels) to add around detected text region when cropping
            warm_model: Send a tiny request on init to keep/loading the model
        """
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ocr_engine = ocr_engine.lower()
        self.use_preprocessing = use_preprocessing
        self.crop_for_ocr = crop_for_ocr
        self.crop_margin = int(max(0, crop_margin))

        # Reuse HTTP connections
        self.session = requests.Session()
        
        # Initialize OCR engines
        if self.ocr_engine == "easyocr":
            lang_key = "en"  # extendable in future
            if lang_key not in EnhancedBookMetadataExtractor._easyocr_reader_cache:
                EnhancedBookMetadataExtractor._easyocr_reader_cache[lang_key] = easyocr.Reader(['en'])
            self.easyocr_reader = EnhancedBookMetadataExtractor._easyocr_reader_cache[lang_key]
        
        # Load the enhanced prompt from file
        if prompt_file is None:
            prompt_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "prompts", 
                "enhanced_book_metadata_prompt.txt"
            )
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        # Optionally warm the model so first inference is faster
        if warm_model:
            try:
                self._warm_ollama_model()
            except Exception as e:
                print(f"Warning: model warm-up skipped due to error: {e}")

    def _warm_ollama_model(self) -> None:
        """Send a tiny request to prompt the Ollama server to load the model."""
        payload = {
            "model": self.model,
            "prompt": "ping",
            "stream": False
        }
        try:
            resp = self.session.post(self.ollama_url, json=payload, timeout=10)
            # Ignore content; just ensure request completes
            if resp.status_code != 200:
                raise RuntimeError(f"Warm-up status {resp.status_code}")
            print("🔥 Model warm-up request sent")
        except Exception as e:
            # Surface as warning to not block processing
            raise e

    def _auto_crop_text_region(self, image_path: str, margin: int) -> Optional[str]:
        """Detect and crop the dominant text region. Returns new image path or None if no crop.

        Heuristic: threshold to text mask, close gaps, find largest contour, crop with margin.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # Binary inverse so text is white
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)
        # Morphologically close to connect letters into lines/blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,3))
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Remove small noise
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2, iterations=1)
        # Find contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Choose largest reasonable contour by area
        img_area = float(h * w)
        best = None
        best_area = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            # Filter improbable regions
            if area < 0.01 * img_area:
                continue
            aspect = cw / float(ch + 1e-6)
            if 0.2 <= aspect <= 10.0:  # exclude extremely thin shapes
                if area > best_area:
                    best_area = area
                    best = (x, y, cw, ch)
        if best is None:
            return None
        x, y, cw, ch = best
        # Expand with margin but keep within image
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w, x + cw + margin)
        y1 = min(h, y + ch + margin)
        # If crop is almost full image, skip to avoid extra IO
        crop_area = float((x1 - x0) * (y1 - y0))
        if crop_area > 0.9 * img_area:
            return None
        cropped = img[y0:y1, x0:x1]
        # Write to temp alongside source
        temp_dir = os.path.join(os.path.dirname(image_path), "temp_preprocessed")
        os.makedirs(temp_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(temp_dir, f"{base}_cropped.png")
        cv2.imwrite(out_path, cropped)
        return out_path
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def extract_text_with_ocr(self, image_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from an image using OCR and return both text and heuristic metadata."""
        print(f"    🔍 Starting OCR processing for: {os.path.basename(image_path)}")
        
        # Apply preprocessing if enabled
        preprocessed_image_path = image_path
        temp_files_to_cleanup: List[str] = []
        if self.use_preprocessing and PREPROCESSING_AVAILABLE:
            print(f"    📝 Applying image preprocessing...")
            # Create temporary preprocessed image
            temp_dir = os.path.join(os.path.dirname(image_path), "temp_preprocessed")
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            preprocessed_image_path = os.path.join(temp_dir, f"{base_name}_preprocessed.png")
            
            try:
                _, preprocessed_image_path, steps = preprocess_for_book_cover(image_path, preprocessed_image_path)
                print(f"    ✓ Preprocessing completed. Steps applied: {', '.join(steps)}")
                temp_files_to_cleanup.append(preprocessed_image_path)
            except Exception as e:
                print(f"    ⚠️  Preprocessing failed for {image_path}: {e}")
                preprocessed_image_path = image_path
        elif self.use_preprocessing and not PREPROCESSING_AVAILABLE:
            print(f"    ⚠️  Preprocessing requested but not available for {image_path}")
        else:
            print(f"    📷 Using original image (preprocessing disabled)")

        # Optional: auto-crop likely text region for OCR to reduce noise
        crop_image_path = preprocessed_image_path
        if self.crop_for_ocr:
            try:
                cropped = self._auto_crop_text_region(preprocessed_image_path, self.crop_margin)
                if cropped and os.path.exists(cropped):
                    crop_image_path = cropped
                    temp_files_to_cleanup.append(cropped)
                    print(f"    ✂️  Auto-cropped image for OCR: {os.path.basename(cropped)}")
                else:
                    print(f"    ⚠️  Auto-cropping produced no improvement; using original for OCR")
            except Exception as e:
                print(f"    ⚠️  Auto-cropping failed: {e}")
        
        # Extract text using selected OCR engine
        text = ""
        print(f"    🤖 Running {self.ocr_engine.upper()} OCR...")
        try:
            if self.ocr_engine == "easyocr":
                results = self.easyocr_reader.readtext(crop_image_path)
                text = " ".join([result[1] for result in results])
                print(f"    ✓ EasyOCR found {len(results)} text regions")
            elif self.ocr_engine == "tesseract":
                image = Image.open(crop_image_path)
                text = pytesseract.image_to_string(image)
                print(f"    ✓ Tesseract OCR completed")
            else:
                raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
        except Exception as e:
            print(f"    ❌ OCR failed for {image_path}: {e}")
            text = ""
        
        # Display OCR results
        if text.strip():
            print(f"    📄 OCR Text Extracted ({len(text)} characters):")
            print(f"    " + "="*60)
            # Show first 500 characters of OCR text, with line breaks preserved
            display_text = text[:500] + "..." if len(text) > 500 else text
            for line in display_text.split('\n'):
                if line.strip():  # Only show non-empty lines
                    print(f"    | {line.strip()}")
            print(f"    " + "="*60)
        else:
            print(f"    ⚠️  No text extracted from OCR")
        
        # Extract heuristic metadata from the OCR text
        print(f"    🔍 Extracting heuristic metadata from OCR text...")
        heuristic_metadata = extract_book_metadata_from_text(text) if text else {}
        
        if heuristic_metadata:
            print(f"    📊 Heuristic metadata found:")
            for key, value in heuristic_metadata.items():
                if value:
                    print(f"    | {key}: {value}")
        else:
            print(f"    ⚠️  No heuristic metadata extracted")
        
        # Clean up temporary files
        try:
            for tmp in temp_files_to_cleanup:
                if tmp != image_path and os.path.exists(tmp):
                    os.remove(tmp)
            # Attempt to clean the temp directory if empty
            if temp_files_to_cleanup:
                temp_dir = os.path.dirname(temp_files_to_cleanup[0])
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                print(f"    🧹 Cleaned up temporary OCR artifacts")
        except Exception:
            pass  # Ignore cleanup errors
        
        return text, heuristic_metadata
    
    def create_enhanced_prompt(self, ocr_texts: List[str], heuristic_metadata: Dict[str, Any]) -> str:
        """Create an enhanced prompt that includes OCR context."""
        print(f"📝 Building enhanced prompt with OCR context...")
        
        ocr_context = ""
        if ocr_texts:
            print(f"📋 Adding OCR context from {len(ocr_texts)} information pages")
            ocr_context = "\n\nADDITIONAL OCR CONTEXT FROM INFORMATION PAGES:\n"
            for i, text in enumerate(ocr_texts, 1):
                if text.strip():
                    ocr_context += f"\nPage {i+1} OCR Text:\n{text.strip()}\n"
                    print(f"   ✓ Added OCR text from page {i+1} ({len(text)} characters)")
        else:
            print(f"⚠️  No OCR text available for context")
        
        heuristic_context = ""
        if heuristic_metadata:
            print(f"📊 Adding heuristic metadata context:")
            for key, value in heuristic_metadata.items():
                if value:
                    print(f"   ✓ {key}: {value}")
            heuristic_context = f"\n\nHEURISTIC METADATA EXTRACTED FROM OCR:\n{json.dumps(heuristic_metadata, indent=2)}\n"
            heuristic_context += "\nUse this heuristic data as additional context, but prioritize what you can directly see in the images. The OCR may contain errors."
        else:
            print(f"⚠️  No heuristic metadata available for context")
        
        enhanced_prompt = self.prompt_template + ocr_context + heuristic_context
        
        print(f"✅ Enhanced prompt created ({len(enhanced_prompt)} characters total)")
        print(f"📄 ENHANCED PROMPT PREVIEW:")
        print("="*80)
        print(enhanced_prompt)
        print("="*80)
        
        return enhanced_prompt
    
    def extract_metadata_from_images(self, image_paths: List[str], ocr_image_indices: List[int] = None) -> Dict[str, Any]:
        """
        Extract metadata from multiple book images with OCR enhancement.
        
        Args:
            image_paths: List of image file paths
            ocr_image_indices: List of indices (0-based) of images to run OCR on. 
                              If None, defaults to [1, 2] (2nd and 3rd images)
        """
        if not image_paths:
            raise Exception("No image paths provided")
        
        # Default to processing 2nd and 3rd images with OCR (indices 1 and 2)
        if ocr_image_indices is None:
            ocr_image_indices = [1, 2] if len(image_paths) > 2 else [1] if len(image_paths) > 1 else []
        
        # Extract OCR text from specified images
        ocr_texts = []
        combined_heuristic_metadata = {}
        
        print(f"\n🔍 OCR PROCESSING PHASE")
        print(f"=" * 50)
        print(f"📋 Running OCR on {len(ocr_image_indices)} information pages...")
        print(f"📂 OCR target indices: {ocr_image_indices}")
        
        for idx in ocr_image_indices:
            if 0 <= idx < len(image_paths):
                print(f"\n📖 Processing OCR for image {idx + 1}: {os.path.basename(image_paths[idx])}")
                ocr_text, heuristic_meta = self.extract_text_with_ocr(image_paths[idx])
                if ocr_text.strip():
                    ocr_texts.append(ocr_text)
                    print(f"    ✅ OCR text added to context")
                    # Merge heuristic metadata, preferring non-null values
                    for key, value in heuristic_meta.items():
                        if value and (key not in combined_heuristic_metadata or not combined_heuristic_metadata[key]):
                            combined_heuristic_metadata[key] = value
                            print(f"    📊 Merged heuristic: {key} = {value}")
                else:
                    print(f"    ⚠️  No usable OCR text from this image")
            else:
                print(f"    ❌ Invalid OCR index {idx} (image not found)")
        
        print(f"\n📊 OCR PROCESSING SUMMARY:")
        print(f"   • OCR texts collected: {len(ocr_texts)}")
        print(f"   • Heuristic metadata fields: {len([k for k, v in combined_heuristic_metadata.items() if v])}")
        if combined_heuristic_metadata:
            for key, value in combined_heuristic_metadata.items():
                if value:
                    print(f"   • {key}: {value}")
        
        # Create enhanced prompt with OCR context
        print(f"\n🤖 OLLAMA PROCESSING PHASE")
        print(f"=" * 50)
        enhanced_prompt = self.create_enhanced_prompt(ocr_texts, combined_heuristic_metadata)
        
        # Encode all images for Ollama
        print(f"\n📸 Encoding {len(image_paths)} images for vision model...")
        images = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"   📷 Encoding image {i}: {os.path.basename(img_path)}")
            encoded = self.encode_image(img_path)
            images.append(encoded)
            print(f"      ✓ Encoded ({len(encoded)} characters)")
        
        # Create the request payload
        payload = {
            "model": self.model,
            "prompt": enhanced_prompt,
            "stream": False,
            "images": images
        }
        
        print(f"\n🚀 Sending request to Ollama...")
        print(f"   • Model: {self.model}")
        print(f"   • Images: {len(images)}")
        print(f"   • Prompt length: {len(enhanced_prompt)} characters")
        print(f"   • OCR context included: {'Yes' if ocr_texts else 'No'}")
        print(f"   • Heuristic context included: {'Yes' if combined_heuristic_metadata else 'No'}")
        
        # Send request to Ollama
        response = self.session.post(self.ollama_url, json=payload)
        
        if response.status_code != 200:
            print(f"❌ Ollama API error: {response.status_code}")
            raise Exception(f"Error from Ollama API: {response.text}")
        
        print(f"✅ Received response from Ollama")
        
        # Extract the response
        result = response.json()
        response_text = result.get("response", "")
        
        print(f"\n📄 OLLAMA RAW RESPONSE:")
        print("="*80)
        print(response_text)
        print("="*80)
        
        print(f"\n🔧 PARSING RESPONSE...")
        print(f"   📏 Raw response length: {len(response_text)} characters")
        
        # Parse the JSON from the response
        try:
            # Remove any markdown formatting
            print(f"   🧹 Cleaning markdown formatting...")
            response_text = response_text.replace("```json", "").replace("```", "")
            
            # Try to find JSON in the response
            print(f"   🔍 Searching for JSON structure...")
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                print(f"   ✓ JSON found at positions {json_start}-{json_end}")
                json_str = response_text[json_start:json_end+1]
                print(f"   📏 Extracted JSON length: {len(json_str)} characters")
                
                # Replace template placeholders with null values
                print(f"   🔄 Replacing template placeholders...")
                json_str = json_str.replace('"string | null"', 'null')
                json_str = json_str.replace('"integer | null"', 'null')
                json_str = json_str.replace('"float | null"', 'null')
                json_str = json_str.replace('"YYYY | null"', 'null')
                json_str = json_str.replace('["string", "..."] | []', '[]')
                
                print(f"   📋 Parsing cleaned JSON...")
                metadata = json.loads(json_str)
                print(f"   ✅ JSON parsed successfully")
            else:
                print(f"   ⚠️  No JSON braces found, attempting direct parse...")
                metadata = json.loads(response_text)
                print(f"   ✅ Direct JSON parse successful")
                
            # Validate against schema
            print(f"   🔍 Validating against schema...")
            jsonschema.validate(instance=metadata, schema=METADATA_SCHEMA)
            print(f"   ✅ Schema validation passed")
            
            # Add processing metadata
            print(f"   📊 Adding processing metadata...")
            metadata["_processing_info"] = {
                "ocr_engine": self.ocr_engine,
                "preprocessing_used": self.use_preprocessing,
                "ocr_images_processed": len(ocr_texts),
                "total_images": len(image_paths),
                "heuristic_metadata_found": bool(combined_heuristic_metadata)
            }
            
            print(f"\n✅ EXTRACTION SUCCESSFUL!")
            print(f"📊 FINAL METADATA SUMMARY:")
            print(f"   • Title: {metadata.get('title', 'N/A')}")
            print(f"   • Authors: {', '.join(metadata.get('authors', [])) or 'N/A'}")
            print(f"   • Publisher: {metadata.get('publisher', 'N/A')}")
            print(f"   • Publication Date: {metadata.get('publication_date', 'N/A')}")
            print(f"   • ISBN-13: {metadata.get('isbn_13', 'N/A')}")
            print(f"   • ISBN-10: {metadata.get('isbn_10', 'N/A')}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return a structured error response with heuristic fallback
            print(f"\n❌ JSON PARSING FAILED!")
            print(f"   Error: {e}")
            print(f"🔄 FALLING BACK TO HEURISTIC METADATA...")
            
            fallback_metadata = {
                "title": combined_heuristic_metadata.get("title"),
                "subtitle": None,
                "authors": [combined_heuristic_metadata.get("author")] if combined_heuristic_metadata.get("author") else [],
                "publisher": combined_heuristic_metadata.get("publisher"),
                "publication_date": combined_heuristic_metadata.get("year"),
                "isbn_10": None,
                "isbn_13": combined_heuristic_metadata.get("isbn") if combined_heuristic_metadata.get("isbn") and len(combined_heuristic_metadata.get("isbn", "").replace("-", "")) == 13 else None,
                "asin": None,
                "edition": None,
                "binding_type": None,
                "language": None,
                "page_count": None,
                "categories": [],
                "description": None,
                "condition_keywords": [],
                "price": {
                    "currency": None,
                    "amount": float(combined_heuristic_metadata.get("price", 0)) if combined_heuristic_metadata.get("price") else None
                },
                "_processing_info": {
                    "ocr_engine": self.ocr_engine,
                    "preprocessing_used": self.use_preprocessing,
                    "ocr_images_processed": len(ocr_texts),
                    "total_images": len(image_paths),
                    "heuristic_metadata_found": bool(combined_heuristic_metadata),
                    "fallback_used": True,
                    "ollama_error": str(e)
                }
            }
            
            print(f"📊 FALLBACK METADATA SUMMARY:")
            print(f"   • Title: {fallback_metadata.get('title', 'N/A')}")
            print(f"   • Authors: {', '.join(fallback_metadata.get('authors', [])) or 'N/A'}")
            print(f"   • Publisher: {fallback_metadata.get('publisher', 'N/A')}")
            print(f"   • Publication Date: {fallback_metadata.get('publication_date', 'N/A')}")
            print(f"   ⚠️  Using heuristic fallback due to Ollama parsing failure")
            
            return fallback_metadata
            
        except jsonschema.exceptions.ValidationError as e:
            raise Exception(f"JSON validation failed: {e}")
    
    def process_book_directory(self, book_dir: str, ocr_image_indices: List[int] = None) -> Dict[str, Any]:
        """Process all images in a book directory with OCR enhancement."""
        print(f"\n📂 PROCESSING BOOK DIRECTORY")
        print(f"=" * 50)
        print(f"📁 Directory: {book_dir}")
        
        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        
        print(f"🔍 Scanning for image files...")
        for file in sorted(os.listdir(book_dir)):  # Sort to ensure consistent ordering
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_paths.append(os.path.join(book_dir, file))
        
        if not image_paths:
            print(f"❌ No image files found in {book_dir}")
            raise Exception(f"No image files found in {book_dir}")
        
        print(f"✅ Found {len(image_paths)} images:")
        for i, path in enumerate(image_paths):
            file_size = os.path.getsize(path) / 1024  # Size in KB
            print(f"   {i+1}. {os.path.basename(path)} ({file_size:.1f} KB)")
        
        # Show OCR processing plan
        if ocr_image_indices is None:
            ocr_image_indices = [1, 2] if len(image_paths) > 2 else [1] if len(image_paths) > 1 else []
        
        print(f"\n📋 OCR PROCESSING PLAN:")
        if ocr_image_indices:
            print(f"   OCR will be applied to {len(ocr_image_indices)} images:")
            for idx in ocr_image_indices:
                if 0 <= idx < len(image_paths):
                    print(f"   • Index {idx}: {os.path.basename(image_paths[idx])}")
                else:
                    print(f"   ⚠️  Index {idx}: OUT OF RANGE")
        else:
            print(f"   ⚠️  No OCR processing planned")
        
        # Extract metadata from the images
        return self.extract_metadata_from_images(image_paths, ocr_image_indices)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Extract book metadata using enhanced OCR + Ollama pipeline")
    parser.add_argument("--book-dir", type=str, help="Directory containing book images")
    parser.add_argument("--image", type=str, nargs="+", help="Path to book image(s)")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model to use")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--prompt-file", type=str, help="Custom prompt file path")
    parser.add_argument("--ocr-engine", type=str, choices=["easyocr", "tesseract"], default="easyocr", help="OCR engine to use")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable image preprocessing")
    parser.add_argument("--ocr-indices", type=int, nargs="+", help="Indices of images to run OCR on (0-based, default: 1 2)")
    parser.add_argument("--show-raw", action="store_true", help="Show raw Ollama response")
    parser.add_argument("--crop-ocr", action="store_true", help="Auto-crop text regions before OCR")
    parser.add_argument("--crop-margin", type=int, default=16, help="Margin pixels around detected text when cropping (default: 16)")
    parser.add_argument("--no-warm-model", action="store_true", help="Disable model warm-up on startup")
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = EnhancedBookMetadataExtractor(
        model=args.model, 
        prompt_file=args.prompt_file,
        ocr_engine=args.ocr_engine,
        use_preprocessing=not args.no_preprocessing,
        crop_for_ocr=args.crop_ocr,
        crop_margin=args.crop_margin,
        warm_model=not args.no_warm_model
    )
    
    try:
        # Process based on input type
        if args.book_dir:
            metadata = extractor.process_book_directory(args.book_dir, args.ocr_indices)
        elif args.image:
            metadata = extractor.extract_metadata_from_images(args.image, args.ocr_indices)
        else:
            parser.error("Either --book-dir or --image must be provided")
            return
        
        # Output the metadata
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Enhanced metadata saved to {args.output}")
        else:
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
