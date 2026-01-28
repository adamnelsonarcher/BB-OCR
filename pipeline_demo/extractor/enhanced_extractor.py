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
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import requests
from PIL import Image
import jsonschema
import easyocr
import pytesseract
import cv2
import numpy as np
import tempfile

# Load .env if present so env-based flags work when launching via UI/server
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Add the parent directories to the path to import from other modules
parent_dir = os.path.dirname(os.path.abspath(__file__))
ocr_testing_dir = os.path.join(os.path.dirname(parent_dir), "ocr_testing")
sys.path.append(ocr_testing_dir)

# Import preprocessing module
try:
    from preprocessing.image_preprocessor import preprocess_for_book_cover
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import OCR testing modules: {e}")
    print("Make sure the ocr_testing directory is available in the parent directory.")
    PREPROCESSING_AVAILABLE = False
    
    # Define fallback functions
    def preprocess_for_book_cover(image_path, output_path=None):
        """Fallback function when preprocessing is not available."""
        return image_path, output_path, ["original"]

# Heuristic OCR->JSON preview removed by request; keep placeholders off
HEUR_HEURISTICS_AVAILABLE = False
def _heuristic_book_extract(_: str) -> Dict[str, Any]:
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
        "year": {"type": ["string", "null"]},
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
            "type": ["object", "null"],
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
                 crop_for_ocr: bool = False, crop_margin: int = 128, warm_model: bool = True,
                 edge_crop_percent: float = 0.0, ollama_timeout_seconds: float = 300.0,
                 max_ocr_chars_per_image: int = 330, llm_backend: str = "ollama"):
        """Initialize the extractor with the specified model, OCR engine, and preprocessing options.

        Args:
            model: Ollama model name
            prompt_file: Path to enhanced prompt template
            ocr_engine: "easyocr" or "tesseract"
            use_preprocessing: Apply image preprocessing before OCR
            crop_for_ocr: Auto-crop text regions before OCR to reduce noise
            crop_margin: Margin (in pixels) to add around detected text region when cropping
            warm_model: Send a tiny request on init to keep/loading the model
            edge_crop_percent: Percentage [0-45] to crop from each edge (centered crop) for OCR
        """
        self.model = model
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.ocr_engine = ocr_engine.lower()
        self.use_preprocessing = use_preprocessing
        self.crop_for_ocr = crop_for_ocr
        self.crop_margin = int(max(0, crop_margin))
        self.edge_crop_percent = float(max(0.0, min(45.0, edge_crop_percent)))
        self.ollama_timeout_seconds = float(max(5.0, ollama_timeout_seconds))
        self.llm_backend = (llm_backend or "ollama").lower()
        # Per-image OCR length cap (texts longer than this are ignored for context)
        self.max_ocr_chars_per_image = int(max(1, max_ocr_chars_per_image))
        self._trace_sink: Optional[Callable[[Dict[str, Any]], None]] = None
        # Debug flag to draw auto-crop overlays and skip cropping when enabled
        try:
            self.debug_autocrop = str(os.getenv("BB_OCR_DEBUG_AUTOCROP", "")).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            self.debug_autocrop = False

        # Reuse HTTP connections
        self.session = requests.Session()
        # Avoid inheriting proxy env that can break local connections on Windows
        try:
            self.session.trust_env = False
        except Exception:
            pass
        
        # Initialize OCR engines
        if self.ocr_engine == "easyocr":
            # Default to CPU-only to work on machines without GPUs.
            # Set BB_OCR_EASYOCR_GPU=1 to enable GPU if desired.
            try:
                use_gpu = str(os.getenv("BB_OCR_EASYOCR_GPU", "")).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                use_gpu = False
            lang_key = "en"  # extendable in future
            cache_key = f"{lang_key}|gpu={1 if use_gpu else 0}"
            if cache_key not in EnhancedBookMetadataExtractor._easyocr_reader_cache:
                EnhancedBookMetadataExtractor._easyocr_reader_cache[cache_key] = easyocr.Reader(["en"], gpu=use_gpu)
            self.easyocr_reader = EnhancedBookMetadataExtractor._easyocr_reader_cache[cache_key]
        
        # Load the enhanced prompt from file
        if prompt_file is None:
            prompt_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "prompts", 
                "enhanced_book_metadata_prompt.txt"
            )
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        # Optionally warm Ollama model only (Gemini/OpenAI don't need this)
        if warm_model and self.llm_backend == "ollama":
            try:
                self._warm_ollama_model()
            except Exception as e:
                print(f"Warning: model warm-up skipped due to error: {e}")
    def set_trace_sink(self, sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Set a callback that receives the current trace dictionary as processing progresses."""
        self._trace_sink = sink

    def _emit_trace(self, trace: Dict[str, Any]) -> None:
        try:
            if self._trace_sink is not None:
                # send a shallow copy to avoid mutation races
                self._trace_sink(dict(trace))
        except Exception:
            pass
    def _image_to_data_url(self, path: str, fallback_ext: str = "png", *, max_dim: int = 800) -> Optional[str]:
        """Read image, downscale to max_dim, and return data URL base64 string."""
        try:
            if not path or not os.path.exists(path):
                return None
            # Downscale for UI preview to avoid huge base64 payloads
            try:
                img = Image.open(path)
                img.thumbnail((max_dim, max_dim))
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="PNG")
                data = buf.getvalue()
                mime = "image/png"
                b64 = base64.b64encode(data).decode("utf-8")
                return f"data:{mime};base64,{b64}"
            except Exception:
                # Fallback: raw bytes
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = (os.path.splitext(path)[1] or f".{fallback_ext}").lower().lstrip(".")
                mime = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "bmp": "image/bmp", "gif": "image/gif", "tiff": "image/tiff"
                }.get(ext, "image/png")
                return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    def _get_temp_dir(self) -> str:
        """Return a process-wide temp directory for image artifacts outside the project tree."""
        base = os.path.join(tempfile.gettempdir(), "bb_ocr_pipeline_temp")
        os.makedirs(base, exist_ok=True)
        return base


    def _warm_ollama_model(self) -> None:
        """Send a tiny request to prompt the Ollama server to load the model."""
        payload = {
            "model": self.model,
            "prompt": "ping",
            "stream": False
        }
        try:
            # Give CPU-only machines more time to load the model.
            resp = self.session.post(self.ollama_url, json=payload, timeout=(5, 30))
            # Ignore content; just ensure request completes
            if resp.status_code != 200:
                raise RuntimeError(f"Warm-up status {resp.status_code}")
            print("🔥 Model warm-up request sent")
        except Exception as e:
            # Do not block the pipeline for demo usage
            print(f"Warning: model warm-up failed: {e}; continuing anyway")
            return

    def _auto_crop_text_region(self, image_path: str, margin: int) -> Optional[str]:
        """Detect and crop the dominant text region. Returns new image path or None if no crop.

        Robust heuristic:
        - Build a composite text mask using adaptive (mean/gaussian), Otsu, and gradient cues
        - Run morphology with multiple kernel sizes to connect text lines into blocks
        - Collect plausible text contours and union their bounding boxes
        - Clamp the crop to avoid extreme over/under-cropping and apply margin
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Light denoise + contrast to help thresholding
        gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray_blur)
        # Thresholds (binary inverse so text is white)
        thr_mean = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)
        thr_gaus = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        _, thr_otsu = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Gradient cue (emphasize text edges)
        gradx = cv2.Sobel(gray_eq, cv2.CV_16S, 1, 0, ksize=3)
        grady = cv2.Sobel(gray_eq, cv2.CV_16S, 0, 1, ksize=3)
        grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(gradx), 1.0, cv2.convertScaleAbs(grady), 1.0, 0))
        _, thr_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Composite mask
        mask = cv2.bitwise_or(cv2.bitwise_or(thr_mean, thr_gaus), cv2.bitwise_or(thr_otsu, thr_grad))
        # Morphology with multiple kernels to connect text blocks
        def morph_pass(src, kclose, kopen, kdil):
            closed = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kclose, iterations=2)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kopen, iterations=1)
            dilated = cv2.dilate(opened, kdil, iterations=1)
            return dilated
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (11,3))
        k4 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
        variant1 = morph_pass(mask, k1, k2, k3)
        variant2 = morph_pass(mask, k4, k2, k3)
        merged = cv2.bitwise_or(variant1, variant2)
        # Find contours on merged mask
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        img_area = float(h * w)
        # Collect boxes using only area-based filtering (no aspect/edge heuristics)
        min_area_frac = 0.0001  # ignore specks
        max_area_frac = 0.10    # ignore huge blobs
        boxes: List[Tuple[int,int,int,int]] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = float(cw * ch)
            if area < (min_area_frac * img_area):
                continue
            if area > (max_area_frac * img_area):
                continue
            boxes.append((x, y, cw, ch))
        if not boxes:
            # Fallback: increase sensitivity to very small text on otherwise blank pages
            contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours2:
                x, y, cw, ch = cv2.boundingRect(c)
                area = float(cw * ch)
                """if area < 0.0003 * img_area:
                    continue
                aspect = cw / float(ch + 1e-6)
                if aspect < 0.08 or aspect > 60.0:
                    continue
                boxes.append((x, y, cw, ch))"""
            if not boxes:
                return None
        # Union of boxes
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        # Clamp extreme sizes: inflate too-small, deflate too-large
        area = float((x1 - x0) * (y1 - y0))
        min_area = 0.12 * img_area
        max_area = 0.95 * img_area
        if area < min_area:
            pad = int(0.03 * max(w, h))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad)
            y1 = min(h, y1 + pad)
        elif area > max_area:
            # Keep union as-is; do not drop boxes—red box must include all green boxes
            pass
        # Save pre-margin union box for debug overlay
        pre_margin_x0, pre_margin_y0, pre_margin_x1, pre_margin_y1 = x0, y0, x1, y1

        # Apply margin and bounds (final crop region)
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(w, x1 + margin)
        y1 = min(h, y1 + margin)
        if x1 <= x0 or y1 <= y0:
            return None
        
        
        # Debug overlay: draw boxes and union rect to a sidecar image (do not early-return)
        if getattr(self, "debug_autocrop", False):
            annotated = img.copy()
            # draw individual boxes (green)
            try:
                for (bx, by, bw, bh) in boxes:
                    cv2.rectangle(annotated, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
                # draw union box pre-margin (red)
                cv2.rectangle(annotated, (int(pre_margin_x0), int(pre_margin_y0)), (int(pre_margin_x1), int(pre_margin_y1)), (0, 0, 255), 3)
                # draw final crop box post-margin (gray)
                cv2.rectangle(annotated, (int(x0), int(y0)), (int(x1), int(y1)), (180, 180, 180), 4)
            except Exception:
                pass
            temp_dir = self._get_temp_dir()
            base = os.path.splitext(os.path.basename(image_path))[0]
            debug_path = os.path.join(temp_dir, f"{base}_autocrop_debug.png")
            try:
                cv2.imwrite(debug_path, annotated)
            except Exception:
                pass
            # In debug mode, return the annotated full image instead of a cropped region
            return debug_path
    
    
        cropped = img[y0:y1, x0:x1]
        # Write to temp outside project tree to avoid dev server reloads
        temp_dir = self._get_temp_dir()
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(temp_dir, f"{base}_cropped.png")
        cv2.imwrite(out_path, cropped)
        return out_path

    def _central_edge_crop(self, image_path: str, percent: float) -> Optional[str]:
        """Crop a centered rectangle by removing `percent` from each edge. Returns new path or None."""
        if percent <= 0.0:
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        # Compute pixel margins
        mx = int(round(w * (percent / 100.0)))
        my = int(round(h * (percent / 100.0)))
        # Ensure valid crop area
        x0 = max(0, mx)
        y0 = max(0, my)
        x1 = min(w, w - mx)
        y1 = min(h, h - my)
        if x1 - x0 < max(16, w * 0.2) or y1 - y0 < max(16, h * 0.2):
            return None
        cropped = img[y0:y1, x0:x1]
        temp_dir = self._get_temp_dir()
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(temp_dir, f"{base}_edgecrop_{int(percent)}.png")
        cv2.imwrite(out_path, cropped)
        return out_path
    
    def _encode_image_for_model(self, image_path: str, *, max_dim: int = 1600, jpeg_quality: int = 85) -> str:
        """Encode image to base64 for model: downscale to max_dim and compress to JPEG."""
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((max_dim, max_dim))
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=int(max(50, min(95, jpeg_quality))))
            data = buf.getvalue()
            return base64.b64encode(data).decode("utf-8")
        except Exception:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    
    def extract_text_with_ocr(self, image_path: str, trace_image: Optional[Dict[str, Any]] = None, trace_global: Optional[Dict[str, Any]] = None, *, step_log: Optional[List[Dict[str, Any]]] = None, image_index: Optional[int] = None) -> str:
        """Extract text from an image using OCR and return text only."""
        print(f"    🔍 Starting OCR processing for: {os.path.basename(image_path)}")
        
        # Apply preprocessing if enabled
        preprocessed_image_path = image_path
        temp_files_to_cleanup: List[str] = []
        if trace_image is not None:
            trace_image.setdefault("original_b64", self._image_to_data_url(image_path))
        if self.use_preprocessing and PREPROCESSING_AVAILABLE:
            print(f"    📝 Applying image preprocessing...")
            # Create temporary preprocessed image in system temp
            temp_dir = self._get_temp_dir()
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            preprocessed_image_path = os.path.join(temp_dir, f"{base_name}_preprocessed.png")
            
            try:
                _, preprocessed_image_path, steps = preprocess_for_book_cover(image_path, preprocessed_image_path)
                print(f"    ✓ Preprocessing completed. Steps applied: {', '.join(steps)}")
                temp_files_to_cleanup.append(preprocessed_image_path)
                if trace_image is not None:
                    trace_image["preprocessing_steps"] = steps
                    trace_image["preprocessed_b64"] = self._image_to_data_url(preprocessed_image_path)
                    if step_log is not None:
                        step_log.append({"step": "preprocess", "image_index": None, "images": {"preprocessed_b64": trace_image.get("preprocessed_b64")}, "info": {"steps": steps}})
                    if trace_global is not None:
                        self._emit_trace(trace_global)
            except Exception as e:
                print(f"    ⚠️  Preprocessing failed for {image_path}: {e}")
                preprocessed_image_path = image_path
        elif self.use_preprocessing and not PREPROCESSING_AVAILABLE:
            print(f"    ⚠️  Preprocessing requested but not available for {image_path}")
        else:
            print(f"    📷 Using original image (preprocessing disabled)")

        # Optional: either central edge crop (simple) or auto text region crop
        crop_image_path = preprocessed_image_path
        # Apply simple centered edge crop first if configured
        if self.edge_crop_percent > 0.0:
            try:
                central = self._central_edge_crop(preprocessed_image_path, self.edge_crop_percent)
                if central and os.path.exists(central):
                    crop_image_path = central
                    temp_files_to_cleanup.append(central)
                    print(f"    ✂️  Edge-cropped image for OCR: {os.path.basename(central)} ({self.edge_crop_percent:.1f}%)")
                    if trace_image is not None:
                        trace_image["edge_cropped_b64"] = self._image_to_data_url(central)
                        if step_log is not None:
                            step_log.append({"step": "edge_crop", "image_index": None, "images": {"edge_cropped_b64": trace_image.get("edge_cropped_b64")}})
                        if trace_global is not None:
                            self._emit_trace(trace_global)
            except Exception as e:
                print(f"    ⚠️  Edge-cropping failed: {e}")
        # Optionally try auto-crop of text region on the current crop (after edge-crop if any)
        if self.crop_for_ocr:
            try:
                cropped = self._auto_crop_text_region(crop_image_path, self.crop_margin)
                if cropped and os.path.exists(cropped):
                    crop_image_path = cropped
                    temp_files_to_cleanup.append(cropped)
                    print(f"    ✂️  Auto-cropped image for OCR: {os.path.basename(cropped)}")
                    if trace_image is not None:
                        trace_image["auto_cropped_b64"] = self._image_to_data_url(cropped)
                        if step_log is not None:
                            step_log.append({"step": "auto_crop", "image_index": None, "images": {"auto_cropped_b64": trace_image.get("auto_cropped_b64")}})
                        if trace_global is not None:
                            self._emit_trace(trace_global)
                else:
                    print(f"    ⚠️  Auto-cropping produced no improvement; using current crop for OCR")
            except Exception as e:
                print(f"    ⚠️  Auto-cropping failed: {e}")
        
        # Optionally downscale the crop for faster OCR while preserving readability
        ocr_input_path = crop_image_path
        try:
            try:
                img = Image.open(crop_image_path)
                # Use a higher limit for non-cover pages to preserve text fidelity
                # Cover (index 0) can be downscaled more aggressively; others keep larger size
                # CPU-friendly cap for non-cover pages to avoid huge OCR work
                max_dim = 1600 if (image_index is None or image_index == 0) else 2400
                if max(img.size) > max_dim:
                    img = img.convert("RGB")
                    img.thumbnail((max_dim, max_dim))
                    from io import BytesIO
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=(90 if (image_index is None or image_index == 0) else 95))
                    data = buf.getvalue()
                    temp_dir = self._get_temp_dir()
                    base_name = os.path.splitext(os.path.basename(crop_image_path))[0]
                    ocr_input_path = os.path.join(temp_dir, f"{base_name}_ocr_ds.jpg")
                    with open(ocr_input_path, "wb") as f:
                        f.write(data)
                    temp_files_to_cleanup.append(ocr_input_path)
                    print(f"    🗜️  Downscaled image for OCR ({max_dim}px): {os.path.basename(ocr_input_path)}")
            except Exception as _:
                ocr_input_path = crop_image_path
        except Exception:
            ocr_input_path = crop_image_path

        # Extract text using selected OCR engine
        text = ""
        print(f"    🤖 Running {self.ocr_engine.upper()} OCR...")
        try:
            if self.ocr_engine == "easyocr":
                # Use a smaller EasyOCR configuration to reduce memory usage
                results = self.easyocr_reader.readtext(ocr_input_path, paragraph=False, batch_size=1, workers=0)
                text = " ".join([result[1] for result in results])
                print(f"    ✓ EasyOCR found {len(results)} text regions")
            elif self.ocr_engine == "tesseract":
                image = Image.open(ocr_input_path)
                text = pytesseract.image_to_string(image)
                print(f"    ✓ Tesseract OCR completed")
            else:
                raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
        except Exception as e:
            print(f"    ❌ OCR failed for {image_path}: {e}")
            text = ""
        
        if trace_image is not None:
            trace_image["ocr_text"] = text
            if step_log is not None:
                step_log.append({"step": "ocr", "image_index": None, "info": {"chars": len(text)}})
            if trace_global is not None:
                self._emit_trace(trace_global)
        # Display OCR results
        if text.strip():
            print(f"    📄 OCR Text Extracted ({len(text)} characters):")
            print(f"    " + "="*60)
            display_text = text[:500] + "..." if len(text) > 500 else text
            for line in display_text.split('\n'):
                if line.strip():
                    print(f"    | {line.strip()}")
            print(f"    " + "="*60)
        else:
            print(f"    ⚠️  No text extracted from OCR")
        
        # Clean up temporary files
        try:
            for tmp in temp_files_to_cleanup:
                if tmp != image_path and os.path.exists(tmp):
                    os.remove(tmp)
            # Skip removing the shared temp directory
            print(f"    🧹 Cleaned up temporary OCR artifacts")
        except Exception:
            pass  # Ignore cleanup errors
        
        return text
    
    def create_enhanced_prompt(self, ocr_texts: List[str]) -> str:
        """Create an enhanced prompt that includes OCR context only."""
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
        
        enhanced_prompt = self.prompt_template + ocr_context
        
        print(f"✅ Enhanced prompt created ({len(enhanced_prompt)} characters total)")
        print(f"📄 ENHANCED PROMPT PREVIEW:")
        print("="*80)
        print(enhanced_prompt)
        print("="*80)
        
        return enhanced_prompt
    
    def extract_metadata_from_images(self, image_paths: List[str], ocr_image_indices: List[int] = None, *, capture_trace: bool = False, trace_sink: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Extract metadata from multiple book images with OCR enhancement.
        
        Args:
            image_paths: List of image file paths
            ocr_image_indices: List of indices (0-based) of images to run OCR on. 
                              If None, defaults to [1, 2] (2nd and 3rd images)
            capture_trace: If True, include detailed processing trace in the result under `_trace`
        """
        if not image_paths:
            raise Exception("No image paths provided")
        
        # Default to processing 2nd and 3rd images with OCR (indices 1 and 2)
        if ocr_image_indices is None:
            ocr_image_indices = [1, 2] if len(image_paths) > 2 else [1] if len(image_paths) > 1 else []
        
        # Extract OCR text from specified images
        ocr_texts = []
        trace: Dict[str, Any] = {"images": [], "steps": []} if capture_trace else {}
        old_sink = self._trace_sink
        if trace_sink is not None:
            self._trace_sink = trace_sink
        if capture_trace:
            # seed per-image trace with originals
            for p in image_paths:
                trace["images"].append({"original_b64": self._image_to_data_url(p)})
            trace["steps"].append({"step": "seed_images", "info": {"count": len(image_paths)}})
            self._emit_trace(trace)

            # Prepare processed previews for ALL images before OCR (preprocess / crop only)
            # This allows the UI to show processed thumbnails even for non-OCR images
            for idx, p in enumerate(image_paths):
                if idx == 0:
                    # Skip first image for preprocessing preview
                    continue
                try:
                    trace_img = trace["images"][idx]
                    temp_files_to_cleanup: List[str] = []
                    preview_base_path = p
                    # Preprocessing (if available and enabled)
                    if self.use_preprocessing and PREPROCESSING_AVAILABLE:
                        try:
                            temp_dir = self._get_temp_dir()
                            base_name = os.path.splitext(os.path.basename(p))[0]
                            pre_path = os.path.join(temp_dir, f"{base_name}_pre_preview.png")
                            _, pre_path, steps = preprocess_for_book_cover(p, pre_path)
                            temp_files_to_cleanup.append(pre_path)
                            trace_img["preprocessing_steps"] = steps
                            trace_img["preprocessed_b64"] = self._image_to_data_url(pre_path)
                            trace["steps"].append({"step": "preprocess_preview", "image_index": idx, "info": {"steps": steps}})
                            self._emit_trace(trace)
                            preview_base_path = pre_path
                        except Exception:
                            preview_base_path = p
                    # Edge crop (simple centered crop)
                    if self.edge_crop_percent > 0.0:
                        try:
                            central = self._central_edge_crop(preview_base_path, self.edge_crop_percent)
                            if central and os.path.exists(central):
                                temp_files_to_cleanup.append(central)
                                trace_img["edge_cropped_b64"] = self._image_to_data_url(central)
                                trace["steps"].append({"step": "edge_crop_preview", "image_index": idx})
                                self._emit_trace(trace)
                                preview_base_path = central
                        except Exception:
                            pass
                    # Auto text crop (heuristic), only if not already edge-cropped to a smaller region
                    if self.crop_for_ocr:
                        try:
                            cropped = self._auto_crop_text_region(preview_base_path, self.crop_margin)
                            if cropped and os.path.exists(cropped):
                                temp_files_to_cleanup.append(cropped)
                                trace_img["auto_cropped_b64"] = self._image_to_data_url(cropped)
                                trace["steps"].append({"step": "auto_crop_preview", "image_index": idx})
                                self._emit_trace(trace)
                        except Exception:
                            pass
                finally:
                    try:
                        for tmp in list(locals().get('temp_files_to_cleanup', [])):
                            if tmp and os.path.exists(tmp):
                                os.remove(tmp)
                    except Exception:
                        pass
        
        
        print(f"\n🔍 OCR PROCESSING PHASE")
        print(f"=" * 50)
        print(f"📋 Running OCR on {len(ocr_image_indices)} information pages...")
        print(f"📂 OCR target indices: {ocr_image_indices}")
        
        for idx in ocr_image_indices:
            if 0 <= idx < len(image_paths):
                print(f"\n📖 Processing OCR for image {idx + 1}: {os.path.basename(image_paths[idx])}")
                if capture_trace:
                    trace["steps"].append({"step": "start_ocr", "image_index": idx})
                    self._emit_trace(trace)
                trace_img_dict = trace["images"][idx] if capture_trace else None
                step_log = trace.get("steps") if capture_trace else None
                ocr_text = self.extract_text_with_ocr(image_paths[idx], trace_image=trace_img_dict, trace_global=trace if capture_trace else None, step_log=step_log, image_index=idx)
                if ocr_text.strip():
                    text_len = len(ocr_text)
                    if text_len > self.max_ocr_chars_per_image:
                        print(f"    ⏭️  Skipping OCR text for context (length {text_len} > {self.max_ocr_chars_per_image})")
                        if capture_trace:
                            try:
                                trace["steps"].append({"step": "ocr_skip_long", "image_index": idx, "info": {"chars": text_len, "limit": self.max_ocr_chars_per_image}})
                                # Mark image trace for UI awareness
                                if isinstance(trace_img_dict, dict):
                                    trace_img_dict["ocr_skipped_long"] = True
                                    trace_img_dict["ocr_chars"] = text_len
                                self._emit_trace(trace)
                            except Exception:
                                pass
                    else:
                        ocr_texts.append(ocr_text)
                        print(f"    ✅ OCR text added to context ({text_len} chars)")
                else:
                    print(f"    ⚠️  No usable OCR text from this image")
            else:
                print(f"    ❌ Invalid OCR index {idx} (image not found)")
        
        print(f"\n📊 OCR PROCESSING SUMMARY:")
        print(f"   • OCR texts collected: {len(ocr_texts)}")
        
        # If autocrop debug is enabled, do not send any model requests; return minimal stub
        if getattr(self, "debug_autocrop", False):
            print("\n🧪 Debug autocrop is ON: skipping model request and returning minimal stub")
            if capture_trace:
                try:
                    trace_copy = dict(trace)
                except Exception:
                    trace_copy = trace
            else:
                trace_copy = None
            stub = {
                "title": None,
                "subtitle": None,
                "authors": [],
                "publisher": None,
                "publication_date": None,
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
                "_processing_info": {
                    "ocr_engine": self.ocr_engine,
                    "preprocessing_used": self.use_preprocessing,
                    "ocr_images_processed": len(ocr_texts),
                    "total_images": len(image_paths),
                    "debug_autocrop": True,
                    "model_skipped": True,
                },
            }
            if capture_trace and isinstance(trace_copy, dict):
                stub["_trace"] = trace_copy
            return stub

        # Create enhanced prompt with OCR context
        print(f"\n🤖 OLLAMA PROCESSING PHASE")
        print(f"=" * 50)
        enhanced_prompt = self.create_enhanced_prompt(ocr_texts)
        if capture_trace:
            trace["enhanced_prompt"] = enhanced_prompt
            trace["steps"].append({"step": "build_prompt", "info": {"chars": len(enhanced_prompt)}})
            self._emit_trace(trace)
        # (trace already updated above)
        # Build best-available processed inputs for VLM (preprocess → edge-crop → auto-crop)
        model_input_paths: List[str] = []
        model_temp_files_to_cleanup: List[str] = []
        for idx, p in enumerate(image_paths):
            current_path = p
            try:
                # Preprocessing (if available and enabled)
                if self.use_preprocessing and PREPROCESSING_AVAILABLE:
                    try:
                        temp_dir = self._get_temp_dir()
                        base_name = os.path.splitext(os.path.basename(p))[0]
                        pre_path = os.path.join(temp_dir, f"{base_name}_pre_model.png")
                        _, pre_path, _steps = preprocess_for_book_cover(p, pre_path)
                        model_temp_files_to_cleanup.append(pre_path)
                        current_path = pre_path
                    except Exception:
                        current_path = p
                # Edge crop (simple centered crop)
                if self.edge_crop_percent > 0.0:
                    try:
                        central = self._central_edge_crop(current_path, self.edge_crop_percent)
                        if central and os.path.exists(central):
                            model_temp_files_to_cleanup.append(central)
                            current_path = central
                    except Exception:
                        pass
                # Auto text crop (heuristic)
                if self.crop_for_ocr:
                    try:
                        cropped = self._auto_crop_text_region(current_path, self.crop_margin)
                        if cropped and os.path.exists(cropped):
                            model_temp_files_to_cleanup.append(cropped)
                            current_path = cropped
                    except Exception:
                        pass
            finally:
                model_input_paths.append(current_path)

        # Encode all images for Ollama using processed inputs
        print(f"\n📸 Encoding {len(model_input_paths)} images for vision model...")
        images = []
        for i, (orig_path, model_path) in enumerate(zip(image_paths, model_input_paths), 1):
            print(f"   📷 Encoding image {i}: {os.path.basename(orig_path)} → {os.path.basename(model_path)}")
            idx0 = i - 1
            # IMPORTANT: model_path is already post preprocess/crop; only downscale here.
            # Use a higher cap for non-cover pages to preserve small text for Gemini.
            max_dim_model = 2000 if idx0 == 0 else 3200
            jpeg_q = 88 if idx0 == 0 else 95
            encoded = self._encode_image_for_model(model_path, max_dim=max_dim_model, jpeg_quality=jpeg_q)
            images.append(encoded)
            print(f"      ✓ Encoded ({len(encoded)} characters)")
        # Cleanup temporary processed inputs
        try:
            for tmp in model_temp_files_to_cleanup:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
        except Exception:
            pass
        if capture_trace:
            trace["steps"].append({"step": "encode_images", "info": {"count": len(images)}})
            self._emit_trace(trace)
        
        # Send request via selected LLM backend
        if capture_trace:
            trace["steps"].append({"step": "request_sent", "info": {"model": self.model, "backend": self.llm_backend}})
            self._emit_trace(trace)
        response_text = ""
        if self.llm_backend == "ollama":
            print(f"\n🚀 Sending request to Ollama...")
            print(f"   • Model: {self.model}")
            print(f"   • Images: {len(images)}")
            print(f"   • Prompt length: {len(enhanced_prompt)} characters")
            print(f"   • OCR context included: {'Yes' if ocr_texts else 'No'}")
            payload = {
                "model": self.model,
                "prompt": enhanced_prompt,
                "stream": False,
                "images": images
            }
            response = None
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    connect_timeout = 2.5
                    # CPU inference can be slow; avoid tiny read timeouts
                    read_timeout = max(60.0, self.ollama_timeout_seconds - connect_timeout)
                    response = self.session.post(self.ollama_url, json=payload, timeout=(connect_timeout, read_timeout))
                    if response.status_code == 200:
                        break
                    else:
                        raise Exception(f"Ollama HTTP {response.status_code}")
                except Exception as e:
                    last_err = e
                    wait = 1.0 * (attempt + 1)
                    print(f"   ⚠️ Ollama request failed (attempt {attempt+1}/3): {e}; retrying in {wait:.1f}s")
                    try:
                        import time as _t
                        _t.sleep(wait)
                    except Exception:
                        pass
            if response is None or response.status_code != 200:
                raise Exception(f"Error from Ollama API: {last_err}")
            try:
                result = response.json()
            except Exception as e:
                txt = response.text[:2000]
                raise Exception(f"Failed to parse Ollama JSON: {e}; prefix=\n{txt}")
            response_text = result.get("response", "")
            print(f"✅ Received response from Ollama")
        else:
            # External APIs via llm_providers
            print(f"\n🚀 Sending request to {self.llm_backend}...")
            from llm_providers.client import create_llm_client
            client = create_llm_client(self.llm_backend, session=self.session)
            response_text = client.generate(self.model, enhanced_prompt, images, timeout_seconds=self.ollama_timeout_seconds)
        
        print(f"\n📄 OLLAMA RAW RESPONSE:")
        print("="*80)
        print(response_text)
        print("="*80)
        if capture_trace:
            trace["ollama_raw"] = response_text
            trace["steps"].append({"step": "vlm_raw", "info": {"chars": len(response_text)}})
            self._emit_trace(trace)
        
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
                "total_images": len(image_paths)
            }

            # Heuristic OCR-derived JSON removed; still attach trace if requested
            if capture_trace:
                metadata["_trace"] = trace
                self._emit_trace(trace)
            
            print(f"\n✅ EXTRACTION SUCCESSFUL!")
            print(f"📊 FINAL METADATA SUMMARY:")
            print(f"   • Title: {metadata.get('title', 'N/A')}")
            print(f"   • Authors: {', '.join(metadata.get('authors', [])) or 'N/A'}")
            print(f"   • Publisher: {metadata.get('publisher', 'N/A')}")
            print(f"   • Year: {metadata.get('year', 'N/A')}")
            print(f"   • ISBN-13: {metadata.get('isbn_13', 'N/A')}")
            print(f"   • ISBN-10: {metadata.get('isbn_10', 'N/A')}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return a structured error response with heuristic fallback
            print(f"\n❌ JSON PARSING FAILED!")
            print(f"   Error: {e}")
            print(f"🔄 FALLING BACK TO HEURISTIC METADATA...")
            
            fallback_metadata = {
                "title": None,
                "subtitle": None,
                "authors": [],
                "publisher": None,
                "year": None,
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
                "price": {
                    "currency": None,
                    "amount": None
                },
                "_processing_info": {
                    "ocr_engine": self.ocr_engine,
                    "preprocessing_used": self.use_preprocessing,
                    "ocr_images_processed": len(ocr_texts),
                    "total_images": len(image_paths),
                    "fallback_used": True,
                    "ollama_error": str(e)
                }
            }
            
            print(f"📊 FALLBACK METADATA SUMMARY:")
            print(f"   • Title: {fallback_metadata.get('title', 'N/A')}")
            print(f"   • Authors: {', '.join(fallback_metadata.get('authors', [])) or 'N/A'}")
            print(f"   • Publisher: {fallback_metadata.get('publisher', 'N/A')}")
            print(f"   • Year: {fallback_metadata.get('year', 'N/A')}")
            print(f"   ⚠️  Using minimal fallback due to Ollama parsing failure")
            
            if capture_trace:
                try:
                    fallback_metadata["_trace"] = trace
                    self._emit_trace(trace)
                except Exception:
                    pass
            return fallback_metadata
            
        except jsonschema.exceptions.ValidationError as e:
            if capture_trace:
                try:
                    metadata = {"_error": f"JSON validation failed: {e}", "_trace": trace}
                    self._emit_trace(trace)
                except Exception:
                    pass
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
        result = self.extract_metadata_from_images(image_paths, ocr_image_indices)
        return result


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
    parser.add_argument("--edge-crop", type=float, default=0.0, help="Centered edge crop percent [0-45] applied before OCR")
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
        warm_model=not args.no_warm_model,
        edge_crop_percent=args.edge_crop
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


