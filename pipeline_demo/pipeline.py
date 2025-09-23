#!/usr/bin/env python3
"""
Unified Book Pricing Pipeline (single-file demo)

This script consolidates the full pipeline:
  1) OCR + preprocessing + optional edge/text cropping
  2) Enhanced vision-language extraction via Ollama to produce structured JSON metadata
  3) Provider aggregation (Google Books metadata, AbeBooks price scrape) with timeouts
  4) Best-offer selection and merged JSON output

Usage examples:
  # Process a book directory of images, then aggregate prices
  python pipeline.py extract --book-dir path/to/book_images --output out/book.json
  python pipeline.py price --input out/book.json --output out/book_priced.json

  # One-shot: extract + price
  python pipeline.py run --book-dir path/to/book_images --output out/book_priced.json

Requirements: see requirements.txt in the same folder.
"""

import os
import sys
import re
import json
import base64
import argparse
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
import requests
import jsonschema
from PIL import Image
import cv2
import numpy as np

# OCR engines (install per requirements)
import easyocr
import pytesseract


# -------------------------
# Common schemas and helpers
# -------------------------

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "subtitle": {"type": ["string", "null"]},
        "authors": {"type": "array", "items": {"type": "string"}},
        "publisher": {"type": ["string", "null"]},
        "publication_date": {"type": ["string", "null"]},
        "isbn_10": {"type": ["string", "null"]},
        "isbn_13": {"type": ["string", "null"]},
        "asin": {"type": ["string", "null"]},
        "edition": {"type": ["string", "null"]},
        "binding_type": {"type": ["string", "null"]},
        "language": {"type": ["string", "null"]},
        "page_count": {"type": ["integer", "null"]},
        "categories": {"type": "array", "items": {"type": "string"}},
        "description": {"type": ["string", "null"]},
        "condition_keywords": {"type": "array", "items": {"type": "string"}},
        "price": {
            "type": "object",
            "properties": {
                "currency": {"type": ["string", "null"]},
                "amount": {"type": ["number", "null"]}
            }
        }
    }
}


def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        jsonschema.validate(instance=metadata, schema=METADATA_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        return False, f"Schema validation failed: {e}"
    if metadata.get("title") is None:
        return False, "Missing title"
    for k in ["authors", "categories", "condition_keywords"]:
        if metadata.get(k) is None:
            metadata[k] = []
    i10 = metadata.get("isbn_10")
    if i10 and (not isinstance(i10, str) or len(i10.replace("-", "")) != 10):
        return False, f"Invalid ISBN-10 format: {i10}"
    i13 = metadata.get("isbn_13")
    if i13 and (not isinstance(i13, str) or len(i13.replace("-", "")) != 13):
        return False, f"Invalid ISBN-13 format: {i13}"
    return True, "ok"


def ensure_output_dir(path: Optional[str]) -> None:
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


# -------------------------
# Image preprocessing utils
# -------------------------

def preprocess_for_book_cover(image_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[str], List[str]]:
    # Load
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    steps = ["original"]
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append("grayscale")
    # resize 1.5x
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
    steps.append("resize(1.5x)")
    # gentle denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 5)
    steps.append("denoise")
    # moderate contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    steps.append("clahe")
    # gentle sharpen (PIL-like unsharp mask approximation)
    pil = Image.fromarray(gray)
    sharpened = pil.filter(Image.Filter.UnsharpMask(radius=1.0, percent=20, threshold=3))
    gray = np.array(sharpened)
    steps.append("sharpen(0.2)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, gray)
    return gray, output_path, steps


def central_edge_crop(image_path: str, percent: float) -> Optional[str]:
    if percent <= 0.0:
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    mx = int(round(w * (percent / 100.0)))
    my = int(round(h * (percent / 100.0)))
    x0, y0, x1, y1 = max(0, mx), max(0, my), min(w, w - mx), min(h, h - my)
    if x1 - x0 < max(16, w * 0.2) or y1 - y0 < max(16, h * 0.2):
        return None
    cropped = img[y0:y1, x0:x1]
    temp_dir = os.path.join(os.path.dirname(image_path), "temp_preprocessed")
    os.makedirs(temp_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(temp_dir, f"{base}_edgecrop_{int(percent)}.png")
    cv2.imwrite(out_path, cropped)
    return out_path


def auto_crop_text_region(image_path: str, margin: int) -> Optional[str]:
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = float(h * w)
    best, best_area = None, 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.01 * img_area:
            continue
        aspect = cw / float(ch + 1e-6)
        if 0.2 <= aspect <= 10.0 and area > best_area:
            best_area, best = area, (x, y, cw, ch)
    if best is None:
        return None
    x, y, cw, ch = best
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + cw + margin), min(h, y + ch + margin)
    crop_area = float((x1 - x0) * (y1 - y0))
    if crop_area > 0.9 * img_area:
        return None
    cropped = img[y0:y1, x0:x1]
    temp_dir = os.path.join(os.path.dirname(image_path), "temp_preprocessed")
    os.makedirs(temp_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(temp_dir, f"{base}_cropped.png")
    cv2.imwrite(out_path, cropped)
    return out_path


# -------------------------
# OCR + Ollama extractor
# -------------------------

class UnifiedExtractor:
    _easyocr_reader = None

    def __init__(self, model: str = "gemma3:4b", ocr_engine: str = "easyocr", use_preprocessing: bool = True,
                 crop_for_ocr: bool = False, crop_margin: int = 16, edge_crop_percent: float = 0.0,
                 warm_model: bool = True) -> None:
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ocr_engine = ocr_engine.lower()
        self.use_preprocessing = use_preprocessing
        self.crop_for_ocr = crop_for_ocr
        self.crop_margin = int(max(0, crop_margin))
        self.edge_crop_percent = float(max(0.0, min(45.0, edge_crop_percent)))
        self.session = requests.Session()
        if self.ocr_engine == "easyocr":
            if UnifiedExtractor._easyocr_reader is None:
                UnifiedExtractor._easyocr_reader = easyocr.Reader(['en'])
            self.easy_reader = UnifiedExtractor._easyocr_reader
        if warm_model:
            try:
                self.session.post(self.ollama_url, json={"model": self.model, "prompt": "ping", "stream": False}, timeout=10)
            except Exception:
                pass

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _extract_text(self, path: str) -> str:
        img_for_ocr = path
        temp_to_cleanup: List[str] = []
        if self.use_preprocessing:
            try:
                temp_dir = os.path.join(os.path.dirname(path), "temp_preprocessed")
                os.makedirs(temp_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                pre_path = os.path.join(temp_dir, f"{base}_pre.png")
                _, pre_saved, _ = preprocess_for_book_cover(path, pre_path)
                if pre_saved and os.path.exists(pre_saved):
                    img_for_ocr = pre_saved
                    temp_to_cleanup.append(pre_saved)
            except Exception:
                pass
        if self.edge_crop_percent > 0.0:
            try:
                ec = central_edge_crop(img_for_ocr, self.edge_crop_percent)
                if ec and os.path.exists(ec):
                    img_for_ocr = ec
                    temp_to_cleanup.append(ec)
            except Exception:
                pass
        if self.crop_for_ocr:
            try:
                ac = auto_crop_text_region(img_for_ocr, self.crop_margin)
                if ac and os.path.exists(ac):
                    img_for_ocr = ac
                    temp_to_cleanup.append(ac)
            except Exception:
                pass
        text = ""
        try:
            if self.ocr_engine == "easyocr":
                results = self.easy_reader.readtext(img_for_ocr)
                text = " ".join(r[1] for r in results)
            elif self.ocr_engine == "tesseract":
                image = Image.open(img_for_ocr)
                text = pytesseract.image_to_string(image)
        except Exception:
            text = ""
        try:
            for t in temp_to_cleanup:
                if t != path and os.path.exists(t):
                    os.remove(t)
            d = os.path.dirname(temp_to_cleanup[0]) if temp_to_cleanup else None
            if d and os.path.exists(d) and not os.listdir(d):
                os.rmdir(d)
        except Exception:
            pass
        return text

    def _build_prompt(self, ocr_texts: List[str]) -> str:
        base_prompt = (
            "You are an assistant that extracts structured book metadata as strict JSON with keys: "
            "title, subtitle, authors (array), publisher, publication_date, isbn_10, isbn_13, asin, "
            "edition, binding_type, language, page_count, categories (array), description, condition_keywords (array), price {currency, amount}.\n"
            "Use OCR evidence below when available. If unknown, use null."
        )
        if ocr_texts:
            ctx = "\n\nADDITIONAL OCR CONTEXT:\n" + "\n\n".join(ocr_texts[:3])
        else:
            ctx = ""
        return base_prompt + ctx

    def extract_from_images(self, image_paths: List[str], ocr_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        if not image_paths:
            raise ValueError("No images provided")
        if ocr_indices is None:
            ocr_indices = [1, 2] if len(image_paths) > 2 else ([1] if len(image_paths) > 1 else [])
        ocr_texts: List[str] = []
        for idx in ocr_indices:
            if 0 <= idx < len(image_paths):
                txt = self._extract_text(image_paths[idx])
                if txt.strip():
                    ocr_texts.append(txt)
        prompt = self._build_prompt(ocr_texts)
        images_b64 = [self._encode_image(p) for p in image_paths]
        payload = {"model": self.model, "prompt": prompt, "stream": False, "images": images_b64}
        resp = self.session.post(self.ollama_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error: {resp.status_code} - {resp.text[:200]}")
        response_text = resp.json().get("response", "")
        # Clean and parse JSON
        response_text = response_text.replace("```json", "").replace("```", "")
        js, je = response_text.find("{"), response_text.rfind("}")
        if js >= 0 and je >= 0:
            jtxt = response_text[js:je+1]
            jtxt = (jtxt
                .replace('"string | null"', 'null')
                .replace('"integer | null"', 'null')
                .replace('"float | null"', 'null')
                .replace('"YYYY | null"', 'null')
                .replace('["string", "..."] | []', '[]'))
            metadata = json.loads(jtxt)
        else:
            metadata = json.loads(response_text)
        jsonschema.validate(instance=metadata, schema=METADATA_SCHEMA)
        metadata.setdefault("_processing_info", {})
        metadata["_processing_info"].update({
            "ocr_engine": self.ocr_engine,
            "preprocessing_used": self.use_preprocessing,
            "ocr_images_processed": len(ocr_texts),
            "total_images": len(image_paths)
        })
        return metadata

    def process_book_dir(self, book_dir: str, ocr_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
        paths = [os.path.join(book_dir, f) for f in sorted(os.listdir(book_dir)) if any(f.lower().endswith(e) for e in exts)]
        if not paths:
            raise ValueError(f"No images in {book_dir}")
        return self.extract_from_images(paths, ocr_indices)


# -------------------------
# Providers (Google Books, AbeBooks)
# -------------------------

class GoogleBooksProvider:
    BASE = "https://www.googleapis.com/books/v1/volumes"
    async def lookup(self, *, title: Optional[str], authors: List[str], isbn_13: Optional[str], isbn_10: Optional[str],
                    publisher: Optional[str], publication_date: Optional[str]) -> List[Dict[str, Any]]:
        key = os.getenv("GOOGLE_BOOKS_API_KEY")
        q_parts: List[str] = []
        if isbn_13: q_parts.append(f"isbn:{isbn_13}")
        if isbn_10: q_parts.append(f"isbn:{isbn_10}")
        if title: q_parts.append(f"intitle:{title}")
        for a in authors or []:
            if a: q_parts.append(f"inauthor:{a}")
        if publisher: q_parts.append(f"inpublisher:{publisher}")
        q = "+".join(p.replace(" ", "+") for p in q_parts) or (title or "")
        params = {"q": q, "maxResults": 5}
        if key: params["key"] = key
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(self.BASE, params=params)
            r.raise_for_status()
            data = r.json()
        items = data.get("items", []) or []
        offers: List[Dict[str, Any]] = []
        for it in items:
            vi = it.get("volumeInfo", {})
            ids = vi.get("industryIdentifiers", [])
            isbn13 = next((i.get("identifier") for i in ids if i.get("type") == "ISBN_13"), None)
            isbn10 = next((i.get("identifier") for i in ids if i.get("type") == "ISBN_10"), None)
            offers.append({
                "provider": "google_books",
                "listing_id": it.get("id"),
                "title": vi.get("title"),
                "authors": vi.get("authors", []),
                "publisher": vi.get("publisher"),
                "publication_date": vi.get("publishedDate"),
                "description": vi.get("description"),
                "page_count": vi.get("pageCount"),
                "categories": vi.get("categories"),
                "language": vi.get("language"),
                "isbn_13": isbn13,
                "isbn_10": isbn10,
                "currency": None,
                "amount": None,
                "url": vi.get("infoLink") or it.get("selfLink"),
                "source": "metadata",
            })
        return offers


def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _extract_year(text: str) -> Optional[str]:
    m = re.search(r"(18|19|20)\d{2}", text or "")
    return m.group(0) if m else None


class AbeBooksHtmlProvider:
    BASE = "https://www.abebooks.com/servlet/SearchResults"
    async def lookup(self, *, title: Optional[str], authors: List[str], isbn_13: Optional[str], isbn_10: Optional[str],
                    publisher: Optional[str], publication_date: Optional[str]) -> List[Dict[str, Any]]:
        primary_author = authors[0] if authors else None
        year = _extract_year(publication_date or "")
        parts = [p for p in [title, primary_author, year] if p]
        if not parts and not title:
            return []
        keywords = " ".join(parts)
        params = {"kn": keywords, "sortby": "17"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            r = await client.get(self.BASE, params=params)
            r.raise_for_status()
            html = r.text
        # Parse with simple fallbacks (avoid external parser dep for demo): regex-based light parsing
        # For robustness in production, use BeautifulSoup. Here keep single-file minimal deps.
        offers: List[Dict[str, Any]] = []
        # Very loose extraction of blocks containing price and link
        for m in re.finditer(r"<a[^>]+href=\"(?P<href>[^\"]+)\"[^>]*>(?P<title>.*?)</a>[\s\S]*?(?P<price>(?:USD|\$|£|€)\s*[0-9]+(?:[.,][0-9]{2})?)", html, re.IGNORECASE):
            href = m.group("href")
            title_text = re.sub(r"<[^>]+>", " ", m.group("title")).strip()
            price_text = m.group("price")
            amount = None
            currency = None
            pm = re.search(r"([A-Z]{3}|\$|£|€)\s*([0-9]+(?:[.,][0-9]{2})?)", price_text)
            if pm:
                cur, amt = pm.group(1), pm.group(2)
                currency = {"$": "USD", "£": "GBP", "€": "EUR"}.get(cur, cur)
                try:
                    amount = float(amt.replace(",", ""))
                except Exception:
                    amount = None
            offer = {
                "provider": "abebooks",
                "listing_id": href,
                "title": title_text,
                "authors": [primary_author] if primary_author else [],
                "publisher": None,
                "publication_date": publication_date,
                "isbn_13": None,
                "isbn_10": None,
                "currency": currency,
                "amount": amount,
                "url": href if href.startswith("http") else ("https://www.abebooks.com" + href),
                "source": "scrape",
            }
            offers.append(offer)
        # Score basic relevance
        q_title = _norm(title)
        q_author = _norm(primary_author or "")
        q_year = year
        def score(o: Dict[str, Any]) -> float:
            s = 0.0
            if q_title and _norm(o.get("title")) == q_title:
                s += 3.0
            elif q_title and q_title in _norm(o.get("title")):
                s += 1.5
            if q_author and (o.get("authors") and q_author in _norm(o["authors"][0] or "")):
                s += 1.5
            if q_year and o.get("publication_date"):
                oy = _extract_year(o.get("publication_date") or "")
                if oy == q_year:
                    s += 1.0
            return s
        for o in offers:
            o["score"] = score(o)
        offers.sort(key=lambda x: (x.get("score", 0.0), -(x.get("amount") or 0)), reverse=True)
        return offers[:10]


DEFAULT_PROVIDERS = [
    ("google_books", GoogleBooksProvider),
    ("abebooks", AbeBooksHtmlProvider),
]


async def aggregate_offers(*, title: Optional[str], authors: List[str], isbn_13: Optional[str], isbn_10: Optional[str],
                           publisher: Optional[str], publication_date: Optional[str], providers: Optional[List[str]] = None,
                           timeout_seconds: float = 8.0) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    chosen = [p for p in DEFAULT_PROVIDERS if (providers is None or p[0] in providers)]
    tasks = []
    for name, klass in chosen:
        prov = klass()
        tasks.append((name, prov.lookup(title=title, authors=authors, isbn_13=isbn_13, isbn_10=isbn_10,
                                        publisher=publisher, publication_date=publication_date)))
    offers: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    async def run_named(name: str, coro):
        try:
            return name, await asyncio.wait_for(coro, timeout=timeout_seconds)
        except Exception as e:
            return name, e
    results = await asyncio.gather(*(run_named(n, c) for n, c in tasks))
    for name, res in results:
        if isinstance(res, Exception):
            errors[name] = str(res)
        else:
            for o in res:
                o.setdefault("provider", name)
                offers.append(o)
    # de-dup
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for o in offers:
        key = (o.get("provider"), o.get("listing_id"), o.get("isbn_13"), o.get("isbn_10"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(o)
    return uniq, errors


def choose_best_offer(query: Dict[str, Any], offers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not offers:
        return None
    qi13 = (query.get("isbn_13") or "").replace("-", "").replace(" ", "")
    qi10 = (query.get("isbn_10") or "").replace("-", "").replace(" ", "")
    qtitle = (query.get("title") or "").strip().lower()
    for o in offers:
        oi13 = (o.get("isbn_13") or "").replace("-", "").replace(" ", "")
        oi10 = (o.get("isbn_10") or "").replace("-", "").replace(" ", "")
        if qi13 and oi13 == qi13:
            return o
        if qi10 and oi10 == qi10:
            return o
    if qtitle:
        m = next((o for o in offers if (o.get("title") or "").strip().lower() == qtitle), None)
        if m:
            return m
    return offers[0]


def merge_with_offer(src: Dict[str, Any], best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(src)) if src else {}
    ensure_keys = [
        'title','subtitle','authors','publisher','publication_date','isbn_13','isbn_10','asin','edition','binding_type','language','page_count','categories','description','condition_keywords','price'
    ]
    for k in ensure_keys:
        if k not in merged:
            merged[k] = None
    if merged.get('price') is None or not isinstance(merged.get('price'), dict):
        merged['price'] = { 'currency': None, 'amount': None }
    def pick(a, b):
        if a is None: return b
        if isinstance(a, str) and not a.strip(): return b
        if isinstance(a, list) and not a: return b
        return a
    if best:
        merged['title'] = pick(merged.get('title'), best.get('title'))
        merged['subtitle'] = pick(merged.get('subtitle'), best.get('subtitle'))
        merged['authors'] = pick(merged.get('authors'), best.get('authors') if isinstance(best.get('authors'), list) else None)
        merged['publisher'] = pick(merged.get('publisher'), best.get('publisher'))
        merged['publication_date'] = pick(merged.get('publication_date'), best.get('publication_date'))
        merged['isbn_13'] = pick(merged.get('isbn_13'), best.get('isbn_13'))
        merged['isbn_10'] = pick(merged.get('isbn_10'), best.get('isbn_10'))
        merged['description'] = pick(merged.get('description'), best.get('description'))
        merged['page_count'] = pick(merged.get('page_count'), best.get('page_count'))
        merged['categories'] = pick(merged.get('categories'), best.get('categories') if isinstance(best.get('categories'), list) else None)
        merged['language'] = pick(merged.get('language'), best.get('language'))
        merged['info_url'] = best.get('url')
        merged['source_provider'] = best.get('provider')
        if best.get('currency') and best.get('amount') is not None:
            merged['price'] = { 'currency': best.get('currency'), 'amount': best.get('amount') }
    return merged


# -------------------------
# CLI commands
# -------------------------

def cmd_extract(args) -> int:
    extractor = UnifiedExtractor(
        model=args.model,
        ocr_engine=args.ocr_engine,
        use_preprocessing=not args.no_preprocessing,
        crop_for_ocr=args.crop_ocr,
        crop_margin=args.crop_margin,
        edge_crop_percent=args.edge_crop,
        warm_model=not args.no_warm_model,
    )
    if not args.book_dir and not args.images:
        print("Either --book-dir or --images is required", file=sys.stderr)
        return 2
    if args.book_dir:
        metadata = extractor.process_book_dir(args.book_dir, args.ocr_indices)
    else:
        metadata = extractor.extract_from_images(args.images, args.ocr_indices)
    ok, msg = validate_metadata(metadata)
    if not ok:
        metadata["_validation_error"] = msg
    if args.output:
        ensure_output_dir(args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {args.output}")
    else:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_price(args) -> int:
    if not args.input:
        print("--input is required (metadata JSON)", file=sys.stderr)
        return 2
    src = _load_json(args.input)
    async def run():
        offers, errors = await aggregate_offers(
            title=src.get("title"),
            authors=src.get("authors") or [],
            isbn_13=src.get("isbn_13"),
            isbn_10=src.get("isbn_10"),
            publisher=src.get("publisher"),
            publication_date=src.get("publication_date"),
            providers=args.providers,
            timeout_seconds=args.timeout
        )
        best = choose_best_offer(src, offers)
        merged = merge_with_offer(src, best)
        out = { "query": src, "offers": offers, "errors": errors, "best": best, "merged": merged }
        if args.output:
            ensure_output_dir(args.output)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"Saved priced result to {args.output}")
        else:
            print(json.dumps(out, indent=2, ensure_ascii=False))
    asyncio.run(run())
    return 0


def cmd_run(args) -> int:
    temp_out = args.temp_output or None
    if temp_out is None and args.output:
        base_dir = os.path.dirname(os.path.abspath(args.output))
        temp_out = os.path.join(base_dir, "_temp_extracted.json")
    # extract
    eargs = argparse.Namespace(
        model=args.model, ocr_engine=args.ocr_engine, no_preprocessing=args.no_preprocessing,
        crop_ocr=args.crop_ocr, crop_margin=args.crop_margin, edge_crop=args.edge_crop,
        no_warm_model=args.no_warm_model, book_dir=args.book_dir, images=args.images,
        ocr_indices=args.ocr_indices, output=temp_out
    )
    rc = cmd_extract(eargs)
    if rc != 0:
        return rc
    # price
    pargs = argparse.Namespace(
        input=temp_out, output=args.output, providers=args.providers, timeout=args.timeout
    )
    rc = cmd_price(pargs)
    # cleanup temp
    try:
        if temp_out and os.path.exists(temp_out):
            os.remove(temp_out)
    except Exception:
        pass
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified book pricing pipeline (extract + price)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_extract(sp):
        sp.add_argument("--book-dir", type=str, help="Directory with book images")
        sp.add_argument("--images", nargs="+", help="Specific image paths")
        sp.add_argument("--model", type=str, default="gemma3:4b")
        sp.add_argument("--ocr-engine", choices=["easyocr", "tesseract"], default="easyocr")
        sp.add_argument("--no-preprocessing", action="store_true")
        sp.add_argument("--ocr-indices", type=int, nargs="+")
        sp.add_argument("--crop-ocr", action="store_true")
        sp.add_argument("--crop-margin", type=int, default=16)
        sp.add_argument("--edge-crop", type=float, default=0.0)
        sp.add_argument("--no-warm-model", action="store_true")

    sp1 = sub.add_parser("extract", help="Extract metadata from images")
    add_common_extract(sp1)
    sp1.add_argument("--output", "-o", type=str)
    sp1.set_defaults(func=cmd_extract)

    sp2 = sub.add_parser("price", help="Aggregate provider offers and merge")
    sp2.add_argument("--input", "-i", type=str, required=True, help="Input metadata JSON")
    sp2.add_argument("--providers", nargs="+", default=None)
    sp2.add_argument("--timeout", type=float, default=8.0)
    sp2.add_argument("--output", "-o", type=str)
    sp2.set_defaults(func=cmd_price)

    sp3 = sub.add_parser("run", help="Extract then price in one go")
    add_common_extract(sp3)
    sp3.add_argument("--providers", nargs="+", default=None)
    sp3.add_argument("--timeout", type=float, default=8.0)
    sp3.add_argument("--output", "-o", type=str)
    sp3.add_argument("--temp-output", type=str)
    sp3.set_defaults(func=cmd_run)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


