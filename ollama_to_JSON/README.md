# 📚 Book Metadata & Pricing Pipeline

## **Overview**

We're building a **book metadata extraction and pricing pipeline**.

The goal:

* Take an **image** of a book (cover, title page, back cover, dust jacket, etc.).
* Use a **vision-language model** (**Ollama + Gemma3:4b**) to extract **structured JSON metadata** similar to an Amazon listing.
* Later, integrate **pricing logic** by comparing against Amazon, Biblio, ABE, Alibris, and other sources.
* Eventually, support **batch ingestion** of hundreds of book images for bulk cataloging and valuation.

---

## **Problem Statement**

Our store needs to:

1. **Catalog books automatically**

   * Extract ISBN, title, author, publisher, publication date, binding, edition, and other attributes.
   * Capture **condition-related keywords** (e.g., "ex-library," "dust jacket clipped," "book club edition").
   * Structure this into **clean JSON** for downstream systems.

2. **Estimate pricing intelligently** *(future step)*

   * Use ISBNs or title/author searches to find comparable listings.
   * Exclude "print-on-demand," damaged, or ex-library copies unless relevant.
   * Match **cover images** and **edition details** for accuracy.
   * Support uniform pricing logic across **Amazon, Biblio, ABE, and Alibris**.

---

## **Phase 1: Metadata Extraction Pipeline**

### **Input**

* A book image (cover, title page, or back cover).

### **Process**

* Send the image to **Gemma3:4b** using Ollama.
* Use a carefully crafted **system prompt** to force strict JSON output.
* Parse and validate the JSON locally.

### **Output**

```json
{
  "title": "string | null",
  "subtitle": "string | null",
  "authors": ["string", "..."] | [],
  "publisher": "string | null",
  "publication_date": "YYYY-MM-DD | YYYY | null",
  "isbn_10": "string | null",
  "isbn_13": "string | null",
  "asin": "string | null",
  "edition": "string | null",
  "binding_type": "string | null",
  "language": "string | null",
  "page_count": "integer | null",
  "categories": ["string", "..."] | [],
  "description": "string | null",
  "condition_keywords": ["string", "..."] | [],
  "price": {
    "currency": "string | null",
    "amount": "float | null"
  }
}
```

---

## **Installation**

### Prerequisites

1. **Python 3.7+**
2. **Ollama** installed and running locally
3. **Gemma3:4b** model pulled in Ollama

### Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running with the Gemma3:4b model:
   ```
   ollama pull gemma3:4b
   ```

---

## **Usage**

### Single Image Processing

To process a single book image:

```bash
python extractor.py --image path/to/book_image.jpg --output book_metadata.json
```

### Multiple Images for One Book

To process multiple images of the same book:

```bash
python extractor.py --image image1.jpg image2.jpg image3.jpg --output book_metadata.json
```

### Process a Book Directory

To process all images in a book directory:

```bash
python extractor.py --book-dir path/to/book_directory --output book_metadata.json
```

### Batch Processing

To process multiple book directories at once:

```bash
python batch_processor.py path/to/books_directory --output-dir path/to/output_directory --output-file all_books.json
```

#### Batch Processing Options

- `--workers N`: Use N worker threads for parallel processing (default: 1)
- `--model MODEL_NAME`: Use a different Ollama model (default: gemma3:4b)
- `--prompt-file PATH`: Use a custom prompt file

---

## **Development Milestones**

* [x] **M1:** Create `prompts/book_metadata_prompt.txt`
* [x] **M2:** Create `extractor.py` to call Ollama with an image
* [x] **M3:** Implement JSON parsing & validation
* [x] **M4:** Add CLI interface for single-image runs
* [x] **M5:** Implement batch processing for multiple books
* [ ] **M6:** Prepare for pricing integration

---

## **Key Decisions**

* **VLM Model**: [Gemma3:4b](https://ollama.ai/library/gemma3) via Ollama
* **Strict JSON Schema**: Required for downstream pricing logic
* **Pricing Integration**: Will be modular and opt-in
* **Batch Support**: Built after single-image pipeline

---

## **Future Work**

* **Phase 2**: Pricing Engine
* **Phase 3**: Bulk Ingestion Optimization
* **Phase 4**: Multi-Platform Sync