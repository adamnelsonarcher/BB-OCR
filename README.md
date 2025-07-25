# Character Recognition Automated Metadata (CRAM)

**Author:** Adam Nelson-Archer  
**Project Start:** June 2025  
**Client:** Becker Books, Houston TX

---

## Overview

The Character Recognition Automated Metadata (CRAM) is a hybrid metadata extraction system designed to streamline the cataloging of books—especially those published before 1970 and lacking standard barcodes or ISBNs. It operates as an augmentation layer for BookTrakker, the commercial inventory management system currently in use by Becker Books.

BBOCR allows a user to take 1–2 images of a book (cover + title page) and automatically extract key metadata such as **title**, **author**, and **publication year**, without requiring manual entry.

The system is designed with future goals in mind: lightweight, locally runnable, minimally dependent on paid APIs, and eventually contributing to a community-maintained metadata database.

---

## System Highlights

- **OCR-based Metadata Extraction** (EasyOCR, Tesseract 5)
- **Custom Heuristics Layer** for intelligent field parsing
- **Search & Enrichment** using local DB + external APIs
- **Human-in-the-Loop Verification** before integration
- **BookTrakker Integration** via CSV or RPA bot automation
- **Label Printing** for physically tagging books post-processing

For a full visual breakdown, see the formal system architecture diagram and swimlane description in  
[`BBOCR_Formal_7-2-25.pdf`](./BBOCR_Formal_7-2-25.pdf)

---

## Project Structure

```plaintext
BB-OCR/
├── gradio_GPT/                # Gradio demo interface
├── ocr_engines/               # EasyOCR, Tesseract tests and preprocessing
├── dataset/                   # Cover/title images and labeled metadata
├── heuristics/                # Rule-based text parsing
├── experiments/               # Evaluation logs, metrics
├── notes/                     # Observations, research notes
└── README.md
```

---

## Current Goals (July 2025)

* [ ] Curate an initial dataset of 300 book cover/title images
* [ ] Benchmark pretrained OCR models (EasyOCR, Tesseract 5)
* [ ] Develop and evaluate a custom heuristic layer for field extraction
* [ ] Begin caching metadata locally with PostgreSQL or SQLite
* [ ] Set up real-world testing pipeline with Gradio interface

---

## Integration Targets

The final output of BBOCR feeds directly into BookTrakker via:

* **CSV Import Pipelines**
* **RPA Automation Layer** using PyAutoGUI or similar
* **Label Printing Support** for physical inventory tagging

---

## Technologies

* **OCR Engines**: EasyOCR, Tesseract 5
* **Preprocessing**: OpenCV
* **Regex + NLP**: Custom Python scripts for intelligent structuring
* **Database**: SQLite or PostgreSQL
* **API Sources**: Google Books, OpenLibrary, AbeBooks (planned fallback)
* **UI/Verification**: Gradio, PyQt (experimental)
