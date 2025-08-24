# Book Metadata Extractor

A simple tool to extract structured metadata from book images using Ollama.

## Requirements

- Python 3.7+
- Ollama installed and running locally
- Gemma3:4b model pulled in Ollama

## Setup

1. Install dependencies:
   ```
   pip install -r ollama_to_JSON/requirements.txt
   ```

2. Make sure Ollama is running with the Gemma3:4b model:
   ```
   ollama pull gemma3:4b
   ```

## Usage

### Process a single book

The simplest way to process a book is to use the batch file:

```
process_book.bat [BOOK_ID]
```

For example:

```
process_book.bat 1
```

This will:
1. Process all images in the `books/1` directory
2. Extract metadata using Ollama and Gemma3:4b
3. Validate the extracted metadata
4. Save the results to `output/book_1.json`
5. Display a summary of the extracted information

### Advanced Usage

You can also use the Python script directly for more options:

```
python process_book.py [BOOK_ID] --output-dir [OUTPUT_DIR] --model [MODEL_NAME]
```

For example:

```
python process_book.py 1 --output-dir my_results --model llava
```

## Output Format

The extracted metadata is saved as a JSON file with the following structure:

```json
{
  "title": "Book Title",
  "subtitle": "Book Subtitle",
  "authors": ["Author Name"],
  "publisher": "Publisher Name",
  "publication_date": "2023",
  "isbn_10": "1234567890",
  "isbn_13": "9781234567890",
  "asin": "B123456789",
  "edition": "First Edition",
  "binding_type": "Hardcover",
  "language": "English",
  "page_count": 300,
  "categories": ["Fiction", "Mystery"],
  "description": "Book description...",
  "condition_keywords": ["dust jacket", "signed copy"],
  "price": {
    "currency": "USD",
    "amount": 19.99
  }
}
```

## Validation

The script validates the extracted metadata to ensure:
- It conforms to the expected JSON schema
- Required fields are present
- ISBN formats are valid
- Arrays are properly formatted

If validation fails, the metadata is still saved but with an additional `_validation_error` field.