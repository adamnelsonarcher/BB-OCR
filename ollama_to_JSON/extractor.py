#!/usr/bin/env python3
"""
Book Metadata Extractor

This script uses Ollama with Gemma3:4b to extract structured book metadata from images.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import requests
from PIL import Image
import jsonschema

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

class BookMetadataExtractor:
    """Extract book metadata from images using Ollama."""
    
    def __init__(self, model: str = "gemma3:4b", prompt_file: str = None):
        """Initialize the extractor with the specified model and prompt."""
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Load the prompt from file
        if prompt_file is None:
            prompt_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "prompts", 
                "book_metadata_prompt.txt"
            )
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt = f.read()
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def extract_metadata_from_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Extract metadata from multiple book images."""
        # Encode all images
        images = [self.encode_image(img_path) for img_path in image_paths]
        
        # Create the request payload
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": False,
            "images": images
        }
        
        # Send request to Ollama
        response = requests.post(self.ollama_url, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from Ollama API: {response.text}")
        
        # Extract the response
        result = response.json()
        
        # Parse the JSON from the response
        try:
            # The response might contain the JSON string within a larger text
            response_text = result.get("response", "")
            
            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = response_text[json_start:json_end+1]
                metadata = json.loads(json_str)
            else:
                metadata = json.loads(response_text)
                
            # Validate against schema
            jsonschema.validate(instance=metadata, schema=METADATA_SCHEMA)
            return metadata
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON from response: {e}\nResponse: {response_text}")
        except jsonschema.exceptions.ValidationError as e:
            raise Exception(f"JSON validation failed: {e}")
    
    def process_book_directory(self, book_dir: str) -> Dict[str, Any]:
        """Process all images in a book directory."""
        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        
        for file in os.listdir(book_dir):
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_paths.append(os.path.join(book_dir, file))
        
        if not image_paths:
            raise Exception(f"No image files found in {book_dir}")
        
        # Extract metadata from the images
        return self.extract_metadata_from_images(image_paths)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Extract book metadata from images using Ollama")
    parser.add_argument("--book-dir", type=str, help="Directory containing book images")
    parser.add_argument("--image", type=str, nargs="+", help="Path to book image(s)")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model to use")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--prompt-file", type=str, help="Custom prompt file path")
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = BookMetadataExtractor(model=args.model, prompt_file=args.prompt_file)
    
    try:
        # Process based on input type
        if args.book_dir:
            metadata = extractor.process_book_directory(args.book_dir)
        elif args.image:
            metadata = extractor.extract_metadata_from_images(args.image)
        else:
            parser.error("Either --book-dir or --image must be provided")
            return
        
        # Output the metadata
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Metadata saved to {args.output}")
        else:
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
