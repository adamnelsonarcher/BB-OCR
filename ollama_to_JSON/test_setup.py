#!/usr/bin/env python3
"""
Test script to verify the setup of the book metadata extraction pipeline.
"""

import os
import sys
import importlib.util

def check_module(module_name):
    """Check if a module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Check the setup and dependencies."""
    print("Checking setup for Book Metadata Extraction Pipeline...")
    
    # Check required Python modules
    required_modules = ["requests", "PIL", "jsonschema", "tqdm"]
    missing_modules = []
    
    for module in required_modules:
        if not check_module(module):
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install -r requirements.txt")
    else:
        print("All required Python modules are installed.")
    
    # Check if the prompt file exists
    prompt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "prompts", 
        "book_metadata_prompt.txt"
    )
    
    if os.path.exists(prompt_path):
        print("Prompt file found.")
    else:
        print(f"Prompt file not found at {prompt_path}")
    
    # Check Ollama connection
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("Ollama server is running.")
            
            # Check if Gemma3:4b is available
            models = response.json().get("models", [])
            gemma_available = any(model.get("name", "").startswith("gemma3:4b") for model in models)
            
            if gemma_available:
                print("Gemma3:4b model is available in Ollama.")
            else:
                print("Gemma3:4b model not found. Please pull it using: ollama pull gemma3:4b")
        else:
            print("Could not connect to Ollama server.")
            print("Please make sure Ollama is running on http://localhost:11434")
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        print("Please make sure Ollama is installed and running.")
    
    print("\nSetup check complete.")


if __name__ == "__main__":
    main()
