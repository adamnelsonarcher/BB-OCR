#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cache info script for VLM testing.
This script provides information about the Hugging Face cache directory and allows clearing it.
"""

import os
import argparse
import shutil
from huggingface_hub import HfFolder

def get_cache_info():
    """
    Get information about the Hugging Face cache directory.
    
    Returns:
        dict: Dictionary containing cache information
    """
    # Get the cache directory
    cache_dir = HfFolder.get_cache_dir()
    
    # Check if the cache directory exists
    if not os.path.exists(cache_dir):
        return {
            "cache_dir": cache_dir,
            "exists": False,
            "size": 0,
            "models": []
        }
    
    # Get the size of the cache directory
    total_size = 0
    models = []
    
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
        
        # Look for model directories
        if "models--" in dirpath:
            model_dir = os.path.basename(dirpath)
            if model_dir.startswith("models--"):
                model_name = model_dir.replace("models--", "").replace("--", "/")
                models.append(model_name)
    
    # Convert size to MB
    size_mb = total_size / (1024 * 1024)
    
    return {
        "cache_dir": cache_dir,
        "exists": True,
        "size": size_mb,
        "models": list(set(models))  # Remove duplicates
    }

def clear_cache():
    """
    Clear the Hugging Face cache directory.
    """
    # Get the cache directory
    cache_dir = HfFolder.get_cache_dir()
    
    # Check if the cache directory exists
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    # Clear the cache directory
    try:
        shutil.rmtree(cache_dir)
        print(f"Cache directory cleared: {cache_dir}")
    except Exception as e:
        print(f"Error clearing cache directory: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Hugging Face cache information and management")
    parser.add_argument("--clear", action="store_true",
                        help="Clear the Hugging Face cache directory")
    args = parser.parse_args()
    
    if args.clear:
        clear_cache()
    else:
        # Get cache information
        cache_info = get_cache_info()
        
        # Print cache information
        print(f"Hugging Face Cache Directory: {cache_info['cache_dir']}")
        
        if cache_info["exists"]:
            print(f"Cache Size: {cache_info['size']:.2f} MB")
            print(f"Models in Cache: {len(cache_info['models'])}")
            
            # Print list of models
            if cache_info["models"]:
                print("\nList of Models:")
                for model in sorted(cache_info["models"]):
                    print(f"- {model}")
        else:
            print("Cache directory does not exist.")

if __name__ == "__main__":
    main()