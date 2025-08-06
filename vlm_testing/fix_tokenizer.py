#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix the tokenizer files in the local LLaVA model directory.
This script downloads fresh tokenizer files from Hugging Face and saves them to the local model directory.
"""

import os
import shutil
import sys
import subprocess
import pkg_resources

def check_and_install_dependencies():
    """Check if required packages are installed and install them if not."""
    required_packages = ['transformers', 'sentencepiece', 'torch', 'pillow']
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"✓ {package} is already installed")
        except pkg_resources.DistributionNotFound:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} has been installed")

# Check and install dependencies
print("Checking dependencies...")
check_and_install_dependencies()

# Now import the required packages
from transformers import LlamaTokenizer, CLIPImageProcessor, LlavaProcessor

def fix_tokenizer(local_model_path="./models/llava-7b", source_model_id="llava-hf/llava-1.5-7b-hf"):
    """
    Fix the tokenizer files in the local model directory by downloading fresh files from Hugging Face.
    
    Args:
        local_model_path (str): Path to the local model directory
        source_model_id (str): Hugging Face model ID to download tokenizer from
    """
    print(f"Fixing tokenizer files in {local_model_path}")
    print(f"Downloading fresh tokenizer files from {source_model_id}")
    
    # Create a temporary directory to download the tokenizer
    temp_dir = "./temp_tokenizer"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Download the tokenizer and image processor
        print("Downloading tokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        
        print("Downloading image processor...")
        image_processor = CLIPImageProcessor.from_pretrained(source_model_id, trust_remote_code=True)
        image_processor.save_pretrained(temp_dir)
        
        # Create and save processor config
        print("Creating processor...")
        processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(temp_dir)
        
        # Copy the tokenizer files to the local model directory
        print(f"Copying tokenizer files to {local_model_path}")
        tokenizer_files = [
            "tokenizer.json", 
            "tokenizer.model", 
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "preprocessor_config.json",
            "processor_config.json"
        ]
        
        for file in tokenizer_files:
            src_file = os.path.join(temp_dir, file)
            dst_file = os.path.join(local_model_path, file)
            
            if os.path.exists(src_file):
                print(f"Copying {file}...")
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: {file} not found in downloaded files")
        
        print("Tokenizer files successfully updated!")
        print("\nYou can now run your LLaVA model with the fixed tokenizer files.")
        print("Use the following code to load the model:")
        print("\nfrom transformers import AutoModelForCausalLM, LlamaTokenizer, CLIPImageProcessor, LlavaProcessor")
        print("model_path = './models/llava-7b'")
        print("tokenizer = LlamaTokenizer.from_pretrained(model_path)")
        print("image_processor = CLIPImageProcessor.from_pretrained(model_path)")
        print("processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)")
        print("model = AutoModelForCausalLM.from_pretrained(model_path)")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have the required dependencies installed:")
        print("   pip install transformers sentencepiece torch pillow")
        print("2. Check if you have write permissions to the model directory")
        print("3. If the error persists, try running with administrator privileges")
        print("4. Alternatively, use the Hugging Face Hub directly in your code:")
        print("   model_id = 'llava-hf/llava-1.5-7b-hf'")
        print("   processor = AutoProcessor.from_pretrained(model_id)")
        print("   model = AutoModelForCausalLM.from_pretrained(model_id)")
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix tokenizer files for LLaVA model")
    parser.add_argument("--model_path", type=str, default="./models/llava-7b",
                        help="Path to the local model directory")
    parser.add_argument("--source_model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Hugging Face model ID to download tokenizer from")
    
    args = parser.parse_args()
    
    fix_tokenizer(args.model_path, args.source_model)