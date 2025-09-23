#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model setup script for VLM testing.
This script downloads and sets up either BLIP-2 or LLaVA models for book cover metadata extraction.
"""

import os
import argparse
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import logging

# Set logging level to reduce verbosity
logging.set_verbosity_error()

def setup_blip2(model_path=None):
    """
    Set up BLIP-2 model for inference.
    
    Args:
        model_path (str, optional): Path to save the model. If None, uses default cache.
    
    Returns:
        tuple: (processor, model) for BLIP-2
    """
    print("Setting up BLIP-2 model...")
    
    # Use a smaller BLIP-2 model to reduce memory requirements and avoid compatibility issues
    model_name = "Salesforce/blip2-flan-t5-xl"
    
    # Download and set up the model
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"BLIP-2 model loaded successfully on {device}")
    return processor, model

def setup_llava(model_path=None):
    """
    Set up LLaVA model for inference.
    
    Args:
        model_path (str, optional): Path to save the model. If None, uses default cache.
    
    Returns:
        tuple: (processor, model) for LLaVA
    """
    print("Setting up LLaVA model...")
    
    # Use LLaVA-1.5 with 7B parameters (smaller version)
    model_name = "llava-hf/llava-1.5-7b-hf"
    
    # Download and set up the model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print(f"LLaVA model loaded successfully")
    return processor, model

def main():
    parser = argparse.ArgumentParser(description="Set up VLM models for book cover metadata extraction")
    parser.add_argument("--model", type=str, choices=["blip2", "llava"], default="blip2", 
                        help="VLM model to use (default: blip2)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to save the model (default: use cache)")
    args = parser.parse_args()
    
    # Set up the selected model
    if args.model == "blip2":
        processor, model = setup_blip2(args.model_path)
    else:
        processor, model = setup_llava(args.model_path)
    
    print(f"Model setup complete. Using {args.model} model.")

if __name__ == "__main__":
    main() 