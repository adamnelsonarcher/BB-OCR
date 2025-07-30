#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script for VLM testing.
This script tests basic VLM functionality on a single image.
"""

import os
import time
import json
import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

def run_inference(image_path, prompts):
    """
    Run inference on an image using BLIP model.
    
    Args:
        image_path (str): Path to the image file
        prompts (list): List of prompts to use for inference
    
    Returns:
        dict: Dictionary containing inference results
    """
    print(f"Processing image: {image_path}")
    
    # Start timing for model loading
    model_load_start = time.time()
    
    # Initialize the model - using original BLIP which has better compatibility
    print("Loading BLIP model...")
    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # End timing for model loading
    model_load_end = time.time()
    model_load_time = model_load_end - model_load_start
    print(f"Model loading time: {model_load_time:.2f} seconds")
    print(f"Model loaded on {device}")
    
    # Load the image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None
    
    results = {}
    total_time = 0
    
    # Process each prompt
    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        
        # Start timing
        start_time = time.time()
        
        # Process the image and prompt
        inputs = processor(image, prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(**inputs)
            
        # Decode the generated text
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # End timing
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        
        # Store the result
        prompt_key = prompt.split("?")[0].strip().lower().replace(" ", "_")
        results[prompt_key] = {
            "prompt": prompt,
            "response": generated_text,
            "inference_time": inference_time
        }
        
        print(f"Response: {generated_text}")
        print(f"Inference time: {inference_time:.2f} seconds")
    
    # Add total inference time
    results["total_inference_time"] = total_time
    
    # Add timing information
    timing_info = {
        "model_load_time": model_load_time,
        "inference_time": total_time,
        "total_time": model_load_time + total_time
    }
    
    # Create the final results dictionary
    final_results = {
        "image_path": image_path,
        "model_type": "blip-vqa-base",
        "timing": timing_info,
        "results": results
    }
    
    # Save the results
    os.makedirs("results/json", exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    output_file = os.path.join("results/json", f"{base_name}_blip_simple_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nTotal processing time: {model_load_time + total_time:.2f} seconds")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Simple test for VLM on a single image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file to process")
    args = parser.parse_args()
    
    # Default prompts for book covers
    prompts = [
        "What is the title of this book?",
        "Who is the author of this book?",
        "Who is the publisher of this book?"
    ]
    
    # Run inference
    run_inference(args.image, prompts)

if __name__ == "__main__":
    main()