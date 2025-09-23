#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CogVLM test script for VLM testing.
This script tests CogVLM on book cover images.
"""

import os
import time
import json
import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

def run_inference(image_path, prompts):
    """
    Run inference on an image using CogVLM model.
    
    Args:
        image_path (str): Path to the image file
        prompts (list): List of prompts to use for inference
    
    Returns:
        dict: Dictionary containing inference results
    """
    print(f"Processing image: {image_path}")
    
    # Start timing for model loading
    model_load_start = time.time()
    
    # Initialize the model - using CogVLM which is excellent for visual understanding
    print("Loading CogVLM model...")
    model_name = "THUDM/cogvlm-chat-hf"
    
    # Download and set up the model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # End timing for model loading
    model_load_end = time.time()
    model_load_time = model_load_end - model_load_start
    print(f"Model loading time: {model_load_time:.2f} seconds")
    
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
        
        # Create a more detailed prompt for better results
        detailed_prompt = f"Look at this book cover image carefully and answer the following question accurately: {prompt}"
        
        # Process the image and prompt
        inputs = processor(text=detailed_prompt, images=image, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
            )
        
        # Decode the generated text
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response - remove the prompt if present
        if detailed_prompt in generated_text:
            generated_text = generated_text.replace(detailed_prompt, "").strip()
        
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
        "model_type": "cogvlm",
        "timing": timing_info,
        "results": results
    }
    
    # Save the results
    os.makedirs("results/json", exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    output_file = os.path.join("results/json", f"{base_name}_cogvlm_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nTotal processing time: {model_load_time + total_time:.2f} seconds")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="CogVLM test for VLM on a single image")
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