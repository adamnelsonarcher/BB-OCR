#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for VLM testing.
This script runs inference on book cover images using BLIP-2 or LLaVA models.
"""

import os
import json
import time
import argparse
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model setup functions
from model_setup import setup_blip2, setup_llava

def run_blip2_inference(processor, model, image_path, prompts):
    """
    Run inference on an image using BLIP-2 model.
    
    Args:
        processor: BLIP-2 processor
        model: BLIP-2 model
        image_path (str): Path to the image file
        prompts (list): List of prompts to use for inference
    
    Returns:
        dict: Dictionary containing inference results
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    results = {}
    total_time = 0
    
    # Process each prompt
    for prompt in prompts:
        # Start timing
        start_time = time.time()
        
        # Process the image and prompt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                min_length=1,
                top_p=0.9,
            )
        
        # Decode the generated text
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
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
    
    # Add total inference time
    results["total_inference_time"] = total_time
    
    return results

def run_llava_inference(processor, model, image_path, prompts):
    """
    Run inference on an image using LLaVA model.
    
    Args:
        processor: LLaVA processor
        model: LLaVA model
        image_path (str): Path to the image file
        prompts (list): List of prompts to use for inference
    
    Returns:
        dict: Dictionary containing inference results
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    results = {}
    total_time = 0
    
    # Process each prompt
    for prompt in prompts:
        # Start timing
        start_time = time.time()
        
        # Process the image and prompt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.2,
                do_sample=True,
            )
        
        # Decode the generated text
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        # Extract the actual response (remove the prompt)
        if "<answer>" in generated_text.lower():
            # Extract text between <answer> and </answer> if present
            answer_start = generated_text.lower().find("<answer>") + len("<answer>")
            answer_end = generated_text.lower().find("</answer>")
            if answer_end > answer_start:
                generated_text = generated_text[answer_start:answer_end].strip()
        else:
            # Try to extract the response after the prompt
            prompt_parts = prompt.split("?")
            if len(prompt_parts) > 1 and prompt_parts[0] in generated_text:
                generated_text = generated_text[generated_text.find(prompt_parts[0]) + len(prompt_parts[0]):].strip()
                if generated_text.startswith("?"):
                    generated_text = generated_text[1:].strip()
        
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
    
    # Add total inference time
    results["total_inference_time"] = total_time
    
    return results

def process_images(model_type, image_dir, output_dir, prompts=None):
    """
    Process all images in the directory using the specified model.
    
    Args:
        model_type (str): Type of model to use ("blip2" or "llava")
        image_dir (str): Directory containing images to process
        output_dir (str): Directory to save results
        prompts (list, optional): List of prompts to use for inference
    """
    # Default prompts if not provided
    if prompts is None:
        prompts = [
            "What is the title of this book?",
            "Who is the author of this book?",
            "Who is the publisher of this book?"
        ]
    
    # Set up the model
    if model_type == "blip2":
        processor, model = setup_blip2()
        inference_func = run_blip2_inference
    else:
        processor, model = setup_llava()
        inference_func = run_llava_inference
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process each image
    results = {}
    for image_file in tqdm(image_files, desc=f"Processing images with {model_type}"):
        image_path = os.path.join(image_dir, image_file)
        
        # Run inference
        try:
            image_results = inference_func(processor, model, image_path, prompts)
            
            # Store the results
            results[image_file] = {
                "image_path": image_path,
                "results": image_results
            }
            
            # Save the results for this image
            image_output_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_{model_type}_results.json")
            with open(image_output_file, 'w') as f:
                json.dump(results[image_file], f, indent=2)
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            results[image_file] = {
                "image_path": image_path,
                "error": str(e)
            }
    
    # Save all results to a single file
    output_file = os.path.join(output_dir, f"{model_type}_all_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference on book cover images using VLM models")
    parser.add_argument("--model", type=str, choices=["blip2", "llava"], default="blip2",
                        help="VLM model to use (default: blip2)")
    parser.add_argument("--image_dir", type=str, default="../data/images",
                        help="Directory containing images to process (default: ../data/images)")
    parser.add_argument("--output_dir", type=str, default="../results/json",
                        help="Directory to save results (default: ../results/json)")
    args = parser.parse_args()
    
    # Get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    image_dir = os.path.join(project_dir, args.image_dir.lstrip("../"))
    output_dir = os.path.join(project_dir, args.output_dir.lstrip("../"))
    
    # Process the images
    process_images(args.model, image_dir, output_dir)

if __name__ == "__main__":
    main() 