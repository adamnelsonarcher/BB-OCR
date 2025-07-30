#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLIP-2 test script for VLM testing.
This script tests BLIP-2 with a smaller backbone on book cover images.
"""

import os
import time
import json
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def run_inference(image_path, prompts):
    """
    Run inference on an image using BLIP-2 model with a smaller backbone.
    
    Args:
        image_path (str): Path to the image file
        prompts (list): List of prompts to use for inference
    
    Returns:
        dict: Dictionary containing inference results
    """
    print(f"Processing image: {image_path}")
    
    # Start timing for model loading
    model_load_start = time.time()
    
    # Initialize the model - using BLIP-2 with a smaller backbone
    print("Loading BLIP-2 model...")
    
    try:
        # Using Salesforce's BLIP-2 with FLAN-T5 XL backbone
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        model_name = "Salesforce/blip2-flan-t5-xl"
        processor = Blip2Processor.from_pretrained(model_name, use_fast=False)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"Using model: {model_name}")
        print(f"Model loaded on {device}")
        
    except Exception as e:
        print(f"Error loading BLIP-2 model: {str(e)}")
        print("Trying alternative model...")
        
        # Fallback to a more compatible model
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            model_name = "microsoft/git-base"
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(model_name)
            
            # Move model to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            
            print(f"Using fallback model: {model_name}")
            print(f"Model loaded on {device}")
            
        except Exception as e:
            print(f"Error loading fallback model: {str(e)}")
            return None
    
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
        
        try:
            # Process the image and prompt
            if "blip2" in model_name.lower():
                # BLIP-2 specific processing - use question answering format
                # For BLIP-2, we need to use a specific format for VQA tasks
                inputs = processor(image, f"Question: {prompt} Answer:", return_tensors="pt").to(device)
                
                # Generate text
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,  # Generate new tokens, not including prompt
                        num_beams=5,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,  # Discourage repetition
                    )
                
                # Decode the generated text
                generated_text = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the response - remove the prompt part
                if "Answer:" in generated_text:
                    generated_text = generated_text.split("Answer:")[1].strip()
                elif prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                
            else:
                # GIT model specific processing
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                
                # Generate text
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=50,
                    )
                
                # Decode the generated text
                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            generated_text = f"Error: {str(e)}"
        
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
        "model_type": model_name,
        "timing": timing_info,
        "results": results
    }
    
    # Save the results
    os.makedirs("results/json", exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    output_file = os.path.join("results/json", f"{base_name}_blip2_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nTotal processing time: {model_load_time + total_time:.2f} seconds")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="BLIP-2 test for VLM on a single image")
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