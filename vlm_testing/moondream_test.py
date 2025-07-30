#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Moondream test script for VLM testing.
This script tests the Moondream model on book cover images.
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
    Run inference on an image using Moondream model.
    
    Args:
        image_path (str): Path to the image file
        prompts (list): List of prompts to use for inference
    
    Returns:
        dict: Dictionary containing inference results
    """
    print(f"Processing image: {image_path}")
    
    # Start timing for model loading
    model_load_start = time.time()
    
    # Initialize the model - using Moondream which is a lightweight but powerful VLM
    print("Loading Moondream model...")
    model_name = "vikhyatk/moondream2"
    
    # Download and set up the model
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
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
            # Process the image
            image_embeds = processor.image_processor(image, return_tensors="pt").to(device)
            
            # Encode the image
            with torch.no_grad():
                image_output = model.vision_encoder(**image_embeds)
                image_embeds = image_output.last_hidden_state
            
            # Process the prompt with the image
            inputs = processor.text_processor(
                f"<image>\n{prompt}",
                return_tensors="pt",
                add_special_tokens=False
            ).to(device)
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    image_embeds=image_embeds,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.1,
                )
            
            # Decode the generated text
            generated_text = processor.text_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response - remove the prompt if present
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
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
        "model_type": "moondream2",
        "timing": timing_info,
        "results": results
    }
    
    # Save the results
    os.makedirs("results/json", exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    output_file = os.path.join("results/json", f"{base_name}_moondream_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nTotal processing time: {model_load_time + total_time:.2f} seconds")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Moondream test for VLM on book cover images")
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