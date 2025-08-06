#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaVA-7B test script for VLM testing.
This script tests LLaVA-7B model on book cover images.
"""

import os
import time
import json
import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, CLIPImageProcessor, LlamaTokenizer
from transformers import LlavaProcessor

def run_inference(image_path, prompts):
    """
    Run inference on an image using LLaVA-7B model.
    
    Args:
        image_path (str): Path to the image file
        prompts (list): List of prompts to use for inference
    
    Returns:
        dict: Dictionary containing inference results
    """
    print(f"Processing image: {image_path}")
    
    # Start timing for model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_load_start = time.time()
    
    
    # Initialize the model - using LLaVA-7B
    print("Loading LLaVA-7B model...")
    local_model_path = "./models/llava-7b"
    
    try:
        # First try to load from local path with fixed tokenizer
        print(f"Trying to load from local path: {local_model_path}")
        
        try:
            # Load the image processor, tokenizer, and model separately
            image_processor = CLIPImageProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True
            )
            
            tokenizer = LlamaTokenizer.from_pretrained(
                local_model_path, 
                trust_remote_code=True
            )
            
            # Create the processor from the components
            processor = LlavaProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer
            )
            
            # Load the model from local path
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                use_safetensors=True
            )
            print("Successfully loaded model from local path!")
            
        except Exception as local_error:
            # If local loading fails, fall back to Hugging Face Hub
            print(f"Local loading failed: {str(local_error)}")
            print("Falling back to Hugging Face Hub...")
            
            # Use a known working model from Hugging Face Hub
            model_id = "llava-hf/llava-1.5-7b-hf"
            print(f"Loading model from {model_id}...")
            
            # Load from Hugging Face Hub
            image_processor = CLIPImageProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            tokenizer = LlamaTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True
            )
            
            # Add the image token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if "<image>" not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
            
            # Create the processor from the components
            processor = LlavaProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer
            )
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("Successfully loaded model from Hugging Face Hub!")
            print("To fix your local model files, run: python fix_tokenizer.py")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

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
            # For LLaVA, use specific prompt format
            formatted_prompt = f"<image>\nLook at this book cover and answer: {prompt}"
            
            # Process the image and prompt
            inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(device)
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
            
            # Decode the generated text using tokenizer directly
            generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response - extract the answer part
            if formatted_prompt in generated_text:
                generated_text = generated_text.replace(formatted_prompt, "").strip()
            elif prompt in generated_text:
                # Try to extract the response after the prompt
                start_idx = generated_text.find(prompt) + len(prompt)
                generated_text = generated_text[start_idx:].strip()
            
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
        "model_type": "llava-1.5-7b",
        "timing": timing_info,
        "results": results
    }
    
    # Save the results
    os.makedirs("results/json", exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    output_file = os.path.join("results/json", f"{base_name}_llava_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nTotal processing time: {model_load_time + total_time:.2f} seconds")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="LLaVA-7B test for VLM on book cover images")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file to process")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache models (default: HF cache)")
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