#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for VLM testing.
This script runs all available VLM models on a single image and compares the results.
"""

import os
import time
import json
import argparse
import subprocess
import pandas as pd
from tabulate import tabulate

def run_model(model_script, image_path):
    """
    Run a model script on an image.
    
    Args:
        model_script (str): Path to the model script
        image_path (str): Path to the image file
    
    Returns:
        int: Return code of the subprocess
    """
    print(f"\n{'='*80}")
    print(f"Running {os.path.basename(model_script)} on {os.path.basename(image_path)}")
    print(f"{'='*80}\n")
    
    cmd = ["python", model_script, "--image", image_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def collect_results(image_path):
    """
    Collect results from all model runs on an image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        dict: Dictionary containing collected results
    """
    base_name = os.path.basename(image_path).split('.')[0]
    results_dir = os.path.join("results", "json")
    
    # List of possible result files
    result_files = {
        "blip_simple": f"{base_name}_blip_simple_results.json",
        "blip2_opt": f"{base_name}_blip2_opt_results.json",
        "llava": f"{base_name}_llava_results.json",
        "git": f"{base_name}_git_results.json",
        "pix2struct": f"{base_name}_pix2struct_results.json",
        "moondream": f"{base_name}_moondream_results.json",
        "vqa": f"{base_name}_vqa_results.json"
    }
    
    collected_results = {}
    
    # Collect results from each file if it exists
    for model_name, result_file in result_files.items():
        file_path = os.path.join(results_dir, result_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                collected_results[model_name] = data
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    return collected_results

def generate_comparison_table(collected_results):
    """
    Generate a comparison table from collected results.
    
    Args:
        collected_results (dict): Dictionary containing collected results
    
    Returns:
        str: Formatted comparison table
    """
    # Extract data for the table
    data = []
    
    # Define the prompts to look for
    prompts = [
        "what_is_the_title_of_this_book",
        "who_is_the_author_of_this_book",
        "who_is_the_publisher"
    ]
    
    # Extract data for each model
    for model_name, results in collected_results.items():
        model_type = results.get("model_type", model_name)
        timing = results.get("timing", {})
        model_results = results.get("results", {})
        
        # Extract responses for each prompt
        responses = {}
        for prompt in prompts:
            if prompt in model_results:
                responses[prompt] = model_results[prompt].get("response", "N/A")
            else:
                responses[prompt] = "N/A"
        
        # Add row to data
        data.append({
            "Model": model_type,
            "Load Time (s)": timing.get("model_load_time", "N/A"),
            "Inference Time (s)": timing.get("inference_time", "N/A"),
            "Total Time (s)": timing.get("total_time", "N/A"),
            "Title": responses.get("what_is_the_title_of_this_book", "N/A"),
            "Author": responses.get("who_is_the_author_of_this_book", "N/A"),
            "Publisher": responses.get("who_is_the_publisher", "N/A")
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format the table
    table = tabulate(df, headers="keys", tablefmt="grid", showindex=False)
    return table

def save_comparison_report(image_path, collected_results):
    """
    Save a comparison report to a file.
    
    Args:
        image_path (str): Path to the image file
        collected_results (dict): Dictionary containing collected results
    """
    base_name = os.path.basename(image_path).split('.')[0]
    output_file = os.path.join("results", f"{base_name}_comparison_report.md")
    
    # Generate the report
    report = []
    report.append(f"# VLM Comparison Report for {base_name}")
    report.append("")
    report.append(f"Image: {image_path}")
    report.append("")
    report.append("## Comparison Table")
    report.append("")
    
    # Add the comparison table
    table = generate_comparison_table(collected_results)
    report.append("```")
    report.append(table)
    report.append("```")
    report.append("")
    
    # Add detailed results for each model
    report.append("## Detailed Results")
    report.append("")
    
    for model_name, results in collected_results.items():
        model_type = results.get("model_type", model_name)
        report.append(f"### {model_type}")
        report.append("")
        
        # Add timing information
        timing = results.get("timing", {})
        report.append("#### Timing")
        report.append("")
        report.append(f"- Model Load Time: {timing.get('model_load_time', 'N/A'):.2f} seconds")
        report.append(f"- Inference Time: {timing.get('inference_time', 'N/A'):.2f} seconds")
        report.append(f"- Total Time: {timing.get('total_time', 'N/A'):.2f} seconds")
        report.append("")
        
        # Add results for each prompt
        model_results = results.get("results", {})
        report.append("#### Results")
        report.append("")
        
        for prompt_key, prompt_data in model_results.items():
            if prompt_key != "total_inference_time":
                report.append(f"**Prompt:** {prompt_data.get('prompt', prompt_key)}")
                report.append(f"**Response:** {prompt_data.get('response', 'N/A')}")
                report.append(f"**Time:** {prompt_data.get('inference_time', 'N/A'):.2f} seconds")
                report.append("")
    
    # Write the report to a file
    with open(output_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"\nComparison report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark all VLM models on a single image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file to process")
    parser.add_argument("--models", type=str, nargs="+", default=["all"],
                        help="List of models to run (default: all)")
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results/json", exist_ok=True)
    
    # List of available model scripts
    available_models = {
        "blip_simple": "simple_test.py",
        "blip2_opt": "blip2_opt_test.py",
        "llava": "llava_test.py",
        "git": "git_test.py",
        "pix2struct": "pix2struct_test.py",
        "moondream": "moondream_test.py"
    }
    
    # Determine which models to run
    if "all" in args.models:
        models_to_run = list(available_models.keys())
    else:
        models_to_run = [model for model in args.models if model in available_models]
    
    # Run each model
    for model in models_to_run:
        model_script = available_models[model]
        run_model(model_script, args.image)
    
    # Collect results
    collected_results = collect_results(args.image)
    
    # Print comparison table
    if collected_results:
        table = generate_comparison_table(collected_results)
        print("\n" + table)
        
        # Save comparison report
        save_comparison_report(args.image, collected_results)
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()