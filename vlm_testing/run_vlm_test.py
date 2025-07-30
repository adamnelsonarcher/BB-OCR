#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for VLM testing pipeline.
This script runs the VLM testing pipeline for book cover metadata extraction.
"""

import os
import argparse
import subprocess
import time
from pathlib import Path

def run_command(command, description=None):
    """
    Run a command and print its output.
    
    Args:
        command (list): Command to run
        description (str, optional): Description of the command
    
    Returns:
        int: Return code of the command
    """
    if description:
        print(f"\n=== {description} ===\n")
    
    print(f"Running: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Run VLM testing on book cover images")
    parser.add_argument("--model", type=str, choices=["blip2", "llava"], default="blip2",
                        help="VLM model to use (default: blip2)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image file to process (if not specified, process all images in data/images)")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation step")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation step")
    args = parser.parse_args()
    
    # Get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(script_dir, "scripts")
    
    # Step 1: Data preparation (if not skipped and no specific image is provided)
    if not args.skip_data_prep and not args.image:
        data_prep_cmd = ["python", os.path.join(scripts_dir, "prepare_data.py")]
        if run_command(data_prep_cmd, "Data Preparation") != 0:
            print("Data preparation failed. Exiting.")
            return 1
    
    # Step 2: Inference
    if args.image:
        # Process a single image
        inference_cmd = [
            "python", 
            os.path.join(scripts_dir, "run_inference.py"), 
            "--model", args.model,
            "--image", args.image
        ]
        description = f"Running Inference on {os.path.basename(args.image)} with {args.model.upper()}"
    else:
        # Process all images in the data directory
        inference_cmd = [
            "python", 
            os.path.join(scripts_dir, "run_inference.py"), 
            "--model", args.model
        ]
        description = f"Running Inference on all images with {args.model.upper()}"
    
    if run_command(inference_cmd, description) != 0:
        print("Inference failed. Exiting.")
        return 1
    
    # Step 3: Evaluation (if not skipped)
    if not args.skip_evaluation:
        results_file = os.path.join(script_dir, "results", "json", f"{args.model}_all_results.json")
        ground_truth_file = os.path.join(script_dir, "data", "ground_truth.json")
        output_file = os.path.join(script_dir, "results", f"{args.model}_evaluation_report.md")
        
        # Check if the results file exists
        if os.path.exists(results_file):
            eval_cmd = [
                "python", 
                os.path.join(scripts_dir, "evaluate_results.py"),
                "--results_file", results_file,
                "--ground_truth_file", ground_truth_file,
                "--output_file", output_file
            ]
            
            if run_command(eval_cmd, "Evaluation") != 0:
                print("Evaluation failed. Exiting.")
                return 1
        else:
            print(f"Results file not found: {results_file}")
            print("Skipping evaluation step.")
    
    print("\n=== VLM Testing Pipeline Completed Successfully ===\n")
    
    # Print summary
    if args.image:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        result_file = os.path.join(script_dir, "results", "json", f"{base_name}_{args.model}_results.json")
        print(f"Results saved to: {result_file}")
    else:
        all_results_file = os.path.join(script_dir, "results", "json", f"{args.model}_all_results.json")
        print(f"All results saved to: {all_results_file}")
        
        if not args.skip_evaluation:
            print(f"Evaluation report saved to: {os.path.join(script_dir, 'results', f'{args.model}_evaluation_report.md')}")
    
    return 0

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    exit(exit_code) 