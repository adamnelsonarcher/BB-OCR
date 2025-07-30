#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for VLM testing pipeline.
This script runs the entire VLM testing pipeline from data preparation to evaluation.
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
    parser = argparse.ArgumentParser(description="Run the entire VLM testing pipeline")
    parser.add_argument("--model", type=str, choices=["blip2", "llava"], default="blip2",
                        help="VLM model to use (default: blip2)")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation step")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference step")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation step")
    args = parser.parse_args()
    
    # Get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(script_dir, "scripts")
    
    # Step 1: Data preparation
    if not args.skip_data_prep:
        data_prep_cmd = ["python", os.path.join(scripts_dir, "prepare_data.py")]
        if run_command(data_prep_cmd, "Data Preparation") != 0:
            print("Data preparation failed. Exiting.")
            return 1
    
    # Step 2: Inference
    if not args.skip_inference:
        inference_cmd = ["python", os.path.join(scripts_dir, "run_inference.py"), "--model", args.model]
        if run_command(inference_cmd, f"Running Inference with {args.model.upper()}") != 0:
            print("Inference failed. Exiting.")
            return 1
    
    # Step 3: Evaluation
    if not args.skip_evaluation:
        results_file = os.path.join(script_dir, "results", "json", f"{args.model}_all_results.json")
        ground_truth_file = os.path.join(script_dir, "data", "ground_truth.json")
        output_file = os.path.join(script_dir, "results", f"{args.model}_evaluation_report.md")
        
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
    
    print("\n=== VLM Testing Pipeline Completed Successfully ===\n")
    
    # Print summary
    if not args.skip_evaluation:
        print(f"Evaluation report saved to: {os.path.join(script_dir, 'results', f'{args.model}_evaluation_report.md')}")
    
    return 0

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    exit(exit_code) 