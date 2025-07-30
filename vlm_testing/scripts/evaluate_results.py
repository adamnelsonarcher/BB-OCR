#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for VLM testing.
This script evaluates the results of VLM inference against ground truth data.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from pathlib import Path

def load_results(results_file):
    """
    Load VLM results from a JSON file.
    
    Args:
        results_file (str): Path to the results JSON file
    
    Returns:
        dict: Dictionary containing VLM results
    """
    with open(results_file, 'r') as f:
        return json.load(f)

def load_ground_truth(ground_truth_file):
    """
    Load ground truth data from a JSON file.
    
    Args:
        ground_truth_file (str): Path to the ground truth JSON file
    
    Returns:
        dict: Dictionary containing ground truth data
    """
    with open(ground_truth_file, 'r') as f:
        return json.load(f)

def evaluate_accuracy(results, ground_truth, fuzzy_threshold=80):
    """
    Evaluate the accuracy of VLM results against ground truth data.
    
    Args:
        results (dict): Dictionary containing VLM results
        ground_truth (dict): Dictionary containing ground truth data
        fuzzy_threshold (int): Threshold for fuzzy matching (0-100)
    
    Returns:
        dict: Dictionary containing evaluation results
    """
    evaluation = {
        "total_images": 0,
        "total_fields": 0,
        "exact_matches": 0,
        "fuzzy_matches": 0,
        "accuracy_exact": 0.0,
        "accuracy_fuzzy": 0.0,
        "average_inference_time": 0.0,
        "details": {}
    }
    
    total_inference_time = 0.0
    
    # Map prompt keys to ground truth fields
    field_mapping = {
        "what_is_the_title_of_this_book": "title",
        "who_is_the_author_of_this_book": "author",
        "who_is_the_publisher": "publisher"
    }
    
    # Evaluate each image
    for image_name, image_data in results.items():
        # Skip images with errors
        if "error" in image_data:
            continue
        
        # Get ground truth data for this image
        if image_name not in ground_truth:
            print(f"Warning: No ground truth data for {image_name}")
            continue
        
        gt_data = ground_truth[image_name]
        
        # Initialize image evaluation data
        image_eval = {
            "image_path": image_data["image_path"],
            "inference_time": image_data["results"]["total_inference_time"],
            "fields": {}
        }
        
        # Add to total inference time
        total_inference_time += image_data["results"]["total_inference_time"]
        
        # Evaluate each field
        for prompt_key, field_name in field_mapping.items():
            if prompt_key not in image_data["results"] or field_name not in gt_data:
                continue
            
            vlm_value = image_data["results"][prompt_key]["response"]
            gt_value = gt_data[field_name]
            
            # Calculate match scores
            exact_match = (vlm_value.lower() == gt_value.lower())
            fuzzy_score = fuzz.ratio(vlm_value.lower(), gt_value.lower())
            fuzzy_match = (fuzzy_score >= fuzzy_threshold)
            
            # Update counters
            evaluation["total_fields"] += 1
            if exact_match:
                evaluation["exact_matches"] += 1
            if fuzzy_match:
                evaluation["fuzzy_matches"] += 1
            
            # Store field evaluation data
            image_eval["fields"][field_name] = {
                "vlm_value": vlm_value,
                "gt_value": gt_value,
                "exact_match": exact_match,
                "fuzzy_score": fuzzy_score,
                "fuzzy_match": fuzzy_match
            }
        
        # Store image evaluation data
        evaluation["details"][image_name] = image_eval
        evaluation["total_images"] += 1
    
    # Calculate overall metrics
    if evaluation["total_fields"] > 0:
        evaluation["accuracy_exact"] = evaluation["exact_matches"] / evaluation["total_fields"] * 100
        evaluation["accuracy_fuzzy"] = evaluation["fuzzy_matches"] / evaluation["total_fields"] * 100
    
    if evaluation["total_images"] > 0:
        evaluation["average_inference_time"] = total_inference_time / evaluation["total_images"]
    
    return evaluation

def generate_report(evaluation, output_file=None):
    """
    Generate a report from the evaluation results.
    
    Args:
        evaluation (dict): Dictionary containing evaluation results
        output_file (str, optional): Path to save the report
    
    Returns:
        str: Report text
    """
    # Create a report
    report = []
    report.append("# VLM Evaluation Report")
    report.append("")
    report.append(f"Total images evaluated: {evaluation['total_images']}")
    report.append(f"Total fields evaluated: {evaluation['total_fields']}")
    report.append(f"Exact matches: {evaluation['exact_matches']} ({evaluation['accuracy_exact']:.2f}%)")
    report.append(f"Fuzzy matches: {evaluation['fuzzy_matches']} ({evaluation['accuracy_fuzzy']:.2f}%)")
    report.append(f"Average inference time per image: {evaluation['average_inference_time']:.2f} seconds")
    report.append("")
    
    # Add details for each image
    report.append("## Details by Image")
    report.append("")
    
    for image_name, image_eval in evaluation["details"].items():
        report.append(f"### {image_name}")
        report.append(f"- Inference time: {image_eval['inference_time']:.2f} seconds")
        report.append("")
        
        # Add field details
        for field_name, field_eval in image_eval["fields"].items():
            match_status = "✓" if field_eval["exact_match"] else "✗"
            report.append(f"- {field_name}: {match_status}")
            report.append(f"  - VLM: \"{field_eval['vlm_value']}\"")
            report.append(f"  - GT:  \"{field_eval['gt_value']}\"")
            report.append(f"  - Fuzzy score: {field_eval['fuzzy_score']}")
            report.append("")
    
    # Join the report lines
    report_text = "\n".join(report)
    
    # Save the report if an output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
    
    return report_text

def generate_comparison_chart(evaluations, output_file=None):
    """
    Generate a comparison chart for multiple VLM models.
    
    Args:
        evaluations (dict): Dictionary containing evaluation results for multiple models
        output_file (str, optional): Path to save the chart
    """
    # Extract data for the chart
    models = list(evaluations.keys())
    exact_accuracy = [evaluations[model]["accuracy_exact"] for model in models]
    fuzzy_accuracy = [evaluations[model]["accuracy_fuzzy"] for model in models]
    inference_times = [evaluations[model]["average_inference_time"] for model in models]
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        "Model": models,
        "Exact Accuracy (%)": exact_accuracy,
        "Fuzzy Accuracy (%)": fuzzy_accuracy,
        "Avg. Inference Time (s)": inference_times
    })
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot accuracy
    df.plot(x="Model", y=["Exact Accuracy (%)", "Fuzzy Accuracy (%)"], kind="bar", ax=ax1)
    ax1.set_title("Accuracy Comparison")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    
    # Plot inference time
    df.plot(x="Model", y="Avg. Inference Time (s)", kind="bar", ax=ax2, color="green")
    ax2.set_title("Inference Time Comparison")
    ax2.set_ylabel("Average Inference Time (s)")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart if an output file is specified
    if output_file:
        plt.savefig(output_file)
        print(f"Comparison chart saved to {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM results against ground truth data")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to the VLM results JSON file")
    parser.add_argument("--ground_truth_file", type=str, default="../data/ground_truth.json",
                        help="Path to the ground truth JSON file (default: ../data/ground_truth.json)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the evaluation report (default: None)")
    parser.add_argument("--fuzzy_threshold", type=int, default=80,
                        help="Threshold for fuzzy matching (0-100, default: 80)")
    args = parser.parse_args()
    
    # Get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    results_file = os.path.join(project_dir, args.results_file.lstrip("../"))
    ground_truth_file = os.path.join(project_dir, args.ground_truth_file.lstrip("../"))
    
    if args.output_file:
        output_file = os.path.join(project_dir, args.output_file.lstrip("../"))
    else:
        output_file = None
    
    # Load the results and ground truth data
    results = load_results(results_file)
    ground_truth = load_ground_truth(ground_truth_file)
    
    # Evaluate the results
    evaluation = evaluate_accuracy(results, ground_truth, args.fuzzy_threshold)
    
    # Generate and print the report
    report = generate_report(evaluation, output_file)
    print(report)

if __name__ == "__main__":
    main() 