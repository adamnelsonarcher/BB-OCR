#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparison script for VLM testing.
This script compares VLM results with OCR results for book cover metadata extraction.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_vlm_results(vlm_results_file):
    """
    Load VLM results from a JSON file.
    
    Args:
        vlm_results_file (str): Path to the VLM results JSON file
    
    Returns:
        dict: Dictionary containing VLM results
    """
    with open(vlm_results_file, 'r') as f:
        return json.load(f)

def load_ocr_results(ocr_dir):
    """
    Load OCR results from JSON files in the directory.
    
    Args:
        ocr_dir (str): Path to the directory containing OCR result JSON files
    
    Returns:
        dict: Dictionary containing OCR results
    """
    ocr_results = {}
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(ocr_dir) if f.lower().endswith('.json')]
    
    # Load each JSON file
    for json_file in json_files:
        with open(os.path.join(ocr_dir, json_file), 'r') as f:
            data = json.load(f)
            
            # Extract the image name from the file name
            image_name = json_file.replace('ocr_comparison_', '').replace('.json', '')
            
            # Store the OCR results
            ocr_results[f"{image_name}.png"] = data
    
    return ocr_results

def extract_ocr_metadata(ocr_results):
    """
    Extract metadata from OCR results.
    
    Args:
        ocr_results (dict): Dictionary containing OCR results
    
    Returns:
        dict: Dictionary containing extracted metadata
    """
    metadata = {}
    
    # Process each image
    for image_name, image_data in ocr_results.items():
        # Initialize metadata for this image
        metadata[image_name] = {
            "easyocr": {
                "title": None,
                "author": None,
                "publisher": None,
                "processing_time": None
            },
            "tesseract": {
                "title": None,
                "author": None,
                "publisher": None,
                "processing_time": None
            }
        }
        
        # Extract metadata from EasyOCR results
        if "easyocr" in image_data:
            # Processing time
            metadata[image_name]["easyocr"]["processing_time"] = image_data["easyocr"].get("processing_time", 0)
            
            # Metadata
            if "book_metadata" in image_data["easyocr"]:
                metadata[image_name]["easyocr"]["title"] = image_data["easyocr"]["book_metadata"].get("title", None)
                metadata[image_name]["easyocr"]["author"] = image_data["easyocr"]["book_metadata"].get("author", None)
                metadata[image_name]["easyocr"]["publisher"] = image_data["easyocr"]["book_metadata"].get("publisher", None)
        
        # Extract metadata from Tesseract results
        if "tesseract" in image_data:
            # Processing time
            metadata[image_name]["tesseract"]["processing_time"] = image_data["tesseract"].get("processing_time", 0)
            
            # Metadata
            if "book_metadata" in image_data["tesseract"]:
                metadata[image_name]["tesseract"]["title"] = image_data["tesseract"]["book_metadata"].get("title", None)
                metadata[image_name]["tesseract"]["author"] = image_data["tesseract"]["book_metadata"].get("author", None)
                metadata[image_name]["tesseract"]["publisher"] = image_data["tesseract"]["book_metadata"].get("publisher", None)
    
    return metadata

def extract_vlm_metadata(vlm_results):
    """
    Extract metadata from VLM results.
    
    Args:
        vlm_results (dict): Dictionary containing VLM results
    
    Returns:
        dict: Dictionary containing extracted metadata
    """
    metadata = {}
    
    # Process each image
    for image_name, image_data in vlm_results.items():
        # Skip images with errors
        if "error" in image_data:
            continue
        
        # Initialize metadata for this image
        metadata[image_name] = {
            "title": None,
            "author": None,
            "publisher": None,
            "processing_time": None
        }
        
        # Extract metadata from VLM results
        results = image_data.get("results", {})
        
        # Title
        if "what_is_the_title_of_this_book" in results:
            metadata[image_name]["title"] = results["what_is_the_title_of_this_book"].get("response", None)
        
        # Author
        if "who_is_the_author_of_this_book" in results:
            metadata[image_name]["author"] = results["who_is_the_author_of_this_book"].get("response", None)
        
        # Publisher
        if "who_is_the_publisher" in results:
            metadata[image_name]["publisher"] = results["who_is_the_publisher"].get("response", None)
        
        # Processing time
        metadata[image_name]["processing_time"] = results.get("total_inference_time", 0)
    
    return metadata

def compare_results(vlm_metadata, ocr_metadata, ground_truth):
    """
    Compare VLM and OCR results against ground truth.
    
    Args:
        vlm_metadata (dict): Dictionary containing VLM metadata
        ocr_metadata (dict): Dictionary containing OCR metadata
        ground_truth (dict): Dictionary containing ground truth data
    
    Returns:
        dict: Dictionary containing comparison results
    """
    comparison = {
        "vlm": {
            "total_fields": 0,
            "correct_fields": 0,
            "accuracy": 0.0,
            "average_time": 0.0
        },
        "easyocr": {
            "total_fields": 0,
            "correct_fields": 0,
            "accuracy": 0.0,
            "average_time": 0.0
        },
        "tesseract": {
            "total_fields": 0,
            "correct_fields": 0,
            "accuracy": 0.0,
            "average_time": 0.0
        },
        "details": {}
    }
    
    # Fields to compare
    fields = ["title", "author", "publisher"]
    
    # Process each image
    for image_name in set(vlm_metadata.keys()) & set(ocr_metadata.keys()) & set(ground_truth.keys()):
        # Initialize comparison details for this image
        comparison["details"][image_name] = {
            "vlm": {"correct": 0, "total": 0, "time": 0},
            "easyocr": {"correct": 0, "total": 0, "time": 0},
            "tesseract": {"correct": 0, "total": 0, "time": 0}
        }
        
        # Compare VLM results
        vlm_data = vlm_metadata[image_name]
        comparison["details"][image_name]["vlm"]["time"] = vlm_data["processing_time"]
        comparison["vlm"]["average_time"] += vlm_data["processing_time"]
        
        for field in fields:
            if vlm_data[field] is not None:
                comparison["vlm"]["total_fields"] += 1
                comparison["details"][image_name]["vlm"]["total"] += 1
                
                if vlm_data[field].lower() == ground_truth[image_name][field].lower():
                    comparison["vlm"]["correct_fields"] += 1
                    comparison["details"][image_name]["vlm"]["correct"] += 1
        
        # Compare EasyOCR results
        easyocr_data = ocr_metadata[image_name]["easyocr"]
        comparison["details"][image_name]["easyocr"]["time"] = easyocr_data["processing_time"]
        comparison["easyocr"]["average_time"] += easyocr_data["processing_time"]
        
        for field in fields:
            if easyocr_data[field] is not None:
                comparison["easyocr"]["total_fields"] += 1
                comparison["details"][image_name]["easyocr"]["total"] += 1
                
                if easyocr_data[field].lower() == ground_truth[image_name][field].lower():
                    comparison["easyocr"]["correct_fields"] += 1
                    comparison["details"][image_name]["easyocr"]["correct"] += 1
        
        # Compare Tesseract results
        tesseract_data = ocr_metadata[image_name]["tesseract"]
        comparison["details"][image_name]["tesseract"]["time"] = tesseract_data["processing_time"]
        comparison["tesseract"]["average_time"] += tesseract_data["processing_time"]
        
        for field in fields:
            if tesseract_data[field] is not None:
                comparison["tesseract"]["total_fields"] += 1
                comparison["details"][image_name]["tesseract"]["total"] += 1
                
                if tesseract_data[field].lower() == ground_truth[image_name][field].lower():
                    comparison["tesseract"]["correct_fields"] += 1
                    comparison["details"][image_name]["tesseract"]["correct"] += 1
    
    # Calculate average times and accuracies
    num_images = len(comparison["details"])
    if num_images > 0:
        comparison["vlm"]["average_time"] /= num_images
        comparison["easyocr"]["average_time"] /= num_images
        comparison["tesseract"]["average_time"] /= num_images
    
    # Calculate accuracies
    if comparison["vlm"]["total_fields"] > 0:
        comparison["vlm"]["accuracy"] = comparison["vlm"]["correct_fields"] / comparison["vlm"]["total_fields"] * 100
    
    if comparison["easyocr"]["total_fields"] > 0:
        comparison["easyocr"]["accuracy"] = comparison["easyocr"]["correct_fields"] / comparison["easyocr"]["total_fields"] * 100
    
    if comparison["tesseract"]["total_fields"] > 0:
        comparison["tesseract"]["accuracy"] = comparison["tesseract"]["correct_fields"] / comparison["tesseract"]["total_fields"] * 100
    
    return comparison

def generate_comparison_report(comparison, output_file=None):
    """
    Generate a comparison report.
    
    Args:
        comparison (dict): Dictionary containing comparison results
        output_file (str, optional): Path to save the report
    
    Returns:
        str: Report text
    """
    # Create a report
    report = []
    report.append("# VLM vs. OCR Comparison Report")
    report.append("")
    
    # Add summary table
    report.append("## Summary")
    report.append("")
    report.append("| Metric | VLM | EasyOCR | Tesseract |")
    report.append("| --- | --- | --- | --- |")
    report.append(f"| Accuracy | {comparison['vlm']['accuracy']:.2f}% | {comparison['easyocr']['accuracy']:.2f}% | {comparison['tesseract']['accuracy']:.2f}% |")
    report.append(f"| Correct Fields | {comparison['vlm']['correct_fields']} / {comparison['vlm']['total_fields']} | {comparison['easyocr']['correct_fields']} / {comparison['easyocr']['total_fields']} | {comparison['tesseract']['correct_fields']} / {comparison['tesseract']['total_fields']} |")
    report.append(f"| Avg. Processing Time | {comparison['vlm']['average_time']:.2f}s | {comparison['easyocr']['average_time']:.2f}s | {comparison['tesseract']['average_time']:.2f}s |")
    report.append("")
    
    # Add details for each image
    report.append("## Details by Image")
    report.append("")
    
    for image_name, image_data in comparison["details"].items():
        report.append(f"### {image_name}")
        report.append("")
        
        # Add image comparison table
        report.append("| Method | Accuracy | Processing Time |")
        report.append("| --- | --- | --- |")
        
        vlm_accuracy = 0 if image_data["vlm"]["total"] == 0 else image_data["vlm"]["correct"] / image_data["vlm"]["total"] * 100
        easyocr_accuracy = 0 if image_data["easyocr"]["total"] == 0 else image_data["easyocr"]["correct"] / image_data["easyocr"]["total"] * 100
        tesseract_accuracy = 0 if image_data["tesseract"]["total"] == 0 else image_data["tesseract"]["correct"] / image_data["tesseract"]["total"] * 100
        
        report.append(f"| VLM | {vlm_accuracy:.2f}% ({image_data['vlm']['correct']} / {image_data['vlm']['total']}) | {image_data['vlm']['time']:.2f}s |")
        report.append(f"| EasyOCR | {easyocr_accuracy:.2f}% ({image_data['easyocr']['correct']} / {image_data['easyocr']['total']}) | {image_data['easyocr']['time']:.2f}s |")
        report.append(f"| Tesseract | {tesseract_accuracy:.2f}% ({image_data['tesseract']['correct']} / {image_data['tesseract']['total']}) | {image_data['tesseract']['time']:.2f}s |")
        report.append("")
    
    # Join the report lines
    report_text = "\n".join(report)
    
    # Save the report if an output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
    
    return report_text

def generate_comparison_chart(comparison, output_file=None):
    """
    Generate a comparison chart.
    
    Args:
        comparison (dict): Dictionary containing comparison results
        output_file (str, optional): Path to save the chart
    """
    # Create a DataFrame for easier plotting
    methods = ["VLM", "EasyOCR", "Tesseract"]
    accuracies = [
        comparison["vlm"]["accuracy"],
        comparison["easyocr"]["accuracy"],
        comparison["tesseract"]["accuracy"]
    ]
    times = [
        comparison["vlm"]["average_time"],
        comparison["easyocr"]["average_time"],
        comparison["tesseract"]["average_time"]
    ]
    
    df = pd.DataFrame({
        "Method": methods,
        "Accuracy (%)": accuracies,
        "Avg. Processing Time (s)": times
    })
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot accuracy
    df.plot(x="Method", y="Accuracy (%)", kind="bar", ax=ax1, color="blue")
    ax1.set_title("Accuracy Comparison")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    
    # Plot processing time
    df.plot(x="Method", y="Avg. Processing Time (s)", kind="bar", ax=ax2, color="green")
    ax2.set_title("Processing Time Comparison")
    ax2.set_ylabel("Average Processing Time (s)")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart if an output file is specified
    if output_file:
        plt.savefig(output_file)
        print(f"Comparison chart saved to {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare VLM results with OCR results")
    parser.add_argument("--vlm_results", type=str, required=True,
                        help="Path to the VLM results JSON file")
    parser.add_argument("--ocr_dir", type=str, default="../../ocr_testing/results/json",
                        help="Path to the directory containing OCR result JSON files")
    parser.add_argument("--ground_truth", type=str, default="../data/ground_truth.json",
                        help="Path to the ground truth JSON file")
    parser.add_argument("--output_report", type=str, default="../results/vlm_vs_ocr_report.md",
                        help="Path to save the comparison report")
    parser.add_argument("--output_chart", type=str, default="../results/images/vlm_vs_ocr_chart.png",
                        help="Path to save the comparison chart")
    args = parser.parse_args()
    
    # Get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    vlm_results_file = os.path.join(project_dir, args.vlm_results.lstrip("../"))
    ocr_dir = os.path.join(os.path.dirname(project_dir), args.ocr_dir.lstrip("../"))
    ground_truth_file = os.path.join(project_dir, args.ground_truth.lstrip("../"))
    output_report_file = os.path.join(project_dir, args.output_report.lstrip("../"))
    output_chart_file = os.path.join(project_dir, args.output_chart.lstrip("../"))
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_report_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_chart_file), exist_ok=True)
    
    # Load the results and ground truth data
    vlm_results = load_vlm_results(vlm_results_file)
    ocr_results = load_ocr_results(ocr_dir)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Extract metadata from the results
    vlm_metadata = extract_vlm_metadata(vlm_results)
    ocr_metadata = extract_ocr_metadata(ocr_results)
    
    # Compare the results
    comparison = compare_results(vlm_metadata, ocr_metadata, ground_truth)
    
    # Generate and print the report
    report = generate_comparison_report(comparison, output_report_file)
    print(report)
    
    # Generate the comparison chart
    generate_comparison_chart(comparison, output_chart_file)

if __name__ == "__main__":
    main() 