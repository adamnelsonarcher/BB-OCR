#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for VLM testing.
This script prepares the book cover dataset and creates ground truth metadata for evaluation.
"""

import os
import json
import argparse
import shutil
from pathlib import Path

def create_sample_ground_truth(output_file, image_dir):
    """
    Create a sample ground truth JSON file for the book covers in the dataset.
    In a real scenario, this would be filled with actual metadata.
    
    Args:
        output_file (str): Path to save the ground truth JSON file
        image_dir (str): Directory containing book cover images
    """
    # Get all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create a dictionary to store ground truth data
    ground_truth = {}
    
    # Sample metadata for the test images
    # In a real scenario, this would be actual metadata for each book
    sample_metadata = {
        "book1.png": {
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "publisher": "Scribner"
        },
        "book2.png": {
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "publisher": "J. B. Lippincott & Co."
        },
        "book3.png": {
            "title": "1984",
            "author": "George Orwell",
            "publisher": "Secker & Warburg"
        },
        "book4.png": {
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "publisher": "T. Egerton, Whitehall"
        },
        "book5.png": {
            "title": "The Catcher in the Rye",
            "author": "J.D. Salinger",
            "publisher": "Little, Brown and Company"
        },
        "book6.png": {
            "title": "Moby-Dick",
            "author": "Herman Melville",
            "publisher": "Harper & Brothers"
        }
    }
    
    # Fill in the ground truth data
    for image_file in image_files:
        if image_file in sample_metadata:
            ground_truth[image_file] = sample_metadata[image_file]
        else:
            # For images without predefined metadata, create placeholder entries
            ground_truth[image_file] = {
                "title": "Unknown Title",
                "author": "Unknown Author",
                "publisher": "Unknown Publisher"
            }
    
    # Save the ground truth data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Ground truth data saved to {output_file}")
    print(f"Created metadata for {len(ground_truth)} book covers")

def copy_dataset_images(source_dir, target_dir):
    """
    Copy book cover images from the source directory to the target directory.
    
    Args:
        source_dir (str): Source directory containing book cover images
        target_dir (str): Target directory to copy images to
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files in the source directory
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Copy each image to the target directory
    for image_file in image_files:
        source_path = os.path.join(source_dir, image_file)
        target_path = os.path.join(target_dir, image_file)
        shutil.copy2(source_path, target_path)
    
    print(f"Copied {len(image_files)} images from {source_dir} to {target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for VLM testing")
    parser.add_argument("--source_dir", type=str, default="../dataset",
                        help="Source directory containing book cover images (default: ../dataset)")
    parser.add_argument("--target_dir", type=str, default="data/images",
                        help="Target directory to copy images to (default: data/images)")
    parser.add_argument("--output_file", type=str, default="data/ground_truth.json",
                        help="Path to save the ground truth JSON file (default: data/ground_truth.json)")
    args = parser.parse_args()
    
    # Get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    source_dir = os.path.join(os.path.dirname(project_dir), args.source_dir)
    target_dir = os.path.join(project_dir, args.target_dir)
    output_file = os.path.join(project_dir, args.output_file)
    
    # Create the target directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Copy dataset images
    copy_dataset_images(source_dir, target_dir)
    
    # Create ground truth metadata
    create_sample_ground_truth(output_file, target_dir)

if __name__ == "__main__":
    main() 