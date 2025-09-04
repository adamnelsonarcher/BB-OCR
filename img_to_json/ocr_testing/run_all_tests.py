#!/usr/bin/env python
import os
import subprocess
import glob
import argparse

def run_tests_on_all_images(use_preprocessing=False):
    """
    Run OCR comparison tests on all book images in the dataset folder
    
    Args:
        use_preprocessing (bool): Whether to apply preprocessing to the images
    """
    # Get all PNG images in the dataset folder
    image_files = glob.glob("../dataset/*.png")
    
    if not image_files:
        print("No image files found in the dataset folder")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create results directories if they don't exist
    os.makedirs("results/json", exist_ok=True)
    os.makedirs("results/images", exist_ok=True)
    
    # Run the comparison script on each image
    for image_file in image_files:
        print(f"\n\n{'='*80}")
        print(f"Processing {image_file}")
        print(f"{'='*80}\n")
        
        # Run the comparison script
        cmd = ["python", "ocr_engines/compare_ocr_engines.py", "--image", image_file]
        
        # Add preprocessing flag if enabled
        if use_preprocessing:
            cmd.append("--preprocess")
            
        subprocess.run(cmd)
        
        print(f"\nCompleted processing {image_file}")
        print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Run OCR comparison tests on all images')
    parser.add_argument('--preprocess', '-p', action='store_true',
                        help='Apply preprocessing to the images before OCR')
    
    args = parser.parse_args()
    
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_tests_on_all_images(args.preprocess)

if __name__ == "__main__":
    main() 