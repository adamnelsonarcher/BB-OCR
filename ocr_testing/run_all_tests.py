#!/usr/bin/env python
import os
import subprocess
import glob

def run_tests_on_all_images():
    """
    Run OCR comparison tests on all book images in the dataset folder
    """
    # Get all PNG images in the dataset folder
    image_files = glob.glob("../dataset/*.png")
    
    if not image_files:
        print("No image files found in the dataset folder")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Run the comparison script on each image
    for image_file in image_files:
        print(f"\n\n{'='*80}")
        print(f"Processing {image_file}")
        print(f"{'='*80}\n")
        
        # Run the comparison script
        cmd = ["python", "ocr_engines/compare_ocr_engines.py", "--image", image_file]
        subprocess.run(cmd)
        
        print(f"\nCompleted processing {image_file}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_tests_on_all_images() 