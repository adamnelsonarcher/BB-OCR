#!/usr/bin/env python
import os
import argparse
import cv2
import matplotlib.pyplot as plt
from image_preprocessor import ImagePreprocessor, preprocess_for_book_cover

def display_preprocessing_steps(image_path, output_dir=None):
    """
    Apply various preprocessing steps to an image and display the results.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str, optional): Directory to save the preprocessed images
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    preprocessor.load_image(image_path)
    
    # Original image
    original = preprocessor.get_image().copy()
    
    # Apply grayscale
    preprocessor.to_grayscale()
    grayscale = preprocessor.get_image().copy()
    
    # Apply contrast enhancement
    preprocessor.increase_contrast(1.5)
    contrast = preprocessor.get_image().copy()
    
    # Apply denoising
    preprocessor.denoise(strength=5)
    denoised = preprocessor.get_image().copy()
    
    # Apply sharpening
    preprocessor.sharpen()
    sharpened = preprocessor.get_image().copy()
    
    # Apply thresholding
    preprocessor.threshold(method="adaptive")
    thresholded = preprocessor.get_image().copy()
    
    # Display the results
    plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Grayscale
    plt.subplot(2, 3, 2)
    plt.title("Grayscale")
    plt.imshow(grayscale, cmap='gray')
    plt.axis('off')
    
    # Contrast
    plt.subplot(2, 3, 3)
    plt.title("Contrast Enhanced")
    plt.imshow(contrast, cmap='gray')
    plt.axis('off')
    
    # Denoised
    plt.subplot(2, 3, 4)
    plt.title("Denoised")
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')
    
    # Sharpened
    plt.subplot(2, 3, 5)
    plt.title("Sharpened")
    plt.imshow(sharpened, cmap='gray')
    plt.axis('off')
    
    # Thresholded
    plt.subplot(2, 3, 6)
    plt.title("Thresholded")
    plt.imshow(thresholded, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if output directory is specified
    if output_dir:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_preprocessing_steps.png")
        plt.savefig(output_path)
        print(f"Saved preprocessing steps to {output_path}")
    
    plt.show()
    
    # Save individual images if output directory is specified
    if output_dir:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save original
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), original)
        
        # Save grayscale
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_grayscale.png"), grayscale)
        
        # Save contrast enhanced
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_contrast.png"), contrast)
        
        # Save denoised
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_denoised.png"), denoised)
        
        # Save sharpened
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_sharpened.png"), sharpened)
        
        # Save thresholded
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_thresholded.png"), thresholded)
        
        print(f"Saved individual preprocessing steps to {output_dir}")
    
    return thresholded

def test_book_cover_preprocessing(image_path, output_dir=None):
    """
    Test the book cover preprocessing function.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str, optional): Directory to save the preprocessed images
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Get the base name of the image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Set output path if output directory is specified
    output_path = None
    if output_dir:
        output_path = os.path.join(output_dir, f"{base_name}_preprocessed.png")
    
    # Apply preprocessing
    preprocessed_image, saved_path, steps = preprocess_for_book_cover(image_path, output_path)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.title("Original")
    original = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Preprocessed
    plt.subplot(1, 2, 2)
    plt.title("Preprocessed")
    if len(preprocessed_image.shape) == 3:
        plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(preprocessed_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if output directory is specified
    if output_dir:
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(comparison_path)
        print(f"Saved comparison to {comparison_path}")
    
    plt.show()
    
    # Print the steps applied
    print("Preprocessing steps applied:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")
    
    return preprocessed_image

def main():
    parser = argparse.ArgumentParser(description='Test image preprocessing for OCR')
    parser.add_argument('--image', '-i', type=str, required=True, 
                        help='Path to the input image')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Directory to save the preprocessed images')
    parser.add_argument('--mode', '-m', type=str, choices=['steps', 'book_cover'], 
                        default='book_cover',
                        help='Test mode: "steps" to show all preprocessing steps, '
                             '"book_cover" to test the book cover preprocessing function')
    
    args = parser.parse_args()
    
    if args.mode == 'steps':
        display_preprocessing_steps(args.image, args.output)
    else:
        test_book_cover_preprocessing(args.image, args.output)

if __name__ == "__main__":
    main() 