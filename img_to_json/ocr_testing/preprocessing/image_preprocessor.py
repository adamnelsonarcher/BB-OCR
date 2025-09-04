#!/usr/bin/env python
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter

class ImagePreprocessor:
    """
    A class to preprocess images for better OCR results.
    Implements various techniques like grayscale conversion, noise removal,
    histogram equalization, etc.
    """
    
    def __init__(self):
        self.preprocessed_image = None
        self.original_image = None
        self.steps_applied = []
    
    def load_image(self, image_path):
        """
        Load an image from the given path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            self: For method chaining
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.preprocessed_image = self.original_image.copy()
        self.steps_applied = ["original"]
        return self
    
    def to_grayscale(self):
        """Convert image to grayscale"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        self.preprocessed_image = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2GRAY)
        self.steps_applied.append("grayscale")
        return self
    
    def denoise(self, strength=7):
        """Remove noise using Gaussian blur with moderate strength"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Use a smaller kernel size (3x3) with moderate strength
        self.preprocessed_image = cv2.GaussianBlur(self.preprocessed_image, (3, 3), strength)
        self.steps_applied.append(f"denoise(strength={strength})")
        return self
    
    def equalize_histogram(self):
        """
        Apply histogram equalization to improve contrast.
        This is more effective than harsh thresholding for text recognition.
        """
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Ensure image is grayscale
        if len(self.preprocessed_image.shape) == 3:
            self.to_grayscale()
            
        # Apply histogram equalization
        self.preprocessed_image = cv2.equalizeHist(self.preprocessed_image)
        self.steps_applied.append("equalize_histogram")
        return self
    
    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        This is better than regular histogram equalization for non-uniform lighting.
        
        Args:
            clip_limit (float): Threshold for contrast limiting
            tile_grid_size (tuple): Size of grid for histogram equalization
        """
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Ensure image is grayscale
        if len(self.preprocessed_image.shape) == 3:
            self.to_grayscale()
            
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE
        self.preprocessed_image = clahe.apply(self.preprocessed_image)
        self.steps_applied.append(f"clahe(clip_limit={clip_limit})")
        return self
    
    def gentle_threshold(self, block_size=11, constant=2):
        """
        Apply a gentle adaptive threshold that preserves more detail.
        Uses a larger block size and smaller constant for better text preservation.
        """
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Ensure image is grayscale
        if len(self.preprocessed_image.shape) == 3:
            self.to_grayscale()
            
        # Apply adaptive thresholding with THRESH_BINARY (not THRESH_BINARY_INV)
        self.preprocessed_image = cv2.adaptiveThreshold(
            self.preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, constant
        )
        self.steps_applied.append(f"gentle_threshold(block_size={block_size}, constant={constant})")
        return self
    
    def increase_contrast(self, factor=1.5):
        """Increase the contrast of the image with a moderate factor"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Convert to PIL Image for easier contrast adjustment
        if len(self.preprocessed_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.preprocessed_image)
            
        # Enhance contrast with a moderate factor
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced_img = enhancer.enhance(factor)
        
        # Convert back to OpenCV format
        if len(self.preprocessed_image.shape) == 3:
            self.preprocessed_image = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
        else:
            self.preprocessed_image = np.array(enhanced_img)
            
        self.steps_applied.append(f"increase_contrast(factor={factor})")
        return self
    
    def sharpen(self, amount=0.3):
        """
        Sharpen the image to improve text clarity.
        Uses a moderate amount to avoid introducing noise.
        """
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Convert to PIL Image for easier sharpening
        if len(self.preprocessed_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.preprocessed_image)
            
        # Apply gentle sharpening filter
        sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(amount*100), threshold=3))
        
        # Convert back to OpenCV format
        if len(self.preprocessed_image.shape) == 3:
            self.preprocessed_image = cv2.cvtColor(np.array(sharpened), cv2.COLOR_RGB2BGR)
        else:
            self.preprocessed_image = np.array(sharpened)
            
        self.steps_applied.append(f"sharpen(amount={amount})")
        return self
    
    def remove_borders(self, border_size=5):
        """Remove potential borders from the image with a smaller border size"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        h, w = self.preprocessed_image.shape[:2]
        self.preprocessed_image = self.preprocessed_image[
            border_size:h-border_size, 
            border_size:w-border_size
        ]
        
        self.steps_applied.append(f"remove_borders(size={border_size})")
        return self
    
    def resize(self, scale_factor=2.0):
        """
        Resize the image to improve OCR accuracy.
        Enlarging small text can help OCR engines recognize it better.
        """
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        h, w = self.preprocessed_image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        self.preprocessed_image = cv2.resize(
            self.preprocessed_image, (new_w, new_h), 
            interpolation=cv2.INTER_CUBIC
        )
        
        self.steps_applied.append(f"resize(scale_factor={scale_factor})")
        return self
    
    def save_image(self, output_path):
        """Save the preprocessed image to the given path"""
        if self.preprocessed_image is None:
            raise ValueError("No image to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        cv2.imwrite(output_path, self.preprocessed_image)
        return output_path
    
    def get_image(self):
        """Return the preprocessed image"""
        return self.preprocessed_image
    
    def get_steps_applied(self):
        """Return the list of preprocessing steps applied"""
        return self.steps_applied


def preprocess_for_book_cover(image_path, output_path=None):
    """
    Apply a series of preprocessing steps optimized for book covers.
    Uses histogram equalization instead of harsh thresholding.
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the preprocessed image
        
    Returns:
        tuple: (preprocessed_image, output_path, steps_applied)
    """
    preprocessor = ImagePreprocessor()
    
    # Apply a series of preprocessing steps optimized for book covers
    preprocessor.load_image(image_path)
    preprocessor.to_grayscale()
    preprocessor.resize(scale_factor=1.5)  # Enlarge image slightly
    preprocessor.denoise(strength=5)       # Gentle denoising
    preprocessor.increase_contrast(1.3)    # Moderate contrast enhancement
    preprocessor.clahe(clip_limit=2.0)     # Better than regular histogram equalization
    preprocessor.sharpen(amount=0.2)       # Gentle sharpening
    
    # Save the image if output path is provided
    if output_path:
        preprocessor.save_image(output_path)
    
    return (
        preprocessor.get_image(),
        output_path,
        preprocessor.get_steps_applied()
    ) 