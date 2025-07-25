#!/usr/bin/env python
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter

class ImagePreprocessor:
    """
    A class to preprocess images for better OCR results.
    Implements various techniques like grayscale conversion, noise removal,
    thresholding, deskewing, etc.
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
    
    def denoise(self, strength=10):
        """Remove noise using Gaussian blur"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        self.preprocessed_image = cv2.GaussianBlur(self.preprocessed_image, (5, 5), strength)
        self.steps_applied.append(f"denoise(strength={strength})")
        return self
    
    def threshold(self, method="adaptive"):
        """
        Apply thresholding to the image.
        
        Args:
            method (str): Thresholding method to use.
                          Options: 'simple', 'otsu', 'adaptive'
        """
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Ensure image is grayscale
        if len(self.preprocessed_image.shape) == 3:
            self.to_grayscale()
            
        if method == "simple":
            _, self.preprocessed_image = cv2.threshold(
                self.preprocessed_image, 127, 255, cv2.THRESH_BINARY
            )
        elif method == "otsu":
            _, self.preprocessed_image = cv2.threshold(
                self.preprocessed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "adaptive":
            self.preprocessed_image = cv2.adaptiveThreshold(
                self.preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            raise ValueError(f"Unknown thresholding method: {method}")
            
        self.steps_applied.append(f"threshold({method})")
        return self
    
    def deskew(self):
        """Detect and correct skew in the image"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Ensure image is grayscale
        if len(self.preprocessed_image.shape) == 3:
            gray = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.preprocessed_image.copy()
            
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour by area
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the minimum area rectangle that encloses the contour
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Determine the angle to rotate (we want text to be horizontal)
            if angle < -45:
                angle = 90 + angle
            
            # Rotate the image
            (h, w) = self.preprocessed_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            self.preprocessed_image = cv2.warpAffine(
                self.preprocessed_image, M, (w, h), 
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            
        self.steps_applied.append("deskew")
        return self
    
    def increase_contrast(self, factor=2.0):
        """Increase the contrast of the image"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Convert to PIL Image for easier contrast adjustment
        if len(self.preprocessed_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.preprocessed_image)
            
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced_img = enhancer.enhance(factor)
        
        # Convert back to OpenCV format
        if len(self.preprocessed_image.shape) == 3:
            self.preprocessed_image = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
        else:
            self.preprocessed_image = np.array(enhanced_img)
            
        self.steps_applied.append(f"increase_contrast(factor={factor})")
        return self
    
    def sharpen(self):
        """Sharpen the image to improve text clarity"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        # Convert to PIL Image for easier sharpening
        if len(self.preprocessed_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.preprocessed_image)
            
        # Apply sharpening filter
        sharpened = pil_img.filter(ImageFilter.SHARPEN)
        
        # Convert back to OpenCV format
        if len(self.preprocessed_image.shape) == 3:
            self.preprocessed_image = cv2.cvtColor(np.array(sharpened), cv2.COLOR_RGB2BGR)
        else:
            self.preprocessed_image = np.array(sharpened)
            
        self.steps_applied.append("sharpen")
        return self
    
    def remove_borders(self, border_size=10):
        """Remove potential borders from the image"""
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
            
        h, w = self.preprocessed_image.shape[:2]
        self.preprocessed_image = self.preprocessed_image[
            border_size:h-border_size, 
            border_size:w-border_size
        ]
        
        self.steps_applied.append(f"remove_borders(size={border_size})")
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
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the preprocessed image
        
    Returns:
        tuple: (preprocessed_image, output_path, steps_applied)
    """
    preprocessor = ImagePreprocessor()
    
    # Apply a series of preprocessing steps
    preprocessor.load_image(image_path)
    preprocessor.to_grayscale()
    preprocessor.increase_contrast(1.5)
    preprocessor.denoise(strength=5)
    preprocessor.sharpen()
    preprocessor.threshold(method="adaptive")
    
    # Save the image if output path is provided
    if output_path:
        preprocessor.save_image(output_path)
    
    return (
        preprocessor.get_image(),
        output_path,
        preprocessor.get_steps_applied()
    ) 