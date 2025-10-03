#!/usr/bin/env python
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter

class ImagePreprocessor:
    """
    Preprocessing steps for better OCR results (grayscale, denoise, CLAHE, sharpen, resize).
    """
    
    def __init__(self):
        self.preprocessed_image = None
        self.original_image = None
        self.steps_applied = []
    
    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.preprocessed_image = self.original_image.copy()
        self.steps_applied = ["original"]
        return self
    
    def to_grayscale(self):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        self.preprocessed_image = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2GRAY)
        self.steps_applied.append("grayscale")
        return self
    
    def denoise(self, strength=7):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        self.preprocessed_image = cv2.GaussianBlur(self.preprocessed_image, (3, 3), strength)
        self.steps_applied.append(f"denoise(strength={strength})")
        return self
    
    def equalize_histogram(self):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        if len(self.preprocessed_image.shape) == 3:
            self.to_grayscale()
        self.preprocessed_image = cv2.equalizeHist(self.preprocessed_image)
        self.steps_applied.append("equalize_histogram")
        return self
    
    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        if len(self.preprocessed_image.shape) == 3:
            self.to_grayscale()
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.preprocessed_image = clahe.apply(self.preprocessed_image)
        self.steps_applied.append(f"clahe(clip_limit={clip_limit})")
        return self
    
    def gentle_threshold(self, block_size=11, constant=2):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        if len(self.preprocessed_image.shape) == 3:
            self.to_grayscale()
        self.preprocessed_image = cv2.adaptiveThreshold(
            self.preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, constant
        )
        self.steps_applied.append(f"gentle_threshold(block_size={block_size}, constant={constant})")
        return self
    
    def increase_contrast(self, factor=1.5):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        if len(self.preprocessed_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.preprocessed_image)
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced_img = enhancer.enhance(factor)
        if len(self.preprocessed_image.shape) == 3:
            self.preprocessed_image = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
        else:
            self.preprocessed_image = np.array(enhanced_img)
        self.steps_applied.append(f"increase_contrast(factor={factor})")
        return self

    def increase_brightness(self, factor=1.2):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        if len(self.preprocessed_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.preprocessed_image)
        enhancer = ImageEnhance.Brightness(pil_img)
        bright_img = enhancer.enhance(factor)
        if len(self.preprocessed_image.shape) == 3:
            self.preprocessed_image = cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)
        else:
            self.preprocessed_image = np.array(bright_img)
        self.steps_applied.append(f"increase_brightness(factor={factor})")
        return self
    
    def sharpen(self, amount=0.3):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        if len(self.preprocessed_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.preprocessed_image)
        sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(amount*100), threshold=3))
        if len(self.preprocessed_image.shape) == 3:
            self.preprocessed_image = cv2.cvtColor(np.array(sharpened), cv2.COLOR_RGB2BGR)
        else:
            self.preprocessed_image = np.array(sharpened)
        self.steps_applied.append(f"sharpen(amount={amount})")
        return self
    
    def remove_borders(self, border_size=5):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        h, w = self.preprocessed_image.shape[:2]
        self.preprocessed_image = self.preprocessed_image[border_size:h-border_size, border_size:w-border_size]
        self.steps_applied.append(f"remove_borders(size={border_size})")
        return self
    
    def resize(self, scale_factor=2.0):
        if self.preprocessed_image is None:
            raise ValueError("No image loaded")
        h, w = self.preprocessed_image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        self.preprocessed_image = cv2.resize(self.preprocessed_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        self.steps_applied.append(f"resize(scale_factor={scale_factor})")
        return self
    
    def save_image(self, output_path):
        if self.preprocessed_image is None:
            raise ValueError("No image to save")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, self.preprocessed_image)
        return output_path
    
    def get_image(self):
        return self.preprocessed_image
    
    def get_steps_applied(self):
        return self.steps_applied

def preprocess_for_book_cover(image_path, output_path=None):
    preprocessor = ImagePreprocessor()
    preprocessor.load_image(image_path)
    preprocessor.to_grayscale()
    preprocessor.resize(scale_factor=1.5)
    preprocessor.denoise(strength=5)
    preprocessor.increase_brightness(1.2)
    preprocessor.increase_contrast(1.8)
    preprocessor.clahe(clip_limit=2.5)
    preprocessor.sharpen(amount=0.25)
    if output_path:
        preprocessor.save_image(output_path)
    return (preprocessor.get_image(), output_path, preprocessor.get_steps_applied())


