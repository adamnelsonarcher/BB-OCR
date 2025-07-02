Becker Books OCR Tool (Milestone 1)
This project is the first deliverable for the Becker Books OSV Fellowship. Its goal is to create a functional prototype that proves the core "Image to Structured Data" pipeline (Steps 1-7 of the system architecture).

This initial version uses Gradio to create a simple web UI for testing and demonstration. The core processing logic is contained within a single Python script (app.py).

Prerequisites
Before you begin, ensure you have the following installed:

Python 3.8 or newer.

pip (Python's package installer).

Setup Instructions (for Cursor)
These steps will guide you through setting up the project environment from scratch. Using a virtual environment is highly recommended to keep project dependencies isolated.

5. Create the Application File
In the Cursor file explorer on the left, create a new file named app.py. Paste the following code into it:

import gradio as gr
from PIL import Image
# import cv2 # Uncomment when you start using OpenCV
# import easyocr # Uncomment when you start using EasyOCR

# This is your core logic function.
# For now, it ignores the input and returns a hardcoded sample JSON.
# NEXT STEP: Replace this placeholder logic with your real pipeline.
def process_book_image(book_image):
    """
    Takes a book image, processes it, and returns structured metadata.
    This function is the heart of your "Image to Structured Data" pipeline.
    """
    
    # --- Placeholder Logic ---
    print("Processing image...")
    # In a real scenario, you would save the image and process it.
    # For example: book_image.save("temp_image.png")
    
    extracted_data = {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "year": "1925",
        "listing_image_path": "images/placeholder.jpg"
    }
    
    print("Processing complete.")
    return extracted_data

# Create the Gradio interface
demo = gr.Interface(
    fn=process_book_image,
    inputs=gr.Image(type="pil", label="Upload Book Cover"),
    outputs=gr.JSON(label="Extracted Metadata"),
    title="Becker Books OCR Tool",
    description="Milestone 1: Scan & Identify. Upload a book cover to extract its metadata."
)

# Launch the web application
if __name__ == "__main__":
    demo.launch()


Running the Application
With your virtual environment still active, run the application from the Cursor terminal:

python app.py

The terminal will display a message like Running on local URL: http://127.0.0.1:7860.

Ctrl+Click (or Cmd+Click) the URL in the terminal to open the web UI in your browser.

You should now see the simple Gradio interface where you can upload an image and see the placeholder JSON output.

Next Steps
The primary goal now is to build out the real data processing logic inside the process_book_image function in app.py.

Image Preprocessing: Use OpenCV to implement the image cleaning steps (deskew, crop, contrast).

OCR Extraction: Use EasyOCR to extract raw text from the cleaned image.

Heuristics Engine: Write the Python and Regex rules to parse the raw text and populate the extracted_data dictionary dynamically.

You can use Cursor's AI features to help generate code snippets for each of these steps. For example, you could highlight the process_book_image function and prompt the AI with: "Using OpenCV, convert the input 'book_image' to grayscale and apply adaptive thresholding."