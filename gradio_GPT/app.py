import gradio as gr
import os
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_image(image):
    """
    Preprocess the image for better OCR results.
    Applies grayscale conversion, noise reduction, and contrast enhancement.
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding for better text extraction
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to PIL Image for consistency
    processed_image = Image.fromarray(adaptive_thresh)
    
    return processed_image

def encode_image_to_base64(image):
    """Convert PIL image to base64 string for OpenAI API."""
    buffer = BytesIO()
    # Save as RGB to ensure compatibility
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def extract_metadata_with_gpt(image):
    """
    Use GPT-4o to analyze the book image and extract metadata.
    """
    try:
        # Encode image
        base64_image = encode_image_to_base64(image)
        
        # Create the prompt for book metadata extraction
        prompt = """
        You are an expert book cataloger and bibliographer. Analyze this book cover image and extract the following metadata in JSON format:

        {
            "title": "The book's title",
            "author": "The author's name",
            "year": "Publication year (if visible)",
            "publisher": "Publisher name (if visible)",
            "genre": "Genre or category",
            "condition": "Visual condition assessment",
            "isbn": "ISBN if visible",
            "edition": "Edition information if available",
            "language": "Language of the book",
            "confidence_score": "Your confidence in the extraction (0-100)",
            "notes": "Any additional observations"
        }

        Guidelines:
        - Only include information that is clearly visible on the cover
        - If information is not visible or unclear, use "Not visible" or "Unknown"
        - Be conservative with condition assessment
        - Pay attention to spine text, front cover, and any visible text
        - For confidence_score, rate how certain you are about the overall extraction
        - Return only valid JSON, no additional text
        """
        
        # Make API call to GPT-4 Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1  # Low temperature for consistent extraction
        )
        
        # Parse the response
        content = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            metadata = json.loads(content)
            # Add processing timestamp
            metadata["processed_at"] = "2024-01-01"  # You can use datetime.now().isoformat() 
            metadata["processing_method"] = "GPT-4o"
            return metadata
        except json.JSONDecodeError:
            # If JSON parsing fails, return error info
            return {
                "error": "Failed to parse GPT response as JSON",
                "raw_response": content,
                "processing_method": "GPT-4o"
            }
            
    except Exception as e:
        return {
            "error": f"GPT processing failed: {str(e)}",
            "processing_method": "GPT-4o"
        }

def process_book_image(book_image):
    """
    Main processing function: takes a book image and returns structured metadata.
    This is the heart of the "Image to Structured Data" pipeline.
    """
    
    if book_image is None:
        return {"error": "No image provided"}
    
    print("Processing image...")
    print(f"Image size: {book_image.size}")
    print(f"Image mode: {book_image.mode}")
    
    # Step 1: Preprocess the image for better analysis
    processed_image = preprocess_image(book_image)
    print("Image preprocessing complete.")
    
    # Step 2: Use GPT-4o to extract metadata
    print("Extracting metadata with GPT-4o...")
    extracted_data = extract_metadata_with_gpt(book_image)  # Use original image for GPT
    
    print("Processing complete.")
    return extracted_data

def create_demo_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .upload-area {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    """
    
    # Create interface
    demo = gr.Interface(
        fn=process_book_image,
        inputs=[
            gr.Image(
                type="pil", 
                label="📚 Upload Book Cover Image",
                height=400,
                elem_classes=["upload-area"]
            )
        ],
        outputs=[
            gr.JSON(
                label="📋 Extracted Metadata",
                elem_id="metadata-output"
            )
        ],
        title="🔍 Becker Books OCR Tool",
        description="""
        **Instructions:**
        1. Upload a clear image of a book cover
        2. Wait for AI processing (may take 10-30 seconds)
        3. Review the extracted metadata in JSON format
        
        **Note:** Make sure to set your OpenAI API key in the `.env` file.
        """,
        theme=gr.themes.Soft(),
        css=css,
        examples=[
            # You can add example images here later
        ],
        allow_flagging="never",
        analytics_enabled=False
    )
    
    return demo

# Health check function
def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: OpenAI API key not found!")
        print("Please create a .env file with your OpenAI API key.")
        print("Copy .env.example to .env and add your key.")
        return False
    else:
        print("✅ OpenAI API key found.")
        return True

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting Becker Books OCR Tool...")
    print("=" * 50)
    
    # Check API key
    if not check_api_key():
        print("❌ Please configure OpenAI API key to continue.")
        print("The app will start but GPT features won't work.")
    
    # Create and launch the demo
    demo = create_demo_interface()
    
    print("🌐 Launching web interface...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True if you want to create a public link
        show_error=True,
        quiet=False
    ) 