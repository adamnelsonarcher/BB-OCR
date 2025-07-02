import gradio as gr
import os
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
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
        You are an expert book cataloger and bibliographer with deep knowledge of publishing history, typography, and book design. Analyze this book cover image and extract metadata using both visible text and intelligent inference from visual clues.

        Extract the following metadata in JSON format:

        {
            "title": "The book's title",
            "author": "The author's name", 
            "year": "Publication year",
            "publisher": "Publisher name",
            "genre": "Genre or category",
            "condition": "Visual condition assessment",
            "isbn": "ISBN if visible",
            "edition": "Edition information",
            "language": "Language of the book",
            "confidence_score": "Your confidence in the extraction (0-100)",
            "notes": "Additional observations and reasoning"
        }

        CRITICAL INFERENCE GUIDELINES:
        - NEVER use "Not visible" unless absolutely impossible to infer anything
        - ALWAYS make educated guesses using visual, contextual, and historical clues
        - Use typography styles, design trends, and aesthetic elements to determine publication decade
        - Infer publisher from design patterns, author associations, and market positioning
        - Consider bestseller badges, awards, price stickers as era and publisher indicators
        - Use author's typical publishing history and book series context
        - Analyze cover art style, color palettes, and layout to determine genre and era
        - Make reasonable assumptions about edition type from cover design and quality
        - Use context clues like "New York Times Bestseller" to infer timeframe and publisher caliber
        - Cross-reference visual design with known publishing trends by decade
        - For popular books, use your knowledge of when major titles were published
        - Be confident in reasonable inferences - explaining your logic in notes
        - Only mark fields as unknown when truly no reasonable inference exists

        EXAMPLES OF GOOD INFERENCE:
        - Modern sans-serif typography + bestseller badge = likely 2000s-2010s publication
        - Art deco design elements = likely 1920s-1940s or modern retro edition
        - Mass market paperback appearance = likely major publisher like Penguin, Bantam, etc.
        - Award badges and quotes = indicates established publisher and recent publication

        Return only valid JSON, no additional text.
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
        
        # Get usage statistics for cost calculation
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # Calculate cost (GPT-4o pricing as of 2024)
        # Input: $0.005 per 1K tokens, Output: $0.015 per 1K tokens
        input_cost = (input_tokens / 1000) * 0.005
        output_cost = (output_tokens / 1000) * 0.015
        total_cost = input_cost + output_cost
        
        # Try to parse as JSON
        try:
            # Remove markdown code block formatting if present
            if content.strip().startswith('```json'):
                # Extract JSON from markdown code block
                content = content.strip()
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    content = content[start_idx:end_idx]
            elif content.strip().startswith('```'):
                # Handle generic code block
                lines = content.strip().split('\n')
                content = '\n'.join(lines[1:-1])  # Remove first and last line (``` markers)
            
            metadata = json.loads(content)
            # Add processing timestamp and cost info
            metadata["processed_at"] = "2024-01-01"  # You can use datetime.now().isoformat() 
            metadata["processing_method"] = "GPT-4o"
            metadata["tokens_used"] = f"{total_tokens} ({input_tokens} input + {output_tokens} output)"
            metadata["estimated_cost"] = f"${total_cost:.4f}"
            return metadata
        except json.JSONDecodeError:
            # If JSON parsing fails, return error info
            return {
                "error": "Failed to parse GPT response as JSON",
                "raw_response": content,
                "processing_method": "GPT-4o",
                "tokens_used": f"{total_tokens} ({input_tokens} input + {output_tokens} output)",
                "estimated_cost": f"${total_cost:.4f}"
            }
            
    except Exception as e:
        return {
            "error": f"GPT processing failed: {str(e)}",
            "processing_method": "GPT-4o"
        }

def format_metadata_as_table(metadata):
    """
    Convert the metadata JSON to a nicely formatted table.
    """
    if "error" in metadata:
        # Handle error case
        error_df = pd.DataFrame([
            ["❌ Error", metadata.get("error", "Unknown error")],
            ["🤖 Method", metadata.get("processing_method", "Unknown")]
        ], columns=["Field", "Value"])
        return error_df
    
    # Create a formatted table from the metadata
    table_data = []
    
    # Define the order and display names for fields
    field_mapping = {
        "title": "Title",
        "author": "Author", 
        "year": "Year",
        "publisher": "Publisher",
        "genre": "Genre",
        "condition": "Condition",
        "isbn": "ISBN",
        "edition": "Edition",
        "language": "Language",
        "confidence_score": "Confidence Score",
        "notes": "Notes",
        "processed_at": "Processed At",
        "processing_method": "Processing Method",
        "tokens_used": "Tokens Used",
        "estimated_cost": "Estimated Cost"
    }
    
    # Build the table data
    for key, display_name in field_mapping.items():
        if key in metadata:
            value = metadata[key]
            # Format confidence score with percentage
            if key == "confidence_score":
                value = f"{value}%"
            table_data.append([display_name, str(value)])
    
    # Create DataFrame
    df = pd.DataFrame(table_data, columns=["Field", "Value"])
    return df

def process_book_image(book_image):
    """
    Main processing function: takes a book image and returns both raw response and structured metadata.
    """
    
    if book_image is None:
        return "No image provided", pd.DataFrame([["❌ Error", "No image provided"]], columns=["Field", "Value"])
    
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
    
    # Step 3: Format the data as a table
    formatted_table = format_metadata_as_table(extracted_data)
    
    # Return both raw and formatted data as separate values
    raw_response = json.dumps(extracted_data, indent=2)
    return raw_response, formatted_table

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
            gr.Textbox(
                label="Raw GPT Response",
                lines=10,
                max_lines=15
            ),
            gr.Dataframe(
                label="📋 Formatted Metadata",
                headers=["Field", "Value"],
                datatype=["str", "str"],
                row_count=15,
                col_count=(2, "fixed"),
                interactive=False,
                wrap=True
            )
        ],
        title="GUI Baseline, GPT Wrapper",
        description="""
        **Instructions:**
        1. Upload a clear image of a book cover
        2. Wait for AI processing (may take 10-30 seconds)
        3. Review the extracted metadata in the formatted table
        
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
    print("Starting Becker Books OCR Tool...")
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
        share=True,  # Set to True to create a public link
        show_error=True,
        quiet=False
    ) 