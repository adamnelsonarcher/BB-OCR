# Quick Setup Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

### 3. Run the Application
```bash
python app.py
```

### 4. Open in Browser
Visit: http://127.0.0.1:7860

## 🔧 Features

- **GPT-4 Vision Integration**: Uses OpenAI's latest vision model for accurate text extraction
- **Image Preprocessing**: OpenCV-based image enhancement for better results  
- **Structured JSON Output**: Clean, consistent metadata format
- **Confidence Scoring**: AI-powered reliability assessment
- **Modern UI**: Beautiful Gradio interface with custom styling

## 📋 Expected JSON Output

```json
{
  "title": "The Great Gatsby",
  "author": "F. Scott Fitzgerald", 
  "year": "1925",
  "publisher": "Scribner",
  "genre": "Classic Literature",
  "condition": "Good",
  "isbn": "Not visible",
  "edition": "First Edition",
  "language": "English",
  "confidence_score": 85,
  "notes": "Classic American novel cover"
}
```

## 🛠️ Troubleshooting

- **API Key Issues**: Make sure your `.env` file is in the project root
- **Slow Processing**: GPT-4 Vision can take 10-30 seconds per image
- **Installation Issues**: Try using a virtual environment

## 🎯 Next Steps

This is Milestone 1 - the foundation for the complete OCR pipeline. Future enhancements will include:
- Local database integration
- Pricing engine
- Batch processing
- Advanced image preprocessing 