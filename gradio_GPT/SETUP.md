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

- **GPT-4o Integration**: Uses OpenAI's latest multimodal model for accurate text extraction with intelligent inference
- **Smart Inference**: Makes educated guesses from typography, design elements, and visual style cues
- **Formatted Table Output**: Clean, organized display with visual icons and structured data
- **Confidence Scoring**: AI-powered reliability assessment with detailed reasoning
- **Modern UI**: Beautiful Gradio interface with custom styling

## 📋 Expected Table Output

The results are displayed as a formatted table with fields like:

| Field | Value |
|-------|-------|
| 📚 Title | The Great Gatsby |
| ✍️ Author | F. Scott Fitzgerald |
| 📅 Year | 1925 |
| 🏢 Publisher | Scribner |
| 📖 Genre | Classic Literature |
| 🔍 Condition | Good |
| 🔢 ISBN | Not visible |
| 📃 Edition | First Edition |
| 🌐 Language | English |
| 🎯 Confidence Score | 85% |
| 📝 Notes | Classic American novel cover with art deco design elements |

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