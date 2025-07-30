We are testing the use of a Vision-Language Model (VLM), such as BLIP-2 or LLaVA, to extract metadata (title, author, publisher) from book cover images. The goal is to benchmark this approach against traditional OCR tools like Tesseract and EasyOCR, which have proven unreliable on stylized covers.

The test should involve:

1. **Model Setup**

   * Set up either BLIP-2 or LLaVA locally using a Conda environment or Docker.
   * Download the pre-trained model weights.
   * Ensure the model can accept image input and return text responses to specific prompts.

2. **Data Preparation**

   * Use a test dataset of 50–100 book cover images.
   * Organize these images in a folder and prepare a JSON or CSV file containing the expected metadata (ground truth) for comparison.

3. **Inference Script**

   * Write a Python script that:

     * Loads an image
     * Sends it through the VLM using prompts like:

       * "What is the title of this book?"
       * "Who is the author of this book?"
       * "Who is the publisher?"
     * Records the output and inference time
     * Stores everything in a JSON log per image

4. **Evaluation Script**

   * Create another script that compares model output to the ground truth metadata.
   * It should measure:

     * Accuracy (exact match or fuzzy match)
     * Average inference time per image

5. **Output**

   * Save a structured log (JSON or CSV) with fields like:

     * image name
     * VLM output
     * expected metadata
     * inference time
     * whether it matched or not
   * Optional: generate a table or chart comparing accuracy and speed vs. OCR

The main focus is to determine if a local VLM can outperform OCR in metadata extraction without incurring API costs. We are aiming for a fast, scalable solution that could process 800,000+ book covers locally. Start with just 1 model and 50 images.

## Implementation

We have implemented the VLM testing framework with the following components:

### Directory Structure

```
vlm_testing/
├── data/                  # Data directory
│   ├── images/            # Book cover images
│   └── ground_truth.json  # Ground truth metadata
├── models/                # Model directory for cached models
├── results/               # Results directory
│   ├── images/            # Output images and charts
│   └── json/              # JSON results from VLM inference
├── scripts/               # Python scripts
│   ├── model_setup.py     # Model setup script
│   ├── prepare_data.py    # Data preparation script
│   ├── run_inference.py   # Inference script
│   ├── evaluate_results.py # Evaluation script
│   └── compare_with_ocr.py # OCR comparison script
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment file
├── run_vlm_test.py        # Main script to run the pipeline
├── run_comparison.bat     # Windows batch script to run the pipeline
├── run_comparison.sh      # Linux/Mac shell script to run the pipeline
└── test_setup.py          # Script to verify the installation
```

### Installation

#### Option 1: Using pip with a virtual environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Option 2: Using conda environment

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate vlm-testing
   ```

#### Verify Installation

Run the test script to verify that all required packages are installed:

```bash
python test_setup.py
```

This script checks:
- Python version
- Required packages and their versions
- CUDA-compatible GPU availability

### Usage

#### Running the Entire Pipeline

To run the entire VLM testing pipeline:

```bash
python run_vlm_test.py --model blip2
```

Options:
- `--model`: VLM model to use (`blip2` or `llava`, default: `blip2`)
- `--skip_data_prep`: Skip data preparation step
- `--skip_inference`: Skip inference step
- `--skip_evaluation`: Skip evaluation step

#### Running the Entire Pipeline and Comparing with OCR

On Windows:
```bash
run_comparison.bat
```

On Linux/Mac:
```bash
./run_comparison.sh
```

#### Running Individual Steps

1. Data Preparation:
   ```bash
   python scripts/prepare_data.py
   ```

2. Model Setup and Inference:
   ```bash
   python scripts/run_inference.py --model blip2
   ```

3. Evaluation:
   ```bash
   python scripts/evaluate_results.py --results_file results/json/blip2_all_results.json
   ```

4. Comparison with OCR:
   ```bash
   python scripts/compare_with_ocr.py --vlm_results results/json/blip2_all_results.json
   ```

### Results

After running the pipeline, you'll find the following results:

- Individual VLM results in `results/json/`
- Evaluation report in `results/blip2_evaluation_report.md`
- VLM vs. OCR comparison report in `results/vlm_vs_ocr_report.md`
- Comparison chart in `results/images/vlm_vs_ocr_chart.png`

### Hardware Requirements

- **CPU**: A modern multi-core CPU (8+ cores recommended for faster processing)
- **RAM**: At least 16GB RAM (32GB+ recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for faster inference)
- **Storage**: At least 10GB free space for model weights and results

### Notes on Model Selection

- **BLIP-2**: More efficient, requires less VRAM, and is faster for inference
- **LLaVA**: More accurate for complex visual understanding tasks but requires more VRAM

For initial testing, we recommend starting with BLIP-2 to establish a baseline before trying LLaVA.