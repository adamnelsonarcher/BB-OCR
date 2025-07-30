#!/bin/bash

echo "==================================="
echo "VLM vs OCR Comparison Pipeline"
echo "==================================="

# Run the VLM testing pipeline
echo "Running VLM testing pipeline with BLIP-2..."
python run_vlm_test.py --model blip2

# Compare with OCR results
echo "Comparing VLM results with OCR..."
python scripts/compare_with_ocr.py --vlm_results results/json/blip2_all_results.json

echo "==================================="
echo "Pipeline completed successfully!"
echo "==================================="
echo "Results are available in:"
echo "- VLM Evaluation: results/blip2_evaluation_report.md"
echo "- VLM vs OCR Comparison: results/vlm_vs_ocr_report.md"
echo "- Comparison Chart: results/images/vlm_vs_ocr_chart.png"
echo "===================================" 