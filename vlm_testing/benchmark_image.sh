#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./benchmark_image.sh path/to/image.png"
    exit 1
fi

echo "==================================="
echo "VLM Benchmark for Single Image"
echo "==================================="
echo "Image: $1"

# Get the filename without extension
filename=$(basename -- "$1")
filename_noext="${filename%.*}"

# Run benchmark with BLIP-2
echo
echo "Running benchmark with BLIP-2..."
python run_vlm_test.py --model blip2 --image "$1" --skip_evaluation

# Run benchmark with LLaVA
echo
echo "Running benchmark with LLaVA..."
python run_vlm_test.py --model llava --image "$1" --skip_evaluation

echo
echo "==================================="
echo "Benchmark completed!"
echo "==================================="
echo "Results are available in:"
echo "- BLIP-2: results/json/${filename_noext}_blip2_results.json"
echo "- LLaVA: results/json/${filename_noext}_llava_results.json"
echo "===================================" 