@echo off
if "%1"=="" (
    echo Usage: benchmark_image.bat path\to\image.png
    exit /b 1
)

echo ===================================
echo VLM Benchmark for Single Image
echo ===================================
echo Image: %1

REM Run benchmark with BLIP-2
echo.
echo Running benchmark with BLIP-2...
python run_vlm_test.py --model blip2 --image %1 --skip_evaluation

REM Run benchmark with LLaVA
echo.
echo Running benchmark with LLaVA...
python run_vlm_test.py --model llava --image %1 --skip_evaluation

echo.
echo ===================================
echo Benchmark completed!
echo ===================================
echo Results are available in:
echo - BLIP-2: results\json\%~n1_blip2_results.json
echo - LLaVA: results\json\%~n1_llava_results.json
echo ===================================

pause 