@echo off
REM Batch script to process book images on Windows

echo Book Metadata Extraction Pipeline
echo ================================

IF "%1"=="" (
    echo Usage: process_books.bat [books_directory] [output_directory]
    echo Example: process_books.bat C:\path\to\books C:\path\to\output
    exit /b 1
)

set BOOKS_DIR=%1
set OUTPUT_DIR=%2

IF "%OUTPUT_DIR%"=="" (
    set OUTPUT_DIR=output
)

echo Processing books from: %BOOKS_DIR%
echo Saving results to: %OUTPUT_DIR%

python batch_processor.py %BOOKS_DIR% --output-dir %OUTPUT_DIR% --output-file %OUTPUT_DIR%\all_books.json

echo.
echo Processing complete!
