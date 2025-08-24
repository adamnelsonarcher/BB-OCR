@echo off
REM Batch script to process a single book directory on Windows

echo Book Metadata Extraction - Single Book
echo =====================================

IF "%1"=="" (
    echo Usage: process_single_book.bat [book_directory] [output_file]
    echo Example: process_single_book.bat C:\path\to\book_images book_metadata.json
    exit /b 1
)

set BOOK_DIR=%1
set OUTPUT_FILE=%2

IF "%OUTPUT_FILE%"=="" (
    set OUTPUT_FILE=book_metadata.json
)

echo Processing book from: %BOOK_DIR%
echo Saving result to: %OUTPUT_FILE%

python extractor.py --book-dir %BOOK_DIR% --output %OUTPUT_FILE%

echo.
echo Processing complete!
