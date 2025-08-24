@echo off
REM Simple batch file to process a book by ID

echo Book Metadata Extractor
echo ======================

IF "%1"=="" (
    echo Usage: process_book.bat [BOOK_ID]
    echo Example: process_book.bat 1
    echo.
    echo Available books:
    dir /b books
    exit /b 1
)

set BOOK_ID=%1

echo Processing book %BOOK_ID%...
echo.

python process_book.py %BOOK_ID%

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Processing failed!
) ELSE (
    echo.
    echo Processing complete!
    echo Results saved to output/book_%BOOK_ID%.json
)
