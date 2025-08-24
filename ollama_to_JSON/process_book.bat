@echo off
REM Simple batch file to process a book by ID

IF "%1"=="list" (
    echo Listing available Ollama models...
    python process_book.py 1 --model list
    exit /b
)

IF "%1"=="" (
    echo Usage: process_book.bat [BOOK_ID] [MODEL]
    echo.
    echo Examples:
    echo   process_book.bat 1            - Process book 1 with default model
    echo   process_book.bat 2 llava      - Process book 2 with llava model
    echo   process_book.bat list         - List available models
    echo.
    echo Available books:
    dir /b books
    exit /b
)

set BOOK_ID=%1
set MODEL=%2

IF "%MODEL%"=="" (
    echo Processing book %BOOK_ID% with default model...
    python process_book.py %BOOK_ID%
) ELSE (
    echo Processing book %BOOK_ID% with model %MODEL%...
    python process_book.py %BOOK_ID% --model %MODEL%
)