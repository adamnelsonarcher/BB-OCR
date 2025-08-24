@echo off
REM Book Metadata Extraction Tool
REM Simple batch file for Windows users

cd %~dp0\ollama_to_JSON
python book.py %*
cd ..
