#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the installation of required packages for VLM testing.
"""

import sys
import importlib
import subprocess
import platform

def check_package(package_name):
    """
    Check if a package is installed and get its version.
    
    Args:
        package_name (str): Name of the package to check
    
    Returns:
        tuple: (is_installed, version)
    """
    try:
        module = importlib.import_module(package_name)
        try:
            version = module.__version__
        except AttributeError:
            version = "Unknown"
        return True, version
    except ImportError:
        return False, None

def check_gpu():
    """
    Check if a CUDA-compatible GPU is available.
    
    Returns:
        bool: True if a CUDA-compatible GPU is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def main():
    # List of required packages
    required_packages = [
        "torch",
        "torchvision",
        "transformers",
        "PIL",
        "matplotlib",
        "pandas",
        "sklearn",
        "tqdm",
        "numpy",
        "cv2",
        "fuzzywuzzy"
    ]
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    # Check packages
    print("\nChecking required packages:")
    all_packages_installed = True
    
    for package in required_packages:
        is_installed, version = check_package(package)
        status = "✓" if is_installed else "✗"
        version_str = f"v{version}" if version else "Not installed"
        print(f"  {status} {package}: {version_str}")
        
        if not is_installed:
            all_packages_installed = False
    
    # Check GPU
    print("\nChecking GPU:")
    has_gpu = check_gpu()
    print(f"  {'✓' if has_gpu else '✗'} CUDA-compatible GPU: {'Available' if has_gpu else 'Not available'}")
    
    # Print summary
    print("\nSummary:")
    if all_packages_installed:
        print("  ✓ All required packages are installed")
    else:
        print("  ✗ Some required packages are missing")
    
    if has_gpu:
        print("  ✓ CUDA-compatible GPU is available")
    else:
        print("  ✗ CUDA-compatible GPU is not available (inference will be slower)")
    
    # Print overall status
    print("\nOverall status:")
    if all_packages_installed:
        print("  ✓ Setup is complete, you can run the VLM testing pipeline")
    else:
        print("  ✗ Setup is incomplete, please install the missing packages")
        print("    Run: pip install -r requirements.txt")
    
    return 0 if all_packages_installed else 1

if __name__ == "__main__":
    sys.exit(main()) 