#!/usr/bin/env python
import argparse
import json
from ocr_testing.hueristics.extractor import extract_metadata_from_text

# Sample OCR text blocks for testing
SAMPLE_INVOICE_TEXT = """
INVOICE
Invoice #: INV-2023-0456
Date: 05/15/2023
Due Date: 06/15/2023

Bill To:
Acme Corporation
123 Business St.
Business City, BC 12345
Contact: John Smith
Email: john.smith@acmecorp.com
Phone: (555) 123-4567

Description                     Quantity    Rate        Amount
-----------------------------------------------------------------
Web Development Services        1           $2,500.00   $2,500.00
Server Maintenance              5           $100.00     $500.00
Domain Renewal                  2           $15.00      $30.00
-----------------------------------------------------------------
                                           Subtotal:    $3,030.00
                                           Tax (10%):   $303.00
                                           Total:       $3,333.00

Payment Details:
Bank Transfer to: ABC Bank
Account #: 123456789
Thank you for your business!
"""

SAMPLE_RECEIPT_TEXT = """
GROCERY MART
123 Food Lane
Foodville, FM 98765
Tel: 987-654-3210

RECEIPT
Transaction #: T-20230610-1234
Date: 10 June 2023
Time: 14:30:25

Cashier: Jane Doe

ITEMS:
1x Milk 1gal          $3.99
2x Bread              $5.98
1x Eggs (dozen)       $2.49
3x Apples             $1.50
1x Chicken Breast     $8.99

Subtotal:             $22.95
Tax (6%):             $1.38
TOTAL:                $24.33

Paid by: VISA ****1234
Amount Paid:          $24.33
Change Due:           $0.00

Thank you for shopping at Grocery Mart!
For customer service: support@grocerymart.com
"""

def test_metadata_extraction(text):
    """
    Test the metadata extraction on the given OCR text.
    
    Args:
        text (str): The OCR text to extract metadata from
        
    Returns:
        dict: The extracted metadata
    """
    print("Input OCR Text:")
    print("-" * 60)
    print(text)
    print("-" * 60)
    
    print("\nExtracting metadata...")
    metadata = extract_metadata_from_text(text)
    
    print("\nExtracted Metadata:")
    print("-" * 60)
    print(json.dumps(metadata, indent=2))
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Test OCR metadata extraction using heuristics')
    parser.add_argument('--sample', '-s', type=str, choices=['invoice', 'receipt'], 
                        help='Use a predefined sample (invoice or receipt)')
    parser.add_argument('--text', '-t', type=str, help='Custom OCR text to analyze')
    parser.add_argument('--file', '-f', type=str, help='Path to a file containing OCR text')
    
    args = parser.parse_args()
    
    # Determine which text to use
    if args.sample == 'invoice':
        text = SAMPLE_INVOICE_TEXT
    elif args.sample == 'receipt':
        text = SAMPLE_RECEIPT_TEXT
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.text:
        text = args.text
    else:
        # Default to invoice sample if no input specified
        text = SAMPLE_INVOICE_TEXT
        print("No input specified, using default invoice sample.")
    
    # Run the test
    test_metadata_extraction(text)

if __name__ == "__main__":
    main() 