#!/usr/bin/env python
import re
import datetime

class OCRMetadataExtractor:
    """
    A class to extract metadata from OCR text using heuristic methods.
    """
    
    def __init__(self):
        # Common regex patterns for metadata extraction
        self.patterns = {
            'date': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
                r'(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})',  # YYYY/MM/DD
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',  # 10 January 2023
            ],
            'email': [
                r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
            ],
            'phone': [
                r'(\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})',  # +1 (123) 456-7890
                r'(\d{3}[\s.-]\d{3}[\s.-]\d{4})'  # 123-456-7890
            ],
            'amount': [
                r'[$€£¥]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1,234.56
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*[$€£¥]'   # 1,234.56$
            ],
            'invoice_number': [
                r'(?:invoice|inv|invoice\s+number|inv\s+no|invoice\s+no)[.:\s#]*([a-zA-Z0-9-]+)',
                r'(?:invoice|inv)[.:\s#]*([a-zA-Z0-9-]+)'
            ],
            'total': [
                r'(?:total|amount\s+due|grand\s+total|balance\s+due)[.:\s]*[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            ]
        }
    
    def extract_metadata(self, text):
        """
        Extract metadata from the given OCR text.
        
        Args:
            text (str): The OCR text to extract metadata from
            
        Returns:
            dict: A dictionary of extracted metadata
        """
        if not text:
            return {}
        
        # Initialize results dictionary
        results = {
            'dates': [],
            'emails': [],
            'phones': [],
            'amounts': [],
            'invoice_numbers': [],
            'total_amount': None
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Extract dates
        for pattern in self.patterns['date']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results['dates'].extend(matches)
        
        # Extract emails
        for pattern in self.patterns['email']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results['emails'].extend(matches)
        
        # Extract phone numbers
        for pattern in self.patterns['phone']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results['phones'].extend(matches)
        
        # Extract monetary amounts
        for pattern in self.patterns['amount']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results['amounts'].extend(matches)
        
        # Extract invoice numbers
        for pattern in self.patterns['invoice_number']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                results['invoice_numbers'].extend(matches)
        
        # Extract total amount
        for pattern in self.patterns['total']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Use the last match as it's more likely to be the grand total
                results['total_amount'] = matches[-1]
                break
        
        # Remove duplicates while preserving order
        for key in results:
            if isinstance(results[key], list):
                results[key] = list(dict.fromkeys(results[key]))
        
        return results

def extract_metadata_from_text(text):
    """
    Convenience function to extract metadata from OCR text.
    
    Args:
        text (str): The OCR text to extract metadata from
        
    Returns:
        dict: A dictionary of extracted metadata
    """
    extractor = OCRMetadataExtractor()
    return extractor.extract_metadata(text) 