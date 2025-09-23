#!/usr/bin/env python
import re

class OCRMetadataExtractor:
    def __init__(self):
        self.patterns = {
            'date': [r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})', r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})'],
            'email': [r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'],
            'phone': [r'(\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', r'(\d{3}[\s.-]\d{3}[\s.-]\d{4})'],
            'amount': [r'[$€£¥]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*[$€£¥]'],
            'invoice_number': [r'(?:invoice|inv|invoice\s+number|inv\s+no|invoice\s+no)[.:\s#]*([a-zA-Z0-9-]+)', r'(?:invoice|inv)[.:\s#]*([a-zA-Z0-9-]+)'],
            'total': [r'(?:total|amount\s+due|grand\s+total|balance\s+due)[.:\s]*[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)']
        }
    def extract_metadata(self, text):
        if not text:
            return {}
        results = {'dates': [], 'emails': [], 'phones': [], 'amounts': [], 'invoice_numbers': [], 'total_amount': None}
        text_lower = text.lower()
        for pattern in self.patterns['date']:
            results['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        for pattern in self.patterns['email']:
            results['emails'].extend(re.findall(pattern, text, re.IGNORECASE))
        for pattern in self.patterns['phone']:
            results['phones'].extend(re.findall(pattern, text, re.IGNORECASE))
        for pattern in self.patterns['amount']:
            results['amounts'].extend(re.findall(pattern, text, re.IGNORECASE))
        for pattern in self.patterns['invoice_number']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                results['invoice_numbers'].extend(matches)
        for pattern in self.patterns['total']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                results['total_amount'] = matches[-1]
                break
        for key in results:
            if isinstance(results[key], list):
                results[key] = list(dict.fromkeys(results[key]))
        return results

def extract_metadata_from_text(text):
    extractor = OCRMetadataExtractor()
    return extractor.extract_metadata(text)


