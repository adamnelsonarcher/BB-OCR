#!/usr/bin/env python
import re

class BookMetadataExtractor:
    def __init__(self):
        self.patterns = {
            'isbn': [r'ISBN(?:-1[03])?:?\s*((?:\d[- ]?){9}[\dX])', r'(?:\d[- ]?){9}[\dX]'],
            'year': [r'\b((?:19|20)\d{2})\b', r'©\s*(\d{4})', r'(?:published|pub|copyright|©)\s*(?:in|:)?\s*(\d{4})'],
            'publisher': [r'(?:published by|publisher|pub[.:])\s*([A-Z][A-Za-z\s&]+(?:Press|Publishing|Books|Publications|Publishers))', r'\b([A-Z][A-Za-z\s&]+(?:Press|Publishing|Books|Publications|Publishers))\b'],
            'price': [r'(?:\$|USD|£|GBP|€|EUR)\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:\$|USD|£|GBP|€|EUR)']
        }
        self.known_publishers = [
            "Penguin", "Random House", "HarperCollins", "Simon & Schuster", 
            "Hachette", "Macmillan", "Scholastic", "Wiley", "Oxford University Press",
            "Cambridge University Press", "MIT Press", "Pearson", "McGraw-Hill",
            "Bloomsbury", "Vintage", "Knopf", "Bantam", "Ballantine", "Del Rey",
            "Tor Books", "Orbit", "DAW", "Baen", "Ace", "Pocket Books"
        ]
    def extract_metadata(self, text):
        if not text:
            return {}
        results = {'title': None, 'author': None, 'publisher': None, 'year': None, 'isbn': None, 'price': None, 'genre': None, 'series': None}
        for pattern in self.patterns['isbn']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                isbn = re.sub(r'[- ]', '', matches[0])
                results['isbn'] = isbn
                break
        for pattern in self.patterns['year']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['year'] = matches[0]
                break
        for pattern in self.patterns['publisher']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['publisher'] = matches[0].strip()
                break
        if not results['publisher']:
            for publisher in self.known_publishers:
                if re.search(r'\b' + re.escape(publisher) + r'\b', text, re.IGNORECASE):
                    results['publisher'] = publisher
                    break
        for pattern in self.patterns['price']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['price'] = matches[0]
                break
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            lines = [s.strip() for s in re.split(r'[.!?]|\s{2,}', text) if s.strip()]
        if lines and not results['title']:
            results['title'] = lines[0]
        author_match = re.search(r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        if author_match:
            results['author'] = author_match.group(1)
        elif len(lines) > 1 and not results['author']:
            results['author'] = lines[1]
        for pattern in [r'(?:series|book)\s*[:#]?\s*(\d+)', r'([A-Za-z]+)\s+series', r'the\s+([A-Za-z]+)\s+(?:trilogy|saga)']:
            series_match = re.search(pattern, text, re.IGNORECASE)
            if series_match:
                results['series'] = series_match.group(1)
                break
        return results

def extract_book_metadata_from_text(text):
    extractor = BookMetadataExtractor()
    metadata = extractor.extract_metadata(text)
    return metadata


