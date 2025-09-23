#!/usr/bin/env python
import re

class BookMetadataExtractor:
    """
    A class to extract book-specific metadata from OCR text using heuristic methods.
    """
    
    def __init__(self):
        # Common patterns for book metadata extraction
        self.patterns = {
            'isbn': [
                r'ISBN(?:-1[03])?:?\s*((?:\d[- ]?){9}[\dX])',  # ISBN-10 or ISBN-13
                r'(?:\d[- ]?){9}[\dX]'  # Just the ISBN number without prefix
            ],
            'year': [
                r'\b((?:19|20)\d{2})\b',  # Years between 1900-2099
                r'©\s*(\d{4})',  # Copyright year
                r'(?:published|pub|copyright|©)\s*(?:in|:)?\s*(\d{4})'  # Published year
            ],
            'publisher': [
                r'(?:published by|publisher|pub[.:])\s*([A-Z][A-Za-z\s&]+(?:Press|Publishing|Books|Publications|Publishers))',
                r'\b([A-Z][A-Za-z\s&]+(?:Press|Publishing|Books|Publications|Publishers))\b'
            ],
            'price': [
                r'(?:\$|USD|£|GBP|€|EUR)\s*(\d+\.?\d*)',  # Price with currency symbol
                r'(\d+\.?\d*)\s*(?:\$|USD|£|GBP|€|EUR)'   # Price with currency symbol after
            ]
        }
        
        # Common book publishers for better matching
        self.known_publishers = [
            "Penguin", "Random House", "HarperCollins", "Simon & Schuster", 
            "Hachette", "Macmillan", "Scholastic", "Wiley", "Oxford University Press",
            "Cambridge University Press", "MIT Press", "Pearson", "McGraw-Hill",
            "Bloomsbury", "Vintage", "Knopf", "Bantam", "Ballantine", "Del Rey",
            "Tor Books", "Orbit", "DAW", "Baen", "Ace", "Pocket Books"
        ]

        # Common book-related keywords for title/author identification
        self.book_keywords = ["novel", "fiction", "book", "story", "author", "edition"]
        
    def extract_metadata(self, text):
        """
        Extract book-specific metadata from the given OCR text.
        
        Args:
            text (str): The OCR text to extract metadata from
            
        Returns:
            dict: A dictionary of extracted metadata
        """
        if not text:
            return {}
        
        # Initialize results dictionary
        results = {
            'title': None,
            'author': None,
            'publisher': None,
            'year': None,
            'isbn': None,
            'price': None,
            'genre': None,
            'series': None
        }
        
        # Extract structured metadata using regex patterns
        self._extract_structured_metadata(text, results)
        
        # Extract title and author using simple heuristics
        self._extract_title_author_simple(text, results)
        
        return results
    
    def _extract_structured_metadata(self, text, results):
        """Extract metadata using regex patterns"""
        # Extract ISBN
        for pattern in self.patterns['isbn']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Clean up the ISBN by removing spaces and dashes
                isbn = re.sub(r'[- ]', '', matches[0])
                results['isbn'] = isbn
                break
        
        # Extract year
        for pattern in self.patterns['year']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['year'] = matches[0]
                break
        
        # Extract publisher
        for pattern in self.patterns['publisher']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['publisher'] = matches[0].strip()
                break
        
        # If no publisher found with regex, try to match known publishers
        if not results['publisher']:
            for publisher in self.known_publishers:
                if re.search(r'\b' + re.escape(publisher) + r'\b', text, re.IGNORECASE):
                    results['publisher'] = publisher
                    break
        
        # Extract price
        for pattern in self.patterns['price']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['price'] = matches[0]
                break
    
    def _extract_title_author_simple(self, text, results):
        """Extract title and author using simple heuristics without NLTK"""
        # Split text into lines and clean them
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # If no lines found, try splitting by periods or multiple spaces
        if not lines:
            lines = [s.strip() for s in re.split(r'[.!?]|\s{2,}', text) if s.strip()]
        
        # First non-empty line might be the title
        if lines and not results['title']:
            results['title'] = lines[0]
        
        # Look for potential author patterns (e.g., "by John Smith")
        author_match = re.search(r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        if author_match:
            results['author'] = author_match.group(1)
        elif len(lines) > 1 and not results['author']:
            # Second line might be the author
            results['author'] = lines[1]
        
        # Look for common book series patterns
        series_patterns = [
            r'(?:series|book)\s*[:#]?\s*(\d+)',
            r'([A-Za-z]+)\s+series',
            r'the\s+([A-Za-z]+)\s+(?:trilogy|saga)'
        ]
        
        for pattern in series_patterns:
            series_match = re.search(pattern, text, re.IGNORECASE)
            if series_match:
                results['series'] = series_match.group(1)
                break
    
    def guess_genre(self, text):
        """
        Attempt to guess the book genre based on keywords.
        This is a very simplistic approach and would need to be improved.
        """
        genre_keywords = {
            'fiction': ['novel', 'fiction', 'story', 'stories', 'thriller', 'mystery'],
            'fantasy': ['magic', 'dragon', 'wizard', 'fantasy', 'mythical', 'epic'],
            'science fiction': ['space', 'alien', 'future', 'sci-fi', 'science fiction', 'dystopian'],
            'romance': ['love', 'romance', 'passion', 'relationship'],
            'biography': ['biography', 'memoir', 'autobiography', 'life story'],
            'history': ['history', 'historical', 'century', 'war', 'revolution'],
            'self-help': ['self-help', 'motivation', 'inspiration', 'success', 'happiness'],
            'business': ['business', 'management', 'leadership', 'entrepreneur', 'finance'],
            'children': ['children', 'kids', 'young reader', 'picture book']
        }
        
        text_lower = text.lower()
        genre_matches = {}
        
        for genre, keywords in genre_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                genre_matches[genre] = count
        
        if genre_matches:
            # Return the genre with the most keyword matches
            return max(genre_matches.items(), key=lambda x: x[1])[0]
        
        return None

def extract_book_metadata_from_text(text):
    """
    Convenience function to extract book metadata from OCR text.
    
    Args:
        text (str): The OCR text to extract metadata from
        
    Returns:
        dict: A dictionary of extracted book metadata
    """
    extractor = BookMetadataExtractor()
    metadata = extractor.extract_metadata(text)
    
    # Try to guess the genre
    genre = extractor.guess_genre(text)
    if genre:
        metadata['genre'] = genre
    
    return metadata 