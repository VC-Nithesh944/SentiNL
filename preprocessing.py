"""
Text Preprocessing Module for SentiNL
Handles cleaning and tokenization of movie reviews
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data (one-time)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Handles all text preprocessing steps for sentiment analysis."""

    def __init__(self, use_stemming=True):
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Remove HTML tags and special characters."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def tokenize(self, text):
        """Split text into individual words."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove common words that don't carry sentiment."""
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]

    def stem_words(self, tokens):
        """Reduce words to their root form."""
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess(self, text):
        """
        Full preprocessing pipeline:
        1. Clean text (remove HTML, special chars)
        2. Tokenize
        3. Remove stopwords
        4. Stem (optional)
        """
        # Step 1: Clean
        text = self.clean_text(text)

        # Step 2: Tokenize
        tokens = self.tokenize(text)

        # Step 3: Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Step 4: Stemming
        if self.use_stemming:
            tokens = self.stem_words(tokens)

        # Join back into string for TF-IDF
        return ' '.join(tokens)


def preprocess_dataset(reviews, use_stemming=True):
    """
    Preprocess a list of reviews.

    Args:
        reviews: List of raw review strings
        use_stemming: Whether to apply stemming

    Returns:
        List of preprocessed reviews
    """
    preprocessor = TextPreprocessor(use_stemming=use_stemming)
    return [preprocessor.preprocess(review) for review in reviews]


# Example usage
if __name__ == "__main__":
    sample_review = "<br />This movie was absolutely fantastic! I loved every minute of it."

    preprocessor = TextPreprocessor(use_stemming=True)
    processed = preprocessor.preprocess(sample_review)

    print("Original:", sample_review)
    print("Processed:", processed)