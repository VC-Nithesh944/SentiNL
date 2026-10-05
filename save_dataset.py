"""
Save IMDB dataset to data folder for inspection
"""
import os
import json
import nltk
from nltk.corpus import movie_reviews

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    nltk.download('movie_reviews', quiet=True)

data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Load and save reviews
print("Loading dataset...")

reviews_data = {
    "positive": [],
    "negative": [],
    "metadata": {
        "source": "NLTK movie_reviews",
        "total_positive": 0,
        "total_negative": 0
    }
}

# Get positive reviews
print("Reading positive reviews...")
for fileid in movie_reviews.fileids('pos'):
    text = movie_reviews.raw(fileid)
    reviews_data["positive"].append({
        "fileid": fileid,
        "text": text[:500]  # First 500 chars for preview
    })
    reviews_data["metadata"]["total_positive"] += 1

# Get negative reviews
print("Reading negative reviews...")
for fileid in movie_reviews.fileids('neg'):
    text = movie_reviews.raw(fileid)
    reviews_data["negative"].append({
        "fileid": fileid,
        "text": text[:500]  # First 500 chars for preview
    })
    reviews_data["metadata"]["total_negative"] += 1

# Save as JSON (with truncated text for file size)
output_path = os.path.join(data_dir, 'imdb_reviews.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(reviews_data, f, indent=2)

print(f"\nSaved to: {output_path}")
print(f"Positive reviews: {reviews_data['metadata']['total_positive']}")
print(f"Negative reviews: {reviews_data['metadata']['total_negative']}")

# Also create a sample file with full text of 5 reviews each
sample_data = {
    "positive_samples": [],
    "negative_samples": []
}

# 5 full positive reviews
for fileid in movie_reviews.fileids('pos')[:5]:
    sample_data["positive_samples"].append({
        "fileid": fileid,
        "text": movie_reviews.raw(fileid)
    })

# 5 full negative reviews
for fileid in movie_reviews.fileids('neg')[:5]:
    sample_data["negative_samples"].append({
        "fileid": fileid,
        "text": movie_reviews.raw(fileid)
    })

sample_path = os.path.join(data_dir, 'sample_reviews.json')
with open(sample_path, 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, indent=2)

print(f"Sample data (5 each): {sample_path}")
print("\nDone! Check the data folder.")