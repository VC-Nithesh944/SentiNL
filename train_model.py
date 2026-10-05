"""
Model Training Script for SentiNL
Downloads IMDB dataset, trains TF-IDF + Logistic Regression model
"""

import os
import pickle
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our preprocessing module
from preprocessing import preprocess_dataset


def download_imdb_dataset(data_dir):
    """
    Download and extract IMDB movie review dataset.
    Uses sklearn's built-in dataset loader.
    """
    print("Loading IMDB dataset...")

    # Use sklearn's built-in IMDB dataset (downloads automatically)
    try:
        from sklearn.datasets import fetch_20newsgroups
        # For IMDB, we'll use the movie review dataset from nltk
        import nltk
        try:
            nltk.data.find('corpora/movie_reviews')
        except LookupError:
            print("Downloading NLTK movie reviews...")
            nltk.download('movie_reviews', quiet=True)

        from nltk.corpus import movie_reviews

        # Load reviews and labels
        reviews = []
        labels = []

        print("Reading positive reviews...")
        for fileid in movie_reviews.fileids('pos'):
            reviews.append(movie_reviews.raw(fileid))
            labels.append('pos')

        print("Reading negative reviews...")
        for fileid in movie_reviews.fileids('neg'):
            reviews.append(movie_reviews.raw(fileid))
            labels.append('neg')

        print(f"Loaded {len(reviews)} reviews")
        return reviews, labels

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def train_model(reviews, labels, use_stemming=True):
    """
    Train the sentiment analysis model.

    Pipeline:
    1. Preprocess text (clean, tokenize, stem)
    2. Vectorize with TF-IDF
    3. Train Logistic Regression

    Args:
        reviews: List of raw review strings
        labels: List of labels ('pos' or 'neg')
        use_stemming: Whether to apply stemming

    Returns:
        Trained vectorizer, trained model, and accuracy score
    """
    print("\n" + "="*50)
    print("TRAINING PIPELINE")
    print("="*50)

    # Step 1: Preprocess text
    print("\n[Step 1] Preprocessing text...")
    print("  - Cleaning HTML tags and special characters")
    print("  - Tokenizing words")
    print("  - Removing stop words")
    if use_stemming:
        print("  - Applying Porter stemming")

    processed_reviews = preprocess_dataset(reviews, use_stemming=use_stemming)
    print(f"  - Preprocessed {len(processed_reviews)} reviews")

    # Step 2: Split data
    print("\n[Step 2] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_reviews, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # Equal pos/neg in both sets
    )
    print(f"  - Training set: {len(X_train)} reviews")
    print(f"  - Test set: {len(X_test)} reviews")

    # Step 3: TF-IDF Vectorization
    print("\n[Step 3] TF-IDF Vectorization...")
    print("  - Converting text to numerical vectors")
    print("  - TF = term frequency in document")
    print("  - IDF = inverse document frequency (rare words matter more)")

    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit vocabulary size
        min_df=2,            # Ignore very rare words
        max_df=0.95,         # Ignore very common words
        ngram_range=(1, 2)   # Unigrams and bigrams
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  - Feature matrix shape: {X_train_tfidf.shape}")

    # Step 4: Train Logistic Regression
    print("\n[Step 4] Training Logistic Regression...")
    print("  - Learning weights for each word/phrase")
    print("  - Positive weight = word indicates positive sentiment")
    print("  - Negative weight = word indicates negative sentiment")

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization strength
        solver='lbfgs',
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)
    print("  - Training complete!")

    # Step 5: Evaluate
    print("\n[Step 5] Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    Predicted:   Neg    Pos")
    print(f"    Actual Neg:  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"    Actual Pos:  {cm[1][0]:4d}  {cm[1][1]:4d}")

    return vectorizer, model, accuracy


def save_model(vectorizer, model, output_dir):
    """Save the trained model and vectorizer to disk."""
    print(f"\nSaving model to {output_dir}...")

    # Save vectorizer
    with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save model
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("  - Saved vectorizer.pkl")
    print("  - Saved model.pkl")


def analyze_model_weights(model, vectorizer, top_n=15):
    """
    Show what the model learned - which words indicate positive/negative sentiment.
    """
    print("\n" + "="*50)
    print("MODEL WEIGHTS ANALYSIS")
    print("="*50)
    print("\nWhat the model learned:")

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get coefficients (weights)
    coefficients = model.coef_[0]

    # Top positive indicators
    print("\nTop words indicating POSITIVE sentiment:")
    top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
    for i, idx in enumerate(top_positive_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:20s} (weight: {coefficients[idx]:.4f})")

    # Top negative indicators
    print("\nTop words indicating NEGATIVE sentiment:")
    top_negative_idx = np.argsort(coefficients)[:top_n]
    for i, idx in enumerate(top_negative_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:20s} (weight: {coefficients[idx]:.4f})")


def main():
    """Main training pipeline."""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Load dataset
    reviews, labels = download_imdb_dataset('data')

    # Train model
    vectorizer, model, accuracy = train_model(reviews, labels, use_stemming=True)

    # Show what the model learned
    analyze_model_weights(model, vectorizer)

    # Save model
    save_model(vectorizer, model, models_dir)

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {models_dir}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()