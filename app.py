"""
SentiNL - Customer Review Sentiment Analyzer
Streamlit web application for analyzing movie review sentiment
"""

import streamlit as st
import pickle
import os

# Import preprocessing
from preprocessing import TextPreprocessor


def load_model(models_dir):
    """Load saved model and vectorizer."""
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    model_path = os.path.join(models_dir, 'model.pkl')

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return vectorizer, model


def predict_sentiment(text, vectorizer, model, preprocessor):
    """Predict sentiment for a given review text."""
    # Preprocess the input text
    processed_text = preprocessor.preprocess(text)

    # Transform to TF-IDF features
    tfidf_features = vectorizer.transform([processed_text])

    # Predict
    prediction = model.predict(tfidf_features)[0]

    # Get confidence scores
    probabilities = model.predict_proba(tfidf_features)[0]
    confidence = max(probabilities)

    return prediction, confidence


# Page configuration
st.set_page_config(
    page_title="SentiNL - Sentiment Analyzer",
    page_icon="💭",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Get models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Load model on startup
@st.cache_resource
def init_model():
    try:
        return load_model(models_dir)
    except FileNotFoundError:
        return None, None

vectorizer, model = init_model()
preprocessor = TextPreprocessor(use_stemming=True)


# Main app
st.title("💭 SentiNL")
st.subheader("Customer Review Sentiment Analyzer")

# Check if model is loaded
if vectorizer is None or model is None:
    st.error("Model not found. Please run `python train_model.py` first to train the model.")
    st.info("Run: `python train_model.py` in the SentiNL directory")
else:
    # Demo examples
    st.markdown("### Quick Examples")
    demo_reviews = {
        "🎬 Great Movie": "This movie was absolutely fantastic! I loved every minute of it. The acting was superb and the storyline kept me engaged throughout.",
        "🎬 Terrible Movie": "What a waste of time. The plot was boring, the acting was terrible, and I couldn't wait for it to end. Total disaster.",
        "🎬 Mixed Feelings": "The movie had some good moments but also many flaws. The acting was great but the pacing was slow. Overall it was okay."
    }

    # Demo buttons in columns
    cols = st.columns(3)
    for i, (label, review) in enumerate(demo_reviews.items()):
        if cols[i].button(label, key=f"demo_{i}"):
            st.session_state.review_text = review

    # Text input
    st.markdown("### Enter Your Review")
    review_text = st.text_area(
        "Paste your customer review below:",
        height=150,
        placeholder="Type or paste your review here...",
        key="review_text"
    )

    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_text.strip():
            with st.spinner("Analyzing..."):
                # Predict
                prediction, confidence = predict_sentiment(
                    review_text, vectorizer, model, preprocessor
                )

                # Map prediction to label
                sentiment_label = "Positive" if prediction == "pos" else "Negative"
                emoji = "😊" if prediction == "pos" else "😞"

                # Display results
                st.markdown("---")
                st.markdown("### 📊 Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Sentiment:**")
                    if prediction == "pos":
                        st.markdown(f"<h2 class='sentiment-positive'>{emoji} {sentiment_label}</h2>",
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 class='sentiment-negative'>{emoji} {sentiment_label}</h2>",
                                   unsafe_allow_html=True)

                with col2:
                    st.markdown("**Confidence:**")
                    conf_percentage = confidence * 100
                    st.markdown(f"### {conf_percentage:.1f}%")

                    # Confidence bar
                    bar_color = "#28a745" if prediction == "pos" else "#dc3545"
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {conf_percentage}%; background-color: {bar_color};"></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Detailed breakdown
                st.markdown("---")
                st.markdown("### 📝 How it works")
                st.info("""
                **Pipeline:**
                1. Text preprocessing (cleaning, tokenization, stemming)
                2. TF-IDF vectorization (convert text to numbers)
                3. Logistic Regression classification
                4. Sentiment prediction with confidence score
                """)
        else:
            st.warning("Please enter a review to analyze.")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "<small>SentiNL - NLP Sentiment Analysis Demo</small>"
    "</div>",
    unsafe_allow_html=True
)