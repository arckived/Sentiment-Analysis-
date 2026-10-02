# Sentiment Analysis of Reddit and Twitter Data

A sentiment classification system that fine-tunes BERT (Hugging Face Transformers) on ~50K labeled Reddit and Twitter posts, then layers on a topic-tracking module using TF-IDF clustering to surface week-over-week shifts in negative sentiment.

## What this project does

1. **Classification**: Fine-tunes a BERT model to classify posts as positive, negative, or neutral.
2. **Preprocessing**: Uses NLTK for text cleaning, tokenization, and stopword removal before training.
3. **Topic-level trend analysis**: Groups posts by topic using TF-IDF clustering, then tracks sentiment over time within each topic, exporting the results as CSV reports. This is the part that turns a plain classifier into something useful for feedback analysis: instead of just "this post is negative," it flags *which topics* are seeing a spike in negative sentiment, and when.

## Results

- **~50,000** labeled posts (Reddit and Twitter), 3-class sentiment (positive / negative / neutral)
- **85% accuracy** on the held-out test set

## Setup

```bash
pip install transformers torch nltk scikit-learn pandas
```

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
```

## Usage

1. Open `sentiment analysis using transformers_NLTK.ipynb`
2. Run the preprocessing cells to clean and tokenize the raw text
3. Run the BERT fine-tuning cells (this trains and evaluates the classifier)
4. Run the TF-IDF clustering cells to generate topic-level trend reports

