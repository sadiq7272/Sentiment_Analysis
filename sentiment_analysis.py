"""
Twitter US Airline Sentiment Analysis
=====================================

An end-to-end NLP + Machine Learning pipeline that classifies tweets about US
airlines as **negative**, **neutral**, or **positive**.

Pipeline
--------
1. Load the Twitter US Airline Sentiment dataset (CSV).
2. Drop low-confidence rows (weakly labelled data hurts training).
3. Clean the text: strip URLs/mentions/non-alphabetic chars, lowercase,
   remove stopwords + punctuation, lemmatize.
4. Encode labels: negative -> 0, neutral -> 1, positive -> 2.
5. Vectorize with CountVectorizer (bag-of-words) or TF-IDF.
6. Train a Multinomial Naive Bayes classifier.
7. Evaluate (accuracy, classification report, confusion matrix).
8. Persist the model + vectorizer so they can be reused for inference.

Usage
-----
    python sentiment_analysis.py --data data/Tweets.csv
    python sentiment_analysis.py --data data/Tweets.csv --vectorizer tfidf
    python sentiment_analysis.py --predict "The flight was delayed 4 hours"

"""

from __future__ import annotations

import argparse
import os
import re
import string
import sys
from typing import Iterable, List, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend so the script runs headless
import matplotlib.pyplot as plt  # noqa: E402

import nltk  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402
from sklearn.feature_extraction.text import (  # noqa: E402
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SENTIMENTS: List[str] = ["negative", "neutral", "positive"]

# Airline handles carry no sentiment signal but dominate the vocabulary,
# so we blacklist them at vectorization time.
AIRLINE_STOPWORDS: List[str] = [
    "virginamerica",
    "united",
    "southwestair",
    "jetblue",
    "usairways",
    "americanair",
    "flight",
    "flights",
    "airline",
    "airlines",
]

# Minimal fallback used only if the NLTK corpora cannot be downloaded
# (e.g. offline machine / CI runner with no network access).
FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now",
}

MODEL_FILENAME = "sentiment_model.joblib"
VECTORIZER_FILENAME = "vectorizer.joblib"


# --------------------------------------------------------------------------- #
# NLTK setup
# --------------------------------------------------------------------------- #

def ensure_nltk_resources(quiet: bool = True) -> set:
    """Download the NLTK corpora we need and return the English stopword set.

    Falls back to a built-in stopword list if the download fails, so the
    pipeline still runs on machines without internet access.
    """
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=quiet)
        except Exception:  # pragma: no cover - network dependent
            pass

    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:  # pragma: no cover - network dependent
        print("[warn] NLTK stopwords unavailable, using built-in fallback list.")
        return set(FALLBACK_STOPWORDS)


# --------------------------------------------------------------------------- #
# Data loading & filtering
# --------------------------------------------------------------------------- #

def load_dataset(path: str) -> pd.DataFrame:
    """Read the Twitter US Airline Sentiment CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download 'Twitter US Airline Sentiment' from Kaggle and place "
            "Tweets.csv inside the data/ folder."
        )

    df = pd.read_csv(path)

    required = {"text", "airline_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")

    print(f"[info] Loaded {len(df):,} rows x {df.shape[1]} columns from '{path}'.")
    return df


def filter_by_confidence(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop rows whose annotator confidence is below `threshold`.

    Low-confidence labels are essentially noise, and Naive Bayes is sensitive
    to noisy word/label co-occurrence counts.
    """
    if "airline_sentiment_confidence" not in df.columns:
        print("[warn] No confidence column found, skipping confidence filter.")
        return df.reset_index(drop=True)

    before = len(df)
    filtered = df[df["airline_sentiment_confidence"] >= threshold].reset_index(drop=True)
    print(
        f"[info] Confidence filter (>= {threshold}): "
        f"removed {before - len(filtered):,} rows, {len(filtered):,} remain."
    )
    return filtered


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

class TextCleaner:
    """Reusable tweet cleaner: URL/mention stripping, lemmatization, stopwords."""

    URL_RE = re.compile(r"http\S+|www\.\S+")
    MENTION_RE = re.compile(r"@\w+")
    NON_ALPHA_RE = re.compile(r"[^a-zA-Z]")
    MULTISPACE_RE = re.compile(r"\s+")

    def __init__(self, stop_words: Iterable[str] | None = None, min_token_len: int = 2):
        self.stop_words = set(stop_words) if stop_words else ensure_nltk_resources()
        self.punctuation = set(string.punctuation)
        self.min_token_len = min_token_len
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """Clean a single string and return space-joined lemmas."""
        if not isinstance(text, str):
            return ""

        text = self.URL_RE.sub(" ", text)
        text = self.MENTION_RE.sub(" ", text)
        text = self.NON_ALPHA_RE.sub(" ", text)
        text = self.MULTISPACE_RE.sub(" ", text).strip().lower()

        tokens = [
            self._lemmatize(token)
            for token in text.split()
            if token not in self.stop_words
            and token not in self.punctuation
            and len(token) >= self.min_token_len
        ]
        return " ".join(tokens)

    def _lemmatize(self, token: str) -> str:
        try:
            return self.lemmatizer.lemmatize(token)
        except LookupError:  # pragma: no cover - wordnet not downloaded
            return token

    def transform(self, texts: Sequence[str]) -> List[str]:
        """Clean an entire corpus."""
        return [self.clean(t) for t in texts]


# --------------------------------------------------------------------------- #
# Feature engineering & model
# --------------------------------------------------------------------------- #

def build_vectorizer(kind: str = "count", max_features: int = 5000):
    """Return a fitted-ready CountVectorizer or TfidfVectorizer."""
    params = dict(
        max_features=max_features,
        stop_words=AIRLINE_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
    )

    if kind == "tfidf":
        return TfidfVectorizer(**params)
    if kind == "count":
        return CountVectorizer(**params)
    raise ValueError(f"Unknown vectorizer '{kind}'. Choose 'count' or 'tfidf'.")


def encode_labels(labels: pd.Series) -> pd.Series:
    """Map the sentiment strings to integer class indices."""
    unknown = set(labels.unique()) - set(SENTIMENTS)
    if unknown:
        raise ValueError(f"Unexpected sentiment label(s): {sorted(unknown)}")
    return labels.apply(SENTIMENTS.index)


def train_model(X_train, y_train, alpha: float = 1.0) -> MultinomialNB:
    """Fit a Multinomial Naive Bayes classifier.

    Naive Bayes is a strong baseline for bag-of-words text classification:
    it is fast, needs little data, and models word/class probabilities directly.
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(model, X_test, y_test) -> Tuple[float, str, np.ndarray]:
    """Print and return accuracy, classification report and confusion matrix."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=SENTIMENTS, digits=4)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\n[result] Test accuracy: {accuracy:.4f}\n")
    print("[result] Classification report:")
    print(report)

    return accuracy, report, matrix


def plot_confusion_matrix(matrix: np.ndarray, output_path: str = "confusion_matrix.png") -> None:
    """Save a labelled confusion-matrix figure to disk."""
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=SENTIMENTS)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix - Airline Sentiment")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[info] Confusion matrix saved to '{output_path}'.")


def top_features(model, vectorizer, n: int = 15) -> None:
    """Print the most indicative tokens for each sentiment class."""
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
    except AttributeError:  # pragma: no cover - very old sklearn
        feature_names = np.array(vectorizer.get_feature_names())

    print(f"\n[info] Top {n} indicative tokens per class:")
    for idx, label in enumerate(SENTIMENTS):
        top_idx = np.argsort(model.feature_log_prob_[idx])[-n:][::-1]
        print(f"  {label:>8}: {', '.join(feature_names[top_idx])}")


# --------------------------------------------------------------------------- #
# Persistence & inference
# --------------------------------------------------------------------------- #

def save_artifacts(model, vectorizer, model_dir: str = "models") -> None:
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, MODEL_FILENAME))
    joblib.dump(vectorizer, os.path.join(model_dir, VECTORIZER_FILENAME))
    print(f"[info] Model + vectorizer saved to '{model_dir}/'.")


def load_artifacts(model_dir: str = "models"):
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    vec_path = os.path.join(model_dir, VECTORIZER_FILENAME)

    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        raise FileNotFoundError(
            f"No saved artifacts in '{model_dir}/'. Train the model first:\n"
            "    python sentiment_analysis.py --data data/Tweets.csv"
        )

    return joblib.load(model_path), joblib.load(vec_path)


def predict_sentiment(texts: Sequence[str], model_dir: str = "models") -> List[dict]:
    """Predict sentiment for raw, uncleaned text using saved artifacts."""
    model, vectorizer = load_artifacts(model_dir)
    cleaner = TextCleaner()

    cleaned = cleaner.transform(texts)
    features = vectorizer.transform(cleaned)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = []
    for text, pred, proba in zip(texts, predictions, probabilities):
        results.append(
            {
                "text": text,
                "sentiment": SENTIMENTS[pred],
                "confidence": float(proba.max()),
                "scores": {label: float(p) for label, p in zip(SENTIMENTS, proba)},
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_pipeline(
    data_path: str,
    confidence: float = 0.5,
    test_size: float = 0.3,
    max_features: int = 5000,
    vectorizer_kind: str = "count",
    alpha: float = 1.0,
    model_dir: str = "models",
    plot_path: str = "outputs/confusion_matrix.png",
    random_state: int = 42,
):
    """Run the full train/evaluate/save pipeline."""
    print("=" * 70)
    print("TWITTER US AIRLINE SENTIMENT ANALYSIS")
    print("=" * 70)

    # 1. Load + filter
    df = load_dataset(data_path)
    df = filter_by_confidence(df, confidence)

    print("\n[info] Class distribution:")
    print(df["airline_sentiment"].value_counts().to_string())

    # 2. Split into features / labels
    X_raw = df["text"]
    y = encode_labels(df["airline_sentiment"])

    # 3. Clean text
    print("\n[info] Cleaning text ...")
    cleaner = TextCleaner()
    X_clean = cleaner.transform(X_raw.tolist())
    print(f"[info] Example -> raw:     {X_raw.iloc[0][:80]}")
    print(f"[info] Example -> cleaned: {X_clean[0][:80]}")

    # 4. Vectorize
    print(f"\n[info] Vectorizing with {vectorizer_kind} (max_features={max_features}) ...")
    vectorizer = build_vectorizer(vectorizer_kind, max_features)
    X = vectorizer.fit_transform(X_clean)
    print(f"[info] Feature matrix shape: {X.shape}")

    # 5. Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[info] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # 6. Train
    print("\n[info] Training MultinomialNB ...")
    model = train_model(X_train, y_train, alpha=alpha)

    # 7. Evaluate
    accuracy, _, matrix = evaluate_model(model, X_test, y_test)
    top_features(model, vectorizer)

    # 8. Persist
    plot_confusion_matrix(matrix, plot_path)
    save_artifacts(model, vectorizer, model_dir)

    print("\n[done] Pipeline finished successfully.")
    return model, vectorizer, accuracy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train / use a sentiment classifier on airline tweets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/Tweets.csv", help="Path to the dataset CSV")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min label confidence")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split fraction")
    parser.add_argument("--max-features", type=int, default=5000, help="Vocabulary size cap")
    parser.add_argument(
        "--vectorizer", choices=["count", "tfidf"], default="count", help="Feature extractor"
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Naive Bayes smoothing")
    parser.add_argument("--model-dir", default="models", help="Where to save artifacts")
    parser.add_argument(
        "--plot-path", default="outputs/confusion_matrix.png", help="Confusion matrix image path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--predict",
        nargs="+",
        metavar="TEXT",
        help="Skip training and classify one or more strings with the saved model",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.predict:
        for result in predict_sentiment(args.predict, args.model_dir):
            scores = "  ".join(f"{k}={v:.3f}" for k, v in result["scores"].items())
            print(f"\ntext      : {result['text']}")
            print(f"sentiment : {result['sentiment'].upper()} ({result['confidence']:.2%})")
            print(f"scores    : {scores}")
        return 0

    run_pipeline(
        data_path=args.data,
        confidence=args.confidence,
        test_size=args.test_size,
        max_features=args.max_features,
        vectorizer_kind=args.vectorizer,
        alpha=args.alpha,
        model_dir=args.model_dir,
        plot_path=args.plot_path,
        random_state=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
