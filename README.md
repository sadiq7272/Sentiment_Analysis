# Twitter US Airline Sentiment Analysis

An end-to-end **Natural Language Processing + Machine Learning** project that classifies tweets about US airlines as **negative**, **neutral**, or **positive**.

Built with `scikit-learn`, `NLTK` and `pandas`. Ships as both a reproducible training script and an annotated Jupyter notebook.

---

## Overview

Airlines receive thousands of mentions a day. Manually reading them does not scale, so this project trains a classifier that reads a tweet and assigns a sentiment label — the same idea behind brand-monitoring and customer-feedback dashboards.

The pipeline:

| Step | What happens | Why |
|---|---|---|
| 1. Load | Read the Twitter US Airline Sentiment CSV | Raw labelled data |
| 2. Filter | Drop rows with annotator confidence `< 0.5` | Weak labels are noise for Naive Bayes |
| 3. Clean | Strip URLs, `@mentions`, digits and punctuation; lowercase; remove stopwords; lemmatize | Shrinks the vocabulary to meaningful tokens |
| 4. Encode | `negative → 0`, `neutral → 1`, `positive → 2` | scikit-learn needs numeric targets |
| 5. Vectorize | `CountVectorizer` (or TF-IDF), 5,000 features, unigrams + bigrams | Turns text into a numeric matrix |
| 6. Train | `MultinomialNB` | Fast, probabilistic, strong bag-of-words baseline |
| 7. Evaluate | Accuracy, per-class precision/recall/F1, confusion matrix | Accuracy alone is misleading on imbalanced data |
| 8. Persist | Save model + vectorizer with `joblib` | Reuse for inference without retraining |

---

## Dataset

**Twitter US Airline Sentiment** — ~14,600 tweets from February 2015 about six major US airlines, each labelled by crowd annotators.

Download it here: <https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment>


```
data/Tweets.csv
```

Columns actually used: `text`, `airline_sentiment`, `airline_sentiment_confidence` (plus `airline` and `negativereason` for the EDA plots).

> The dataset is imbalanced — roughly **63% negative, 21% neutral, 16% positive**. Keep that in mind when reading the accuracy number.

---


## Setup

```bash
# 1. Clone
git clone https://github.com/naqi7272/Sentiment_Analysis.git
cd Sentiment_Analysis

# 2. Virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

NLTK corpora (`stopwords`, `wordnet`) download automatically on first run. To fetch them manually:

```bash
python -c "import nltk; [nltk.download(r) for r in ('stopwords','wordnet','omw-1.4')]"
```

---

## Usage

### Train

```bash
python sentiment_analysis.py --data data/Tweets.csv
```

### Options

```bash
python sentiment_analysis.py \
    --data data/Tweets.csv \
    --vectorizer tfidf \        # count | tfidf
    --confidence 0.5 \          # min annotator confidence
    --test-size 0.3 \
    --max-features 5000 \
    --alpha 1.0 \               # Naive Bayes smoothing
    --model-dir models
```

### Use it as a library

```python
from sentiment_analysis import run_pipeline, predict_sentiment

model, vectorizer, accuracy = run_pipeline("data/Tweets.csv")

results = predict_sentiment(["Thanks for the smooth flight!"])
print(results[0]["sentiment"])   # -> positive
```

### Notebook

```bash
jupyter notebook Sentiment___Analysis.ipynb
```

The notebook adds exploratory plots (sentiment by airline, top complaint reasons), cross-validation, top indicative tokens per class, and a side-by-side comparison of Naive Bayes vs Logistic Regression vs LinearSVC.

---


**Reading the results:** the model handles negative tweets well (most training data, distinctive vocabulary like *delayed*, *cancelled*, *hours*, *worst*). Neutral is the hardest class — neutral tweets are mostly logistics questions that share vocabulary with both other classes.

TF-IDF features with Logistic Regression typically add a couple of points over the Naive Bayes baseline.

---

## How it works

**Text cleaning.** Tweets are noisy: handles, links, emoji, inconsistent casing. Everything non-alphabetic is stripped, tokens are lowercased, stopwords removed, and each remaining token is lemmatized (`completing → complete`, `flights → flight`) so different inflections collapse into one feature.

**Airline handles are blacklisted.** `@united`, `@JetBlue` etc. are the most frequent tokens in the corpus but carry zero sentiment signal — they only tell you which airline was tagged. Leaving them in wastes vocabulary slots and lets the model learn airline identity instead of sentiment.

**Why Multinomial Naive Bayes.** It models `P(word | class)` directly from token counts and applies Bayes' rule. That assumption (features independent given the class) is wrong for language, but it works remarkably well for text classification, trains in milliseconds, and needs far less data than a neural model.

**Bigrams matter.** `ngram_range=(1, 2)` lets the model see `not happy` and `great service` as single features rather than losing the relationship between adjacent words.

---

## Tech stack

`Python` · `pandas` · `NumPy` · `NLTK` · `scikit-learn` · `Matplotlib` · `seaborn` · `joblib`

---

## License

Released under the [MIT License](LICENSE).
