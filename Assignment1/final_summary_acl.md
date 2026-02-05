# Sentiment140 Preprocessing Summary (ACL Style)

**Author:** MingHsiang Lee  
**Course:** INFO 7610  
**Assignment:** Text Data Analysis and Preprocessing  
**Date:** 2026-02-04

## Abstract
This report summarizes the preprocessing pipeline applied to the Sentiment140 dataset (1.6M tweets). The workflow includes duplicate removal, text cleaning, tokenization, stopword removal, lemmatization (with stemming for comparison), and dataset splitting into train/validation/test sets. Final statistics and output artifacts are reported for downstream sentiment analysis.

## 1. Introduction
Sentiment140 is a large-scale dataset of tweets labeled for sentiment. We preprocess the data to reduce noise, normalize text, and generate consistent inputs for machine learning models.

## 2. Dataset
- **Source:** Kaggle – Sentiment140 (KazAnova)
- **File:** `training.1600000.processed.noemoticon.csv` (renamed to `Sentiment140.csv`)
- **Labels:** 0 = negative, 4 = positive (converted to binary: 0/1)

## 3. Preprocessing Pipeline
### 3.1 Deduplication
- Removed duplicate tweets based on identical text content.

### 3.2 Missing/Empty Text Handling
- Dropped rows with missing text.
- Detected tweets that became empty after cleaning; saved to `empty_text_cleaned.csv` and removed from the dataset.

### 3.3 Text Cleaning
Applied the following steps:
- URL removal
- HTML tag removal
- @mention removal
- Hashtag symbol removal (kept word)
- Emoji removal
- Special character removal
- Whitespace normalization
- Lowercasing

### 3.4 Tokenization & Normalization
- Tokenization using NLTK `TweetTokenizer`.
- Stopword removal (NLTK English stopwords).
- Lemmatization using WordNet (primary normalized text).
- Stemming using PorterStemmer (comparison only).

## 4. Data Statistics (Final)
**All values can be updated after re-running the notebook.**
- **Final dataset size:** 1,XXX,XXX tweets
- **Negative (0):** XXX,XXX (XX.XX%)
- **Positive (1):** XXX,XXX (XX.XX%)
- **Avg tokens per tweet:** XX.XX
- **Text length (mean):** XX.XX characters

## 5. Data Splits (Stratified)
- **Train:** 70% (XXX,XXX)
- **Validation:** 15% (XXX,XXX)
- **Test:** 15% (XXX,XXX)
- **Stratification:** Maintained label distribution across splits.

## 6. Output Files
- `sentiment140_train.csv`
- `sentiment140_val.csv`
- `sentiment140_test.csv`
- `sentiment140_processed.csv`
- `empty_text_cleaned.csv` (if any)

## 7. Notes
- The original `Sentiment140.csv` is not modified.
- The processed text used for modeling is the lemmatized, stopword-removed version.

---
**Update instructions:** After rerunning the notebook, replace the placeholder numbers in Sections 4–5 with the latest values.
