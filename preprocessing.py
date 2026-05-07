"""
preprocessing.py — NLP Text Preprocessing Pipeline
====================================================
Modular, reusable preprocessing for the Document Intelligence System.
"""

import re
import string
import nltk

# Ensure required NLTK data is available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# Singleton resources (loaded once)
# ---------------------------------------------------------------------------
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(
    text: str,
    remove_numbers: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
) -> str:
    """Clean and normalise a single document string.

    Pipeline steps:
        1. Lowercase
        2. Remove URLs and email addresses
        3. Remove punctuation (optional)
        4. Remove numbers (optional)
        5. Remove extra whitespace
        6. Tokenize
        7. Remove stopwords (optional)
        8. Lemmatize (optional)
        9. Rejoin into a single string

    Returns
    -------
    str
        Cleaned, space-separated token string ready for vectorisation.
    """
    if not text or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs & emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # 3. Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenize
    tokens = word_tokenize(text)

    # 7. Stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    # 8. Lemmatization
    if apply_lemmatization:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # 9. Rejoin
    return " ".join(tokens)


def preprocess_documents(documents: list[str], **kwargs) -> list[str]:
    """Apply *preprocess_text* to every document in a list.

    Parameters
    ----------
    documents : list[str]
        Raw document strings.
    **kwargs
        Forwarded to :func:`preprocess_text`.

    Returns
    -------
    list[str]
        List of cleaned document strings (same order).
    """
    return [preprocess_text(doc, **kwargs) for doc in documents]
