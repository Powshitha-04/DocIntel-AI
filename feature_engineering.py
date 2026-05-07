"""
feature_engineering.py — TF-IDF Vectorization & Keyword Extraction
===================================================================
Converts preprocessed text into numerical representations and extracts
the most significant terms per document.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_matrix(
    documents: list[str],
    max_features: int | None = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple:
    """Fit a TF-IDF vectorizer on *documents* and return the matrix.

    Parameters
    ----------
    documents : list[str]
        Pre-processed document strings.
    max_features : int | None
        Cap on vocabulary size.
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction.

    Returns
    -------
    tuple[scipy.sparse matrix, TfidfVectorizer]
        The TF-IDF sparse matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # apply 1 + log(tf)
        smooth_idf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


def get_top_keywords(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    doc_index: int,
    n: int = 10,
) -> pd.DataFrame:
    """Extract the top-*n* keywords for a single document.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer (provides feature names).
    tfidf_matrix : sparse matrix
        TF-IDF matrix produced by :func:`build_tfidf_matrix`.
    doc_index : int
        Row index of the target document.
    n : int
        Number of top keywords to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``keyword``, ``tfidf_score`` (sorted descending).
    """
    feature_names = vectorizer.get_feature_names_out()
    row = tfidf_matrix[doc_index].toarray().flatten()
    top_indices = row.argsort()[::-1][:n]

    keywords = [feature_names[i] for i in top_indices if row[i] > 0]
    scores = [round(float(row[i]), 4) for i in top_indices if row[i] > 0]

    return pd.DataFrame({"keyword": keywords, "tfidf_score": scores})


def get_all_keywords(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    n: int = 10,
) -> dict[int, pd.DataFrame]:
    """Extract top keywords for **every** document.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from document index to its keyword DataFrame.
    """
    return {
        i: get_top_keywords(vectorizer, tfidf_matrix, i, n)
        for i in range(tfidf_matrix.shape[0])
    }
