"""
similarity_engine.py — Cosine Similarity Computation
=====================================================
Builds pairwise similarity matrices and ranks the most similar document pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(
    tfidf_matrix,
    doc_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise cosine similarity for all documents.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF matrix (n_docs × n_features).
    doc_labels : list[str] | None
        Human-readable document names. Defaults to Doc 1, Doc 2, …

    Returns
    -------
    pd.DataFrame
        Symmetric similarity matrix with labelled rows/columns.
    """
    sim = cosine_similarity(tfidf_matrix)
    n = sim.shape[0]

    if doc_labels is None:
        doc_labels = [f"Doc {i + 1}" for i in range(n)]

    return pd.DataFrame(sim, index=doc_labels, columns=doc_labels)


def get_top_pairs(
    sim_matrix: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """Return the top-*n* most similar document pairs (excluding self-pairs).

    Parameters
    ----------
    sim_matrix : pd.DataFrame
        Symmetric similarity matrix from :func:`compute_similarity_matrix`.
    n : int
        Number of top pairs to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``Document A``, ``Document B``, ``Similarity Score``.
    """
    labels = sim_matrix.columns.tolist()
    pairs: list[dict] = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append(
                {
                    "Document A": labels[i],
                    "Document B": labels[j],
                    "Similarity Score": round(float(sim_matrix.iloc[i, j]), 4),
                }
            )

    df = pd.DataFrame(pairs)
    df = df.sort_values("Similarity Score", ascending=False).head(n).reset_index(drop=True)
    return df
