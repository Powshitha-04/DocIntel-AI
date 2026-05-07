"""
clustering.py — KMeans Document Clustering
============================================
Groups documents into clusters based on TF-IDF vectors and provides
2-D projections for visualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_documents(
    tfidf_matrix,
    n_clusters: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Run KMeans clustering on the TF-IDF matrix.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF matrix (n_docs × n_features).
    n_clusters : int
        Desired number of clusters.  Automatically clamped to
        ``[2, n_docs]`` to prevent sklearn errors.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Cluster label for each document (shape ``(n_docs,)``).
    """
    n_docs = tfidf_matrix.shape[0]
    n_clusters = max(2, min(n_clusters, n_docs))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(tfidf_matrix)
    return km.labels_


def get_cluster_dataframe(
    labels: np.ndarray,
    doc_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Build a tidy DataFrame mapping documents to their cluster.

    Returns
    -------
    pd.DataFrame
        Columns: ``Document``, ``Cluster``.
    """
    n = len(labels)
    if doc_labels is None:
        doc_labels = [f"Doc {i + 1}" for i in range(n)]

    return pd.DataFrame({"Document": doc_labels, "Cluster": labels})


def get_2d_projection(tfidf_matrix, random_state: int = 42) -> np.ndarray:
    """Reduce TF-IDF vectors to 2-D via PCA for scatter-plot visualisation.

    Returns
    -------
    np.ndarray
        Shape ``(n_docs, 2)``.
    """
    n_components = min(2, tfidf_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(tfidf_matrix.toarray())
