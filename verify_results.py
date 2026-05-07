"""
verify_results.py — End-to-End Verification Script
====================================================
Tests each module of the Document Intelligence System with known inputs
and validates correctness of outputs.
"""

import sys
import numpy as np
import pandas as pd

# ── Test documents with KNOWN semantic relationships ──────────────────────
# Doc A & B are about machine learning → should be highly similar
# Doc C is about cooking → should be dissimilar to A & B
# Doc D is about machine learning (different wording) → moderately similar to A/B
DOCS_RAW = {
    "ML_Basics": (
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn from data without being explicitly programmed. "
        "Supervised learning uses labeled training data to build predictive models. "
        "Neural networks and deep learning are powerful machine learning techniques."
    ),
    "ML_Advanced": (
        "Deep learning algorithms use neural networks with many layers to discover "
        "patterns in large datasets. Artificial intelligence and machine learning "
        "are transforming data science. Training models requires significant "
        "computational resources and labeled data."
    ),
    "Cooking_Recipe": (
        "To make a delicious pasta, boil water with salt and cook the noodles for "
        "eight minutes. Meanwhile, sauté garlic and tomatoes in olive oil. "
        "Season with basil, oregano, and pepper. Serve the pasta with fresh "
        "parmesan cheese on top."
    ),
    "ML_Applications": (
        "Machine learning applications include image recognition, natural language "
        "processing, and recommendation systems. These algorithms process vast "
        "amounts of data to make accurate predictions. AI models are deployed "
        "across healthcare, finance, and technology sectors."
    ),
}

doc_names = list(DOCS_RAW.keys())
raw_texts = list(DOCS_RAW.values())

PASS = 0
FAIL = 0


def check(condition: bool, description: str):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {description}")
    else:
        FAIL += 1
        print(f"  [FAIL] {description}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. PREPROCESSING MODULE")
print("=" * 70)

from preprocessing import preprocess_text, preprocess_documents

cleaned = preprocess_documents(raw_texts)

# Basic sanity checks
check(len(cleaned) == len(raw_texts), f"Output length matches input ({len(cleaned)} == {len(raw_texts)})")
check(all(isinstance(d, str) for d in cleaned), "All outputs are strings")
check(all(len(d) > 0 for d in cleaned), "No empty cleaned documents")

# Preprocessing quality checks
sample = cleaned[0]
check(sample == sample.lower(), "Text is lowercased")
check(not any(char.isdigit() for char in sample), "Numbers are removed")
check("the" not in sample.split() and "is" not in sample.split() and "a" not in sample.split(),
      "Common stopwords removed ('the', 'is', 'a')")

# Check that meaningful words survive preprocessing
check("machine" in cleaned[0] or "learning" in cleaned[0],
      "Key terms survive preprocessing (e.g., 'machine' or 'learning')")
check("pasta" in cleaned[2] or "cook" in cleaned[2] or "noodle" in cleaned[2],
      "Cooking terms survive preprocessing (e.g., 'pasta' / 'cook' / 'noodle')")

# Edge case: empty input
check(preprocess_text("") == "", "Empty string returns empty string")
check(preprocess_text("   ") == "", "Whitespace-only returns empty string")

print(f"\n  Sample cleaned doc (ML_Basics): '{cleaned[0][:100]}...'")
print(f"  Sample cleaned doc (Cooking):   '{cleaned[2][:100]}...'")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING MODULE (TF-IDF)")
print("=" * 70)

from feature_engineering import build_tfidf_matrix, get_top_keywords, get_all_keywords

tfidf_matrix, vectorizer = build_tfidf_matrix(cleaned)

check(tfidf_matrix.shape[0] == len(cleaned), f"TF-IDF rows == num docs ({tfidf_matrix.shape[0]})")
check(tfidf_matrix.shape[1] > 0, f"TF-IDF has features ({tfidf_matrix.shape[1]} features)")

# Check TF-IDF values are valid (between 0 and 1 for normalized vectors)
dense = tfidf_matrix.toarray()
check(np.all(dense >= 0), "All TF-IDF values are non-negative")
check(np.max(dense) <= 1.0 + 1e-6, f"Max TF-IDF value is ≤ 1.0 (got {np.max(dense):.4f})")

# Check that L2 norms are ~1 (TfidfVectorizer normalizes by default)
norms = np.linalg.norm(dense, axis=1)
check(np.allclose(norms, 1.0, atol=1e-4), f"Rows are L2-normalized (norms ≈ 1.0, got {norms})")

# Keyword extraction
kw_df_0 = get_top_keywords(vectorizer, tfidf_matrix, doc_index=0, n=10)
check(isinstance(kw_df_0, pd.DataFrame), "get_top_keywords returns a DataFrame")
check("keyword" in kw_df_0.columns and "tfidf_score" in kw_df_0.columns,
      "DataFrame has 'keyword' and 'tfidf_score' columns")
check(len(kw_df_0) > 0, f"Keywords extracted for doc 0 ({len(kw_df_0)} keywords)")
check(kw_df_0["tfidf_score"].is_monotonic_decreasing,
      "Keywords are sorted by score (descending)")

# Check that ML-related keywords appear for ML docs
ml_keywords_found = any("learn" in kw or "machine" in kw or "neural" in kw or "data" in kw
                        for kw in kw_df_0["keyword"].values)
check(ml_keywords_found, "ML-relevant keywords found for ML_Basics document")

# Check cooking keywords for cooking doc
kw_df_2 = get_top_keywords(vectorizer, tfidf_matrix, doc_index=2, n=10)
cooking_keywords_found = any("pasta" in kw or "cook" in kw or "garlic" in kw or "tomato" in kw
                             for kw in kw_df_2["keyword"].values)
check(cooking_keywords_found, "Cooking-relevant keywords found for Cooking_Recipe document")

# get_all_keywords
all_kw = get_all_keywords(vectorizer, tfidf_matrix, n=5)
check(len(all_kw) == len(cleaned), f"get_all_keywords returns dict for all {len(cleaned)} docs")

print(f"\n  Top 5 keywords for ML_Basics:     {kw_df_0['keyword'].head(5).tolist()}")
print(f"  Top 5 keywords for Cooking_Recipe: {kw_df_2['keyword'].head(5).tolist()}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SIMILARITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. SIMILARITY ENGINE MODULE")
print("=" * 70)

from similarity_engine import compute_similarity_matrix, get_top_pairs

sim_matrix = compute_similarity_matrix(tfidf_matrix, doc_labels=doc_names)

check(isinstance(sim_matrix, pd.DataFrame), "Similarity matrix is a DataFrame")
check(sim_matrix.shape == (4, 4), f"Matrix shape is 4×4 (got {sim_matrix.shape})")
check(list(sim_matrix.columns) == doc_names, "Columns match document names")
check(list(sim_matrix.index) == doc_names, "Index matches document names")

# Diagonal should be 1.0 (self-similarity)
diag_vals = np.diag(sim_matrix.values)
check(np.allclose(diag_vals, 1.0, atol=1e-4),
      f"Diagonal values are 1.0 (self-similarity): {diag_vals}")

# Matrix should be symmetric
check(np.allclose(sim_matrix.values, sim_matrix.values.T, atol=1e-6),
      "Matrix is symmetric")

# All values should be between 0 and 1
check(np.all(sim_matrix.values >= -1e-6) and np.all(sim_matrix.values <= 1.0 + 1e-6),
      "All similarity values in [0, 1]")

# SEMANTIC CORRECTNESS CHECKS:
sim_ml_basics_advanced = sim_matrix.loc["ML_Basics", "ML_Advanced"]
sim_ml_basics_cooking  = sim_matrix.loc["ML_Basics", "Cooking_Recipe"]
sim_ml_basics_apps     = sim_matrix.loc["ML_Basics", "ML_Applications"]
sim_ml_advanced_cooking = sim_matrix.loc["ML_Advanced", "Cooking_Recipe"]

# ML docs should be more similar to each other than to cooking
check(sim_ml_basics_advanced > sim_ml_basics_cooking,
      f"ML_Basics↔ML_Advanced ({sim_ml_basics_advanced:.4f}) > ML_Basics↔Cooking ({sim_ml_basics_cooking:.4f})")

check(sim_ml_basics_apps > sim_ml_basics_cooking,
      f"ML_Basics↔ML_Applications ({sim_ml_basics_apps:.4f}) > ML_Basics↔Cooking ({sim_ml_basics_cooking:.4f})")

check(sim_ml_advanced_cooking < 0.15,
      f"ML_Advanced↔Cooking should be very low ({sim_ml_advanced_cooking:.4f} < 0.15)")

# Top pairs
top_pairs = get_top_pairs(sim_matrix, n=6)
check(isinstance(top_pairs, pd.DataFrame), "get_top_pairs returns a DataFrame")
check(len(top_pairs) == 6, f"Returns 6 pairs (got {len(top_pairs)})")
check(top_pairs["Similarity Score"].is_monotonic_decreasing,
      "Pairs are sorted by similarity (descending)")

# The top pair should be between ML documents (not involving cooking)
top_doc_a = top_pairs.iloc[0]["Document A"]
top_doc_b = top_pairs.iloc[0]["Document B"]
check("Cooking" not in top_doc_a and "Cooking" not in top_doc_b,
      f"Most similar pair does NOT involve Cooking doc: {top_doc_a}↔{top_doc_b}")

print(f"\n  Full Similarity Matrix:")
for idx, row_name in enumerate(doc_names):
    vals = " | ".join(f"{sim_matrix.iloc[idx, j]:.4f}" for j in range(len(doc_names)))
    print(f"    {row_name:20s}: {vals}")

print(f"\n  Top pairs:")
for _, row in top_pairs.iterrows():
    print(f"    {row['Document A']:20s} ↔ {row['Document B']:20s} = {row['Similarity Score']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CLUSTERING MODULE")
print("=" * 70)

from clustering import cluster_documents, get_cluster_dataframe, get_2d_projection

labels = cluster_documents(tfidf_matrix, n_clusters=2)

check(isinstance(labels, np.ndarray), "cluster_documents returns numpy array")
check(len(labels) == len(cleaned), f"Labels length matches docs ({len(labels)})")
check(set(labels).issubset({0, 1}), f"Labels are in {{0, 1}} for 2 clusters (got {set(labels)})")

# Cooking doc should be in a DIFFERENT cluster than all ML docs
cooking_cluster = labels[2]  # index 2 = Cooking_Recipe
ml_clusters = [labels[0], labels[1], labels[3]]
check(all(ml_c != cooking_cluster for ml_c in ml_clusters),
      f"Cooking_Recipe (cluster {cooking_cluster}) is in a different cluster than ML docs ({ml_clusters})")

# All ML docs should be in the same cluster
check(labels[0] == labels[1] == labels[3],
      f"All ML docs in same cluster: ML_Basics={labels[0]}, ML_Advanced={labels[1]}, ML_Applications={labels[3]}")

# Cluster DataFrame
cluster_df = get_cluster_dataframe(labels, doc_labels=doc_names)
check(isinstance(cluster_df, pd.DataFrame), "get_cluster_dataframe returns DataFrame")
check("Document" in cluster_df.columns and "Cluster" in cluster_df.columns,
      "DataFrame has 'Document' and 'Cluster' columns")
check(len(cluster_df) == 4, f"DataFrame has {len(cluster_df)} rows (expected 4)")

# 2D Projection
coords = get_2d_projection(tfidf_matrix)
check(isinstance(coords, np.ndarray), "get_2d_projection returns numpy array")
check(coords.shape == (4, 2), f"Projection shape is (4, 2) (got {coords.shape})")
check(not np.any(np.isnan(coords)), "No NaN values in projection")

# Edge case: n_clusters > n_docs should be clamped
labels_clamped = cluster_documents(tfidf_matrix, n_clusters=20)
check(len(set(labels_clamped)) <= len(cleaned),
      f"n_clusters clamped: requested 20, got {len(set(labels_clamped))} clusters for {len(cleaned)} docs")

print(f"\n  Cluster assignments:")
for name, label in zip(doc_names, labels):
    print(f"    {name:20s} → Cluster {label}")
print(f"\n  2D Projection (first 2 coords per doc):")
for name, (x, y) in zip(doc_names, coords):
    print(f"    {name:20s} → ({x:+.4f}, {y:+.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION (Full Pipeline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. FULL PIPELINE INTEGRATION TEST")
print("=" * 70)

# Run the exact same pipeline as app.py
cleaned_2 = preprocess_documents(raw_texts)
tfidf_2, vec_2 = build_tfidf_matrix(cleaned_2)
sim_2 = compute_similarity_matrix(tfidf_2, doc_labels=doc_names)
top_2 = get_top_pairs(sim_2, n=min(10, len(doc_names) * (len(doc_names) - 1) // 2))
kw_2 = get_all_keywords(vec_2, tfidf_2, n=10)
labels_2 = cluster_documents(tfidf_2, n_clusters=3)
cluster_df_2 = get_cluster_dataframe(labels_2, doc_labels=doc_names)
coords_2 = get_2d_projection(tfidf_2)

# Check pipeline doesn't crash and produces valid outputs
check(sim_2.shape == (4, 4), "Pipeline: similarity matrix shape correct")
check(len(top_2) > 0, "Pipeline: top pairs generated")
check(len(kw_2) == 4, "Pipeline: keywords for all docs")
check(len(labels_2) == 4, "Pipeline: cluster labels for all docs")
check(coords_2.shape == (4, 2), "Pipeline: 2D projection shape correct")

# Metrics calculation (same as app.py lines 337-340)
triu_vals = sim_2.values[np.triu_indices(4, k=1)]
avg_sim = float(np.mean(triu_vals))
max_sim = float(np.max(triu_vals))
n_unique = len(set(labels_2))

check(0 <= avg_sim <= 1, f"Avg similarity in [0,1]: {avg_sim:.4f}")
check(0 <= max_sim <= 1, f"Max similarity in [0,1]: {max_sim:.4f}")
check(max_sim >= avg_sim, f"Max similarity ({max_sim:.4f}) >= Avg ({avg_sim:.4f})")
check(n_unique >= 1, f"At least 1 cluster: {n_unique}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"VERIFICATION COMPLETE:  ✅ {PASS} passed  |  ❌ {FAIL} failed")
print("=" * 70)

if FAIL > 0:
    print("\n⚠️  Some checks FAILED. Please review the output above.")
    sys.exit(1)
else:
    print("\n🎉 All checks passed! The system is producing correct results.")
    sys.exit(0)
