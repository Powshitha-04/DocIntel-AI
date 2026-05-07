"""
app.py — Document Intelligence System Dashboard
=================================================
Production-style Streamlit interface for semantic text analysis,
similarity detection, keyword extraction, and document clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local modules
from preprocessing import preprocess_documents
from feature_engineering import build_tfidf_matrix, get_top_keywords, get_all_keywords
from similarity_engine import compute_similarity_matrix, get_top_pairs
from clustering import cluster_documents, get_cluster_dataframe, get_2d_projection

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocIntel — Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean white + blue professional SaaS theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Root variables ---------- */
:root {
    --primary:    #2563eb;
    --primary-lt: #3b82f6;
    --accent:     #0ea5e9;
    --bg:         #f0f4f8;
    --card:       #ffffff;
    --text:       #1e293b;
    --text-sec:   #64748b;
    --border:     #e2e8f0;
    --success:    #10b981;
    --warning:    #f59e0b;
    --danger:     #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Main background */
.stApp {
    background: var(--bg);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stTextArea textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37,99,235,.35);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(37,99,235,.12);
}
div[data-testid="stMetric"] label {
    color: var(--text-sec) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--card);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    color: var(--text-sec);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    color: white !important;
    border-radius: 9px !important;
}

/* Dataframes */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* Headings */
h1, h2, h3, h4 {
    color: var(--text) !important;
}

/* Card wrapper */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    margin-bottom: 1rem;
}

/* Hero title bar */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1e293b 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(37,99,235,.25) 0%, transparent 70%);
}
.hero h1 { color: white !important; margin: 0; font-size: 1.9rem; }
.hero p  { color: #94a3b8; margin: .4rem 0 0; font-size: 1rem; }

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "results" not in st.session_state:
    st.session_state.results = None

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — input controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Document Input")
    st.caption("Add documents via text or file upload, then analyze.")

    st.markdown("---")

    # ── Manual text input ────
    st.markdown("### ✏️ Manual Entry")
    num_docs = st.slider("Number of text inputs", 1, 10, 3, key="num_docs")
    manual_docs: dict[str, str] = {}
    for i in range(num_docs):
        txt = st.text_area(
            f"Document {i + 1}",
            height=100,
            key=f"doc_{i}",
            placeholder=f"Paste or type document {i + 1} content here …",
        )
        if txt.strip():
            manual_docs[f"Doc {i + 1}"] = txt

    st.markdown("---")

    # ── File upload ────
    st.markdown("### 📁 File Upload")
    uploaded_files = st.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True,
    )
    file_docs: dict[str, str] = {}
    if uploaded_files:
        for uf in uploaded_files:
            content = uf.read().decode("utf-8", errors="replace")
            if content.strip():
                file_docs[uf.name] = content

    st.markdown("---")

    # ── Clustering controls ────
    st.markdown("### ⚙️ Settings")
    n_clusters = st.slider("Clusters (KMeans)", 2, 10, 3, key="n_clusters")
    top_n_keywords = st.slider("Keywords per document", 5, 20, 10, key="top_k")

    st.markdown("---")

    # ── Analyze button ────
    analyze = st.button("🚀 Analyze Documents", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Merge all documents
# ──────────────────────────────────────────────────────────────────────────────
all_docs = {**manual_docs, **file_docs}

# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>🔍 DocIntel — Document Intelligence</h1>
    <p>Semantic analysis · Similarity detection · Keyword extraction · Clustering</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────
if analyze:
    if len(all_docs) < 2:
        st.warning("⚠️ Please provide at least **2 documents** to run the analysis.")
    else:
        with st.spinner("🔄 Processing documents …"):
            doc_names = list(all_docs.keys())
            raw_texts = list(all_docs.values())

            # 1. Preprocess
            cleaned = preprocess_documents(raw_texts)

            # 2. TF-IDF
            tfidf_matrix, vectorizer = build_tfidf_matrix(cleaned)

            # 3. Similarity
            sim_matrix = compute_similarity_matrix(tfidf_matrix, doc_labels=doc_names)
            top_pairs = get_top_pairs(sim_matrix, n=min(10, len(doc_names) * (len(doc_names) - 1) // 2))

            # 4. Keywords
            kw_dict = get_all_keywords(vectorizer, tfidf_matrix, n=top_n_keywords)

            # 5. Clustering
            labels = cluster_documents(tfidf_matrix, n_clusters=n_clusters)
            cluster_df = get_cluster_dataframe(labels, doc_labels=doc_names)
            coords = get_2d_projection(tfidf_matrix)

        # Store results
        st.session_state.results = {
            "doc_names": doc_names,
            "sim_matrix": sim_matrix,
            "top_pairs": top_pairs,
            "kw_dict": kw_dict,
            "cluster_df": cluster_df,
            "coords": coords,
            "labels": labels,
            "tfidf_matrix": tfidf_matrix,
        }

# ──────────────────────────────────────────────────────────────────────────────
# Dashboard – display results
# ──────────────────────────────────────────────────────────────────────────────
res = st.session_state.results

if res is None:
    # Landing state
    st.markdown(
        """
<div class="card" style="text-align:center;padding:3rem;">
    <h2 style="margin-bottom:.5rem;">👋 Welcome to DocIntel</h2>
    <p style="color:#64748b;font-size:1.05rem;">
        Add at least <strong>2 documents</strong> in the sidebar and click
        <strong>Analyze Documents</strong> to get started.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

doc_names = res["doc_names"]
sim_matrix = res["sim_matrix"]
top_pairs = res["top_pairs"]
kw_dict = res["kw_dict"]
cluster_df = res["cluster_df"]
coords = res["coords"]
labels = res["labels"]

# ── Metric cards ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
n_docs = len(doc_names)
# Average pairwise similarity (upper triangle, no diagonal)
triu_vals = sim_matrix.values[np.triu_indices(n_docs, k=1)]
avg_sim = float(np.mean(triu_vals)) if len(triu_vals) > 0 else 0.0
max_sim = float(np.max(triu_vals)) if len(triu_vals) > 0 else 0.0
n_unique_clusters = len(set(labels))

c1.metric("📄 Documents", n_docs)
c2.metric("📊 Avg Similarity", f"{avg_sim:.2%}")
c3.metric("🔥 Max Similarity", f"{max_sim:.2%}")
c4.metric("🗂️ Clusters", n_unique_clusters)

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_sim, tab_kw, tab_clust = st.tabs(
    ["📊 Similarity Analysis", "🔑 Keywords", "🗂️ Clusters"]
)

# ─── Tab 1 : Similarity ──────────────────────────────────────────────────
with tab_sim:
    col_heat, col_pairs = st.columns([3, 2])

    with col_heat:
        st.markdown("#### Similarity Heatmap")
        fig_heat = px.imshow(
            sim_matrix.values,
            x=doc_names,
            y=doc_names,
            color_continuous_scale="Blues",
            zmin=0,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_heat.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_colorbar_title="Score",
            font=dict(family="Inter"),
            plot_bgcolor="white",
            height=420,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_pairs:
        st.markdown("#### Top Similar Pairs")
        if not top_pairs.empty:
            for _, row in top_pairs.iterrows():
                score = row["Similarity Score"]
                color = (
                    "var(--success)"
                    if score >= 0.5
                    else "var(--warning)" if score >= 0.25 else "var(--danger)"
                )
                st.markdown(
                    f"""
<div class="card" style="padding:1rem 1.2rem;display:flex;justify-content:space-between;align-items:center;">
    <div>
        <strong>{row['Document A']}</strong> &harr; <strong>{row['Document B']}</strong>
    </div>
    <div style="font-size:1.25rem;font-weight:700;color:{color};">{score:.2%}</div>
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No pairs to show.")

    # Full matrix table
    with st.expander("📋 Full Similarity Matrix (table)"):
        st.dataframe(sim_matrix.style.format("{:.4f}"), use_container_width=True)

# ─── Tab 2 : Keywords ────────────────────────────────────────────────────
with tab_kw:
    for idx, name in enumerate(doc_names):
        kw_df = kw_dict[idx]
        if kw_df.empty:
            continue

        st.markdown(f"#### 📄 {name}")
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig_kw = px.bar(
                kw_df,
                x="tfidf_score",
                y="keyword",
                orientation="h",
                color="tfidf_score",
                color_continuous_scale="Blues",
            )
            fig_kw.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=max(250, 28 * len(kw_df)),
                font=dict(family="Inter"),
                plot_bgcolor="white",
                xaxis_title="TF-IDF Score",
                yaxis_title="",
            )
            st.plotly_chart(fig_kw, use_container_width=True)

        with col_table:
            st.dataframe(
                kw_df.reset_index(drop=True),
                use_container_width=True,
                height=max(250, 28 * len(kw_df)),
            )

        st.markdown("---")

# ─── Tab 3 : Clusters ────────────────────────────────────────────────────
with tab_clust:
    col_scatter, col_tbl = st.columns([3, 2])

    with col_scatter:
        st.markdown("#### 2-D Cluster Projection (PCA)")
        scatter_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        scatter_df["Document"] = doc_names
        scatter_df["Cluster"] = [str(l) for l in labels]

        fig_scatter = px.scatter(
            scatter_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            text="Document",
            color_discrete_sequence=px.colors.qualitative.Bold,
            size_max=14,
        )
        fig_scatter.update_traces(
            textposition="top center",
            marker=dict(size=14, line=dict(width=1.5, color="white")),
        )
        fig_scatter.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter"),
            plot_bgcolor="#f8fafc",
            height=420,
            legend_title_text="Cluster",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_tbl:
        st.markdown("#### Cluster Assignments")
        styled = cluster_df.style.apply(
            lambda s: [
                f"background-color: {px.colors.qualitative.Bold[int(v) % len(px.colors.qualitative.Bold)]}22"
                for v in s
            ]
            if s.name == "Cluster"
            else [""] * len(s),
            axis=0,
        )
        st.dataframe(styled, use_container_width=True, height=420)
