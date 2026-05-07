# 🔍 DocIntel — Document Intelligence System

DocIntel is a production-style Streamlit dashboard for semantic text analysis. It leverages machine learning to provide deep insights into document collections through similarity detection, keyword extraction, and automated clustering.

---

## 🚀 Key Features

- **Semantic Analysis**: Understand the underlying meaning of your documents beyond simple keyword matching.
- **Similarity Heatmaps**: Visualize pairwise cosine similarity between all uploaded documents.
- **Keyword Extraction**: Automated identification of the most significant terms using TF-IDF weighting.
- **Document Clustering**: Automatically group related documents using KMeans clustering.
- **2D Projection**: Visualize document relationships in a 2D space using Principal Component Analysis (PCA).
- **Interactive UI**: Clean, professional SaaS-style interface with real-time analysis.

---

## 🛠️ Technology Stack

- **Core**: Python 3.8+
- **Frontend**: Streamlit
- **ML/NLP**: Scikit-learn (TF-IDF, KMeans, PCA, Cosine Similarity)
- **NLP Preprocessing**: NLTK (Tokenization, Lemmatization, Stopword removal)
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly Express & Graph Objects

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Powshitha-04/-Eyedentify.git
   cd AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources** (The app does this automatically on first run, but you can do it manually):
   ```python
   import nltk
   nltk.download(['punkt', 'stopwords', 'wordnet'])
   ```

---

## 🏃 Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   *If `streamlit` is not in your PATH, use:*
   ```bash
   python -m streamlit run app.py
   ```

2. **Upload Documents**:
   - Use the sidebar to paste text manually or upload `.txt` files.
   - Adjust clustering and keyword settings as needed.
   - Click **🚀 Analyze Documents** to generate insights.

---

## 🧪 Verification

To verify that the analysis engine is producing correct results, run the automated test suite:
```bash
python verify_results.py
```

---

## 📂 Project Structure

- `app.py`: Main Streamlit dashboard and UI logic.
- `preprocessing.py`: NLP cleaning pipeline (Lemmatization, Stopwords).
- `feature_engineering.py`: TF-IDF vectorization and keyword extraction.
- `similarity_engine.py`: Cosine similarity computation.
- `clustering.py`: KMeans clustering and PCA projection.
- `verify_results.py`: End-to-end verification script.
