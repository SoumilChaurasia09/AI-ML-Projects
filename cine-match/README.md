# CineMinds AI: Conversational RAG Movie Recommendation Platform

CineMinds AI is an advanced, high-fidelity movie recommendation system inspired by streaming platforms like Netflix and Prime Video. It combines content-based filtering algorithms, personalized user taste profiles, and a conversational RAG (Retrieval-Augmented Generation) assistant in a premium dark-mode glassmorphic interface.

---

## Key Features

1. **Netflix-Style UI Dashboard**: Includes card sliders categorized by genres, custom glassmorphism styling, scale hover transformations, and a hero featured banner loaded with custom cinematic art.
2. **Interactive Detail Modal**: Allows users to read synopsis descriptions, check cast lists, view IMDB ratings, look up similar movies calculated in real-time, and Like/Dislike items.
3. **Conversational RAG Co-pilot**: A sidebar chat interface allowing users to type natural queries (e.g. *"Show me realistic space exploration dramas like Interstellar"*). The engine retrieves database chunks, maps them, and generates recommendations.
4. **Neural Vector Space visualizer**: An interactive HTML5 Canvas that projects movies as coordinates clustered by genre. When you search, the query vector shoots connection lines directly to retrieved movies with thickness relative to their Cosine Similarity score.
5. **Mathematical Trace Panel**: An analytics panel explaining the vector dot products, TF-IDF calculation, and profile vector subtraction formula in real-time.
6. **Dual-Mode execution**:
   - **Client-Side (Local) Mode**: Runs completely in the browser with no server requirements, performing client-side vector search and fallback heuristic chat. Direct Gemini API keys can be saved securely in browser local storage.
   - **Python AI Backend Mode**: A FastAPI server running scikit-learn for TF-IDF indexing and the official Google GenAI library to call Gemini API.

---

## File Structure

```
cine-match/
├── index.html           # Core HTML dashboard
├── styles.css           # Premium glassmorphic styling
├── README.md            # Project guide and documentation
├── assets/
│   └── whiplash_banner.png # Generated high-fidelity hero banner art
├── data/
│   └── movies.js        # Catalog database (25+ iconic movies)
├── js/
│   ├── recommender.js   # Content filtering & user profile vectorizer
│   ├── visualizer.js    # Canvas-based 2D vector space projection
│   ├── chat.js          # Chat manager and Gemini REST fetch controller
│   └── app.js           # UI binder, sliders, and tab switcher
└── backend/
    ├── app.py           # FastAPI server with ML vector processing
    ├── movies.py        # Python representation of the database
    └── requirements.txt # Python package requirements list
```

---

## How to Run

### Method A: Client-Side Local Mode (Immediate)
1. Double-click the `index.html` file or drag it directly into any modern web browser.
2. The application works instantly! Try Liking/Disliking movies to see your "Recommended for You" feed update in real-time.
3. Switch to the **Neural Vector Map** tab, submit a search in the chat, and watch the query node animate.
4. (Optional) Click the **API Settings** gear in the top header and enter your Gemini API Key. This enables live, conversational RAG answers directly from Gemini!

### Method B: Python AI Backend Mode (Advanced)
If you want to run the full Python AI/ML backend:
1. Ensure Python 3.9+ is installed.
2. Open your terminal in the `backend/` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Set your Gemini API key in your environment:
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your-api-key-here"
   
   # Linux/macOS
   export GEMINI_API_KEY="your-api-key-here"
   ```
5. Run the FastAPI development server:
   ```bash
   python app.py
   ```
6. Open `index.html` in your browser. The header status badge will automatically switch to **"Python API Connected"**, routing search, profile calculations, and chat queries to your local Python server!

---

## Mathematical Formulation

### 1. Cosine Similarity Vector Search
Vector relevance between a search query vector $Q$ and a movie vector $D$ is computed as the normalized dot product in terms space:

$$\text{CosineSim}(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|} = \frac{\sum_{i=1}^n q_i d_i}{\sqrt{\sum_{i=1}^n q_i^2} \sqrt{\sum_{i=1}^n d_i^2}}$$

### 2. User Taste Profiling
We build the user preference vector $U$ based on liked movie vectors $D_{liked}$ and disliked movie vectors $D_{disliked}$:

$$U = \frac{1}{N_{likes}} \sum_{k \in \text{likes}} D_k - \frac{0.3}{M_{dislikes}} \sum_{j \in \text{dislikes}} D_j$$

Movies are ranked by computing $\text{CosineSim}(U, D_{unviewed})$.
