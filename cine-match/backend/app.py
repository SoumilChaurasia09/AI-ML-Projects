import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from dotenv import load_dotenv

# Load movie database
from movies import MOVIES_DATABASE

# Load environment variables
load_dotenv()

app = FastAPI(title="CineMinds AI Recommendation Backend")

# Enable CORS so local HTML file can call backend endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for local sandbox access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# ML Vector Space Setup (TF-IDF & Cosine Similarity)
# ----------------------------------------------------

# Compile documents text sources
corpus = []
for movie in MOVIES_DATABASE:
    text_src = " ".join([
        movie["title"],
        movie["title"], # double weight title
        movie["description"],
        " ".join(movie["genres"]),
        " ".join(movie["keywords"]),
        movie.get("tagline", "")
    ])
    corpus.append(text_src)

# Fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# Helper to find movie index by ID
def get_movie_idx(movie_id: str) -> int:
    for idx, m in enumerate(MOVIES_DATABASE):
        if m["id"] == movie_id:
            return idx
    raise ValueError(f"Movie with ID {movie_id} not found.")

# ----------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------

class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5

class UserProfileSchema(BaseModel):
    likes: List[str]
    dislikes: List[str]
    limit: Optional[int] = 5

class ChatRequest(BaseModel):
    message: str

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------

@app.get("/health")
def health_check():
    """Health check for frontend to detect python backend existence"""
    return {"status": "ok", "message": "CineMinds Python API running successfully"}

@app.get("/movies")
def get_movies():
    """Retrieve full movies database"""
    return MOVIES_DATABASE

@app.post("/search")
def search_movies(search_data: SearchQuery):
    """
    RAG Retrieval Search: Runs TF-IDF semantic vector search on prompt.
    Includes rating bias and exact genre overlap checks.
    """
    query = search_data.query
    limit = search_data.limit

    if not query.strip():
        return []

    # Vectorize query
    query_vec = vectorizer.transform([query])
    
    # Compute Cosine Similarities (linear_kernel is equivalent to dot product since TF-IDF output is normalized)
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()

    query_tokens = set(re.findall(r'\w+', query.lower()))

    results = []
    for idx, movie in enumerate(MOVIES_DATABASE):
        semantic_score = float(cosine_similarities[idx])

        # Genre boost: check if query contains any of the genres explicitly
        genre_overlap = 0.0
        for genre in movie["genres"]:
            if genre.lower() in query_tokens:
                genre_overlap = 1.0
                break

        # Rating score normalized (approx. between 0 and 1)
        rating_score = max(0.0, (movie["rating"] - 5.0) / 5.0)

        # Hybrid weighting (Content similarity, genre match boost, popularity bias)
        total_score = (semantic_score * 0.7) + (genre_overlap * 0.2) + (rating_score * 0.1)

        results.append({
            "movie": movie,
            "score": total_score,
            "details": {
                "semantic": semantic_score,
                "genreBoost": genre_overlap,
                "rating": rating_score
            }
        })

    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

@app.post("/recommend")
def get_recommendations(profile: UserProfileSchema):
    """
    Content-Based User taste profile matcher.
    Averages liked movie vectors, subtracts disliked movie vectors, 
    and returns similarity suggestions.
    """
    likes = profile.likes
    dislikes = profile.dislikes
    limit = profile.limit

    if not likes:
        # Cold start fallback: return highly rated movies
        sorted_popular = sorted(MOVIES_DATABASE, key=lambda x: x["rating"], reverse=True)
        return sorted_popular[:limit]

    # Initialize empty profile vector
    profile_vector = np.zeros((1, tfidf_matrix.shape[1]))

    # Average vectors of liked movies
    for movie_id in likes:
        try:
            idx = get_movie_idx(movie_id)
            profile_vector += tfidf_matrix[idx].toarray()
        except ValueError:
            continue
    profile_vector /= len(likes)

    # Subtract vectors of disliked movies (with smaller penalty weight)
    if dislikes:
        dislike_vector = np.zeros((1, tfidf_matrix.shape[1]))
        for movie_id in dislikes:
            try:
                idx = get_movie_idx(movie_id)
                dislike_vector += tfidf_matrix[idx].toarray()
            except ValueError:
                continue
        dislike_vector /= len(dislikes)
        profile_vector -= (dislike_vector * 0.3)

    # Compute similarity against all items
    similarities = linear_kernel(profile_vector, tfidf_matrix).flatten()

    # Filter out rated movies
    rated_set = set(likes + dislikes)
    recommendations = []
    for idx, movie in enumerate(MOVIES_DATABASE):
        if movie["id"] not in rated_set:
            recommendations.append({
                "movie": movie,
                "score": float(similarities[idx])
            })

    # Sort and return
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return [r["movie"] for r in recommendations[:limit]]

@app.post("/chat")
def chat_rag(chat_req: ChatRequest):
    """
    Conversational RAG using the Google GenAI library.
    Retrieves matching movie documents, injects context into system instructions,
    and calls Gemini 1.5 Flash to respond.
    """
    # 1. Retrieve documents matching query
    search_results = search_movies(SearchQuery(query=chat_req.message, limit=4))
    
    if not search_results:
        return {"response": "I couldn't find any relevant movies in our catalog to base a recommendation on. Try query terms like 'space', 'action', or 'love'."}

    # 2. Check if GEMINI_API_KEY is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Fallback response explaining missing API key
        top_picks = [f"[{r['movie']['title']}](movie://{r['movie']['id']})" for r in search_results]
        return {
            "response": (
                f"I found some relevant matches in our database: {', '.join(top_picks)}.\n\n"
                f"*Note: Python Backend RAG is running in offline mode because the `GEMINI_API_KEY` "
                f"environment variable is not set. Please add your key to the `.env` file in the backend "
                f"directory to enable full conversational LLM features.*"
            )
        }

    try:
        from google import genai
        
        # Initialize Google GenAI client
        client = genai.Client(api_key=api_key)

        # Build movie catalog text context
        movie_context_list = []
        for r in search_results:
            m = r["movie"]
            movie_context_list.append(
                f"[ID: {m['id']}] {m['title']} ({m['year']})\n"
                f"Genres: {', '.join(m['genres'])}\n"
                f"Rating: {m['rating']}\n"
                f"Tagline: \"{m.get('tagline', '')}\"\n"
                f"Description: {m['description']}\n"
                f"Keywords: {', '.join(m['keywords'])}\n"
                f"Vector Match Score: {r['score']:.4f}"
            )
        movie_context = "\n\n".join(movie_context_list)

        prompt = f"""You are "CineMinds AI", a highly sophisticated, friendly movie recommendation concierge.
The user is asking for movie recommendations with the query: "{chat_req.message}"

I have run a TF-IDF semantic search on our database and found the following relevant movies:
{movie_context}

Respond to the user with an engaging, helpful, and conversational response in Markdown. 
Highlight the movies we found, and explain clearly and passionately how their plots, themes, or keywords align with the user's specific request.

CRITICAL FORMATTING RULES:
1. When recommending or naming a movie from our list, you MUST reference its ID using a special link format: [Movie Title](movie://movieID). For example: "If you want psychological tension, you should definitely watch [Inception](movie://m1), which explores dreams...".
2. Do NOT mention IDs in any other format (e.g. do not say "m1" or "ID: m1" in the text).
3. Do not invent movies outside of the list provided. If the matches are weak, recommend the best matches but note that you are tailoring them closely.
4. Keep the explanation punchy, engaging, and professional. Use formatting like bullet points or bold tags."""

        # Call Gemini model
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return {"response": response.text}

    except Exception as e:
        return {"response": f"Encountered an error while calling Gemini API: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
