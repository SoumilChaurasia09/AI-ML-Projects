// Stopwords list to filter out noise words during TF-IDF vectorization
const STOPWORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
  "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
  "can", "cant", "cannot", "could", "couldnt",
  "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during",
  "each",
  "few", "for", "from", "further",
  "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows",
  "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself",
  "lets", "me", "more", "most", "mustnt", "my", "myself",
  "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
  "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such",
  "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very",
  "was", "wasnt", "we", "wed", "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt",
  "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
]);

window.STOPWORDS = STOPWORDS;

// Helper to tokenize and clean text
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "") // remove punctuation except hyphens
    .split(/\s+/)
    .filter(word => word.length > 1 && !STOPWORDS.has(word));
}

class CineRecommender {
  constructor(movies) {
    this.movies = movies;
    this.vocabulary = new Set();
    this.idf = {};
    this.movieVectors = {}; // movie_id -> term_weight_map
    this.buildTFIDF();
  }

  // Precomputes the TF-IDF representation for all movies in the database
  buildTFIDF() {
    const documentRepresentations = {};
    const df = {};
    const totalDocs = this.movies.length;

    // 1. Build term representation for each movie
    this.movies.forEach(movie => {
      // We combine title, description, keywords, and genres to build a rich search context
      const textSource = [
        movie.title,
        movie.title, // double weight for title matches
        movie.description,
        movie.genres.join(" "),
        movie.keywords.join(" "),
        movie.tagline || ""
      ].join(" ");

      const tokens = tokenize(textSource);
      documentRepresentations[movie.id] = tokens;

      // Unique terms in this document for document frequency
      const uniqueTokens = new Set(tokens);
      uniqueTokens.forEach(token => {
        df[token] = (df[token] || 0) + 1;
        this.vocabulary.add(token);
      });
    });

    // 2. Compute Inverse Document Frequency (IDF)
    this.vocabulary.forEach(term => {
      // Adding 1 to numerator and denominator to prevent division by zero (standard smoothing)
      this.idf[term] = Math.log((totalDocs + 1) / (df[term] + 1)) + 1;
    });

    // 3. Compute TF-IDF vectors for each movie
    this.movies.forEach(movie => {
      const tokens = documentRepresentations[movie.id];
      const tf = {};
      tokens.forEach(token => {
        tf[token] = (tf[token] || 0) + 1;
      });

      const vector = {};
      // Calculate normalized weights
      let squareSum = 0;
      Object.keys(tf).forEach(term => {
        const weight = tf[term] * this.idf[term];
        vector[term] = weight;
        squareSum += weight * weight;
      });

      // Normalize vector to unit length
      const magnitude = Math.sqrt(squareSum);
      if (magnitude > 0) {
        Object.keys(vector).forEach(term => {
          vector[term] /= magnitude;
        });
      }

      this.movieVectors[movie.id] = vector;
    });
  }

  // Calculates Cosine Similarity between two term weight maps
  calculateCosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    // Iterate over the smaller vector for speed
    const keysA = Object.keys(vecA);
    const keysB = Object.keys(vecB);
    const target = keysA.length < keysB.length ? vecA : vecB;
    const compare = target === vecA ? vecB : vecA;

    Object.keys(target).forEach(term => {
      if (compare[term]) {
        dotProduct += target[term] * compare[term];
      }
    });

    return dotProduct; // Already normalized since movie vectors are unit vectors
  }

  // Vectorizes a raw text search query
  vectorizeQuery(queryText) {
    const tokens = tokenize(queryText);
    const tf = {};
    tokens.forEach(token => {
      tf[token] = (tf[token] || 0) + 1;
    });

    const vector = {};
    let squareSum = 0;
    Object.keys(tf).forEach(term => {
      if (this.idf[term]) { // Only consider terms present in our corpus vocab
        const weight = tf[term] * this.idf[term];
        vector[term] = weight;
        squareSum += weight * weight;
      }
    });

    const magnitude = Math.sqrt(squareSum);
    if (magnitude > 0) {
      Object.keys(vector).forEach(term => {
        vector[term] /= magnitude;
      });
    }

    return vector;
  }

  // Performs a RAG retrieval (semantic search) returning top K matching movies
  retrieveMovies(queryText, limit = 5, weights = { content: 0.7, genre: 0.2, rating: 0.1 }) {
    const queryVector = this.vectorizeQuery(queryText);
    const queryTokens = new Set(tokenize(queryText));

    const scores = this.movies.map(movie => {
      // 1. Semantic Cosine Similarity
      const semanticScore = this.calculateCosineSimilarity(queryVector, this.movieVectors[movie.id]);

      // 2. Direct genre keyword boost (if query contains "sci-fi", "action", etc.)
      let genreOverlap = 0;
      movie.genres.forEach(genre => {
        if (queryTokens.has(genre.toLowerCase())) {
          genreOverlap = 1.0;
        }
      });

      // 3. IMDB Rating factor (normalized between 0 and 1)
      const ratingScore = (movie.rating - 5.0) / 5.0; // Assume ratings range from 5 to 10

      // Combine weights
      const totalScore = (semanticScore * weights.content) + 
                         (genreOverlap * weights.genre) + 
                         (Math.max(0, ratingScore) * weights.rating);

      return {
        movie,
        score: totalScore,
        details: {
          semantic: semanticScore,
          genreBoost: genreOverlap,
          rating: ratingScore
        }
      };
    });

    // Sort by descending score and limit
    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  // Computes similarity between two movies (Content-Based recommendation)
  getSimilarMovies(movieId, limit = 5) {
    const targetVector = this.movieVectors[movieId];
    const targetMovie = this.movies.find(m => m.id === movieId);
    if (!targetVector || !targetMovie) return [];

    const scores = this.movies
      .filter(m => m.id !== movieId)
      .map(movie => {
        const textSim = this.calculateCosineSimilarity(targetVector, this.movieVectors[movie.id]);
        
        // Calculate genre Jaccard intersection
        const intersection = movie.genres.filter(g => targetMovie.genres.includes(g)).length;
        const union = new Set([...movie.genres, ...targetMovie.genres]).size;
        const genreSim = union > 0 ? intersection / union : 0;

        // Weighted hybrid score
        const hybridScore = (textSim * 0.6) + (genreSim * 0.4);

        return {
          movie,
          score: hybridScore
        };
      });

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(s => s.movie);
  }

  // Generate recommendations based on the user's likes/dislikes profile
  getRecommendationsForUser(userProfile, limit = 5) {
    // If the user hasn't liked anything, return highly rated popular movies as cold start
    if (!userProfile.likes || userProfile.likes.length === 0) {
      return this.movies
        .slice()
        .sort((a, b) => b.rating - a.rating)
        .slice(0, limit);
    }

    // Build the user profile vector in vector space
    // We average the TF-IDF vectors of liked movies and subtract (with smaller weight) the vectors of disliked movies
    const profileVector = {};
    const likedCount = userProfile.likes.length;

    userProfile.likes.forEach(movieId => {
      const vec = this.movieVectors[movieId];
      if (vec) {
        Object.entries(vec).forEach(([term, val]) => {
          profileVector[term] = (profileVector[term] || 0) + (val / likedCount);
        });
      }
    });

    if (userProfile.dislikes && userProfile.dislikes.length > 0) {
      const dislikedCount = userProfile.dislikes.length;
      userProfile.dislikes.forEach(movieId => {
        const vec = this.movieVectors[movieId];
        if (vec) {
          Object.entries(vec).forEach(([term, val]) => {
            profileVector[term] = (profileVector[term] || 0) - (val * 0.3 / dislikedCount);
          });
        }
      });
    }

    // Rank unrated movies by cosine similarity to the profile vector
    const ratedSet = new Set([...(userProfile.likes || []), ...(userProfile.dislikes || [])]);
    const scores = this.movies
      .filter(movie => !ratedSet.has(movie.id))
      .map(movie => {
        const score = this.calculateCosineSimilarity(profileVector, this.movieVectors[movie.id]);
        return {
          movie,
          score
        };
      });

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(s => s.movie);
  }
}

window.CineRecommender = CineRecommender;

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { CineRecommender };
}
