// CineMinds AI - Core UI Controller and Event Router
console.log("CineMinds app.js parsed successfully.");

let userProfile;
try {
  userProfile = JSON.parse(SafeStorage.getItem("cineminds_user_profile")) || { likes: [], dislikes: [] };
} catch(e) {
  userProfile = { likes: [], dislikes: [] };
}
if (!userProfile || !Array.isArray(userProfile.likes)) {
  userProfile = { likes: [], dislikes: [] };
}
if (!Array.isArray(userProfile.dislikes)) {
  userProfile.dislikes = [];
}

// Global variables
let recommender;
let visualizer;
let chatManager;
let currentModalMovieId = null;

function initApp() {
  try {
    // 1. Initialize logic engines
    recommender = new CineRecommender(MOVIES_DATABASE);
    visualizer = new VectorSpaceVisualizer("vectorCanvas", recommender);
    
    chatManager = new CineChatManager(
      "chat-messages-container",
      "chat-input-field",
      "chat-send-btn",
      recommender,
      visualizer
    );

    // Load and apply settings state
    initSettings();

    // 2. Set up event handlers
    initTabSwitching();
    initMovieSliders();
    initModalListeners();
    
    // Render initial dashboard content
    renderFeaturedHero();
    updateRecommendationsRow();
    
    // Set up system check in header status
    checkBackendStatus();
  } catch (err) {
    console.error("CineMinds Initialization Failed:", err);
    // Display error visually on the page for debugging
    const debugDiv = document.createElement("div");
    debugDiv.style.position = "fixed";
    debugDiv.style.top = "0";
    debugDiv.style.left = "0";
    debugDiv.style.width = "100%";
    debugDiv.style.background = "#ef4444";
    debugDiv.style.color = "#ffffff";
    debugDiv.style.padding = "20px";
    debugDiv.style.zIndex = "999999";
    debugDiv.style.fontFamily = "monospace";
    debugDiv.style.fontSize = "12px";
    debugDiv.style.whiteSpace = "pre-wrap";
    debugDiv.innerText = "Fatal Initialization Error: " + err.stack;
    document.body.appendChild(debugDiv);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}

// Update localStorage for user taste profile
function saveUserProfile() {
  SafeStorage.setItem("cineminds_user_profile", JSON.stringify(userProfile));
}

// ----------------------------------------------------
// UI Render Helpers
// ----------------------------------------------------

function renderMovieCard(movie, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const card = document.createElement("div");
  card.className = "movie-card";
  card.setAttribute("data-movie-id", movie.id);
  
  // Choose background style
  let bgStyle = movie.backdropGradient || movie.posterColor;
  if (movie.posterUrl) {
    bgStyle = `url('${movie.posterUrl}')`;
  }
  
  // Content details overlay
  card.innerHTML = `
    <div class="card-poster" style="background: linear-gradient(180deg, rgba(0,0,0,0) 20%, rgba(10, 10, 13, 0.95) 100%), ${bgStyle}; background-size: cover; background-position: center;">
      <span class="card-genre-tag">${movie.genres[0]}</span>
      <div class="card-content">
        <h3 class="card-title">${movie.title}</h3>
        <div class="card-meta">
          <span>${movie.year}</span>
          <span>★ ${movie.rating}</span>
        </div>
      </div>
    </div>
  `;

  // Bind click event to open details modal
  card.addEventListener("click", () => {
    showMovieDetailModal(movie.id);
  });

  container.appendChild(card);
}

function initMovieSliders() {
  // Sci-Fi and Cosmic
  const scifiMovies = MOVIES_DATABASE.filter(m => m.genres.includes("Sci-Fi"))
    .sort((a, b) => b.rating - a.rating);
  scifiMovies.forEach(m => renderMovieCard(m, "row-scifi"));

  // Thrillers and Crime
  const thrillerMovies = MOVIES_DATABASE.filter(m => m.genres.includes("Thriller") || m.genres.includes("Crime"))
    .sort((a, b) => b.rating - a.rating);
  thrillerMovies.forEach(m => renderMovieCard(m, "row-thriller"));

  // Comedy and Romance
  const comedyMovies = MOVIES_DATABASE.filter(m => m.genres.includes("Comedy") || m.genres.includes("Romance"))
    .sort((a, b) => b.rating - a.rating);
  comedyMovies.forEach(m => renderMovieCard(m, "row-comedy"));
}

function renderFeaturedHero() {
  // Use Whiplash or Inception as hero movie
  const featured = MOVIES_DATABASE.find(m => m.isFeatured) || MOVIES_DATABASE[0];
  
  const heroBanner = document.getElementById("hero-banner");
  if (heroBanner && featured) {
    let bgStyle = featured.backdropGradient || '#111';
    if (featured.id === "m13") {
      bgStyle = "url('assets/whiplash_banner.png')";
    } else if (featured.posterUrl) {
      bgStyle = `url('${featured.posterUrl}')`;
    }
    
    heroBanner.style.background = `linear-gradient(90deg, rgba(10, 10, 13, 0.95) 0%, rgba(10, 10, 13, 0.5) 50%, rgba(10, 10, 13, 0) 100%), 
                                   linear-gradient(0deg, rgba(10, 10, 13, 1) 0%, rgba(10, 10, 13, 0.2) 100%), 
                                   ${bgStyle}`;
    heroBanner.style.backgroundSize = "cover";
    heroBanner.style.backgroundPosition = "center";

    document.getElementById("hero-title").innerText = featured.title;
    document.getElementById("hero-year").innerText = featured.year;
    document.getElementById("hero-duration").innerText = featured.duration;
    document.getElementById("hero-rating").innerText = `★ ${featured.rating}`;
    document.getElementById("hero-desc").innerText = featured.description;

    // More Info triggers modal
    const moreBtn = document.getElementById("hero-btn-more");
    moreBtn.onclick = () => showMovieDetailModal(featured.id);

    // Play Trailer simulation
    const trailerBtn = document.getElementById("hero-btn-trailer");
    trailerBtn.onclick = () => {
      alert(`🎥 Simulating movie streaming player interface: Now loading "${featured.title}" (${featured.year}) with 1080p Atmos feed...`);
    };
  }
}

function updateRecommendationsRow() {
  const container = document.getElementById("row-recommendations");
  if (!container) return;
  container.innerHTML = "";

  const recommendations = recommender.getRecommendationsForUser(userProfile, 6);
  recommendations.forEach(m => renderMovieCard(m, "row-recommendations"));
}

// ----------------------------------------------------
// Tab Switching
// ----------------------------------------------------

function initTabSwitching() {
  const tabs = [
    { buttonId: "tab-home", panelId: "panel-home" },
    { buttonId: "tab-vector", panelId: "panel-vector" },
    { buttonId: "tab-math", panelId: "panel-math" }
  ];

  tabs.forEach(tab => {
    const btn = document.getElementById(tab.buttonId);
    btn.addEventListener("click", () => {
      // Remove active classes
      tabs.forEach(t => {
        document.getElementById(t.buttonId).classList.remove("active");
        document.getElementById(t.panelId).classList.remove("active");
      });

      // Add active classes
      btn.classList.add("active");
      document.getElementById(tab.panelId).classList.add("active");

      // Handle custom panel callbacks
      if (tab.panelId === "panel-vector" && visualizer) {
        visualizer.resizeCanvas(); // Make sure canvas scales correctly
      }
    });
  });
}

// ----------------------------------------------------
// Modals Control
// ----------------------------------------------------

function initModalListeners() {
  // Bind Like / Dislike in Detail Modal
  const likeBtn = document.getElementById("modal-like-btn");
  const dislikeBtn = document.getElementById("modal-dislike-btn");

  likeBtn.addEventListener("click", () => toggleUserLike(currentModalMovieId));
  dislikeBtn.addEventListener("click", () => toggleUserDislike(currentModalMovieId));

  // Settings buttons
  const settingsBtn = document.getElementById("settings-btn");
  const settingsModal = document.getElementById("settings-modal");
  const settingsCloseBtn = document.getElementById("settings-close-btn");
  const settingsSaveBtn = document.getElementById("settings-save-btn");
  const settingsResetBtn = document.getElementById("settings-reset-btn");
  const apiKeyField = document.getElementById("settings-api-key");

  settingsBtn.addEventListener("click", () => {
    apiKeyField.value = SafeStorage.getItem("cineminds_gemini_api_key") || "";
    settingsModal.classList.add("active");
  });

  settingsCloseBtn.addEventListener("click", () => {
    settingsModal.classList.remove("active");
  });

  settingsSaveBtn.addEventListener("click", () => {
    const key = apiKeyField.value.trim();
    chatManager.setApiKey(key);
    settingsModal.classList.remove("active");
    
    // Update API indicator details
    const modeText = document.getElementById("rag-mode-subtitle");
    if (key) {
      modeText.innerHTML = "Live Gemini RAG Mode 🔥";
      modeText.style.color = "#10b981";
    } else {
      modeText.innerHTML = "Offline Heuristics Mode";
      modeText.style.color = "var(--text-muted)";
    }
  });

  settingsResetBtn.addEventListener("click", () => {
    apiKeyField.value = "";
  });
}

function showMovieDetailModal(movieId) {
  const movie = MOVIES_DATABASE.find(m => m.id === movieId);
  if (!movie) return;

  currentModalMovieId = movieId;

  // Set basic content
  document.getElementById("modal-movie-title").innerText = movie.title;
  document.getElementById("modal-movie-director").innerText = `Directed by ${movie.director}`;
  document.getElementById("modal-movie-tagline").innerText = movie.tagline ? `"${movie.tagline}"` : "";
  document.getElementById("modal-movie-description").innerText = movie.description;
  document.getElementById("modal-movie-cast").innerText = movie.cast.join(", ");
  document.getElementById("modal-movie-rating").innerText = `★ ${movie.rating}`;
  document.getElementById("modal-movie-meta").innerText = `${movie.year} | ${movie.duration}`;

  // Genres
  const genreContainer = document.getElementById("modal-movie-genres");
  genreContainer.innerHTML = "";
  movie.genres.forEach(g => {
    const tag = document.createElement("span");
    tag.className = "card-genre-tag";
    tag.style.position = "static";
    tag.innerText = g;
    genreContainer.appendChild(tag);
  });

  // Modal header backdrop gradient
  const header = document.getElementById("modal-header-panel");
  let modalBg = movie.backdropGradient || movie.posterColor;
  if (movie.id === "m13") {
    modalBg = "url('assets/whiplash_banner.png')";
  } else if (movie.posterUrl) {
    modalBg = `url('${movie.posterUrl}')`;
  }
  header.style.background = `linear-gradient(180deg, rgba(20, 20, 28, 0.2) 0%, #14141c 100%), ${modalBg}`;
  header.style.backgroundSize = "cover";
  header.style.backgroundPosition = "center";

  // Content-Based Recommendations
  const similarContainer = document.getElementById("modal-similar-list");
  similarContainer.innerHTML = "";
  const similar = recommender.getSimilarMovies(movieId, 5);
  
  similar.forEach(m => {
    const simCard = document.createElement("div");
    simCard.className = "similar-card";
    let simBg = m.backdropGradient || m.posterColor;
    if (m.posterUrl) {
      simBg = `url('${m.posterUrl}')`;
    }
    simCard.style.background = `linear-gradient(180deg, rgba(0,0,0,0) 40%, ${m.posterColor || '#0a0a0d'}e2 100%), ${simBg}`;
    simCard.style.backgroundSize = "cover";
    simCard.style.backgroundPosition = "center";
    simCard.innerHTML = `<span class="similar-title">${m.title}</span>`;
    
    simCard.addEventListener("click", () => {
      showMovieDetailModal(m.id); // Load nested movie details
    });
    
    similarContainer.appendChild(simCard);
  });

  // Update button active state colors
  updateLikeButtonsState(movieId);

  // Show Modal
  document.getElementById("movie-details-modal").classList.add("active");
}

function closeMovieDetailsModal() {
  document.getElementById("movie-details-modal").classList.remove("active");
  currentModalMovieId = null;
}

window.showMovieDetailModal = showMovieDetailModal;
window.closeMovieDetailsModal = closeMovieDetailsModal;

// ----------------------------------------------------
// User Profile Operations
// ----------------------------------------------------

function toggleUserLike(movieId) {
  if (!movieId) return;

  const idx = userProfile.likes.indexOf(movieId);
  if (idx !== -1) {
    userProfile.likes.splice(idx, 1); // remove
  } else {
    userProfile.likes.push(movieId); // add
    // remove from dislike if it was there
    const disIdx = userProfile.dislikes.indexOf(movieId);
    if (disIdx !== -1) userProfile.dislikes.splice(disIdx, 1);
  }

  saveUserProfile();
  updateLikeButtonsState(movieId);
  updateRecommendationsRow();
}

function toggleUserDislike(movieId) {
  if (!movieId) return;

  const idx = userProfile.dislikes.indexOf(movieId);
  if (idx !== -1) {
    userProfile.dislikes.splice(idx, 1); // remove
  } else {
    userProfile.dislikes.push(movieId); // add
    // remove from likes if it was there
    const likeIdx = userProfile.likes.indexOf(movieId);
    if (likeIdx !== -1) userProfile.likes.splice(likeIdx, 1);
  }

  saveUserProfile();
  updateLikeButtonsState(movieId);
  updateRecommendationsRow();
}

function updateLikeButtonsState(movieId) {
  const likeBtn = document.getElementById("modal-like-btn");
  const dislikeBtn = document.getElementById("modal-dislike-btn");

  if (userProfile.likes.includes(movieId)) {
    likeBtn.className = "action-btn active-like";
  } else {
    likeBtn.className = "action-btn";
  }

  if (userProfile.dislikes.includes(movieId)) {
    dislikeBtn.className = "action-btn active-dislike";
  } else {
    dislikeBtn.className = "action-btn";
  }
}

function initSettings() {
  const key = SafeStorage.getItem("cineminds_gemini_api_key");
  const modeText = document.getElementById("rag-mode-subtitle");
  if (key && modeText) {
    modeText.innerHTML = "Live Gemini RAG Mode 🔥";
    modeText.style.color = "#10b981";
  }
}

// ----------------------------------------------------
// Mathematical Trace Explainer (RAG debugger)
// ----------------------------------------------------

function updateMathExplainer(queryText, results) {
  const container = document.getElementById("math-trace-details");
  if (!container) return;

  // Render chunk outputs on Vector Panel Sidebar
  const sidebarList = document.getElementById("retrieved-list");
  if (sidebarList) {
    sidebarList.innerHTML = `<h4>Query: "${queryText}"</h4>`;
    results.forEach((r, idx) => {
      const bullet = idx === 0 ? "🥇" : "🥈";
      sidebarList.innerHTML += `
        <div style="background: rgba(255,255,255,0.03); border: var(--glass-border); padding: 10px; border-radius: 6px;">
          <div style="font-weight: 700; color: #ffffff;">${bullet} ${r.movie.title}</div>
          <div style="color: var(--accent-pink); font-size: 10px; font-weight: 600; margin-top: 2px;">
            Cosine Similarity: ${r.score.toFixed(4)}
          </div>
          <p style="font-size: 10px; margin-top: 4px; line-height: 1.3;">
            "${r.movie.description.substring(0, 100)}..."
          </p>
        </div>
      `;
    });
  }

  // Decompose query vector
  const cleanTokens = queryText.toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .split(/\s+/)
    .filter(word => word.length > 1 && !STOPWORDS.has(word));

  const queryVector = recommender.vectorizeQuery(queryText);

  let traceHTML = `
    <div style="margin-bottom: 15px;">
      <strong>Clean Query Tokens:</strong> 
      ${cleanTokens.map(t => `<code style="background: rgba(236, 72, 153, 0.15); color: #f472b6; padding: 2px 5px; border-radius: 4px; font-size: 10px; margin-right: 4px;">${t}</code>`).join("") || "<em>None (only noise/stopwords)</em>"}
    </div>
    <div style="margin-bottom: 15px;">
      <strong>Query TF-IDF Weights:</strong>
      <table class="math-table" style="width: auto; min-width: 250px; margin-top: 5px;">
        <thead>
          <tr><th>Token</th><th>TF-IDF Weight (Normalized)</th></tr>
        </thead>
        <tbody>
          ${Object.entries(queryVector).map(([term, val]) => `
            <tr><td><strong>${term}</strong></td><td>${val.toFixed(4)}</td></tr>
          `).join("") || "<tr><td colspan='2' style='font-style:italic;'>No tokens found in corpus vocabulary</td></tr>"}
        </tbody>
      </table>
    </div>
    <div>
      <strong>Cosine Dot-Product Calculation (Top 3 Matches):</strong>
      <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 8px;">
  `;

  results.slice(0, 3).forEach((res, rank) => {
    const movie = res.movie;
    const movieVec = recommender.movieVectors[movie.id];
    
    // Find intersection terms
    const intersects = [];
    Object.keys(queryVector).forEach(term => {
      if (movieVec[term]) {
        intersects.push({
          term,
          qWeight: queryVector[term],
          mWeight: movieVec[term],
          product: queryVector[term] * movieVec[term]
        });
      }
    });

    traceHTML += `
      <div style="background: rgba(255,255,255,0.02); border: var(--glass-border); padding: 12px; border-radius: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; margin-bottom: 6px;">
          <span>Rank #${rank + 1}: ${movie.title}</span>
          <span style="color: var(--accent-blue);">Cosine Score: ${res.score.toFixed(4)}</span>
        </div>
        <p style="font-size: 10px; color: var(--text-secondary); margin-bottom: 6px;">
          Formula: CosineSim = Σ (q_term * d_term). Contributing dimensions:
        </p>
        
        <table class="math-table">
          <thead>
            <tr><th>Shared Term</th><th>Q_Weight</th><th>D_Weight</th><th>Product</th></tr>
          </thead>
          <tbody>
            ${intersects.map(i => `
              <tr>
                <td><code style="color:var(--accent-pink);">${i.term}</code></td>
                <td>${i.qWeight.toFixed(4)}</td>
                <td>${i.mWeight.toFixed(4)}</td>
                <td><strong>${i.product.toFixed(4)}</strong></td>
              </tr>
            `).join("") || "<tr><td colspan='4' style='font-style:italic; text-align:center;'>No overlapping vocab terms. Vector score is 0.0000</td></tr>"}
          </tbody>
        </table>
      </div>
    `;
  });

  traceHTML += `</div></div>`;
  container.innerHTML = traceHTML;
}

// ----------------------------------------------------
// Health Check / Python Backend Detect
// ----------------------------------------------------

async function checkBackendStatus() {
  const badge = document.getElementById("backend-status");
  if (!badge) return;

  try {
    const response = await fetch("http://localhost:8000/health", { method: "GET" });
    if (response.ok) {
      const data = await response.json();
      badge.innerHTML = `
        <span class="status-indicator" style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #06b6d4; margin-right: 5px; box-shadow: 0 0 8px #06b6d4;"></span>
        Python API Connected
      `;
      badge.style.borderColor = "var(--accent-blue)";
      
      window.isBackendConnected = true;
      const modeText = document.getElementById("rag-mode-subtitle");
      if (modeText) {
        modeText.innerHTML = "Live Python Backend RAG Mode 🔥";
        modeText.style.color = "#06b6d4";
      }
    }
  } catch (err) {
    window.isBackendConnected = false;
    // Keep Local RAG mode defaults
  }
}
