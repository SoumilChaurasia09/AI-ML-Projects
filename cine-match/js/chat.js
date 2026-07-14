class CineChatManager {
  constructor(chatContainerId, textInputId, sendButtonId, recommender, visualizer) {
    this.container = document.getElementById(chatContainerId);
    this.input = document.getElementById(textInputId);
    this.sendBtn = document.getElementById(sendButtonId);
    this.recommender = recommender;
    this.visualizer = visualizer;
    this.messages = [];
    this.apiKey = SafeStorage.getItem("cineminds_gemini_api_key") || "";

    this.initEvents();
    this.renderWelcomeMessage();
  }

  setApiKey(key) {
    this.apiKey = key;
    SafeStorage.setItem("cineminds_gemini_api_key", key);
  }

  initEvents() {
    this.sendBtn.addEventListener("click", () => this.handleSendMessage());
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.handleSendMessage();
      }
    });
  }

  renderWelcomeMessage() {
    this.addMessage("ai", `Welcome to **CineMinds AI**! 🍿\n\nI am your conversational RAG recommendation co-pilot. You can talk to me like a real streaming concierge. Ask me for recommendations based on your mood, themes, plotlines, or specific settings.\n\nTry asking me: \n* *"I'm in the mood for a mind-bending sci-fi heist movie"* \n* *"Recommend an intense drama about obsession and jazz music"* \n* *"Show me a whimsical romance set in Paris"*`);
  }

  addMessage(sender, text) {
    const msg = { sender, text, timestamp: new Date() };
    this.messages.push(msg);
    this.renderMessage(msg);
    this.scrollToBottom();
  }

  renderMessage(msg) {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-bubble-wrapper ${msg.sender === "user" ? "user-msg" : "ai-msg"}`;

    const icon = document.createElement("div");
    icon.className = "chat-bubble-avatar";
    icon.innerHTML = msg.sender === "user" ? "👤" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.innerHTML = this.parseMarkdown(msg.text);

    // Bind event listeners to custom movie:// links in the rendered HTML
    bubble.querySelectorAll("a").forEach(link => {
      const href = link.getAttribute("href");
      if (href && href.startsWith("movie://")) {
        const movieId = href.replace("movie://", "");
        link.addEventListener("click", (e) => {
          e.preventDefault();
          if (window.showMovieDetailModal) {
            window.showMovieDetailModal(movieId);
          }
        });
      }
    });

    wrapper.appendChild(icon);
    wrapper.appendChild(bubble);
    this.container.appendChild(wrapper);
  }

  parseMarkdown(text) {
    // Basic markdown parsing for bold, italics, lists, and code blocks
    let html = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br>");

    // Custom link format mapping: [Movie Title](movie://movieId) -> <a href="movie://movieId" class="movie-link">Movie Title</a>
    html = html.replace(/\[([^\]]+)\]\((movie:\/\/[a-zA-Z0-9_-]+)\)/g, '<a href="$2" class="movie-chat-link">$1</a>');

    return html;
  }

  showTypingIndicator() {
    const indicator = document.createElement("div");
    indicator.id = "chat-typing-indicator";
    indicator.className = "chat-bubble-wrapper ai-msg typing-msg";
    indicator.innerHTML = `
      <div class="chat-bubble-avatar">🤖</div>
      <div class="chat-bubble typing-bubble">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    `;
    this.container.appendChild(indicator);
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    const indicator = document.getElementById("chat-typing-indicator");
    if (indicator) {
      indicator.remove();
    }
  }

  scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }

  async handleSendMessage() {
    const text = this.input.value.trim();
    if (!text) return;

    this.input.value = "";
    this.addMessage("user", text);
    this.showTypingIndicator();

    try {
      // 1. Perform RAG Retrieval to find matching movie vectors
      const results = this.recommender.retrieveMovies(text, 4);
      
      // Update Neural vector visualizer with query and matching nodes
      if (this.visualizer) {
        this.visualizer.visualizeQuery(text, results);
      }

      // Display matching items details in mathematical explainer panel if active
      if (window.updateMathExplainer) {
        window.updateMathExplainer(text, results);
      }

      // 2. Generate AI recommendation text
      let aiResponse = "";
      if (window.isBackendConnected) {
        aiResponse = await this.generateBackendResponse(text);
      } else if (this.apiKey) {
        aiResponse = await this.generateGeminiResponse(text, results);
      } else {
        aiResponse = await this.generateLocalResponse(text, results);
      }

      this.hideTypingIndicator();
      this.addMessage("ai", aiResponse);

    } catch (err) {
      console.error(err);
      this.hideTypingIndicator();
      this.addMessage("ai", `Sorry, I encountered an error during inference. Make sure your Gemini API key is valid or check your internet connection.\n\n*Error details: ${err.message}*`);
    }
  }

  async generateGeminiResponse(queryText, retrievedResults) {
    const apiURL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${this.apiKey}`;
    
    // Construct rich RAG context
    const movieContext = retrievedResults.map((r, idx) => {
      const m = r.movie;
      return `[ID: ${m.id}] ${m.title} (${m.year})
Genres: ${m.genres.join(", ")}
Rating: ★${m.rating}
Tagline: "${m.tagline}"
Description: ${m.description}
Keywords: ${m.keywords.join(", ")}
Cosine Similarity Score: ${(r.score).toFixed(3)}`;
    }).join("\n\n");

    const prompt = `You are "CineMinds AI", a highly sophisticated, friendly movie recommendation concierge.
The user is asking for movie recommendations with the query: "${queryText}"

I have run a TF-IDF semantic search on our database and found the following relevant movies:
${movieContext}

Respond to the user with an engaging, helpful, and conversational response in Markdown. 
Highlight the movies we found, and explain clearly and passionately how their plots, themes, or keywords align with the user's specific request.

CRITICAL FORMATTING RULES:
1. When recommending or naming a movie from our list, you MUST reference its ID using a special link format: [Movie Title](movie://movieID). For example: "If you want psychological tension, you should definitely watch [Inception](movie://m1), which explores dreams...".
2. Do NOT mention IDs in any other format (e.g. do not say "m1" or "ID: m1" in the text).
3. Do not invent movies outside of the list provided. If the matches are weak, recommend the best matches but note that you are tailoring them closely.
4. Keep the explanation punchy, engaging, and professional. Use formatting like bullet points or bold tags.`;

    const requestBody = {
      contents: [{
        parts: [{ text: prompt }]
      }]
    };

    const response = await fetch(apiURL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error?.message || "HTTP error from Gemini API");
    }

    const data = await response.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || "I was unable to formulate a response. Please try again.";
  }

  // A local rule-based response generator that runs completely client-side without API key
  async generateLocalResponse(queryText, retrievedResults) {
    return new Promise((resolve) => {
      setTimeout(() => {
        if (retrievedResults.length === 0) {
          resolve("I searched our archives but couldn't find any close matches for your request. Try broadening your keywords (e.g. search 'space', 'action', or 'comedy')!");
          return;
        }

        const topMatch = retrievedResults[0];
        
        let reply = `Here are the top matches from our catalog that fit your request: \n\n`;
        
        retrievedResults.forEach((r, idx) => {
          const m = r.movie;
          const bullet = idx === 0 ? "🏆 **Top Pick**" : "🍿 **Option**";
          reply += `${bullet}: [${m.title}](movie://${m.id}) (${m.year}) - *★${m.rating}*\n`;
          reply += `*   **Genres:** ${m.genres.join(", ")}\n`;
          reply += `*   **Why it matches:** ${m.tagline ? `"${m.tagline}" ` : ""}${m.description.substring(0, 140)}...\n\n`;
        });

        reply += `> 💡 *Note: I am running in Offline/Heuristic Mode. To unlock fully personalized, conversational recommendations that discuss themes in depth, click the Settings icon in the top header and enter your Gemini API Key.*`;

        resolve(reply);
      }, 1000); // Simulate network latency for realism
    });
  }

  async generateBackendResponse(queryText) {
    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message: queryText })
    });
    
    if (!response.ok) {
      throw new Error("Failed to get response from local Python backend.");
    }
    
    const data = await response.json();
    return data.response || "I was unable to get a valid response from the Python RAG server.";
  }
}

window.CineChatManager = CineChatManager;

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { CineChatManager };
}
