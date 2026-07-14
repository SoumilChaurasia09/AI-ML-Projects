/* ==========================================
   VIT Bhopal University RAG Chatbox Controller
   Handles UI state, automatic RAG indexing on load,
   and RAG integration (real LLM & simulated).
   ========================================== */

// 1. Application State
const state = {
  activeTab: 'corpus',
  isTrained: false,
  chunkSize: 2,
  overlap: 1,
  
  // Data caches
  documentText: '',
  dataset: [],
  indexedChunks: [],
  
  // RAG Pipeline states
  lastQuery: '',
  retrievedResults: [],
  
  // Settings & Voice
  apiKey: localStorage.getItem('vit_bhopal_gemini_api_key') || '',
  soundEnabled: true,
  isListening: false
};

// DOM Elements cache
const DOM = {
  // Navigation Tabs
  tabButtons: document.querySelectorAll('.tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  
  // Settings Modal
  btnOpenSettings: document.getElementById('open-settings-btn'),
  btnCloseSettings: document.getElementById('close-settings-modal'),
  settingsModal: document.getElementById('settings-modal'),
  btnSaveApi: document.getElementById('save-api-btn'),
  btnClearApi: document.getElementById('clear-api-btn'),
  apiKeyInput: document.getElementById('gemini-api-key-input'),
  systemDot: document.getElementById('system-dot'),
  systemLabel: document.getElementById('system-label'),
  
  // Corpus Tab
  corpusDocText: document.getElementById('corpus-doc-text'),
  
  // Dataset Tab
  searchField: document.getElementById('search-field'),
  categoryFilter: document.getElementById('category-filter'),
  qaTableBody: document.getElementById('qa-table-body'),
  
  // Chat Panel
  chatHistory: document.getElementById('chat-history'),
  chatInput: document.getElementById('chat-input'),
  charCounter: document.getElementById('char-counter'),
  btnSend: document.getElementById('btn-send'),
  suggestionsCategories: document.getElementById('suggestions-categories'),
  suggestionsChips: document.getElementById('suggestions-chips'),
  
  // Bottom Drawer RAG Inspector
  pipelineDrawer: document.getElementById('pipeline-drawer'),
  drawerTrigger: document.getElementById('drawer-trigger'),
  drawerQueryPeek: document.getElementById('drawer-query-peek'),
  drawerMatchesCount: document.getElementById('drawer-matches-count'),
  pipelineTokens: document.getElementById('pipeline-tokens'),
  pipelineRankTable: document.getElementById('pipeline-rank-table-body'),
  pipelinePromptText: document.getElementById('pipeline-prompt-text')
};

// Colors mapping for document sections
const CATEGORY_COLORS = {
  'Admissions': '#00f2fe',
  'Fees': '#10b981',
  'Hostel': '#ffb020',
  'Academics': '#8b5cf6',
  'Placements': '#f43f5e'
};

// ==========================================
// 2. Application Setup
// ==========================================
document.addEventListener('DOMContentLoaded', initApplication);

async function initApplication() {
  setupTabs();
  setupSettingsModal();
  setupSpeechEngine();
  setupChatInputs();
  setupDrawer();
  
  // Load data immediately from data.js variables (bypasses browser CORS fetch file scheme blocks)
  try {
    state.documentText = PROSPECTUS_TEXT;
    DOM.corpusDocText.innerHTML = formatCorpusHTML(state.documentText);
    
    state.dataset = DATASET_DATA;
    populateDatasetTable(state.dataset);
    populateCategoryFilter(state.dataset);
    populateSuggestions(state.dataset);
    
    // Automatic Background Document Indexing
    runDocumentIndexing();
  } catch (err) {
    console.error("ERROR starting application:", err);
  }
}

// Setup tab switches
function setupTabs() {
  DOM.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      state.activeTab = targetTab;
      
      DOM.tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      DOM.tabPanes.forEach(pane => {
        pane.classList.remove('active');
        if (pane.id === `${targetTab}-pane`) {
          pane.classList.add('active');
        }
      });
    });
  });
}

// Background Document Indexing execution
function runDocumentIndexing() {
  // Run RAG chunking algorithm
  state.indexedChunks = chunkProspectus(state.documentText, state.chunkSize, state.overlap);
  
  // Build actual RAG TF-IDF indexes
  indexDocuments(state.indexedChunks);
  state.isTrained = true;
  
  // Update status UI
  updateStatusLabels();
  
  // Counselor welcomes the user in chat history with clickable recommended questions
  appendWelcomeBubble();
}

// Appends counselor welcome bubble with option button shortcuts
function appendWelcomeBubble() {
  const bubbleId = `msg_welcome_${Date.now()}`;
  const bubble = document.createElement('div');
  bubble.className = `chat-message assistant`;
  bubble.id = bubbleId;
  
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  bubble.innerHTML = `
    <div class="msg-header">
      <span>🤖 VIT Bhopal ChatBox</span>
      <span>${time}</span>
    </div>
    <div class="msg-body">
      <p>👋 <strong>Hello! I am your VIT Bhopal University Admissions ChatBox.</strong></p>
      <p>I have automatically loaded and indexed the university admissions prospectus. You can type any question, or select one of these recommended reviewer questions to test my RAG retrieval database:</p>
      <div class="guided-options-group">
        <button class="guided-option-btn" data-q="What is the minimum percentage required for B.Tech admission?">🎟️ B.Tech Admission Eligibility <span>➔</span></button>
        <button class="guided-option-btn" data-q="What is the fee structure for B.Tech in Computer Science & Engineering?">💵 B.Tech CSE Tuition Fees <span>➔</span></button>
        <button class="guided-option-btn" data-q="Are there any scholarships available for top rankers?">🎁 Scholarships & waivers <span>➔</span></button>
        <button class="guided-option-btn" data-q="What is the cost of a single sharing AC room in the hostel?">🏠 Hostel accommodation costs <span>➔</span></button>
        <button class="guided-option-btn" data-q="What is the placement percentage and average salary package?">💼 Placement rates & recruiters <span>➔</span></button>
      </div>
    </div>
  `;
  
  DOM.chatHistory.appendChild(bubble);
  DOM.chatHistory.scrollTop = DOM.chatHistory.scrollHeight;
  
  // Bind click handlers to the recommendation buttons
  bubble.querySelectorAll('.guided-option-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const question = btn.getAttribute('data-q');
      DOM.chatInput.value = question;
      DOM.chatInput.dispatchEvent(new Event('input'));
      handleChatMessageSubmit();
    });
  });
}

// ==========================================
// 3. Document / Dataset View Renderers
// ==========================================
function formatCorpusHTML(text) {
  const sections = text.split(/\[Section:\s*/);
  let html = '';
  
  sections.forEach(section => {
    if (!section.trim()) return;
    const parts = section.split(']');
    if (parts.length < 2) {
      // Render main title as a clean centered header
      html += `
        <div class="corpus-main-header" style="text-align: center; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px dashed var(--border-color);">
          <h2 style="font-size: 1.15rem; font-weight: 700; color: var(--neon-cyan); text-transform: uppercase; letter-spacing: 0.5px;">${section.replace(/===/g, '').trim()}</h2>
        </div>
      `;
    } else {
      const title = parts[0].trim();
      const body = parts.slice(1).join(']').trim();
      
      html += `
        <div class="corpus-section" style="margin-bottom: 20px;">
          <h3 style="color: var(--text-main); font-size: 0.95rem; font-weight: 600; margin-bottom: 6px; border-left: 3px solid var(--neon-cyan); padding-left: 10px;">${title}</h3>
          <p style="color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; white-space: pre-line;">${body}</p>
        </div>
      `;
    }
  });
  
  return html;
}

function populateDatasetTable(data) {
  DOM.qaTableBody.innerHTML = '';
  
  data.forEach(item => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-dim); width: 8%;">${item.id}</td>
      <td style="font-weight: 500; color: var(--text-main); width: 35%;">${item.question}</td>
      <td style="width: 42%;">${item.answer}</td>
      <td style="text-align: right; width: 15%;"><span class="tag-badge" style="background: ${getCategoryColorAlpha(item.category)}; color: ${CATEGORY_COLORS[item.category] || '#fff'}; border: 1px solid ${CATEGORY_COLORS[item.category] || '#fff'}30;">${item.category}</span></td>
    `;
    DOM.qaTableBody.appendChild(row);
  });
  
  // Bind search & filters
  DOM.searchField.addEventListener('input', filterQADataTable);
  DOM.categoryFilter.addEventListener('change', filterQADataTable);
}

function getCategoryColorAlpha(cat) {
  const hex = CATEGORY_COLORS[cat] || '#8b5cf6';
  return hex + '15';
}

function populateCategoryFilter(data) {
  const categories = [...new Set(data.map(item => item.category))];
  DOM.categoryFilter.innerHTML = '<option value="all">All Categories</option>';
  categories.forEach(cat => {
    DOM.categoryFilter.innerHTML += `<option value="${cat}">${cat}</option>`;
  });
}

function filterQADataTable() {
  const query = DOM.searchField.value.toLowerCase();
  const catFilter = DOM.categoryFilter.value;
  
  const filtered = state.dataset.filter(item => {
    const matchesSearch = item.question.toLowerCase().includes(query) || item.answer.toLowerCase().includes(query);
    const matchesCat = catFilter === 'all' || item.category === catFilter;
    return matchesSearch && matchesCat;
  });
  
  DOM.qaTableBody.innerHTML = '';
  if (filtered.length === 0) {
    DOM.qaTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-dim); padding: 24px;">No matching dataset records found.</td></tr>';
  } else {
    populateDatasetTable(filtered);
  }
}

// Populate suggested reviewer question chips sorted by category types
function populateSuggestions(data) {
  if (!DOM.suggestionsCategories || !DOM.suggestionsChips) return;

  // 1. Get unique categories
  const categories = [...new Set(data.map(item => item.category))];
  
  DOM.suggestionsCategories.innerHTML = '';
  
  // 2. Render category selectors
  categories.forEach((cat, idx) => {
    const btn = document.createElement('button');
    btn.className = `chip-btn ${idx === 0 ? 'active' : ''}`;
    // Inline styling for active states
    if (idx === 0) {
      btn.style.borderColor = 'var(--neon-cyan)';
      btn.style.color = 'var(--neon-cyan)';
      btn.style.background = 'rgba(0, 242, 254, 0.08)';
    }
    btn.textContent = cat;
    btn.addEventListener('click', () => {
      // Toggle active states
      DOM.suggestionsCategories.querySelectorAll('button').forEach(b => {
        b.style.borderColor = 'var(--border-color)';
        b.style.color = 'var(--text-muted)';
        b.style.background = 'rgba(22, 30, 49, 0.8)';
      });
      btn.style.borderColor = 'var(--neon-cyan)';
      btn.style.color = 'var(--neon-cyan)';
      btn.style.background = 'rgba(0, 242, 254, 0.08)';
      
      // Render filtered questions
      renderQuestionChipsForCategory(cat, data);
    });
    DOM.suggestionsCategories.appendChild(btn);
  });
  
  // 3. Render questions for first category by default
  if (categories.length > 0) {
    renderQuestionChipsForCategory(categories[0], data);
  }
}

function renderQuestionChipsForCategory(category, data) {
  DOM.suggestionsChips.innerHTML = '';
  const filtered = data.filter(item => item.category === category);
  
  filtered.forEach(item => {
    const q = item.question;
    const chip = document.createElement('button');
    chip.className = 'chip-btn';
    chip.textContent = q;
    chip.addEventListener('click', () => {
      DOM.chatInput.value = q;
      DOM.chatInput.dispatchEvent(new Event('input'));
      handleChatMessageSubmit();
    });
    DOM.suggestionsChips.appendChild(chip);
  });
}

// ==========================================
// 4. Admissions Chatbox Engine
// ==========================================
function setupChatInputs() {
  DOM.btnSend.addEventListener('click', handleChatMessageSubmit);
  DOM.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChatMessageSubmit();
    }
  });
  
  DOM.chatInput.addEventListener('input', () => {
    const len = DOM.chatInput.value.length;
    DOM.charCounter.textContent = `${len} / 1000`;
    DOM.btnSend.disabled = (len === 0);
    
    // Auto height
    DOM.chatInput.style.height = 'auto';
    DOM.chatInput.style.height = (DOM.chatInput.scrollHeight) + 'px';
  });
}

function handleChatMessageSubmit() {
  const query = DOM.chatInput.value.trim();
  if (!query) return;
  
  // Reset input UI
  DOM.chatInput.value = '';
  DOM.charCounter.textContent = '0 / 1000';
  DOM.chatInput.style.height = '48px';
  DOM.btnSend.disabled = true;
  
  // Append user bubble
  appendChatBubble('user', query);
  
  // Show AI typing loader bubble
  const loaderId = appendChatBubble('assistant', `
    <div class="typing-indicator" style="display:flex; gap:4px; font-family:var(--font-mono); font-size:0.75rem; color:var(--text-dim);">
      <span>[Searching vector indexes...]</span>
    </div>
  `);
  
  playSystemSound('send');
  
  // Run retrieval pipeline (Wait 500ms simulated network delay)
  setTimeout(async () => {
    const loaderMsg = document.getElementById(loaderId);
    if (loaderMsg) loaderMsg.remove();
    
    try {
      // 1. Execute Cosine Similarity retrieval from vector DB
      const topK = 3;
      const results = retrieveRelevantChunks(query, topK);
      state.retrievedResults = results;
      
      // Update Pipeline Inspector panel
      updatePipelineInspector(query, results);
      
      // 2. Prompt Builder (System instruction wrapping the context chunks)
      const contextBlocks = results.map(r => `[Category: ${r.chunk.category}]\nContent: ${r.chunk.rawContent}\n---`).join('\n\n');
      const systemPrompt = `You are a helpful and detailed university admissions counselor AI assistant.
Answer the student's question based ONLY on the provided context document segments below.
Provide accurate details about pricing (in INR), credit metrics, and placement rates. 
Always cite the category source logs when stating facts.
If the context segments do not contain sufficient information to answer, politely state that you cannot find this in the official admissions database and offer to direct them to the admissions office.

CONTEXT DOCUMENTS:
${contextBlocks}`;
      
      DOM.pipelinePromptText.value = `${systemPrompt}\n\nSTUDENT QUESTION:\n${query}`;
      
      // 3. Content Generation
      let answerText = "";
      if (state.apiKey) {
        // Online: call Gemini model
        answerText = await generateGeminiLLMResponse(state.apiKey, systemPrompt, query);
      } else {
        // Offline: call local matcher
        answerText = generateLocalAdmissionsResponse(query, results);
      }
      
      // 4. Stream response output
      streamBotResponse(answerText);
    } catch (err) {
      appendChatBubble('assistant', `⚠️ **Admissions Engine Fault:** Failed to process query. ${err.message}`);
      playSystemSound('receive');
    }
  }, 500);
}

function appendChatBubble(sender, markdown) {
  const bubbleId = `msg_${Date.now()}`;
  const bubble = document.createElement('div');
  bubble.className = `chat-message ${sender}`;
  bubble.id = bubbleId;
  
  const headerText = sender === 'user' ? '👤 Student' : '🤖 VIT Bhopal ChatBox';
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  bubble.innerHTML = `
    <div class="msg-header">
      <span>${headerText}</span>
      <span>${time}</span>
    </div>
    <div class="msg-body">${formatMarkdownHTML(markdown)}</div>
  `;
  
  DOM.chatHistory.appendChild(bubble);
  DOM.chatHistory.scrollTop = DOM.chatHistory.scrollHeight;
  return bubbleId;
}

function streamBotResponse(rawText) {
  const bubbleId = `msg_${Date.now()}`;
  const bubble = document.createElement('div');
  bubble.className = `chat-message assistant`;
  bubble.id = bubbleId;
  
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  bubble.innerHTML = `
    <div class="msg-header">
      <span>🤖 VIT Bhopal ChatBox</span>
      <span>${time}</span>
    </div>
    <div class="msg-body"></div>
  `;
  
  DOM.chatHistory.appendChild(bubble);
  const body = bubble.querySelector('.msg-body');
  
  let i = 0;
  let textAccum = '';
  const delay = 10; // 10ms per char stream
  
  function step() {
    if (i < rawText.length) {
      textAccum += rawText.charAt(i);
      body.innerHTML = formatMarkdownHTML(textAccum);
      DOM.chatHistory.scrollTop = DOM.chatHistory.scrollHeight;
      
      // Subtle click sound every 4 characters
      if (state.soundEnabled && i % 4 === 0) {
        playTypingTick();
      }
      
      i++;
      setTimeout(step, delay);
    } else {
      playSystemSound('receive');
      // Read aloud complete message if voice enabled
      speakBotReplyText(rawText);
    }
  }
  
  step();
}

// Simple Markdown-to-HTML parser
function formatMarkdownHTML(text) {
  let escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
    
  // Bold
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // Bullets
  escaped = escaped.replace(/^\*\s+(.+)$/gm, '<li>$1</li>');
  escaped = escaped.replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');
  escaped = escaped.replace(/<\/ul>\s*<ul>/g, '');
  
  // Tables
  escaped = escaped.replace(/\|([\s\S]*?)\|/g, function(match) {
    if (match.includes('\n')) {
      const rows = match.split('\n').filter(r => r.trim() !== '');
      let table = '<table>';
      rows.forEach((row, idx) => {
        if (row.includes('---')) return;
        const cells = row.split('|').map(c => c.trim()).filter((c, cIdx) => cIdx > 0 && cIdx < row.split('|').length - 1);
        table += '<tr>';
        cells.forEach(cell => {
          if (idx === 0) {
            table += `<th>${cell}</th>`;
          } else {
            table += `<td>${cell}</td>`;
          }
        });
        table += '</tr>';
      });
      table += '</table>';
      return table;
    }
    return match;
  });
  
  const lines = escaped.split('\n');
  const processedLines = lines.map(line => {
    if (line.includes('<table') || line.includes('<tr') || line.includes('<td') || line.includes('<th') || line.includes('</table') || line.includes('</tr')) return line;
    return line + '<br>';
  });
  
  return processedLines.join('\n').replace(/(<br>)+$/g, '');
}

// Local simulation heuristics matching
function generateLocalAdmissionsResponse(query, results) {
  const topResult = results[0];
  
  const simLabel = `🤖 *[OFFLINE LOCAL CO-PILOT RESPONSE]*\n\n`;
  
  if (!topResult || topResult.score < 0.08) {
    return simLabel + `I searched the VIT Bhopal University admissions index but could not retrieve enough relevant context to address your query. \n\nTry asking about **admission percentages (60%)**, **tuition fees per semester**, **hostel AC charges**, or **placement stats (98.6%)**.`;
  }
  
  // Search dataset mapping for exact matches first
  const exactMatch = state.dataset.find(item => {
    const qWords = tokenize(item.question);
    const userWords = tokenize(query);
    const overlap = qWords.filter(w => userWords.includes(w)).length;
    return overlap >= qWords.length * 0.75;
  });
  
  if (exactMatch) {
    return simLabel + `**Source: ${exactMatch.category} Archives**\n\n${exactMatch.answer}`;
  }
  
  // Construct dynamic summarization answer based on top matching chunks
  let ans = simLabel + `**Retrieved Admissions Knowledge (Cosine Score: ${topResult.score.toFixed(3)}):**\n\n`;
  ans += `Based on the official **${topResult.chunk.category}** logs:\n"${topResult.chunk.rawContent}"`;
  
  if (results[1] && results[1].score > 0.1) {
    ans += `\n\n*Supporting information from ${results[1].chunk.category}:*\n"${results[1].chunk.rawContent}"`;
  }
  
  return ans;
}

// Calls external generative AI model (Google Gemini)
async function generateGeminiLLMResponse(apiKey, systemPrompt, userQuery) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
  
  const payload = {
    systemInstruction: {
      parts: [
        { text: "You are the VIT Bhopal ChatBox admissions counselor assistant. Keep responses helpful, structured, clear, and based ONLY on the provided context logs." }
      ]
    },
    contents: [
      {
        parts: [
          { text: `${systemPrompt}\n\nStudent Query: ${userQuery}` }
        ]
      }
    ],
    generationConfig: {
      temperature: 0.2,
      maxOutputTokens: 600
    }
  };
  
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  
  if (!res.ok) {
    const errObj = await res.json().catch(() => ({}));
    const errMsg = errObj.error?.message || `Status Code ${res.status}`;
    throw new Error(`Gemini Endpoint: ${errMsg}`);
  }
  
  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error("Empty candidate list returned");
  return text;
}

// ==========================================
// 5. Bottom Drawer RAG Pipeline Inspector
// ==========================================
function setupDrawer() {
  DOM.drawerTrigger.addEventListener('click', () => {
    DOM.pipelineDrawer.classList.toggle('expanded');
  });
}

function updatePipelineInspector(query, results) {
  DOM.drawerQueryPeek.textContent = `Query: "${query.substring(0, 30)}${query.length > 30 ? '...' : ''}"`;
  
  const matches = results.filter(r => r.score > 0.05).length;
  DOM.drawerMatchesCount.textContent = `${matches} chunk${matches !== 1 ? 's' : ''} retrieved`;
  
  // 1. Process Tokens list
  DOM.pipelineTokens.innerHTML = '';
  const tokens = tokenize(query);
  if (tokens.length === 0) {
    DOM.pipelineTokens.innerHTML = `<span style="color:var(--text-dim); font-size:0.7rem;">No search tokens.</span>`;
  } else {
    tokens.forEach(tok => {
      const span = document.createElement('span');
      span.className = 'token-badge';
      span.textContent = tok;
      DOM.pipelineTokens.appendChild(span);
    });
  }
  
  // 2. Populate rank table
  DOM.pipelineRankTable.innerHTML = '';
  if (results.length === 0) {
    DOM.pipelineRankTable.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-dim);">No indexes matching query.</td></tr>';
    return;
  }
  
  results.forEach((res, idx) => {
    const row = document.createElement('tr');
    if (idx === 0 && res.score > 0.05) row.className = 'top-retrieved';
    
    row.innerHTML = `
      <td style="font-family: var(--font-mono);">#${idx + 1}</td>
      <td style="color: var(--neon-cyan); font-weight: 500;">${res.score.toFixed(4)}</td>
      <td><span class="tag-badge" style="background:${getCategoryColorAlpha(res.chunk.category)}; color:${CATEGORY_COLORS[res.chunk.category]}">${res.chunk.category}</span></td>
      <td>
        <div style="font-family: var(--font-mono); font-size:0.65rem; color:var(--text-dim);">${res.chunk.id}</div>
        <div style="font-size:0.75rem; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; max-width: 250px;" title="${res.chunk.content}">${res.chunk.rawContent}</div>
      </td>
    `;
    DOM.pipelineRankTable.appendChild(row);
  });
}

// ==========================================
// 6. Settings API Panel Modal
// ==========================================
function setupSettingsModal() {
  if (DOM.apiKeyInput) DOM.apiKeyInput.value = state.apiKey;
  updateStatusLabels();
  
  if (DOM.btnOpenSettings && DOM.settingsModal) {
    DOM.btnOpenSettings.addEventListener('click', () => {
      if (DOM.apiKeyInput) DOM.apiKeyInput.value = state.apiKey;
      DOM.settingsModal.classList.add('active');
    });
  }
  
  if (DOM.btnCloseSettings && DOM.settingsModal) {
    DOM.btnCloseSettings.addEventListener('click', () => {
      DOM.settingsModal.classList.remove('active');
    });
  }
  
  if (DOM.btnSaveApi && DOM.apiKeyInput && DOM.settingsModal) {
    DOM.btnSaveApi.addEventListener('click', () => {
      const key = DOM.apiKeyInput.value.trim();
      state.apiKey = key;
      if (key) {
        localStorage.setItem('vit_bhopal_gemini_api_key', key);
      } else {
        localStorage.removeItem('vit_bhopal_gemini_api_key');
      }
      updateStatusLabels();
      DOM.settingsModal.classList.remove('active');
    });
  }
  
  if (DOM.btnClearApi && DOM.apiKeyInput && DOM.settingsModal) {
    DOM.btnClearApi.addEventListener('click', () => {
      state.apiKey = '';
      DOM.apiKeyInput.value = '';
      localStorage.removeItem('vit_bhopal_gemini_api_key');
      updateStatusLabels();
      DOM.settingsModal.classList.remove('active');
    });
  }
}

function updateStatusLabels() {
  if (!DOM.systemDot || !DOM.systemLabel) return;
  if (state.apiKey) {
    DOM.systemDot.style.backgroundColor = 'var(--neon-emerald)';
    DOM.systemDot.style.boxShadow = '0 0 8px var(--neon-emerald)';
    DOM.systemLabel.textContent = 'RAG MODE: GEMINI LIVE';
  } else {
    DOM.systemDot.style.backgroundColor = 'var(--neon-emerald)';
    DOM.systemDot.style.boxShadow = '0 0 8px var(--neon-emerald)';
    DOM.systemLabel.textContent = 'RAG MODE: LOCAL SIM';
  }
}

// ==========================================
// 7. Speech Engine Integration (Disabled)
// ==========================================
function setupSpeechEngine() {
  // Voice controls removed from UI
}

function speakBotReplyText(text) {
  // TTS disabled
}

// ==========================================
// 8. Web Audio Sound Effects Synthesizer
// ==========================================
let audioContext = null;

function initAudio() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
}

function playSystemSound(type) {
  if (!state.soundEnabled) return;
  try {
    initAudio();
    const osc = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    osc.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    const now = audioContext.currentTime;
    
    if (type === 'send') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(650, now);
      osc.frequency.exponentialRampToValueAtTime(1050, now + 0.08);
      gainNode.gain.setValueAtTime(0.04, now);
      gainNode.gain.linearRampToValueAtTime(0, now + 0.08);
      osc.start(now);
      osc.stop(now + 0.08);
    } else if (type === 'receive') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(520, now);
      osc.frequency.setValueAtTime(390, now + 0.06);
      gainNode.gain.setValueAtTime(0.05, now);
      gainNode.gain.linearRampToValueAtTime(0, now + 0.15);
      osc.start(now);
      osc.stop(now + 0.15);
    }
  } catch (e) {}
}

function playTypingTick() {
  if (!state.soundEnabled) return;
  try {
    initAudio();
    const osc = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    osc.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    const now = audioContext.currentTime;
    osc.type = 'sine';
    osc.frequency.setValueAtTime(1200, now);
    gainNode.gain.setValueAtTime(0.002, now);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 0.01);
    
    osc.start(now);
    osc.stop(now + 0.01);
  } catch (e) {}
}
