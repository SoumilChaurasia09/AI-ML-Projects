// Imports omitted for browser script compatibility

// DOM Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');
const resetBtn = document.getElementById('resetLanderBtn');
const muteBtn = document.getElementById('muteAudioBtn');
const openSettingsBtn = document.getElementById('openSettingsBtn');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const settingsModal = document.getElementById('settingsModal');
const saveApiBtn = document.getElementById('saveApiBtn');
const clearApiBtn = document.getElementById('clearApiBtn');
const apiKeyInput = document.getElementById('geminiApiKeyInput');
const systemIndicatorDot = document.getElementById('systemIndicatorDot');
const systemIndicatorLabel = document.getElementById('systemIndicatorLabel');

const hudAltitude = document.getElementById('hudAltitude');
const hudFuel = document.getElementById('hudFuel');
const hudVspeed = document.getElementById('hudVspeed');
const hudHspeed = document.getElementById('hudHspeed');
const hudTilt = document.getElementById('hudTilt');

const fillAltitude = document.getElementById('fillAltitude');
const fillFuel = document.getElementById('fillFuel');
const fillVspeed = document.getElementById('fillVspeed');
const fillHspeed = document.getElementById('fillHspeed');
const fillTilt = document.getElementById('fillTilt');

const hudAltitudeCard = document.getElementById('hudAltitudeCard');
const hudFuelCard = document.getElementById('hudFuelCard');
const hudVspeedCard = document.getElementById('hudVspeedCard');
const hudHspeedCard = document.getElementById('hudHspeedCard');
const hudTiltCard = document.getElementById('hudTiltCard');

const flightStatusBadge = document.getElementById('flightStatusBadge');
const chatHistory = document.getElementById('chatHistory');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const contextPeek = document.getElementById('contextPeek');
const contextPeekCount = document.getElementById('contextPeekCount');
const contextPeekDoc = document.getElementById('contextPeekDoc');
const contextPeekScore = document.getElementById('contextPeekScore');

const docLibraryContainer = document.getElementById('docLibrary');
const pipelineTokens = document.getElementById('pipelineTokens');
const pipelineRankTableBody = document.getElementById('pipelineRankTableBody');
const pipelinePromptBox = document.getElementById('pipelinePromptBox');

// DOM Elements 2.0 (Advanced Visuals & Tuning)
const envSelect = document.getElementById('envSelect');
const sliderChunkSize = document.getElementById('sliderChunkSize');
const sliderOverlap = document.getElementById('sliderOverlap');
const labelChunkSize = document.getElementById('labelChunkSize');
const labelOverlap = document.getElementById('labelOverlap');
const chartCanvasInstance = document.getElementById('chartCanvas');
const vectorCanvasInstance = document.getElementById('vectorCanvas');

const gameOverlay = document.getElementById('gameOverlay');
const overlayTitle = document.getElementById('overlayTitle');
const overlayMessage = document.getElementById('overlayMessage');
const overlayResetBtn = document.getElementById('overlayResetBtn');

// Game state variables
let gameInstance = null;
let isMuted = false;
let apiConfig = {
  apiKey: localStorage.getItem('gemini_api_key') || ''
};
let lastFlightTelemetryText = "No simulation flight telemetry recorded yet. Perform a launch first.";

// Telemetry & Vector Space States
let chartCtx = null;
let vectorCtx = null;
let telemetryHistory = [];
let queryNode = { x: 200, y: 90, targetX: 200, targetY: 90, active: false, pulse: 0 };
let queryConnections = [];

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
  // Initialize context variables for canvases
  if (chartCanvasInstance) chartCtx = chartCanvasInstance.getContext('2d');
  if (vectorCanvasInstance) vectorCtx = vectorCanvasInstance.getContext('2d');

  // 1. Initialize local RAG database with slider presets
  const initSize = parseInt(sliderChunkSize.value);
  const initOverlap = parseInt(sliderOverlap.value);
  initializeRAG(initSize, initOverlap);
  injectTelemetryChunk(lastFlightTelemetryText, initSize, initOverlap);
  renderDocumentLibrary();

  // 2. Setup game simulation
  initGame();

  // 3. Setup tabs switching
  setupTabs();

  // 4. Setup API Settings Modal
  setupSettingsModal();

  // 5. Setup RAG assistant Chat Form
  setupChat();

  // 6. Bind Tuner Sliders & Environment Selector
  setupAdvancedControls();
});

// Initialize game instance
function initGame() {
  gameInstance = new LunarLanderGame(
    'gameCanvas',
    onGameStateChange,
    onTelemetryLogged
  );

  // Bind simulation action buttons
  resetBtn.addEventListener('click', () => {
    gameInstance.reset();
  });

  muteBtn.addEventListener('click', () => {
    isMuted = !isMuted;
    if (isMuted) {
      muteBtn.textContent = "UNMUTE SOUND";
      muteBtn.style.color = "var(--neon-amber)";
      if (gameInstance) gameInstance.stopThrustSound();
    } else {
      muteBtn.textContent = "MUTE SOUND";
      muteBtn.style.color = "var(--neon-cyan)";
    }
  });

  // Start Canvas animation loop
  function loop() {
    gameInstance.update();
    gameInstance.draw();
    updateHUD();
    updateTelemetryHistory();
    drawTelemetryChart();
    drawVectorSpace();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

// Update DOM HUD gauges with telemetry from game instance
function updateHUD() {
  if (!gameInstance) return;
  const tel = gameInstance.getTelemetry();

  // 1. Altitude
  hudAltitude.textContent = `${tel.altitude.toFixed(1)} m`;
  const altPct = Math.min(100, (tel.altitude / 340) * 100);
  fillAltitude.style.width = `${altPct}%`;
  
  // 2. Fuel
  hudFuel.textContent = `${tel.fuel.toFixed(1)}%`;
  fillFuel.style.width = `${tel.fuel}%`;
  if (tel.fuel < 25) {
    hudFuelCard.className = "hud-card critical";
  } else if (tel.fuel < 50) {
    hudFuelCard.className = "hud-card warning";
  } else {
    hudFuelCard.className = "hud-card";
  }

  // 3. VSpeed (Limit: 3.0)
  const vsVal = Math.abs(tel.verticalSpeed);
  hudVspeed.textContent = `${tel.verticalSpeed.toFixed(2)} m/s`;
  const vsPct = Math.min(100, (vsVal / 6) * 100);
  fillVspeed.style.width = `${vsPct}%`;
  if (vsVal > 3.0) {
    hudVspeedCard.className = "hud-card critical";
    fillVspeed.style.backgroundColor = "var(--neon-red)";
  } else {
    hudVspeedCard.className = "hud-card safe";
    fillVspeed.style.backgroundColor = "var(--neon-green)";
  }

  // 4. HSpeed (Limit: 1.5)
  const hsVal = Math.abs(tel.horizontalSpeed);
  hudHspeed.textContent = `${tel.horizontalSpeed.toFixed(2)} m/s`;
  const hsPct = Math.min(100, (hsVal / 4) * 100);
  fillHspeed.style.width = `${hsPct}%`;
  if (hsVal > 1.5) {
    hudHspeedCard.className = "hud-card critical";
    fillHspeed.style.backgroundColor = "var(--neon-red)";
  } else {
    hudHspeedCard.className = "hud-card safe";
    fillHspeed.style.backgroundColor = "var(--neon-green)";
  }

  // 5. Tilt (Limit: 5.0)
  const tiltVal = Math.abs(tel.tilt);
  hudTilt.textContent = `${tel.tilt.toFixed(1)}°`;
  const tiltPct = Math.min(100, (tiltVal / 45) * 100);
  fillTilt.style.width = `${tiltPct}%`;
  if (tiltVal > 5.0) {
    hudTiltCard.className = "hud-card critical";
    fillTilt.style.backgroundColor = "var(--neon-red)";
  } else {
    hudTiltCard.className = "hud-card safe";
    fillTilt.style.backgroundColor = "var(--neon-green)";
  }
}

// Callback when game status changes (e.g. landed, crashed)
function onGameStateChange(state, telemetry) {
  // Update badge UI
  flightStatusBadge.className = `status-badge badge-${state}`;
  flightStatusBadge.textContent = state;

  if (state === 'landed') {
    appendChatBubble('assistant', `<p>🚨 <strong>TELEMETRY SIGNAL TOUCHDOWN DETECTED</strong></p>
    <p>Good landing, pilot! Check my analysis or ask: <em>"Analyze my last landing"</em>. Click <strong>RAG Pipeline Explorer</strong> to inspect how RAG indexed this event.</p>
    <div class="inline-telemetry-box telemetry-alert-safe">
      Vz: ${telemetry.verticalSpeed.toFixed(2)} m/s (SAFE)<br>
      Vx: ${telemetry.horizontalSpeed.toFixed(2)} m/s (SAFE)<br>
      Tilt: ${telemetry.tilt.toFixed(1)}° (SAFE) &bull; Fuel remaining: ${telemetry.fuel.toFixed(1)}%
    </div>`);

    // Show visual overlay on game screen
    if (gameOverlay) {
      overlayTitle.textContent = "SAFE TOUCHDOWN!";
      overlayTitle.style.color = "var(--neon-green)";
      overlayTitle.style.textShadow = "0 0 10px rgba(16, 185, 129, 0.5)";
      overlayMessage.textContent = gameInstance.impactMessage || "Landed safely on flat landing pad.";
      gameOverlay.classList.add('active');
    }
  } else if (state === 'crashed') {
    appendChatBubble('assistant', `<p>🚨 <strong>TELEMETRY LOSS DETECTED: IMPACT CRASH</strong></p>
    <p>We lost the lander. Query me with: <em>"Why did I crash?"</em> or <em>"Explain my telemetry"</em> to retrieve the physics limit parameters and troubleshoot the crash.</p>
    <div class="inline-telemetry-box telemetry-alert-critical">
      Vz: ${telemetry.verticalSpeed.toFixed(2)} m/s (Limit: 3.0)<br>
      Vx: ${telemetry.horizontalSpeed.toFixed(2)} m/s (Limit: 1.5)<br>
      Tilt: ${telemetry.tilt.toFixed(1)}° (Limit: 5.0) &bull; Remaining fuel: ${telemetry.fuel.toFixed(1)}%
    </div>`);

    // Show visual overlay on game screen
    if (gameOverlay) {
      overlayTitle.textContent = "MISSION FAILURE (CRASH)";
      overlayTitle.style.color = "var(--neon-red)";
      overlayTitle.style.textShadow = "0 0 10px rgba(239, 68, 68, 0.5)";
      overlayMessage.textContent = gameInstance.impactMessage || "Spacecraft hull breached during high-impact collision.";
      gameOverlay.classList.add('active');
    }
  } else if (state === 'flying') {
    // Hide visual overlay on reset/relaunch
    if (gameOverlay) {
      gameOverlay.classList.remove('active');
    }
  }
}

// Callback when flight logs are saved (gets injected into RAG)
function onTelemetryLogged(telemetryText) {
  lastFlightTelemetryText = telemetryText;
  injectTelemetryChunk(telemetryText);
  // Re-render documents count
  renderDocumentLibrary();
}

// Tab panes switching
function setupTabs() {
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Toggle buttons
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Toggle panes
      const targetPane = btn.getAttribute('data-pane');
      tabPanes.forEach(pane => {
        pane.classList.remove('active');
        if (pane.id === targetPane) {
          pane.classList.add('active');
        }
      });
    });
  });

  // Let context peek open the RAG Inspector tab
  contextPeek.addEventListener('click', () => {
    const pipeTab = document.getElementById('tabPipelineBtn');
    if (pipeTab) pipeTab.click();
  });
}

// API Config Modal handlers
function setupSettingsModal() {
  apiKeyInput.value = apiConfig.apiKey;
  updateApiStatusIndicator();

  openSettingsBtn.addEventListener('click', () => {
    apiKeyInput.value = apiConfig.apiKey;
    settingsModal.classList.add('active');
  });

  closeSettingsBtn.addEventListener('click', () => {
    settingsModal.classList.remove('active');
  });

  saveApiBtn.addEventListener('click', () => {
    const newKey = apiKeyInput.value.trim();
    apiConfig.apiKey = newKey;
    if (newKey) {
      localStorage.setItem('gemini_api_key', newKey);
    } else {
      localStorage.removeItem('gemini_api_key');
    }
    updateApiStatusIndicator();
    settingsModal.classList.remove('active');
    
    appendChatBubble('assistant', `<p>⚙️ <strong>Configuration Updated:</strong> Gemini API Key saved. AGMC co-pilot will now query real-time Gemini LLM models.</p>`);
  });

  clearApiBtn.addEventListener('click', () => {
    apiConfig.apiKey = '';
    apiKeyInput.value = '';
    localStorage.removeItem('gemini_api_key');
    updateApiStatusIndicator();
    settingsModal.classList.remove('active');
    
    appendChatBubble('assistant', `<p>⚙️ <strong>Configuration Updated:</strong> Gemini API Key cleared. Fallen back to <strong>Local Heuristic Co-Pilot</strong> mode.</p>`);
  });
}

function updateApiStatusIndicator() {
  if (apiConfig.apiKey) {
    systemIndicatorDot.style.backgroundColor = "var(--neon-green)";
    systemIndicatorDot.style.boxShadow = "0 0 8px var(--neon-green)";
    systemIndicatorLabel.textContent = "AGMC CO-PILOT: GEMINI LIVE";
  } else {
    systemIndicatorDot.style.backgroundColor = "var(--neon-cyan)";
    systemIndicatorDot.style.boxShadow = "0 0 8px var(--neon-cyan)";
    systemIndicatorLabel.textContent = "AGMC CO-PILOT: LOCAL SIM";
  }
}

// Render Document Library view list
function renderDocumentLibrary() {
  docLibraryContainer.innerHTML = '';
  
  // Combine static documents and the dynamic telemetry file
  const allDocs = [...documents];
  if (lastFlightTelemetryText) {
    allDocs.push({
      id: "telemetry",
      title: "Live Simulator Flight Telemetry",
      category: "Telemetry",
      content: lastFlightTelemetryText
    });
  }

  const ragState = getRAGState();

  allDocs.forEach(doc => {
    const card = document.createElement('div');
    card.className = 'doc-card';
    
    // Count how many chunks belong to this doc
    const docChunks = ragState.chunks.filter(c => c.docId === doc.id);
    
    card.innerHTML = `
      <div class="doc-header">
        <div class="doc-info">
          <span class="doc-title">${doc.title}</span>
          <div class="doc-meta">
            <span class="doc-category">${doc.category}</span>
            <span class="doc-chunks-count">${docChunks.length} chunks indexed</span>
          </div>
        </div>
        <span class="doc-arrow" style="font-size:0.75rem; color:var(--text-muted);">▼</span>
      </div>
      <div class="doc-body">
        <p style="white-space: pre-wrap; font-family: ${doc.id === 'telemetry' ? 'Fira Code, monospace' : 'inherit'}; font-size: ${doc.id === 'telemetry' ? '0.75rem' : '0.8rem'}">${doc.content}</p>
        <div class="chunks-preview-box" style="margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px;">
          <h4 style="font-size: 0.75rem; color: var(--neon-cyan); margin-bottom: 6px; text-transform: uppercase;">Indexed Text Chunks:</h4>
          ${docChunks.map((c, i) => `
            <div style="background: rgba(0,0,0,0.15); border-left: 2px solid var(--border-cyan); padding: 6px; font-size: 0.7rem; margin-bottom: 6px; border-radius: 0 4px 4px 0;">
              <strong style="color: var(--text-muted); font-size: 0.65rem;">CHUNK ${i+1} [ID: ${c.id}]</strong><br>
              ${c.content}
            </div>
          `).join('')}
        </div>
      </div>
    `;

    // Click handler to expand
    card.querySelector('.doc-header').addEventListener('click', () => {
      const isExpanded = card.classList.contains('expanded');
      
      // Collapse all
      document.querySelectorAll('.doc-card').forEach(c => c.classList.remove('expanded'));
      
      if (!isExpanded) {
        card.classList.add('expanded');
        card.querySelector('.doc-arrow').textContent = '▲';
      } else {
        card.querySelector('.doc-arrow').textContent = '▼';
      }
    });

    docLibraryContainer.appendChild(card);
  });
}

// Setup chat interaction
function setupChat() {
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const queryText = chatInput.value.trim();
    if (!queryText) return;

    // Clear input
    chatInput.value = '';

    // Append user message
    appendChatBubble('user', queryText);

    // Show AI status bubble (typing loader)
    const loaderId = appendChatBubble('assistant', `<p style="color: var(--text-muted); font-family: monospace;">[ RAG PIPELINE: AGMC SEARCHING DATABASE... ]</p>`);

    try {
      // Execute Retrieval
      const results = retrieve(queryText, 3);
      
      // Update Pipeline Explorer visuals
      updatePipelineVisualizer(queryText, results);

      // Update Neural Query Node coordinates in 2D vector space
      const validMatches = results.filter(r => r.score > 0.05);
      if (validMatches.length > 0) {
        let sumX = 0, sumY = 0, sumScore = 0;
        validMatches.forEach(m => {
          sumX += m.visualX * m.score;
          sumY += m.visualY * m.score;
          sumScore += m.score;
        });
        queryNode.targetX = sumX / sumScore;
        queryNode.targetY = sumY / sumScore;
        queryNode.active = true;
        queryConnections = validMatches;
      } else {
        queryNode.targetX = 200;
        queryNode.targetY = 90;
        queryNode.active = true;
        queryConnections = [];
      }

      // Extract top result for quick peek drawer
      const topMatch = results[0];
      if (topMatch && topMatch.score > 0.05) {
        contextPeekCount.textContent = `${results.filter(r => r.score > 0.05).length} match(es)`;
        contextPeekDoc.textContent = `${topMatch.chunk.docTitle} [Ch: ${topMatch.chunk.id.split('chunk_')[1]}]`;
        contextPeekScore.textContent = topMatch.score.toFixed(3);
      } else {
        contextPeekCount.textContent = `0 matches`;
        contextPeekDoc.textContent = `No relevant context found`;
        contextPeekScore.textContent = `0.000`;
      }

      // Generate context block for LLM prompt
      const contextText = results.map(r => `Document: ${r.chunk.docTitle} (Category: ${r.chunk.category})\nChunk Content:\n${r.chunk.content}\n---`).join('\n\n');

      // System Prompt defining the Guidance Computer persona
      const systemPrompt = `You are the Apollo Guidance Mission Computer (AGMC) assistant co-pilot installed inside the Lunar Lander Command Dashboard. 
Your tone is technical, historical, and helpful. 
You answer flight telemetry questions and spacecraft specification queries based ONLY on the provided context document logs below. 

If the user asks about their recent simulator flight, collision, landing speed, crash cause, or telemetry, pull details directly from the "Live Simulator Flight Telemetry" context. Analyze the speed or angle relative to limits (Limit Vz descent: 3.0 m/s, Limit Vx drift: 1.5 m/s, Limit Tilt: 5 degrees).

If the provided context does not contain enough information to answer the question, state that you cannot find this information in the indexed Apollo logs, and offer relevant historical details.

CONTEXT DOCUMENTS:
${contextText}`;

      // Update prompt box visual step
      pipelinePromptBox.value = `${systemPrompt}\n\nUSER QUESTION:\n${queryText}`;

      let aiResponseText = "";
      if (apiConfig.apiKey) {
        // Run real Gemini RAG API call
        aiResponseText = await generateGeminiResponse(apiConfig.apiKey, systemPrompt, queryText);
      } else {
        // Run offline local matcher co-pilot
        aiResponseText = generateLocalResponse(queryText, results);
      }

      // Remove loader bubble
      const loaderBubble = document.getElementById(loaderId);
      if (loaderBubble) loaderBubble.remove();

      // Append real response
      appendChatBubble('assistant', markdownToHTML(aiResponseText));

    } catch (err) {
      console.error(err);
      const loaderBubble = document.getElementById(loaderId);
      if (loaderBubble) loaderBubble.remove();
      appendChatBubble('assistant', `<p style="color: var(--neon-red);">⚠️ <strong>ERROR IN GUIDANCE COMPUTER:</strong> Failed to generate response. ${err.message}. If you entered an API key, check your internet or key validity. Falling back to simulation mode.</p>`);
    }
  });
}

// Appends chat bubble to history log
function appendChatBubble(sender, htmlContent) {
  const bubbleId = `msg_${Date.now()}_${Math.floor(Math.random()*1000)}`;
  const msg = document.createElement('div');
  msg.className = `chat-message ${sender}`;
  msg.id = bubbleId;

  const headerSpan = sender === 'assistant' 
    ? `<span>AGMC Co-Pilot</span><span>${new Date().toLocaleTimeString()}</span>`
    : `<span>Commander (Pilot)</span><span>${new Date().toLocaleTimeString()}</span>`;

  msg.innerHTML = `
    <div class="msg-header">${headerSpan}</div>
    <div class="msg-body">${htmlContent}</div>
  `;

  chatHistory.appendChild(msg);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return bubbleId;
}

// Updates RAG Debug visualizer elements
function updatePipelineVisualizer(queryText, results) {
  // 1. Process Tokens
  const tokens = tokenize(queryText);
  pipelineTokens.innerHTML = '';
  if (tokens.length === 0) {
    pipelineTokens.innerHTML = `<span class="token-badge" style="opacity: 0.5;">Empty Tokens</span>`;
  } else {
    tokens.forEach(tok => {
      const b = document.createElement('span');
      b.className = 'token-badge';
      b.textContent = tok;
      pipelineTokens.appendChild(b);
    });
  }

  // 2. Process Cosine Similarity Rank Table
  pipelineRankTableBody.innerHTML = '';
  if (results.length === 0) {
    pipelineRankTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No matches computed.</td></tr>`;
    return;
  }

  results.forEach((res, index) => {
    const row = document.createElement('tr');
    if (index === 0 && res.score > 0.05) row.className = 'top-match';

    const snippet = res.chunk.content.substring(0, 75) + '...';
    
    row.innerHTML = `
      <td style="font-family: monospace;">#${index + 1}</td>
      <td class="match-score">${res.score.toFixed(4)}</td>
      <td><span class="doc-category" style="font-size: 0.6rem;">${res.chunk.category}</span></td>
      <td>
        <div class="chunk-source">${res.chunk.docTitle} [${res.chunk.id}]</div>
        <div class="chunk-text-preview" title="${res.chunk.content}">${snippet}</div>
      </td>
    `;
    pipelineRankTableBody.appendChild(row);
  });
}

// Generate Gemini API response over RAG context
async function generateGeminiResponse(apiKey, systemPrompt, userQuery) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
  
  const payload = {
    systemInstruction: {
      parts: [
        { text: "You are the Apollo Guidance Mission Computer (AGMC) assistant co-pilot installed inside the Lunar Lander Command Dashboard. Your tone is technical, historical, and helpful. You answer flight telemetry questions and spacecraft specification queries based ONLY on the provided context document logs." }
      ]
    },
    contents: [
      {
        parts: [
          { text: systemPrompt + "\n\nUser Question: " + userQuery }
        ]
      }
    ],
    generationConfig: {
      temperature: 0.15,
      maxOutputTokens: 600
    }
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errMsg = errorData.error?.message || `HTTP ${response.status}`;
    throw new Error(errMsg);
  }

  const data = await response.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
    throw new Error("Empty response returned from Gemini endpoint");
  }
  return text;
}

// Heuristic offline model for zero-setup RAG demonstrations
function generateLocalResponse(queryText, results) {
  const queryWords = tokenize(queryText);
  const qStr = queryText.toLowerCase();

  // Find if dynamic telemetry document is top retrieved
  const topResult = results[0];
  const hasValidContext = topResult && topResult.score > 0.05;
  const isTelemetryQuery = qStr.includes('crash') || qStr.includes('fail') || qStr.includes('my flight') || qStr.includes('my landing') || qStr.includes('telemetry') || qStr.includes('went wrong');

  // Prefix showing co-pilot mode status
  const prefix = `🤖 <strong>[OFFLINE SIMULATION CO-PILOT ACTIVE]</strong><br><br>`;

  // A. If telemetry query and we have valid telemetry document
  if (isTelemetryQuery) {
    const telChunk = results.find(r => r.chunk.docId === 'telemetry');
    
    if (telChunk && !lastFlightTelemetryText.includes("No simulation flight telemetry")) {
      // Parse telemetry logs
      const isSuccess = lastFlightTelemetryText.includes("SUCCESSFUL");
      const padMatch = lastFlightTelemetryText.match(/Ground Landing Site:\s*(.*)/);
      const padStr = padMatch ? padMatch[1] : "Unknown";
      
      const vMatch = lastFlightTelemetryText.match(/Landing Vertical Velocity \(Vz\):\s*([-\d.]+)\s*m\/s/);
      const hMatch = lastFlightTelemetryText.match(/Landing Horizontal Velocity \(Vx\):\s*([-\d.]+)\s*m\/s/);
      const tMatch = lastFlightTelemetryText.match(/Attitude Pitch Tilt Angle:\s*([\d.]+)\s*degrees/);
      const fMatch = lastFlightTelemetryText.match(/Propellant fuel remaining:\s*([\d.]+)\%/);
      
      const vSpeed = vMatch ? parseFloat(vMatch[1]) : 0;
      const hSpeed = hMatch ? parseFloat(hMatch[1]) : 0;
      const tilt = tMatch ? parseFloat(tMatch[1]) : 0;
      const fuel = fMatch ? parseFloat(fMatch[1]) : 0;

      let debugExplanation = "";
      if (isSuccess) {
        debugExplanation = `**Mission Analysis:** Touchdown successful on **${padStr}**! Excellent flight management. Your final descent speeds (Vz: ${vSpeed.toFixed(2)} m/s, Vx: ${hSpeed.toFixed(2)} m/s) and tilt attitude (${tilt.toFixed(1)}°) were all safely within limits (Vz < 3.0 m/s, Vx < 1.5 m/s, Tilt < 5°). Fuel usage left you with **${fuel.toFixed(1)}%** hypergolic reserves, indicating high engine throttle efficiency.`;
      } else {
        debugExplanation = `**Failure Debriefing:** The spacecraft suffered a catastrophic structural failure upon landing. Let's inspect the telemetry vectors:
*   **Target site:** landed on *${padStr}*.
*   **Vertical Speed Vz:** ${vSpeed.toFixed(2)} m/s (Limit: 3.0 m/s) ${Math.abs(vSpeed) > 3.0 ? '🚨 [EXCEEDED]' : '✅ [SAFE]'}
*   **Horizontal Drift Vx:** ${hSpeed.toFixed(2)} m/s (Limit: 1.5 m/s) ${Math.abs(hSpeed) > 1.5 ? '🚨 [EXCEEDED]' : '✅ [SAFE]'}
*   **Attitude Tilt:** ${tilt.toFixed(1)}° (Limit: 5.0°) ${tilt > 5.0 ? '🚨 [EXCEEDED]' : '✅ [SAFE]'}
*   **Fuel reserves:** ${fuel.toFixed(1)}%

**Recommended Corrections:** ${Math.abs(vSpeed) > 3.0 ? 'Pulse your engine throttle W/Up arrow in shorter intervals as you approach the surface to reduce vertical landing impact.' : ''} ${Math.abs(hSpeed) > 1.5 ? 'Use minor RCS taps A/D or Left/Right arrows to center the lander and cancel lateral velocity vectors.' : ''} ${tilt > 5.0 ? 'Keep the spacecraft oriented straight up (0°) in the final 20 meters to prevent tipover.' : ''} ${padStr.includes("None") ? 'Ensure you target one of the glowing flat landing pads. Landing on rough mountainous terrain will instantly destroy the spacecraft gear.' : ''}`;
      }

      return prefix + debugExplanation;
    } else {
      return prefix + `No active landing attempts have been registered in this flight session yet. Start the thrusters using **W/Up Arrow**, guide the capsule onto a landing pad, and then ask me to analyze the flight telemetry details.`;
    }
  }

  // B. Search keyword routing based on top retrieved chunks
  if (hasValidContext) {
    const chunk = topResult.chunk;
    const cat = chunk.category;

    if (cat === 'Specifications' || qStr.includes('fuel') || qStr.includes('propellant') || qStr.includes('engine') || qStr.includes('thrust')) {
      return prefix + `**Mission Archives Search: Lunar Module Propulsion Spec**
The descent propulsion stage uses hypergolic propellants: **Aerozine 50** as fuel and **Nitrogen Tetroxide (N2O4)** as the oxidizer. 
These chemicals ignite instantly when mixed together in a vacuum, making them highly reliable since they don't require an electric ignition spark. The Lunar Module Descent Engine is throttleable between 1,050 lbs (4.7 kN) and 9,850 lbs (43.8 kN) of thrust, which allows for delicate hover maneuvers and descent velocity throttling.`;
    }

    if (cat === 'Physics' || qStr.includes('physics') || qStr.includes('gravity') || qStr.includes('atmosphere') || qStr.includes('parachutes')) {
      return prefix + `**Mission Archives Search: Lunar Physics**
Landing on the Moon is governed by unique forces:
1.  **Gravity:** Lunar gravity is **1.62 m/s²** (roughly 1/6th of Earth's gravity). Objects accelerate downwards slower, but still gain dangerous momentum if unchecked.
2.  **No Atmosphere:** There is no air resistance on the Moon. Traditional flight controls and **parachutes are 100% useless**. Aerodynamic deceleration is impossible.
3.  **Active Deceleration:** You must rely entirely on retro-propulsion (rocket firing opposite to velocity vector) to slow down. Without it, the capsule will continue accelerating until it hits the ground.`;
    }

    if (cat === 'Game Manual' || qStr.includes('how to land') || qStr.includes('tips') || qStr.includes('control') || qStr.includes('keys')) {
      return prefix + `**Lunar Lander Flight Operations Guide**
To perform a safe landing in this simulator, consult the flight checklist:
*   **Controls:** Press **UP ARROW** or **W** to engage the main rocket engine. Tap **LEFT/RIGHT ARROWS** or **A/D** to pivot the lander.
*   **Throttle Strategy:** Pulse the engine in short bursts rather than holding it down continuously. This conserves fuel and prevents launching back into orbit.
*   **Safe Limits:** Touchdown velocity must be under **3.0 m/s** vertical descent speed and **1.5 m/s** horizontal drift, with a hull tilt of less than **5.0°**.
*   **Targets:** Land only on the glowing flat colored pads. Narrower pads (Bravo, Charlie) give score multipliers (2x, 5x) but require high steering precision.`;
    }

    if (cat === 'History' || qStr.includes('apollo 11') || qStr.includes('neil armstrong') || qStr.includes('history') || qStr.includes('alarm')) {
      return prefix + `**Historical Landing Archives: Apollo 11**
On July 20, 1969, Apollo 11 Lunar Module *Eagle* landed at Tranquility Base. Mission Commander Neil Armstrong was forced to switch to manual control when the Apollo Guidance Computer became overloaded with radar signals, triggering **1201 and 1202 program alarms**. 
Armstrong navigated the lander past a crater filled with boulders, landing successfully with only about **25 to 30 seconds of descent propellant remaining** in the fuel tanks.`;
    }

    // Generic summarization fallback from top matching chunks
    let summaryText = `**Mission Computer Data Retrieval Match (Score: ${topResult.score.toFixed(3)}):**\n\n`;
    summaryText += `*   *From ${chunk.docTitle} [Category: ${chunk.category}]:*\n    "${chunk.content}"\n\n`;
    
    if (results[1] && results[1].score > 0.05) {
      summaryText += `*   *Supporting context from ${results[1].chunk.docTitle}:*\n    "${results[1].chunk.content}"`;
    }
    return prefix + summaryText;
  }

  // C. Fallback for completely out of context queries
  return prefix + `I could not retrieve any relevant data chunks matching your query from the Apollo Guidance Archives. 

Try asking about **propellant chemistry**, **lunar gravity physics**, **flight control keys**, **descent limits (3.0 m/s)**, or perform a simulation launch and ask: **"Why did I crash?"** to analyze telemetry.`;
}

// Simple Markdown-to-HTML formatter
function markdownToHTML(text) {
  let html = text;

  // Replace double linebreaks with paragraphs
  html = html.split('\n\n').map(para => {
    para = para.trim();
    if (para.startsWith('🤖') || para.startsWith('🚨') || para.startsWith('⚙️')) {
      return para;
    }
    return `<p>${para}</p>`;
  }).join('');

  // Bold text: **word**
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Bullet items: * text or - text
  html = html.replace(/^(?:\*|-)\s+(.+)$/gm, '<li>$1</li>');
  // Wrap consecutive list items in <ul>
  html = html.replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');
  // Remove nested <ul> tags created by regex
  html = html.replace(/<\/ul><ul>/g, '');

  // Inline Code: `code`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  return html;
}

// 2.0 Advanced Upgrades: Plotting and Control Handlers
function setupAdvancedControls() {
  // Environment selector bindings
  if (envSelect) {
    envSelect.addEventListener('change', () => {
      const selectedEnv = envSelect.value;
      if (gameInstance) {
        gameInstance.setEnvironment(selectedEnv);
        // Wipe historical chart telemetry when environment is reset to keep layout clean
        telemetryHistory = [];
        appendChatBubble('assistant', `<p>🪐 <strong>Planetary Orbit Profile Activated:</strong> Lander target switched to **${gameInstance.envConfigs[selectedEnv].label}**. Gravity altered to **${gameInstance.gravity * 100} units**, atmosphere changed to **${gameInstance.envConfigs[selectedEnv].density}**.</p>`);
      }
    });
  }

  // RAG Tuner slider bindings
  if (sliderChunkSize && sliderOverlap) {
    sliderChunkSize.addEventListener('input', () => {
      const sizeVal = parseInt(sliderChunkSize.value);
      labelChunkSize.textContent = `${sizeVal} sentence${sizeVal > 1 ? 's' : ''}`;
      triggerRAGRebuild();
    });

    sliderOverlap.addEventListener('input', () => {
      const overlapVal = parseInt(sliderOverlap.value);
      labelOverlap.textContent = `${overlapVal} sentence${overlapVal > 1 ? 's' : ''}`;
      triggerRAGRebuild();
    });
  }

  // Game Screen Overlay Relaunch button bindings
  if (overlayResetBtn) {
    overlayResetBtn.addEventListener('click', () => {
      if (gameInstance) gameInstance.reset();
    });
  }
}

function triggerRAGRebuild() {
  const size = parseInt(sliderChunkSize.value);
  let overlap = parseInt(sliderOverlap.value);
  
  // Guard check: overlap must be less than chunk size
  if (overlap >= size) {
    overlap = size - 1;
    sliderOverlap.value = overlap;
    labelOverlap.textContent = `${overlap} sentence${overlap > 1 ? 's' : ''}`;
  }
  
  // Re-run chunking
  initializeRAG(size, overlap);
  injectTelemetryChunk(lastFlightTelemetryText, size, overlap);
  renderDocumentLibrary();
  
  // Reset visual query nodes
  queryNode.active = false;
  queryConnections = [];
}

function updateTelemetryHistory() {
  if (!gameInstance) return;
  const tel = gameInstance.getTelemetry();
  telemetryHistory.push({
    altitude: tel.altitude,
    speed: Math.sqrt(gameInstance.vx * gameInstance.vx + gameInstance.vy * gameInstance.vy) * 5,
    fuel: tel.fuel
  });
  if (telemetryHistory.length > 150) {
    telemetryHistory.shift();
  }
}

function drawTelemetryChart() {
  if (!chartCtx || !chartCanvasInstance) return;
  const w = chartCanvasInstance.width;
  const h = chartCanvasInstance.height;
  chartCtx.clearRect(0, 0, w, h);
  
  // Background
  chartCtx.fillStyle = '#0c101d';
  chartCtx.fillRect(0, 0, w, h);
  
  // Draw Grid Lines
  chartCtx.strokeStyle = 'rgba(56, 189, 248, 0.06)';
  chartCtx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    chartCtx.beginPath();
    chartCtx.moveTo(x, 0);
    chartCtx.lineTo(x, h);
    chartCtx.stroke();
  }
  for (let y = 0; y < h; y += 20) {
    chartCtx.beginPath();
    chartCtx.moveTo(0, y);
    chartCtx.lineTo(w, y);
    chartCtx.stroke();
  }
  
  if (telemetryHistory.length < 2) return;
  
  // Draw curves
  // 1. Fuel (Red)
  chartCtx.strokeStyle = 'rgba(239, 68, 68, 0.65)';
  chartCtx.lineWidth = 1.5;
  chartCtx.beginPath();
  telemetryHistory.forEach((pt, idx) => {
    const px = (idx / 150) * w;
    const py = h - (pt.fuel / 100) * (h - 12) - 6;
    if (idx === 0) chartCtx.moveTo(px, py);
    else chartCtx.lineTo(px, py);
  });
  chartCtx.stroke();
  
  // 2. Altitude (Green)
  chartCtx.strokeStyle = 'rgba(16, 185, 129, 0.65)';
  chartCtx.lineWidth = 1.5;
  chartCtx.beginPath();
  telemetryHistory.forEach((pt, idx) => {
    const px = (idx / 150) * w;
    const py = h - (pt.altitude / 340) * (h - 12) - 6;
    if (idx === 0) chartCtx.moveTo(px, py);
    else chartCtx.lineTo(px, py);
  });
  chartCtx.stroke();
  
  // 3. Speed (Amber)
  chartCtx.strokeStyle = 'rgba(245, 158, 11, 0.65)';
  chartCtx.lineWidth = 1.5;
  chartCtx.beginPath();
  telemetryHistory.forEach((pt, idx) => {
    const px = (idx / 150) * w;
    const py = h - (Math.min(15, pt.speed) / 15) * (h - 12) - 6;
    if (idx === 0) chartCtx.moveTo(px, py);
    else chartCtx.lineTo(px, py);
  });
  chartCtx.stroke();
  
  // Text Labels on chart
  chartCtx.fillStyle = 'rgba(255, 255, 255, 0.35)';
  chartCtx.font = '7px monospace';
  chartCtx.textAlign = 'left';
  chartCtx.fillText("ALTITUDE (GREEN)", 10, 10);
  chartCtx.fillText("FUEL % (RED)", 105, 10);
  chartCtx.fillText("SPEED (AMBER)", 180, 10);
}

function drawVectorSpace() {
  if (!vectorCtx || !vectorCanvasInstance) return;
  const w = vectorCanvasInstance.width;
  const h = vectorCanvasInstance.height;
  vectorCtx.clearRect(0, 0, w, h);
  
  // Background Grid
  vectorCtx.fillStyle = '#070a14';
  vectorCtx.fillRect(0, 0, w, h);
  
  vectorCtx.strokeStyle = 'rgba(56, 189, 248, 0.04)';
  vectorCtx.lineWidth = 1;
  for (let x = 0; x < w; x += 30) {
    vectorCtx.beginPath();
    vectorCtx.moveTo(x, 0);
    vectorCtx.lineTo(x, h);
    vectorCtx.stroke();
  }
  for (let y = 0; y < h; y += 30) {
    vectorCtx.beginPath();
    vectorCtx.moveTo(0, y);
    vectorCtx.lineTo(w, y);
    vectorCtx.stroke();
  }
  
  // Draw Category cluster background boundaries/labels
  const anchors = {
    'History': { x: 80, y: 130, label: "History" },
    'Specifications': { x: 80, y: 50, label: "Specs" },
    'Physics': { x: 320, y: 130, label: "Physics" },
    'Game Manual': { x: 320, y: 50, label: "Manual" },
    'Telemetry': { x: 200, y: 90, label: "Live Telemetry" }
  };
  
  vectorCtx.fillStyle = 'rgba(255, 255, 255, 0.2)';
  vectorCtx.font = '8px monospace';
  vectorCtx.textAlign = 'center';
  Object.values(anchors).forEach(anc => {
    vectorCtx.fillText(anc.label, anc.x, anc.y - 20);
    
    // Draw anchor bounds glow circles
    vectorCtx.fillStyle = 'rgba(56, 189, 248, 0.025)';
    vectorCtx.beginPath();
    vectorCtx.arc(anc.x, anc.y, 35, 0, Math.PI * 2);
    vectorCtx.fill();
    vectorCtx.fillStyle = 'rgba(255, 255, 255, 0.2)';
  });
  
  // Get current vector database nodes
  const ragState = getRAGState();
  const colors = {
    'History': '#06b6d4',
    'Specifications': '#eab308',
    'Physics': '#10b981',
    'Game Manual': '#f97316',
    'Telemetry': '#ef4444'
  };
  
  // Draw connections if query is active
  if (queryNode.active) {
    // Interpolate query node visual coordinate towards target coordinate
    queryNode.x += (queryNode.targetX - queryNode.x) * 0.1;
    queryNode.y += (queryNode.targetY - queryNode.y) * 0.1;
    
    queryConnections.forEach(res => {
      const chunkWeight = res.score;
      if (chunkWeight > 0.05) {
        vectorCtx.strokeStyle = `rgba(56, 189, 248, ${chunkWeight * 0.7})`;
        vectorCtx.lineWidth = chunkWeight * 2.5;
        vectorCtx.beginPath();
        vectorCtx.moveTo(queryNode.x, queryNode.y);
        vectorCtx.lineTo(res.visualX, res.visualY);
        vectorCtx.stroke();
      }
    });
  }
  
  // Draw all chunk nodes
  ragState.chunkVectors.forEach(item => {
    const col = colors[item.category] || '#ffffff';
    vectorCtx.fillStyle = col;
    vectorCtx.beginPath();
    vectorCtx.arc(item.visualX, item.visualY, 3.5, 0, Math.PI * 2);
    vectorCtx.fill();
    
    // Draw highlighted circle if top matching
    if (queryNode.active) {
      const isMatch = queryConnections.find(qc => qc.chunk.id === item.chunkId);
      if (isMatch && isMatch.score > 0.05) {
        vectorCtx.strokeStyle = '#ffffff';
        vectorCtx.lineWidth = 1;
        vectorCtx.beginPath();
        vectorCtx.arc(item.visualX, item.visualY, 6, 0, Math.PI * 2);
        vectorCtx.stroke();
      }
    }
  });
  
  // Draw Query Node (White pulsating star)
  if (queryNode.active) {
    queryNode.pulse += 0.1;
    const glowRadius = 5 + Math.sin(queryNode.pulse) * 2;
    
    // Glow Ring
    vectorCtx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    vectorCtx.lineWidth = 1.2;
    vectorCtx.beginPath();
    vectorCtx.arc(queryNode.x, queryNode.y, glowRadius, 0, Math.PI * 2);
    vectorCtx.stroke();
    
    // Center star dot
    vectorCtx.fillStyle = '#ffffff';
    vectorCtx.shadowColor = '#ffffff';
    vectorCtx.shadowBlur = 8;
    vectorCtx.beginPath();
    vectorCtx.arc(queryNode.x, queryNode.y, 3, 0, Math.PI * 2);
    vectorCtx.fill();
    vectorCtx.shadowBlur = 0; // reset
  }
}
