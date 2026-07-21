// UI Elements
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnEval = document.getElementById("btn-eval");
const statusBadge = document.getElementById("status-badge");
const logConsole = document.getElementById("log-console");

// Stats Elements
const statEpisode = document.getElementById("stat-episode");
const statReward = document.getElementById("stat-reward");
const statAvgReward = document.getElementById("stat-avg-reward");
const statLoss = document.getElementById("stat-loss");
const statEpsilon = document.getElementById("stat-epsilon");

// Hyperparameters Elements
const paramEpisodes = document.getElementById("param-episodes");
const paramMaxSteps = document.getElementById("param-max-steps");
const paramLr = document.getElementById("param-lr");
const paramGamma = document.getElementById("param-gamma");
const paramEpsilonDecay = document.getElementById("param-epsilon-decay");
const paramBatchSize = document.getElementById("param-batch-size");
const paramVisualMode = document.getElementById("param-visual-mode");

// Physics Sandbox Elements
const paramGravity = document.getElementById("param-gravity");
const paramLength = document.getElementById("param-length");

// Bubbles
const valEpisodes = document.getElementById("val-episodes");
const valMaxSteps = document.getElementById("val-max-steps");
const valLr = document.getElementById("val-lr");
const valGamma = document.getElementById("val-gamma");
const valEpsilonDecay = document.getElementById("val-epsilon-decay");
const valGravity = document.getElementById("val-gravity");
const valLength = document.getElementById("val-length");

// Telemetry overlays
const telPos = document.getElementById("tel-pos");
const telAngle = document.getElementById("tel-angle");

// Sound Elements
const btnSound = document.getElementById("btn-sound");
const iconSoundOn = document.getElementById("icon-sound-on");
const iconSoundOff = document.getElementById("icon-sound-off");

// Canvas
const canvas = document.getElementById("cartpole-canvas");
const ctx = canvas.getContext("2d");

// WebSocket Reference
let ws;

// 1. SOUND EFFECTS (Web Audio API Retro Synthesizer)
let audioCtx = null;
let isMuted = false;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

// Sound toggle event
btnSound.addEventListener("click", () => {
    isMuted = !isMuted;
    if (isMuted) {
        iconSoundOn.classList.add("hidden");
        iconSoundOff.classList.remove("hidden");
    } else {
        iconSoundOn.classList.remove("hidden");
        iconSoundOff.classList.add("hidden");
        // Resume context if suspended (browser security)
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
    }
});

function playTone(freq, type, duration, vol, slideToFreq = null) {
    if (isMuted) return;
    try {
        initAudio();
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        
        if (slideToFreq !== null) {
            osc.frequency.exponentialRampToValueAtTime(slideToFreq, audioCtx.currentTime + duration);
        }
        
        gainNode.gain.setValueAtTime(vol, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
        
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.warn("Audio Context failed: ", e);
    }
}

function playTick() {
    playTone(180, 'sine', 0.05, 0.05);
}

function playCrash() {
    playTone(250, 'sawtooth', 0.4, 0.15, 60);
}

function playSuccess() {
    // Retro arpeggio chime
    const now = audioCtx ? audioCtx.currentTime : 0;
    try {
        initAudio();
        if (isMuted) return;
        const notes = [261.63, 329.63, 392.00, 523.25]; // C4, E4, G4, C5
        notes.forEach((freq, idx) => {
            setTimeout(() => {
                playTone(freq, 'triangle', 0.15, 0.08);
            }, idx * 80);
        });
    } catch(e){}
}

// 2. NAVIGATION TAB SYSTEM
const tabs = document.querySelectorAll(".tab-link");
const tabContents = document.querySelectorAll(".tab-content");
let currentActiveTab = "tab-training";

tabs.forEach(tab => {
    tab.addEventListener("click", () => {
        tabs.forEach(t => t.classList.remove("active"));
        tabContents.forEach(tc => tc.classList.remove("active"));
        
        tab.classList.add("active");
        currentActiveTab = tab.getAttribute("data-tab");
        document.getElementById(currentActiveTab).classList.add("active");
        
        // Stop manual play if tab changed
        if (currentActiveTab !== "tab-manual") {
            stopManualGame();
        } else {
            initManualGame();
        }
    });
});

// 3. SLIDERS LISTENERS
paramEpisodes.addEventListener("input", (e) => valEpisodes.innerText = e.target.value);
paramMaxSteps.addEventListener("input", (e) => valMaxSteps.innerText = e.target.value);
paramLr.addEventListener("input", (e) => {
    const val = Math.pow(10, parseFloat(e.target.value));
    valLr.innerText = val === 0.001 ? "0.001" : val.toFixed(4);
});
paramGamma.addEventListener("input", (e) => valGamma.innerText = e.target.value);
paramEpsilonDecay.addEventListener("input", (e) => valEpsilonDecay.innerText = e.target.value);
paramGravity.addEventListener("input", (e) => {
    valGravity.innerText = e.target.value;
    // Update live drawing
    drawCartPole(currentX, currentTheta);
});
paramLength.addEventListener("input", (e) => {
    valLength.innerText = e.target.value;
    // Update live drawing
    drawCartPole(currentX, currentTheta);
});

// Helper to manually trigger slider events to update bubbles
function triggerSliderUpdates() {
    paramEpisodes.dispatchEvent(new Event("input"));
    paramMaxSteps.dispatchEvent(new Event("input"));
    paramLr.dispatchEvent(new Event("input"));
    paramGamma.dispatchEvent(new Event("input"));
    paramEpsilonDecay.dispatchEvent(new Event("input"));
    paramGravity.dispatchEvent(new Event("input"));
    paramLength.dispatchEvent(new Event("input"));
}

// 4. PRESETS HANDLERS
const presetBalanced = document.getElementById("preset-balanced");
const presetFast = document.getElementById("preset-fast");
const presetStable = document.getElementById("preset-stable");
const presetBtns = [presetBalanced, presetFast, presetStable];

function selectPreset(btn, config) {
    presetBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    
    paramEpisodes.value = config.episodes;
    paramMaxSteps.value = config.maxSteps;
    paramLr.value = config.lr; 
    paramGamma.value = config.gamma;
    paramEpsilonDecay.value = config.epsilonDecay;
    paramBatchSize.value = config.batchSize;
    paramVisualMode.checked = config.visualMode;
    
    // Also reset sandbox sliders to standard defaults
    paramGravity.value = 9.8;
    paramLength.value = 1.0;
    
    triggerSliderUpdates();
    appendLog("system", `Preset loaded: ${config.name}`);
}

presetBalanced.addEventListener("click", () => {
    selectPreset(presetBalanced, {
        name: "Balanced",
        episodes: 300,
        maxSteps: 500,
        lr: -3.0,
        gamma: 0.99,
        epsilonDecay: 0.995,
        batchSize: 128,
        visualMode: true
    });
});

presetFast.addEventListener("click", () => {
    selectPreset(presetFast, {
        name: "Fast Solver",
        episodes: 200,
        maxSteps: 500,
        lr: -2.7,
        gamma: 0.98,
        epsilonDecay: 0.985,
        batchSize: 64,
        visualMode: false
    });
});

presetStable.addEventListener("click", () => {
    selectPreset(presetStable, {
        name: "Conservative",
        episodes: 500,
        maxSteps: 500,
        lr: -3.3,
        gamma: 0.995,
        epsilonDecay: 0.997,
        batchSize: 256,
        visualMode: true
    });
});

// Cache positions for drawing updates
let currentX = 0.0;
let currentTheta = 0.0;

// 5. PHYSICS CANVAS RENDERER
function drawCartPole(x, theta, lastAction = null) {
    currentX = x;
    currentTheta = theta;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw space grids
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    for (let i = 0; i < canvas.width; i += 40) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
    }
    
    // Physical dimensions & scaling
    const groundY = 200;
    const cartWidth = 90;
    const cartHeight = 35;
    const wheelRadius = 10;
    
    // Get customized pole length and scale it to pixels
    // Default length config is 1.0m total, mapping to 120px scale
    const configTotalLength = parseFloat(paramLength.value);
    const poleLength = configTotalLength * 120;
    const poleWidth = 6;
    
    // Map cart position x (-2.4 to 2.4 units) to canvas pixels
    const boundaryX = 2.4;
    const trackWidth = canvas.width - 120;
    const pxPerUnit = trackWidth / (boundaryX * 2);
    const cartX = canvas.width / 2 + x * pxPerUnit;

    // Draw rails / boundary marks
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2 - boundaryX * pxPerUnit, groundY - 20);
    ctx.lineTo(canvas.width / 2 - boundaryX * pxPerUnit, groundY + 40);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2 + boundaryX * pxPerUnit, groundY - 20);
    ctx.lineTo(canvas.width / 2 + boundaryX * pxPerUnit, groundY + 40);
    ctx.stroke();

    // 1. Draw central track rail
    ctx.beginPath();
    ctx.moveTo(30, groundY + cartHeight / 2 + wheelRadius);
    ctx.lineTo(canvas.width - 30, groundY + cartHeight / 2 + wheelRadius);
    ctx.strokeStyle = "rgba(0, 240, 255, 0.15)";
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // 2. Draw Cart Chassis (Gradient box)
    const chassisGradient = ctx.createLinearGradient(
        cartX - cartWidth/2, groundY - cartHeight/2, 
        cartX + cartWidth/2, groundY + cartHeight/2
    );
    chassisGradient.addColorStop(0, '#1e293b');
    chassisGradient.addColorStop(1, '#0f172a');
    
    ctx.fillStyle = chassisGradient;
    ctx.strokeStyle = varColor('--accent-blue');
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.roundRect(cartX - cartWidth / 2, groundY - cartHeight / 2, cartWidth, cartHeight, 6);
    ctx.fill();
    ctx.stroke();
    
    // Carbon-mesh accent line
    ctx.strokeStyle = "rgba(0, 240, 255, 0.3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cartX - cartWidth / 2 + 10, groundY);
    ctx.lineTo(cartX + cartWidth / 2 - 10, groundY);
    ctx.stroke();
    
    // 3. Draw Wheels with rotating spokes
    const rotationAngle = (x * pxPerUnit) / wheelRadius;
    drawWheel(cartX - 25, groundY + cartHeight / 2 + 2, wheelRadius, rotationAngle);
    drawWheel(cartX + 25, groundY + cartHeight / 2 + 2, wheelRadius, rotationAngle);
    
    // 4. Draw Pole
    const poleTipX = cartX + poleLength * Math.sin(theta);
    const poleTipY = groundY - poleLength * Math.cos(theta);
    
    // Pole Gradient
    const poleGradient = ctx.createLinearGradient(cartX, groundY, poleTipX, poleTipY);
    poleGradient.addColorStop(0, '#fff');
    poleGradient.addColorStop(0.8, '#cbd5e1');
    poleGradient.addColorStop(1, varColor('--accent-purple'));
    
    ctx.strokeStyle = poleGradient;
    ctx.lineWidth = poleWidth;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cartX, groundY);
    ctx.lineTo(poleTipX, poleTipY);
    ctx.stroke();
    
    // 5. Draw Joint Connector pin
    ctx.beginPath();
    ctx.arc(cartX, groundY, 7, 0, 2 * Math.PI);
    ctx.fillStyle = varColor('--accent-blue');
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // 6. Force indicators arrows
    if (lastAction !== null) {
        ctx.fillStyle = lastAction === 1 ? varColor('--accent-blue') : varColor('--accent-purple');
        ctx.beginPath();
        const arrowY = groundY;
        if (lastAction === 1) { 
            ctx.moveTo(cartX - cartWidth/2 - 25, arrowY);
            ctx.lineTo(cartX - cartWidth/2 - 8, arrowY);
            ctx.lineTo(cartX - cartWidth/2 - 8, arrowY - 5);
            ctx.lineTo(cartX - cartWidth/2, arrowY);
            ctx.lineTo(cartX - cartWidth/2 - 8, arrowY + 5);
            ctx.lineTo(cartX - cartWidth/2 - 8, arrowY);
        } else { 
            ctx.moveTo(cartX + cartWidth/2 + 25, arrowY);
            ctx.lineTo(cartX + cartWidth/2 + 8, arrowY);
            ctx.lineTo(cartX + cartWidth/2 + 8, arrowY - 5);
            ctx.lineTo(cartX + cartWidth/2, arrowY);
            ctx.lineTo(cartX + cartWidth/2 + 8, arrowY + 5);
            ctx.lineTo(cartX + cartWidth/2 + 8, arrowY);
        }
        ctx.fill();
    }
    
    // Update Telemetry Header Texts
    telPos.innerText = x.toFixed(2);
    telAngle.innerText = (theta * 180 / Math.PI).toFixed(1);
}

// Wheel drawer helper with rotating spokes
function drawWheel(x, y, radius, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    
    // Outer tire
    ctx.fillStyle = "#1e293b";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(0, 0, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    
    // Center rim
    ctx.fillStyle = varColor('--accent-blue');
    ctx.beginPath();
    ctx.arc(0, 0, 3, 0, 2 * Math.PI);
    ctx.fill();
    
    // Spokes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
        ctx.rotate(Math.PI / 4);
        ctx.beginPath();
        ctx.moveTo(-radius, 0);
        ctx.lineTo(radius, 0);
        ctx.stroke();
    }
    
    ctx.restore();
}

function varColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// 6. CHARTS INITIALIZATIONS
let rewardsChart, lossChart;
let episodesData = [];
let rewardsData = [];
let avgRewardsData = [];
let lossData = [];

function initCharts() {
    const rewardsCtx = document.getElementById("rewards-chart").getContext("2d");
    const lossCtx = document.getElementById("loss-chart").getContext("2d");

    if (rewardsChart) rewardsChart.destroy();
    if (lossChart) lossChart.destroy();

    episodesData = [];
    rewardsData = [];
    avgRewardsData = [];
    lossData = [];

    rewardsChart = new Chart(rewardsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Episode Reward',
                    data: [],
                    borderColor: 'rgba(0, 240, 255, 0.4)',
                    backgroundColor: 'rgba(0, 240, 255, 0.03)',
                    borderWidth: 1.5,
                    pointRadius: 1,
                    fill: true
                },
                {
                    label: '100-Ep Moving Avg',
                    data: [],
                    borderColor: 'rgba(168, 85, 247, 1)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Solved Threshold (475)',
                    data: [],
                    borderColor: 'rgba(239, 68, 68, 0.7)',
                    borderDash: [4, 4],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 9 } }
                },
                y: {
                    min: 0,
                    max: 500,
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 9 } }
                }
            }
        }
    });

    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Huber Loss',
                    data: [],
                    borderColor: 'rgba(16, 185, 129, 0.8)',
                    backgroundColor: 'rgba(16, 185, 129, 0.03)',
                    borderWidth: 1.5,
                    pointRadius: 1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 9 } }
                }
            }
        }
    });
}

// 7. WEBSOCKET CONTROLS
function connectWebSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socketUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(socketUrl);

    ws.onopen = () => {
        appendLog("system", "Connected to agent training service.");
    };

    ws.onclose = () => {
        appendLog("error", "Service disconnected. Retrying in 3 seconds...");
        setTimeout(connectWebSocket, 3000);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        switch (msg.type) {
            case "status":
                handleStatusChange(msg);
                break;
            case "info":
                appendLog("info", msg.message);
                break;
            case "step":
                handleStepUpdate(msg);
                break;
            case "episode":
                handleEpisodeUpdate(msg);
                break;
        }
    };
}

function handleStatusChange(msg) {
    statusBadge.innerText = msg.status;
    statusBadge.className = `badge badge-${msg.status}`;

    if (msg.status === "training" || msg.status === "running") {
        btnStart.disabled = true;
        btnEval.disabled = true;
        btnStop.disabled = false;
    } else {
        btnStart.disabled = false;
        btnEval.disabled = false;
        btnStop.disabled = true;
        if (msg.message) {
            appendLog("system", msg.message);
        }
    }
}

function appendLog(type, text) {
    const div = document.createElement("div");
    div.className = `log-line ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    div.innerText = `[${timestamp}] ${text}`;
    logConsole.appendChild(div);
    logConsole.scrollTop = logConsole.scrollHeight;
}

function handleStepUpdate(msg) {
    const [x, x_dot, theta, theta_dot] = msg.state;
    drawCartPole(x, theta, msg.action);
    // Play sound FX for actions (blip)
    playTick();
}

function handleEpisodeUpdate(msg) {
    statEpisode.innerText = msg.episode;
    statReward.innerText = msg.reward.toFixed(1);
    statAvgReward.innerText = msg.avg_reward.toFixed(1);
    statLoss.innerText = msg.loss.toFixed(4);
    statEpsilon.innerText = msg.epsilon.toFixed(3);

    appendLog("info", `Ep ${msg.episode} | Reward: ${msg.reward.toFixed(1)} | Avg: ${msg.avg_reward.toFixed(1)} | Loss: ${msg.loss.toFixed(4)} | Eps: ${msg.epsilon.toFixed(3)}`);

    // Add data to charts
    episodesData.push(msg.episode);
    rewardsData.push(msg.reward);
    avgRewardsData.push(msg.avg_reward);
    lossData.push(msg.loss);

    rewardsChart.data.labels = episodesData;
    rewardsChart.data.datasets[0].data = rewardsData;
    rewardsChart.data.datasets[1].data = avgRewardsData;
    rewardsChart.data.datasets[2].data = Array(episodesData.length).fill(475.0);
    rewardsChart.update();

    lossChart.data.labels = episodesData;
    lossChart.data.datasets[0].data = lossData;
    lossChart.update();

    if (!paramVisualMode.checked) {
        drawCartPole(0, 0); 
    }
}

// Gather customized physics configuration payload
function getPhysicsConfig() {
    return {
        gravity: parseFloat(paramGravity.value),
        // Gymnasium uses half-length internally (L = total / 2)
        pole_length: parseFloat(paramLength.value) / 2.0
    };
}

// Button starts training
btnStart.addEventListener("click", () => {
    initCharts();
    const lrValue = Math.pow(10, parseFloat(paramLr.value));
    const physics = getPhysicsConfig();
    
    ws.send(JSON.stringify({
        command: "start",
        config: {
            episodes: parseInt(paramEpisodes.value),
            max_steps: parseInt(paramMaxSteps.value),
            lr: lrValue,
            gamma: parseFloat(paramGamma.value),
            epsilon_decay: parseFloat(paramEpsilonDecay.value),
            batch_size: parseInt(paramBatchSize.value),
            visual_mode: paramVisualMode.checked,
            // Pass custom sandbox physics
            gravity: physics.gravity,
            pole_length: physics.pole_length
        }
    }));
});

// Button stops current training/evaluation
btnStop.addEventListener("click", () => {
    ws.send(JSON.stringify({
        command: "stop"
    }));
});

// Button evaluates pre-trained models
btnEval.addEventListener("click", () => {
    initCharts();
    const physics = getPhysicsConfig();
    ws.send(JSON.stringify({
        command: "evaluate",
        config: {
            gravity: physics.gravity,
            pole_length: physics.pole_length
        }
    }));
});


// 8. MANUAL PLAY GAME SYSTEM (Local JavaScript Physics Loop)
let manualActive = false;
let manualLoopId = null;
let keys = {};

// Game Physics state variables
let m_x = 0.0;
let m_x_dot = 0.0;
let m_theta = 0.05; 
let m_theta_dot = 0.0;
let m_score = 0;
let m_high_score = parseInt(localStorage.getItem("cartpole_high_score") || "0");

// Physics constants (will be read dynamically from sliders)
const MASSCART = 1.0;
const MASSPOLE = 0.1;
const TOTAL_MASS = MASSCART + MASSPOLE;
const FORCE_MAG = 10.0;
const DT = 0.02; // 50 Hz updates

const btnManualStart = document.getElementById("btn-manual-start");
const btnManualReset = document.getElementById("btn-manual-reset");
const labelManualScore = document.getElementById("manual-score");
const labelHighScore = document.getElementById("manual-high-score");
const btnManualLeft = document.getElementById("btn-manual-left");
const btnManualRight = document.getElementById("btn-manual-right");

// Initialize High score display
labelHighScore.innerText = m_high_score;

// Keypress handling
window.addEventListener("keydown", (e) => {
    if (currentActiveTab === "tab-manual") {
        keys[e.key] = true;
        if (["ArrowLeft", "ArrowRight", " "].includes(e.key)) {
            e.preventDefault();
        }
    }
});
window.addEventListener("keyup", (e) => {
    if (currentActiveTab === "tab-manual") {
        keys[e.key] = false;
    }
});

// Manual click buttons
let activeButtonForce = 0;
btnManualLeft.addEventListener("mousedown", () => activeButtonForce = -FORCE_MAG);
btnManualLeft.addEventListener("mouseup", () => activeButtonForce = 0);
btnManualLeft.addEventListener("mouseleave", () => activeButtonForce = 0);
btnManualLeft.addEventListener("touchstart", (e) => { e.preventDefault(); activeButtonForce = -FORCE_MAG; });
btnManualLeft.addEventListener("touchend", () => activeButtonForce = 0);

btnManualRight.addEventListener("mousedown", () => activeButtonForce = FORCE_MAG);
btnManualRight.addEventListener("mouseup", () => activeButtonForce = 0);
btnManualRight.addEventListener("mouseleave", () => activeButtonForce = 0);
btnManualRight.addEventListener("touchstart", (e) => { e.preventDefault(); activeButtonForce = FORCE_MAG; });
btnManualRight.addEventListener("touchend", () => activeButtonForce = 0);

function initManualGame() {
    stopManualGame();
    m_x = 0.0;
    m_x_dot = 0.0;
    m_theta = (Math.random() - 0.5) * 0.05; 
    m_theta_dot = 0.0;
    m_score = 0;
    labelManualScore.innerText = "0";
    drawCartPole(m_x, m_theta);
}

btnManualStart.addEventListener("click", () => {
    if (!manualActive) {
        initManualGame();
        manualActive = true;
        btnManualStart.innerText = "Pause Game";
        btnManualReset.disabled = false;
        statusBadge.innerText = "playing";
        statusBadge.className = "badge badge-training";
        appendLog("system", "Manual game started. Use Left/Right Arrow keys to balance!");
        manualLoopId = setInterval(manualPhysicsUpdate, 20); 
    } else {
        stopManualGame();
        btnManualStart.innerText = "Resume Game";
    }
});

btnManualReset.addEventListener("click", () => {
    initManualGame();
    if (manualActive) {
        // Keep running
    } else {
        btnManualStart.innerText = "Start Game";
    }
});

function stopManualGame() {
    manualActive = false;
    if (manualLoopId) {
        clearInterval(manualLoopId);
        manualLoopId = null;
    }
    btnManualStart.innerText = "Play Game";
    statusBadge.innerText = "idle";
    statusBadge.className = "badge badge-idle";
}

// Math logic for physical state updates
function manualPhysicsUpdate() {
    let force = 0.0;
    let lastAction = null;
    
    if (keys["ArrowLeft"] || activeButtonForce < 0) {
        force = -FORCE_MAG;
        lastAction = 0;
        playTick();
    } else if (keys["ArrowRight"] || activeButtonForce > 0) {
        force = FORCE_MAG;
        lastAction = 1;
        playTick();
    }
    
    // Overwrite sandbox constants dynamically from sliders
    const gravity = parseFloat(paramGravity.value);
    const totalLength = parseFloat(paramLength.value);
    const halfLength = totalLength / 2.0; // half length (L) used in standard equations
    const poleMassLength = MASSPOLE * halfLength;
    
    // Euler-Cromer equations of Cart-Pole dynamics
    const costheta = Math.cos(m_theta);
    const sintheta = Math.sin(m_theta);
    
    const temp = (force + poleMassLength * m_theta_dot * m_theta_dot * sintheta) / TOTAL_MASS;
    const thetaacc = (gravity * sintheta - costheta * temp) / (halfLength * (4.0/3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS));
    const xacc = temp - poleMassLength * thetaacc * costheta / TOTAL_MASS;
    
    m_x += DT * m_x_dot;
    m_x_dot += DT * xacc;
    m_theta += DT * m_theta_dot;
    m_theta_dot += DT * thetaacc;
    
    // Game bounds validation
    const angleLimit = 12 * Math.PI / 180; // 12 degrees
    const posLimit = 2.4;
    
    // Increment score
    m_score += 1;
    labelManualScore.innerText = m_score;
    
    // Draw
    drawCartPole(m_x, m_theta, lastAction);
    
    if (Math.abs(m_x) > posLimit || Math.abs(m_theta) > angleLimit) {
        // Game Over
        stopManualGame();
        playCrash();
        appendLog("error", `Game Over! You fell over at score: ${m_score}`);
        
        if (m_score > m_high_score) {
            m_high_score = m_score;
            localStorage.setItem("cartpole_high_score", m_high_score);
            labelHighScore.innerText = m_high_score;
            playSuccess();
            appendLog("system", `New High Score reached: ${m_high_score}!`);
        }
        btnManualStart.innerText = "Play Again";
    }
}

// Run Initializations
initCharts();
connectWebSocket();
drawCartPole(0, 0);
triggerSliderUpdates();
