// Lunar Lander Simulation Engine (HTML5 Canvas & Web Audio API)

class LunarLanderGame {
  constructor(canvasId, onStateChange, onTelemetryLogged) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.onStateChange = onStateChange; // Callback for state updates (flying, landed, crashed)
    this.onTelemetryLogged = onTelemetryLogged; // Callback to inject flight log into RAG
    
    // Physical Constants (Default Moon)
    this.gravity = 0.015; // Scaled gravity (pixels/frame^2)
    this.thrustPower = 0.035; // Engine acceleration
    this.rotationSpeed = 0.035; // Pitch speed (rad/frame)
    this.drag = 1.0; // Vacuum drag
    
    // Environment configurations dictionary
    this.env = 'moon';
    this.envConfigs = {
      moon: { gravity: 0.015, drag: 1.0, label: "Moon", density: "Vacuum (0%)" },
      mars: { gravity: 0.035, drag: 0.995, label: "Mars", density: "Thin Atmosphere (10%)" },
      earth: { gravity: 0.09, drag: 0.98, label: "Earth", density: "Thick Atmosphere (100%)" },
      bennu: { gravity: 0.002, drag: 1.0, label: "Asteroid Bennu", density: "Vacuum (0%)" }
    };
    
    // Lander state variables
    this.x = 0;
    this.y = 0;
    this.vx = 0;
    this.vy = 0;
    this.angle = 0; // In radians. 0 is pointing straight up.
    this.fuel = 100; // Fuel percentage (0 - 100)
    this.state = 'flying'; // 'flying', 'landed', 'crashed'
    this.score = 0;
    
    // Flight logs (history for RAG analysis)
    this.flightTime = 0;
    this.maxSpeedReached = 0;
    this.thrustPulses = 0;
    this.landingPadSelected = null;
    this.impactMessage = "";
    
    // Particles (for thruster exhaust and explosion)
    this.particles = [];
    
    // Controls mapping
    this.keys = {
      up: false,
      left: false,
      right: false
    };
    
    // Terrain and Pads
    this.terrainPoints = [];
    this.pads = [];
    
    // Audio Context (Procedural sound synthesis)
    this.audioCtx = null;
    this.thrustSoundNode = null;
    
    this.setupInput();
    this.initTerrain();
    this.reset();
  }

  // Initialize browser audio context on user action (chrome requires click interaction)
  initAudio() {
    if (this.audioCtx) return;
    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
      console.warn("Web Audio API not supported", e);
    }
  }

  setupInput() {
    window.addEventListener('keydown', (e) => {
      this.initAudio();
      if (['ArrowUp', 'KeyW'].includes(e.code)) this.keys.up = true;
      if (['ArrowLeft', 'KeyA'].includes(e.code)) this.keys.left = true;
      if (['ArrowRight', 'KeyD'].includes(e.code)) this.keys.right = true;
    });

    window.addEventListener('keyup', (e) => {
      if (['ArrowUp', 'KeyW'].includes(e.code)) this.keys.up = false;
      if (['ArrowLeft', 'KeyA'].includes(e.code)) this.keys.left = false;
      if (['ArrowRight', 'KeyD'].includes(e.code)) this.keys.right = false;
    });
  }

  initTerrain() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    // Clear lists
    this.terrainPoints = [];
    this.pads = [];
    
    // Let's create static terrain with flat sections (landing pads)
    // Pad A (Wide, Easy, Multiplier 1x): X: 80 - 180, Y: 360
    // Pad B (Medium, Normal, Multiplier 2x): X: 320 - 390, Y: 320
    // Pad C (Narrow, Hard, Multiplier 5x): X: 520 - 560, Y: 370
    
    this.pads = [
      { id: "alpha", name: "Pad Alpha (Wide)", startX: 60, endX: 160, y: 360, mult: 1, color: "#10b981" },
      { id: "bravo", name: "Pad Bravo (Medium)", startX: 280, endX: 360, y: 310, mult: 2, color: "#f59e0b" },
      { id: "charlie", name: "Pad Charlie (Narrow)", startX: 470, endX: 520, y: 350, mult: 5, color: "#ef4444" }
    ];
    
    // Generate terrain points
    let currentX = 0;
    
    // Start height
    this.terrainPoints.push({ x: 0, y: height - 60 });
    
    const step = 20;
    while (currentX < width) {
      currentX += step;
      let targetY = height - 50 - Math.random() * 60;
      
      // Check if currentX lies within any landing pad, if so, lock it to pad's height
      const pad = this.pads.find(p => currentX >= p.startX && currentX <= p.endX);
      if (pad) {
        // Just before pad starts, smooth it down
        targetY = pad.y;
      }
      
      // Keep boundary limits
      if (currentX > width) currentX = width;
      this.terrainPoints.push({ x: currentX, y: targetY });
    }
  }

  reset() {
    this.x = 280 + Math.random() * 80;
    this.y = 50;
    this.vx = (Math.random() - 0.5) * 1.0;
    this.vy = 0.5 + Math.random() * 0.5;
    this.angle = (Math.random() - 0.5) * 0.4; // Slightly tilted initial angle
    this.fuel = 100;
    this.state = 'flying';
    this.flightTime = 0;
    this.maxSpeedReached = 0;
    this.thrustPulses = 0;
    this.landingPadSelected = null;
    this.impactMessage = "";
    this.particles = [];
    this.stopThrustSound();
    
    if (this.onStateChange) this.onStateChange(this.state, this.getTelemetry());
  }

  setEnvironment(env) {
    if (this.envConfigs[env]) {
      this.env = env;
      const cfg = this.envConfigs[env];
      this.gravity = cfg.gravity;
      this.drag = cfg.drag;
      this.reset();
    }
  }

  getTelemetry() {
    // Return numerical telemetry
    return {
      altitude: Math.max(0, this.canvas.height - 40 - this.y),
      verticalSpeed: this.vy * 5, // scaled for display
      horizontalSpeed: this.vx * 5, // scaled for display
      tilt: (this.angle * 180 / Math.PI), // degrees
      fuel: this.fuel,
      flightTime: this.flightTime / 60 // seconds
    };
  }

  // Updates game state (called in the animation loop)
  update() {
    if (this.state !== 'flying') {
      this.updateParticles();
      return;
    }
    
    this.flightTime++;
    
    // A. Apply Gravity
    this.vy += this.gravity;
    
    // B. Apply Engine Thrust
    if (this.keys.up && this.fuel > 0) {
      this.vy -= this.thrustPower * Math.cos(this.angle);
      this.vx += this.thrustPower * Math.sin(this.angle);
      this.fuel = Math.max(0, this.fuel - 0.15);
      
      if (this.flightTime % 3 === 0) {
        this.thrustPulses++;
      }
      
      // Add flame particles
      this.addThrustParticles();
      
      // Synthesize audio rumble
      this.playThrustSound();
    } else {
      this.stopThrustSound();
    }
    
    // C. Apply Rotation RCS
    if (this.keys.left && this.fuel > 0) {
      this.angle -= this.rotationSpeed;
      this.fuel = Math.max(0, this.fuel - 0.03);
      this.addRotationParticles('right'); // RCS puff
    }
    if (this.keys.right && this.fuel > 0) {
      this.angle += this.rotationSpeed;
      this.fuel = Math.max(0, this.fuel - 0.03);
      this.addRotationParticles('left');
    }
    
    // D. Apply vacuum drag and update positions
    this.vx *= this.drag;
    this.vy *= this.drag;
    this.x += this.vx;
    this.y += this.vy;
    
    // Track stats
    const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy) * 5;
    if (speed > this.maxSpeedReached) {
      this.maxSpeedReached = speed;
    }
    
    // E. Screen boundary wrapping / limits
    if (this.x < 10) { this.x = 10; this.vx = -this.vx * 0.5; }
    if (this.x > this.canvas.width - 10) { this.x = this.canvas.width - 10; this.vx = -this.vx * 0.5; }
    if (this.y < 0) { this.y = 0; this.vy = 0; }
    
    // F. Check Collision
    this.checkCollisions();
    
    this.updateParticles();
  }

  checkCollisions() {
    const bottomY = this.y + 15; // Lander landing gear height
    const terrainHeight = this.getTerrainHeightAt(this.x);
    
    if (bottomY >= terrainHeight) {
      // Impact!
      this.y = terrainHeight - 15; // snap to terrain
      this.stopThrustSound();
      
      // Determine if touchdown was on a pad
      const pad = this.pads.find(p => this.x >= p.startX && this.x <= p.endX);
      
      const speedV = this.vy * 5; // scaled vertical speed
      const speedH = this.vx * 5; // scaled horizontal speed
      const tiltDeg = Math.abs(this.angle * 180 / Math.PI);
      
      const vLimit = 3.0;
      const hLimit = 1.5;
      const tLimit = 5.0;
      
      const vSafe = Math.abs(speedV) <= vLimit;
      const hSafe = Math.abs(speedH) <= hLimit;
      const tSafe = tiltDeg <= tLimit;
      
      if (pad) {
        this.landingPadSelected = pad.name;
        
        if (vSafe && hSafe && tSafe) {
          // SAFE Touchdown!
          this.state = 'landed';
          const points = 200 * pad.mult;
          this.score += points;
          this.impactMessage = `SAFE LANDING! Touchdown on ${pad.name} succeeded. +${points} points.`;
          this.playLandingChime();
        } else {
          // CRASH on landing pad (structural limits exceeded)
          this.state = 'crashed';
          let causes = [];
          if (!vSafe) causes.push(`vertical speed exceeded limits (${speedV.toFixed(2)} m/s vs ${vLimit} m/s)`);
          if (!hSafe) causes.push(`horizontal drift exceeded limits (${speedH.toFixed(2)} m/s vs ${hLimit} m/s)`);
          if (!tSafe) causes.push(`tilt angle exceeded limits (${tiltDeg.toFixed(1)}° vs ${tLimit}°)`);
          this.impactMessage = `HARD LANDING (CRASH) on ${pad.name}: ${causes.join(", ")}. Spacecraft hull breached.`;
          this.triggerExplosion();
        }
      } else {
        // Crash off-pad (crater collision)
        this.state = 'crashed';
        this.landingPadSelected = "None (Crater Terrain)";
        this.impactMessage = "CRITICAL COLLISION: Spacecraft crashed into non-designated landing zone (rough crater terrain). Structural collapse.";
        this.triggerExplosion();
      }
      
      // Inject logs into RAG
      this.logTelemetryToRAG(speedV, speedH, tiltDeg);
      
      if (this.onStateChange) this.onStateChange(this.state, this.getTelemetry());
    }
  }

  getTerrainHeightAt(x) {
    // Find segments matching x
    for (let i = 0; i < this.terrainPoints.length - 1; i++) {
      const p1 = this.terrainPoints[i];
      const p2 = this.terrainPoints[i + 1];
      if (x >= p1.x && x <= p2.x) {
        // Interpolate y
        const pct = (x - p1.x) / (p2.x - p1.x);
        return p1.y + pct * (p2.y - p1.y);
      }
    }
    return this.canvas.height - 40;
  }

  logTelemetryToRAG(speedV, speedH, tiltDeg) {
    const isSuccess = this.state === 'landed';
    const statusText = isSuccess ? "SUCCESSFUL DESCENT & TOUCHDOWN" : "MISSION FAILURE (COLLISION/CRASH)";
    const dateStr = new Date().toISOString().replace('T', ' ').substring(0, 19);
    
    let analysis = "";
    if (isSuccess) {
      analysis = "The pilot executed a controlled and stable retro-burn, effectively nullifying lateral drift while maintaining a descent velocity vector well within structural limits. Throttling and orientation were excellent.";
    } else {
      if (this.landingPadSelected.startsWith("None")) {
        analysis = "Navigation failed to target a flat landing pad. Operating on uneven crater walls caused immediate mechanical tipping and structural collapse. Fuel management was irrelevant due to bad steering vectoring.";
      } else {
        analysis = "The pilot targeted the landing pad correctly, but failed to decelerate or stabilize the attitude. High descending force or wide tilt vectors exceeded the touchdown shock absorber load thresholds, crushing the module.";
      }
    }
    
    const telemetryDoc = `
FLIGHT TERMINATION REPORT: ${statusText}
Timestamp: ${dateStr} UTC
Target Planet Profile: ${this.envConfigs[this.env].label} (Gravity scale: ${this.envConfigs[this.env].gravity * 100} units, Atmosphere density: ${this.envConfigs[this.env].density})
Lander Position: X: ${this.x.toFixed(1)} px, Ground Landing Site: ${this.landingPadSelected}

Final Flight Vector telemetry parameters:
- Altitude at touchdown: 0.0 meters
- Landing Vertical Velocity (Vz): ${speedV.toFixed(2)} m/s (Limit: 3.0 m/s) -> ${Math.abs(speedV) <= 3.0 ? 'SAFE' : 'CRITICAL OVERLOAD'}
- Landing Horizontal Velocity (Vx): ${speedH.toFixed(2)} m/s (Limit: 1.5 m/s) -> ${Math.abs(speedH) <= 1.5 ? 'SAFE' : 'CRITICAL OVERLOAD'}
- Attitude Pitch Tilt Angle: ${tiltDeg.toFixed(1)} degrees (Limit: 5.0°) -> ${tiltDeg <= 5.0 ? 'SAFE' : 'CRITICAL OVERLOAD'}
- Propellant fuel remaining: ${this.fuel.toFixed(1)}%
- Total flight time: ${(this.flightTime / 60).toFixed(2)} seconds
- Peak speed reached: ${this.maxSpeedReached.toFixed(2)} m/s

Post-Mission Command Analysis:
${analysis}
    `.trim();
    
    if (this.onTelemetryLogged) {
      this.onTelemetryLogged(telemetryDoc);
    }
  }

  // Visuals and Rendering
  draw() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    // Clear canvas
    this.ctx.clearRect(0, 0, width, height);
    
    // Draw Space Stars Background (Procedural static grid)
    this.ctx.fillStyle = '#0f172a';
    this.ctx.fillRect(0, 0, width, height);
    
    this.ctx.fillStyle = '#ffffff';
    for (let i = 0; i < 30; i++) {
      const starX = (Math.sin(i * 123.45) * 0.5 + 0.5) * width;
      const starY = (Math.cos(i * 543.21) * 0.5 + 0.5) * height;
      const radius = (Math.sin(i * 88.8) * 0.5 + 0.5) * 1.5;
      this.ctx.beginPath();
      this.ctx.arc(starX, starY, radius, 0, Math.PI * 2);
      this.ctx.fill();
    }
    
    // Draw Landing Pads indicators (Neon glow)
    this.pads.forEach(pad => {
      this.ctx.fillStyle = pad.color + '22'; // 13% opacity overlay
      this.ctx.fillRect(pad.startX, pad.y, pad.endX - pad.startX, height - pad.y);
      
      this.ctx.strokeStyle = pad.color;
      this.ctx.lineWidth = 4;
      this.ctx.shadowColor = pad.color;
      this.ctx.shadowBlur = 10;
      this.ctx.beginPath();
      this.ctx.moveTo(pad.startX, pad.y);
      this.ctx.lineTo(pad.endX, pad.y);
      this.ctx.stroke();
      
      // Reset shadow
      this.ctx.shadowBlur = 0;
      
      // Label pads
      this.ctx.fillStyle = '#94a3b8';
      this.ctx.font = '8px monospace';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(`${pad.mult}x`, (pad.startX + pad.endX) / 2, pad.y - 12);
    });
    
    // Draw Crater Terrain (Sci-fi grid overlay)
    this.ctx.strokeStyle = '#38bdf8'; // neon cyan
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(this.terrainPoints[0].x, this.terrainPoints[0].y);
    for (let i = 1; i < this.terrainPoints.length; i++) {
      this.ctx.lineTo(this.terrainPoints[i].x, this.terrainPoints[i].y);
    }
    this.ctx.stroke();
    
    // Draw radar altimeter scan beam
    if (this.state === 'flying') {
      const groundY = this.getTerrainHeightAt(this.x);
      this.ctx.save();
      this.ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
      this.ctx.lineWidth = 1;
      this.ctx.setLineDash([4, 4]);
      this.ctx.beginPath();
      this.ctx.moveTo(this.x, this.y + 15);
      this.ctx.lineTo(this.x, groundY);
      this.ctx.stroke();
      
      // Pulse radar target circle on ground contact point
      const pulseRad = 3 + Math.sin(this.flightTime * 0.15) * 2;
      this.ctx.strokeStyle = 'rgba(56, 189, 248, 0.75)';
      this.ctx.lineWidth = 1.5;
      this.ctx.setLineDash([]);
      this.ctx.beginPath();
      this.ctx.arc(this.x, groundY, pulseRad, 0, Math.PI * 2);
      this.ctx.stroke();
      this.ctx.restore();
    }
    
    // Fill under terrain
    this.ctx.fillStyle = '#0f172aee'; // transparent dark gray
    this.ctx.beginPath();
    this.ctx.moveTo(0, height);
    this.ctx.lineTo(this.terrainPoints[0].x, this.terrainPoints[0].y);
    for (let i = 1; i < this.terrainPoints.length; i++) {
      this.ctx.lineTo(this.terrainPoints[i].x, this.terrainPoints[i].y);
    }
    this.ctx.lineTo(width, height);
    this.ctx.closePath();
    this.ctx.fill();
    
    // Draw Particles
    this.particles.forEach(p => {
      this.ctx.fillStyle = p.color;
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      this.ctx.fill();
    });
    
    // Draw Lander Spacecraft
    if (this.state !== 'crashed') {
      this.drawLander();
    }
  }

  drawLander() {
    this.ctx.save();
    this.ctx.translate(this.x, this.y);
    this.ctx.rotate(this.angle);
    
    // Lander Capsule Body (Hexagonal design)
    this.ctx.fillStyle = '#cbd5e1'; // metallic grey
    this.ctx.strokeStyle = '#475569';
    this.ctx.lineWidth = 1.5;
    this.ctx.beginPath();
    this.ctx.moveTo(-8, -4);
    this.ctx.lineTo(-4, -12);
    this.ctx.lineTo(4, -12);
    this.ctx.lineTo(8, -4);
    this.ctx.lineTo(6, 6);
    this.ctx.lineTo(-6, 6);
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.stroke();
    
    // Ascent Module Cabin Window (Glowing Cyan)
    this.ctx.fillStyle = '#06b6d4';
    this.ctx.beginPath();
    this.ctx.moveTo(-3, -10);
    this.ctx.lineTo(3, -10);
    this.ctx.lineTo(4, -6);
    this.ctx.lineTo(-4, -6);
    this.ctx.closePath();
    this.ctx.fill();
    
    // Gold Foil Descent Stage (bottom box)
    this.ctx.fillStyle = '#eab308'; // golden foil
    this.ctx.strokeStyle = '#a16207';
    this.ctx.fillRect(-10, 6, 20, 8);
    this.ctx.strokeRect(-10, 6, 20, 8);
    
    // Landing Gear Legs (Left & Right)
    this.ctx.strokeStyle = '#64748b';
    this.ctx.lineWidth = 2;
    // Left Leg
    this.ctx.beginPath();
    this.ctx.moveTo(-8, 12);
    this.ctx.lineTo(-14, 17);
    this.ctx.moveTo(-14, 17);
    this.ctx.lineTo(-11, 19); // Foot pad left
    this.ctx.lineTo(-17, 19);
    this.ctx.stroke();
    
    // Right Leg
    this.ctx.beginPath();
    this.ctx.moveTo(8, 12);
    this.ctx.lineTo(14, 17);
    this.ctx.moveTo(14, 17);
    this.ctx.lineTo(11, 19); // Foot pad right
    this.ctx.lineTo(17, 19);
    this.ctx.stroke();
    
    // Main thruster nozzle (Bottom center)
    this.ctx.fillStyle = '#334155';
    this.ctx.fillRect(-3, 14, 6, 3);
    
    this.ctx.restore();
  }

  // Particle Engines
  addThrustParticles() {
    // Generate exhaust particles
    // Calculate nozzle position based on angle
    const ox = this.x + 16 * Math.sin(this.angle);
    const oy = this.y + 16 * Math.cos(this.angle);
    
    for (let i = 0; i < 2; i++) {
      const scatterAngle = this.angle + Math.PI + (Math.random() - 0.5) * 0.4;
      const speed = 2 + Math.random() * 3;
      this.particles.push({
        x: ox,
        y: oy,
        vx: speed * Math.sin(scatterAngle) + this.vx,
        vy: speed * Math.cos(scatterAngle) + this.vy,
        size: 3 + Math.random() * 4,
        color: `hsl(${20 + Math.random() * 40}, 100%, ${50 + Math.random() * 20}%)`, // Fire gradient (orange/yellow)
        life: 15 + Math.random() * 15
      });
    }
  }

  addRotationParticles(side) {
    const rx = this.x + (side === 'left' ? -10 : 10) * Math.cos(this.angle);
    const ry = this.y + (side === 'left' ? -10 : 10) * Math.sin(this.angle);
    
    const scatterAngle = this.angle + (side === 'left' ? -Math.PI/2 : Math.PI/2) + (Math.random() - 0.5) * 0.2;
    const speed = 1.5;
    
    for (let i = 0; i < 2; i++) {
      this.particles.push({
        x: rx,
        y: ry,
        vx: speed * Math.sin(scatterAngle) + this.vx,
        vy: speed * Math.cos(scatterAngle) + this.vy,
        size: 1.5,
        color: '#f8fafc', // white nitrogen puff
        life: 8 + Math.random() * 6
      });
    }
  }

  triggerExplosion() {
    this.playExplosionSound();
    for (let i = 0; i < 80; i++) {
      const expAngle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 6 + 1;
      this.particles.push({
        x: this.x,
        y: this.y,
        vx: speed * Math.sin(expAngle),
        vy: speed * Math.cos(expAngle),
        size: 2 + Math.random() * 6,
        color: i % 4 === 0 ? '#ef4444' : (i % 2 === 0 ? '#f97316' : '#facc15'), // red/orange/yellow
        life: 40 + Math.random() * 50
      });
    }
  }

  updateParticles() {
    this.particles = this.particles.filter(p => {
      p.x += p.vx;
      p.y += p.vy;
      p.life--;
      p.size = Math.max(0.1, p.size - 0.05);
      return p.life > 0;
    });
  }

  // Audio Synthesizers (Web Audio API)
  playThrustSound() {
    if (!this.audioCtx) return;
    if (this.thrustSoundNode) return; // already playing
    
    try {
      // 1. Create a Noise Buffer for gas sound
      const bufferSize = this.audioCtx.sampleRate * 2;
      const noiseBuffer = this.audioCtx.createBuffer(1, bufferSize, this.audioCtx.sampleRate);
      const output = noiseBuffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
      }
      
      const whiteNoise = this.audioCtx.createBufferSource();
      whiteNoise.buffer = noiseBuffer;
      whiteNoise.loop = true;
      
      // 2. Low Frequency Rumble Oscillator
      const osc = this.audioCtx.createOscillator();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(55, this.audioCtx.currentTime); // Low A rumbling
      
      // 3. Filter to shape the engine sound (descent engine is deep)
      const lpFilter = this.audioCtx.createBiquadFilter();
      lpFilter.type = 'lowpass';
      lpFilter.frequency.setValueAtTime(140, this.audioCtx.currentTime);
      
      // 4. Gain Nodes (mixing rumble and noise)
      const noiseGain = this.audioCtx.createGain();
      noiseGain.gain.setValueAtTime(0.12, this.audioCtx.currentTime);
      
      const oscGain = this.audioCtx.createGain();
      oscGain.gain.setValueAtTime(0.22, this.audioCtx.currentTime);
      
      const masterGain = this.audioCtx.createGain();
      masterGain.gain.setValueAtTime(0.0, this.audioCtx.currentTime);
      masterGain.gain.linearRampToValueAtTime(0.5, this.audioCtx.currentTime + 0.15); // Fade in
      
      // Connect nodes
      whiteNoise.connect(noiseGain);
      osc.connect(oscGain);
      
      noiseGain.connect(lpFilter);
      oscGain.connect(lpFilter);
      
      lpFilter.connect(masterGain);
      masterGain.connect(this.audioCtx.destination);
      
      // Start sounds
      whiteNoise.start();
      osc.start();
      
      this.thrustSoundNode = {
        whiteNoise,
        osc,
        masterGain
      };
    } catch (e) {
      console.warn("Failed to play thrust audio", e);
    }
  }

  stopThrustSound() {
    if (!this.thrustSoundNode) return;
    try {
      const node = this.thrustSoundNode;
      const currTime = this.audioCtx.currentTime;
      node.masterGain.gain.setValueAtTime(node.masterGain.gain.value, currTime);
      node.masterGain.gain.exponentialRampToValueAtTime(0.001, currTime + 0.1); // fade out
      
      setTimeout(() => {
        try {
          node.whiteNoise.stop();
          node.osc.stop();
        } catch (err) {}
      }, 150);
      
      this.thrustSoundNode = null;
    } catch (e) {
      this.thrustSoundNode = null;
    }
  }

  playLandingChime() {
    if (!this.audioCtx) return;
    try {
      const now = this.audioCtx.currentTime;
      // Arpeggio sound indicating success
      const notes = [261.63, 329.63, 392.00, 523.25]; // C4, E4, G4, C5
      notes.forEach((freq, index) => {
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, now + index * 0.12);
        
        gain.gain.setValueAtTime(0, now + index * 0.12);
        gain.gain.linearRampToValueAtTime(0.2, now + index * 0.12 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.001, now + index * 0.12 + 0.4);
        
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        
        osc.start(now + index * 0.12);
        osc.stop(now + index * 0.12 + 0.5);
      });
    } catch (e) {}
  }

  playExplosionSound() {
    if (!this.audioCtx) return;
    try {
      const now = this.audioCtx.currentTime;
      
      // White noise buffer
      const bufferSize = this.audioCtx.sampleRate * 1.5;
      const noiseBuffer = this.audioCtx.createBuffer(1, bufferSize, this.audioCtx.sampleRate);
      const output = noiseBuffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
      }
      
      const source = this.audioCtx.createBufferSource();
      source.buffer = noiseBuffer;
      
      // Filter for explosion (moving from high-mid to deep low pass)
      const lpFilter = this.audioCtx.createBiquadFilter();
      lpFilter.type = 'lowpass';
      lpFilter.frequency.setValueAtTime(800, now);
      lpFilter.frequency.exponentialRampToValueAtTime(40, now + 1.2);
      
      const gain = this.audioCtx.createGain();
      gain.gain.setValueAtTime(0.8, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 1.4);
      
      source.connect(lpFilter);
      lpFilter.connect(gain);
      gain.connect(this.audioCtx.destination);
      
      source.start(now);
      source.stop(now + 1.5);
    } catch (e) {}
  }
}

window.LunarLanderGame = LunarLanderGame;

