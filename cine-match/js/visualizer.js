// Polyfill roundRect for older browser/webview compatibility
if (typeof CanvasRenderingContext2D !== "undefined" && !CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    if (typeof r === "undefined") r = 0;
    if (typeof r === "number") r = [r, r, r, r];
    else if (r.length === 1) r = [r[0], r[0], r[0], r[0]];
    else if (r.length === 2) r = [r[0], r[1], r[0], r[1]];
    else if (r.length === 3) r = [r[0], r[1], r[2], r[1]];
    
    const rLT = r[0];
    const rRT = r[1];
    const rRB = r[2];
    const rLB = r[3];
    
    this.beginPath();
    this.moveTo(x + rLT, y);
    this.lineTo(x + w - rRT, y);
    this.arcTo(x + w, y, x + w, y + rRT, rRT);
    this.lineTo(x + w, y + h - rRB);
    this.arcTo(x + w, y + h, x + w - rRB, y + h, rRB);
    this.lineTo(x + rLB, y + h);
    this.arcTo(x, y + h, x, y + h - rLB, rLB);
    this.lineTo(x, y + rLT);
    this.arcTo(x, y, x + rLT, y, rLT);
    this.closePath();
    return this;
  };
}

class VectorSpaceVisualizer {
  constructor(canvasId, recommender) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.recommender = recommender;
    this.nodes = [];
    this.queryNode = null;
    this.hoveredNode = null;
    this.animationFrameId = null;

    // Simulation parameters
    this.width = this.canvas.width;
    this.height = this.canvas.height;
    
    // Connect events
    this.initEvents();
    // Build initial node positions
    this.generateNodePositions();
    // Start animation loop
    this.startLoop();
  }

  initEvents() {
    window.addEventListener("resize", () => this.resizeCanvas());
    this.resizeCanvas();

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      // Account for potential canvas scaling
      const mouseX = ((e.clientX - rect.left) / rect.width) * this.width;
      const mouseY = ((e.clientY - rect.top) / rect.height) * this.height;

      this.hoveredNode = null;
      for (const node of this.nodes) {
        const dx = node.x - mouseX;
        const dy = node.y - mouseY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= node.radius + 8) {
          this.hoveredNode = node;
          break;
        }
      }

      this.canvas.style.cursor = this.hoveredNode ? "pointer" : "default";
    });

    this.canvas.addEventListener("click", () => {
      if (this.hoveredNode && window.showMovieDetailModal) {
        window.showMovieDetailModal(this.hoveredNode.movie.id);
      }
    });
  }

  resizeCanvas() {
    const parent = this.canvas.parentElement;
    if (parent) {
      this.canvas.width = parent.clientWidth * window.devicePixelRatio;
      this.canvas.height = parent.clientHeight * window.devicePixelRatio;
      this.canvas.style.width = parent.clientWidth + "px";
      this.canvas.style.height = parent.clientHeight + "px";
      this.width = this.canvas.width;
      this.height = this.canvas.height;
      this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      // Recalculate dimensions in CSS space
      this.width = parent.clientWidth;
      this.height = parent.clientHeight;
      this.generateNodePositions();
    }
  }

  generateNodePositions() {
    const movies = this.recommender.movies;
    const padding = 60;
    
    // Define cluster centers based on genre groups
    const clusters = {
      "Sci-Fi": { x: this.width * 0.25, y: this.height * 0.3 },
      "Action": { x: this.width * 0.25, y: this.height * 0.7 },
      "Thriller": { x: this.width * 0.25, y: this.height * 0.7 },
      "Drama": { x: this.width * 0.75, y: this.height * 0.3 },
      "Romance": { x: this.width * 0.75, y: this.height * 0.3 },
      "Comedy": { x: this.width * 0.75, y: this.height * 0.7 },
      "Animation": { x: this.width * 0.5, y: this.height * 0.25 },
      "Default": { x: this.width * 0.5, y: this.height * 0.5 }
    };

    // Keep track of counts per cluster to spiral them outwards
    const clusterCounts = {};
    Object.keys(clusters).forEach(k => clusterCounts[k] = 0);

    this.nodes = movies.map(movie => {
      // Find matching cluster
      let clusterKey = "Default";
      for (const genre of movie.genres) {
        if (clusters[genre]) {
          clusterKey = genre;
          break;
        }
      }

      const center = clusters[clusterKey];
      const index = clusterCounts[clusterKey]++;
      
      // Calculate spiral distribution offset so they cluster nicely without overlapping
      const distanceMultiplier = 35;
      const angle = index * 2.4; // Golden angle approximation for distribution
      const radius = 10 + Math.sqrt(index) * distanceMultiplier;
      
      let x = center.x + Math.cos(angle) * radius;
      let y = center.y + Math.sin(angle) * radius;

      // Keep inside bounds
      x = Math.max(padding, Math.min(this.width - padding, x));
      y = Math.max(padding, Math.min(this.height - padding, y));

      return {
        movie,
        x,
        y,
        baseX: x,
        baseY: y,
        vx: 0,
        vy: 0,
        radius: 6,
        color: movie.posterColor || "#ffffff",
        pulsePhase: Math.random() * Math.PI * 2,
        connections: [] // [{ targetNode, weight }]
      };
    });
  }

  // Trigger search visualization
  visualizeQuery(queryText, retrievedResults) {
    if (!queryText || retrievedResults.length === 0) {
      this.queryNode = null;
      this.nodes.forEach(n => n.connections = []);
      return;
    }

    // Place query node in the mathematical center of gravity of the retrieved movies
    let sumX = 0, sumY = 0;
    retrievedResults.forEach(r => {
      const node = this.nodes.find(n => n.movie.id === r.movie.id);
      if (node) {
        sumX += node.baseX;
        sumY += node.baseY;
      }
    });

    this.queryNode = {
      text: queryText,
      x: sumX / retrievedResults.length,
      y: sumY / retrievedResults.length,
      targetX: sumX / retrievedResults.length,
      targetY: sumY / retrievedResults.length,
      radius: 12,
      pulse: 0
    };

    // If center of gravity is too close to a node, push it to the absolute center to maintain visibility
    if (isNaN(this.queryNode.x)) {
      this.queryNode.x = this.width / 2;
      this.queryNode.y = this.height / 2;
    }

    // Set connection lines from the query node to target nodes
    this.nodes.forEach(n => n.connections = []);
    retrievedResults.forEach(r => {
      const node = this.nodes.find(n => n.movie.id === r.movie.id);
      if (node) {
        node.connections.push({
          weight: r.score,
          details: r.details
        });
      }
    });
  }

  startLoop() {
    const tick = () => {
      this.update();
      this.draw();
      this.animationFrameId = requestAnimationFrame(tick);
    };
    this.animationFrameId = requestAnimationFrame(tick);
  }

  stopLoop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }

  update() {
    const time = Date.now() * 0.002;
    
    // Float movie nodes slightly using sine waves
    this.nodes.forEach(node => {
      node.pulsePhase += 0.03;
      const floatX = Math.sin(node.pulsePhase) * 1.5;
      const floatY = Math.cos(node.pulsePhase * 0.8) * 1.5;
      
      // Pull toward base position + float offset
      const targetX = node.baseX + floatX;
      const targetY = node.baseY + floatY;
      
      node.x += (targetX - node.x) * 0.1;
      node.y += (targetY - node.y) * 0.1;
    });

    // Update query node pulse and positioning
    if (this.queryNode) {
      this.queryNode.pulse = (this.queryNode.pulse + 0.05) % (Math.PI * 2);
      this.queryNode.x += (this.queryNode.targetX - this.queryNode.x) * 0.05;
      this.queryNode.y += (this.queryNode.targetY - this.queryNode.y) * 0.05;
    }
  }

  draw() {
    this.ctx.clearRect(0, 0, this.width, this.height);

    // Draw background grid lines (cyberpunk dashboard aesthetics)
    this.ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
    this.ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < this.width; x += gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.height);
      this.ctx.stroke();
    }
    for (let y = 0; y < this.height; y += gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(this.width, y);
      this.ctx.stroke();
    }

    // Draw cluster label guides
    this.ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
    this.ctx.font = "bold 9px 'JetBrains Mono', monospace";
    this.ctx.textAlign = "center";
    this.ctx.fillText("SCI-FI EXPLORATION", this.width * 0.25, 20);
    this.ctx.fillText("ACTION / CRIME", this.width * 0.25, this.height - 15);
    this.ctx.fillText("DRAMA & REFLECTION", this.width * 0.75, 20);
    this.ctx.fillText("COMEDY / ROMANCE", this.width * 0.75, this.height - 15);
    this.ctx.fillText("ANIMATION", this.width * 0.5, 20);

    // Draw connection lines first (so nodes sit on top of them)
    if (this.queryNode) {
      this.nodes.forEach(node => {
        if (node.connections.length > 0) {
          node.connections.forEach(conn => {
            const similarity = conn.weight; // score
            
            // Draw connection line
            const grad = this.ctx.createLinearGradient(this.queryNode.x, this.queryNode.y, node.x, node.y);
            grad.addColorStop(0, "rgba(139, 92, 246, 0.6)"); // Pulsing purple
            grad.addColorStop(1, node.color + "99"); // Movie node color with opacity

            this.ctx.beginPath();
            this.ctx.strokeStyle = grad;
            this.ctx.lineWidth = Math.max(1, similarity * 5); // thicker line for higher similarity
            this.ctx.moveTo(this.queryNode.x, this.queryNode.y);
            this.ctx.lineTo(node.x, node.y);
            this.ctx.stroke();

            // Draw glowing particle floating along connection line
            const elapsed = (Date.now() / 1500) % 1.0;
            const px = this.queryNode.x + (node.x - this.queryNode.x) * elapsed;
            const py = this.queryNode.y + (node.y - this.queryNode.y) * elapsed;
            
            this.ctx.beginPath();
            this.ctx.fillStyle = "#ffffff";
            this.ctx.arc(px, py, 3, 0, Math.PI * 2);
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = "#a78bfa";
            this.ctx.fill();
            this.ctx.shadowBlur = 0; // reset
          });
        }
      });
    }

    // Draw movie nodes
    this.nodes.forEach(node => {
      const isHovered = (this.hoveredNode === node);
      const isTarget = node.connections.length > 0;
      
      // Calculate node drawing size
      let drawRadius = node.radius;
      if (isHovered) drawRadius += 4;
      else if (isTarget) drawRadius += 2;

      // Outer glow for query targets or hovered nodes
      if (isHovered || isTarget) {
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, drawRadius + 6, 0, Math.PI * 2);
        this.ctx.fillStyle = node.color + "22"; // 10% opacity
        this.ctx.fill();
      }

      // Draw the core circle node
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, drawRadius, 0, Math.PI * 2);
      this.ctx.fillStyle = node.color;
      this.ctx.fill();
      
      // Node outline
      this.ctx.strokeStyle = isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.4)";
      this.ctx.lineWidth = isHovered ? 2 : 1;
      this.ctx.stroke();

      // Render movie titles
      const showLabel = isHovered || isTarget;
      if (showLabel) {
        this.ctx.fillStyle = isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.8)";
        this.ctx.font = isHovered ? "bold 11px 'Outfit', sans-serif" : "9px 'Outfit', sans-serif";
        this.ctx.textAlign = "center";
        
        // Draw backing rect for readability
        const labelText = node.movie.title;
        const textWidth = this.ctx.measureText(labelText).width;
        this.ctx.fillStyle = "rgba(10, 10, 10, 0.85)";
        this.ctx.fillRect(node.x - textWidth/2 - 4, node.y - drawRadius - 18, textWidth + 8, 14);
        
        this.ctx.fillStyle = isHovered ? "#e879f9" : "rgba(255, 255, 255, 0.95)";
        this.ctx.fillText(labelText, node.x, node.y - drawRadius - 8);
      }
    });

    // Draw query node
    if (this.queryNode) {
      // Glow ring
      const pulseRadius = this.queryNode.radius + Math.sin(this.queryNode.pulse) * 4;
      this.ctx.beginPath();
      this.ctx.arc(this.queryNode.x, this.queryNode.y, pulseRadius + 6, 0, Math.PI * 2);
      this.ctx.strokeStyle = "rgba(139, 92, 246, 0.3)";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Core Query Node
      this.ctx.beginPath();
      this.ctx.arc(this.queryNode.x, this.queryNode.y, this.queryNode.radius, 0, Math.PI * 2);
      
      const grad = this.ctx.createRadialGradient(
        this.queryNode.x - 2, this.queryNode.y - 2, 1, 
        this.queryNode.x, this.queryNode.y, this.queryNode.radius
      );
      grad.addColorStop(0, "#c084fc"); // lighter purple
      grad.addColorStop(1, "#7c3aed"); // violet
      this.ctx.fillStyle = grad;
      this.ctx.fill();
      
      this.ctx.strokeStyle = "#ffffff";
      this.ctx.lineWidth = 1.5;
      this.ctx.stroke();

      // Draw "AI Query" Label
      this.ctx.fillStyle = "#ffffff";
      this.ctx.font = "bold 9px 'JetBrains Mono', monospace";
      this.ctx.textAlign = "center";
      
      const queryLabel = `Query: "${this.queryNode.text.substring(0, 15)}${this.queryNode.text.length > 15 ? '...' : ''}"`;
      const labelW = this.ctx.measureText(queryLabel).width;
      
      this.ctx.fillStyle = "rgba(124, 58, 237, 0.9)"; // violet background
      this.ctx.fillRect(this.queryNode.x - labelW/2 - 6, this.queryNode.y + this.queryNode.radius + 4, labelW + 12, 15);
      
      this.ctx.fillStyle = "#ffffff";
      this.ctx.fillText(queryLabel, this.queryNode.x, this.queryNode.y + this.queryNode.radius + 14);
    }

    // Hover Tooltip overlay on top of everything
    if (this.hoveredNode) {
      const node = this.hoveredNode;
      const movie = node.movie;
      
      // Calculate coordinates so it stays inside bounds
      let tooltipX = node.x + 15;
      let tooltipY = node.y - 15;
      const tooltipW = 200;
      const tooltipH = 95;

      if (tooltipX + tooltipW > this.width) tooltipX = node.x - tooltipW - 15;
      if (tooltipY + tooltipH > this.height) tooltipY = this.height - tooltipH - 10;
      if (tooltipY < 10) tooltipY = 10;

      // Draw background card with blur mockup
      this.ctx.fillStyle = "rgba(18, 18, 18, 0.95)";
      this.ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.roundRect(tooltipX, tooltipY, tooltipW, tooltipH, 6);
      this.ctx.fill();
      this.ctx.stroke();

      // Title & Year
      this.ctx.textAlign = "left";
      this.ctx.fillStyle = "#ffffff";
      this.ctx.font = "bold 12px 'Outfit', sans-serif";
      this.ctx.fillText(movie.title, tooltipX + 10, tooltipY + 20);
      
      this.ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
      this.ctx.font = "9px 'Outfit', sans-serif";
      this.ctx.fillText(`${movie.year} | Rating: ★${movie.rating} | ${movie.duration}`, tooltipX + 10, tooltipY + 34);

      // Genres
      this.ctx.fillStyle = "#ec4899"; // pink
      this.ctx.font = "bold 8px 'JetBrains Mono', monospace";
      this.ctx.fillText(movie.genres.join("  /  ").toUpperCase(), tooltipX + 10, tooltipY + 47);

      // Short Description
      this.ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      this.ctx.font = "9px 'Outfit', sans-serif";
      const desc = movie.description.length > 90 ? movie.description.substring(0, 87) + "..." : movie.description;
      this.wrapText(desc, tooltipX + 10, tooltipY + 60, tooltipW - 20, 11);
    }
  }

  // Helper to wrap text inside canvas tooltips
  wrapText(text, x, y, maxWidth, lineHeight) {
    const words = text.split(" ");
    let line = "";
    let lineCount = 0;
    
    for (let n = 0; n < words.length; n++) {
      let testLine = line + words[n] + " ";
      let metrics = this.ctx.measureText(testLine);
      let testWidth = metrics.width;
      
      if (testWidth > maxWidth && n > 0) {
        this.ctx.fillText(line, x, y);
        line = words[n] + " ";
        y += lineHeight;
        lineCount++;
        if (lineCount >= 3) return; // Cap lines
      } else {
        line = testLine;
      }
    }
    this.ctx.fillText(line, x, y);
  }
}

window.VectorSpaceVisualizer = VectorSpaceVisualizer;

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { VectorSpaceVisualizer };
}
