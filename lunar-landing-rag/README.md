# Apollo L.E.M. Guidance Command Center 2.0 (Advanced Edition)

An interactive, high-fidelity **planetary flight simulator** integrated side-by-side with a visual **Retrieval-Augmented Generation (RAG) Mission Guidance Assistant**. 

Built entirely in client-side HTML5, CSS3, and JavaScript, this application runs directly in any browser with **zero installation or backend server requirements**.

---

## Advanced Systems (v2.0)

1.  **Planetary Physics Sandbox**:
    *   Dropdown selector to switch flight environments on the fly.
    *   Simulate descent on the **Moon** (1/6th gravity, vacuum), **Mars** (1/3rd gravity, thin atmosphere), **Earth** (1.0g gravity, thick atmosphere/terminal velocity drag), and **Asteroid Bennu** (microgravity, vacuum).
    *   The RAG co-pilot corpus automatically updates with planetary data, altering gravity profiles and flight limits in telemetry logs.
2.  **Scrolling Telemetry Oscilloscope**:
    *   A secondary scrolling canvas graph plotting real-time curves for **Altitude** (neon green), **Fuel %** (neon red), and **Relative Speed** (neon amber).
    *   Tracks descent rate profiles and visually captures limits exceeded on touchdown crashes.
3.  **Neural Vector Space Visualizer**:
    *   A 2D projection map canvas showing document chunks as color-coded nodes clustered around category hubs (History, Specs, Physics, Manual, Telemetry).
    *   When you query the RAG assistant, a **Query Node** pulsates and projects vector connection lines to the retrieved document chunks, with line thickness and opacity reflecting the Cosine Similarity score.
4.  **Database Chunking Tuner**:
    *   Interactive range sliders in the RAG Explorer panel to adjust **Chunk Size** (1 to 5 sentences) and **Chunk Overlap** (0 to 3 sentences) in real-time.
    *   Dynamically rebuilds the TF-IDF vector database and updates the Mission Archives accordion on the fly.
5.  **Radar Altimeter Scans**:
    *   A scanning altimeter radar beam (dotted pulsing cyan line) projecting from the lander's gear down to the landing contact points.

---

## File Structure

```
lunar-landing-rag/
├── index.html           # Main command dashboard UI and layout
├── styles.css           # Glassmorphic sci-fi CSS styling and slider tracks
├── README.md            # Advanced project architecture and physics equations
├── data/
│   └── documents.js     # Mission static knowledge base documents
└── js/
    ├── game.js          # Canvas game loops, physical presets, radar altimeters, and synth audio
    ├── rag.js           # Sentence chunker, TF-IDF calculation, and 2D node coordinates hashing
    └── app.js           # Dashboard controller, graph scrolling plots, and neural vector space drawings
```

---

## Quick Start (How to Run)

1.  **Open index.html**:
    Double-click the `index.html` file or drag it into any web browser.
2.  **Pilot the Module**:
    *   Select your target preset (e.g. *Mars* or *Earth* to test drag coefficients).
    *   Press **W** / **UP ARROW** to engage thrusters, and **A/D** / **LEFT/RIGHT ARROWS** to rotate.
    *   Watch the HUD gauges and the scrolling curves on the **Telemetry Oscilloscope** graph.
3.  **Inspect the Neural RAG Pipeline**:
    *   Submit a chat question on the **Mission Assistant** tab.
    *   Click the **RAG Pipeline Explorer** tab. 
    *   Observe how the **Query Node** moves to the center-of-gravity of matching document nodes, mapping the query vector's location in semantic space.
    *   Slide the **Chunk Size** or **Overlap** controls to inspect how indexing boundaries resize.

---

## Mathematics & Physics

### RAG Cosine Similarity
Vector relevance is computed via cosine dot products in normalized term space:
$$\text{CosineSim}(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|} = \frac{\sum_i q_i d_i}{\sqrt{\sum_i q_i^2} \sqrt{\sum_i d_i^2}}$$

### Physics Sandbox Formulas
Planetary velocity updating at time $t$ with gravity acceleration $g$, air drag multiplier $d$, thrust power $F$, and pitch angle $\theta$:
$$v_{x, t+1} = (v_{x, t} + F \cdot \sin(\theta)) \cdot d$$
$$v_{y, t+1} = (v_{y, t} + g - F \cdot \cos(\theta)) \cdot d$$
$$x_{t+1} = x_t + v_{x, t+1}$$
$$y_{t+1} = y_t + v_{y, t+1}$$
