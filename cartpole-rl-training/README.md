# Cart-Pole Reinforcement Learning Dashboard

This project is a modular, educational, and high-performance implementation of a **Deep Q-Network (DQN)** agent to solve the classic **Cart-Pole** (`CartPole-v1`) control problem using Farama Gymnasium and PyTorch. 

It features an interactive, real-time **Web UI Dashboard** to configure hyperparameters, visualize physics rendering at 60 FPS, and watch live training curves update dynamically.

---

## 📂 Project Structure

```text
cartpole-rl-training/
│
├── .gitignore              # Ignores environment, python cache, saved models, and video outputs
├── requirements.txt        # Python package dependencies
│
├── dqn_agent.py            # Neural Network, Replay Buffer, and DQN agent logic
├── train.py                # CLI script to train the DQN agent and save weights
├── evaluate.py             # CLI script to run evaluation (live render/video recording)
│
├── server.py               # FastAPI backend web server for WebSockets and API requests
├── static/                 # Frontend Web App assets
│   ├── index.html          # Dashboard page (HTML structure & Chart.js widgets)
│   ├── style.css           # Custom Glassmorphic design stylesheet
│   └── app.js              # Websocket handling, physical canvas rendering, & chart plotting
│
└── cartpole_dqn.ipynb      # Interactive Jupyter Notebook explaining concepts and training inline
```

---

## 🛠️ Installation & Setup

We recommend setting up a virtual environment to manage dependencies locally.

1. **Clone/Navigate to this folder**:
   ```bash
   cd C:/Users/soumi/.gemini/antigravity/scratch/cartpole-rl-training
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   * **PowerShell (Windows)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Command Prompt (Windows)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **Bash (Linux/macOS)**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project uses `pygame-ce` to support Python 3.14+ compatibility for Gymnasium rendering.*

---

## 🚀 How to Run

### 1. Launching the Web UI Dashboard (Recommended)
You can run the interactive web control panel and real-time physical simulator:
```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

**Dashboard Features:**
* **Interactive Sliders**: Dynamically tune learning rate, gamma, epsilon decay, max steps, and training episodes.
* **Dual Rendering Modes**:
  * *Visual Animation Mode* (Checked): Slows physics down (~50 FPS) and streams real-time state coordinates to animate the cart and pole on an HTML5 canvas.
  * *Turbo Mode* (Unchecked): Turns off physics animation to train the agent at maximum CPU speed, updating the charts instantly.
* **Real-time Charts**: Watch episodic rewards and Huber loss plot on live line graphs using Chart.js.
* **Log Console**: Displays background Python terminal logs and training events in real-time.

---

### 2. Command Line Training (Alternative)
To train the agent in the command line using default hyperparameters (saves plots and weights at completion):
```bash
python train.py
```
You can customize hyperparameters via command-line arguments:
```bash
python train.py --episodes 500 --lr 1e-3 --batch_size 128
```

---

### 3. Evaluating a Trained Agent
To run evaluation and visually watch the cart-pole being balanced:
```bash
python evaluate.py --render_mode human
```
To record a high-definition video of the agent's evaluation runs:
```bash
python evaluate.py --record_video
```
This saves video recordings of the evaluation episodes in the `videos/` folder.

---

### 4. Interactive Jupyter Notebook
To run the step-by-step interactive Jupyter Notebook:
```bash
jupyter notebook cartpole_dqn.ipynb
```
This notebook outlines the math behind Q-learning, DQN concepts, and runs the entire training loop inline with visualizations.

---

## 🧠 Reinforcement Learning Concepts Implemented

* **Deep Q-Network (DQN)**: A Multi-Layer Perceptron (MLP) mapping 4-dimensional state observations to 2-dimensional action Q-values.
* **Experience Replay Buffer**: A memory bank storing transitions `(state, action, reward, next_state, done)`. We sample randomly from it during training, breaking temporal correlation and stabilizing gradient updates.
* **Target Network with Soft Update**: A secondary network (`target_net`) calculates target Q-values. It is updated gradually toward the training network (`policy_net`) using Polyak averaging (soft updates, controlled by parameter `tau`), preventing oscillation during learning.
* **Huber Loss (Smooth L1 Loss)**: Minimizes error between predicted and target Q-values. Huber loss behaves like L2 loss for small errors and L1 loss for large errors, making it highly robust to outliers and gradient instability.
