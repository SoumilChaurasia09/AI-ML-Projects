import os
import time
import json
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import gymnasium as gym
import numpy as np
import torch
from dqn_agent import DQNAgent

app = FastAPI(title="Cart-Pole RL Training Dashboard")

# Ensure static files directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# Global state
active_connections: list[WebSocket] = []
training_thread = None
training_active = False
loop = None
message_queue = asyncio.Queue()

class ConnectionManager:
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        active_connections.append(websocket)
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "status": "training" if training_active else "idle"
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in active_connections:
            active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dead in dead_connections:
            self.disconnect(dead)

manager = ConnectionManager()

# Message dispatcher that runs in the main async loop
async def message_dispatcher():
    while True:
        message = await message_queue.get()
        await manager.broadcast(message)
        message_queue.task_done()

@app.on_event("startup")
async def startup_event():
    # Start the message dispatcher task
    asyncio.create_task(message_dispatcher())

def run_training_loop(config: dict, async_loop: asyncio.AbstractEventLoop):
    global training_active
    
    # Extract config
    lr = config.get("lr", 1e-3)
    gamma = config.get("gamma", 0.99)
    episodes = config.get("episodes", 300)
    max_steps = config.get("max_steps", 500)
    epsilon_decay = config.get("epsilon_decay", 0.995)
    buffer_size = config.get("buffer_size", 20000)
    batch_size = config.get("batch_size", 128)
    tau = config.get("tau", 0.005)
    visual_mode = config.get("visual_mode", True)
    
    # Physics sandbox settings
    gravity = config.get("gravity", 9.8)
    pole_length = config.get("pole_length", 0.5)  # half length
    
    # Setup environment
    env = gym.make("CartPole-v1")
    
    # Apply physics customizations directly to Gymnasium unwrapped object
    env.unwrapped.gravity = gravity
    env.unwrapped.length = pole_length
    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        device=device
    )
    
    rewards_history = []
    
    def queue_msg(msg):
        async_loop.call_soon_threadsafe(message_queue.put_nowait, msg)
    
    queue_msg({
        "type": "info", 
        "message": f"Training started on: {device} (Gravity: {gravity}m/s², Pole Length: {pole_length * 2:.1f}m)"
    })
    
    for episode in range(1, episodes + 1):
        if not training_active:
            break
            
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if not training_active:
                break
                
            # Select action
            action = agent.select_action(state)
            
            # Step env
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record and learn
            agent.step(state, action, reward, next_state, done)
            
            # If visual mode is enabled, stream frame-by-frame details
            if visual_mode:
                queue_msg({
                    "type": "step",
                    "episode": episode,
                    "step": step + 1,
                    "state": state.tolist(),
                    "action": int(action),
                    "reward": float(reward)
                })
                # Add delay to make it visualizable (approx 50 FPS)
                time.sleep(0.02)
                
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        if not training_active:
            break
            
        rewards_history.append(episode_reward)
        running_avg = float(np.mean(rewards_history[-100:]))
        
        # Stream episode metrics
        queue_msg({
            "type": "episode",
            "episode": episode,
            "reward": float(episode_reward),
            "avg_reward": running_avg,
            "loss": float(agent.last_loss),
            "epsilon": float(agent.epsilon)
        })
        
        # Check solved condition
        if running_avg >= 475.0 and episode >= 100:
            queue_msg({
                "type": "status",
                "status": "solved",
                "message": f"Environment solved in {episode} episodes with average reward {running_avg:.1f}!"
            })
            # Save final model weights
            os.makedirs("models", exist_ok=True)
            agent.save("models/cartpole_dqn.pth")
            training_active = False
            return
            
    env.close()
    
    # Save model if completed
    if training_active:
        os.makedirs("models", exist_ok=True)
        agent.save("models/cartpole_dqn.pth")
        queue_msg({
            "type": "status",
            "status": "idle",
            "message": "Training finished successfully!"
        })
    else:
        queue_msg({
            "type": "status",
            "status": "stopped",
            "message": "Training stopped."
        })
        
    training_active = False

def run_evaluation_loop(config: dict, async_loop: asyncio.AbstractEventLoop):
    global training_active
    
    # Physics sandbox settings
    gravity = config.get("gravity", 9.8)
    pole_length = config.get("pole_length", 0.5)
    
    def queue_msg(msg):
        async_loop.call_soon_threadsafe(message_queue.put_nowait, msg)
        
    # Check model paths
    model_path = "models/cartpole_dqn.pth" if os.path.exists("models/cartpole_dqn.pth") else "cartpole_dqn.pth"
    if not os.path.exists(model_path):
        queue_msg({
            "type": "info",
            "message": "Error: No trained model weights found! Please train the agent first."
        })
        queue_msg({"type": "status", "status": "idle"})
        training_active = False
        return

    env = gym.make("CartPole-v1")
    
    # Apply same physics parameters to evaluation run!
    env.unwrapped.gravity = gravity
    env.unwrapped.length = pole_length
    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
    
    try:
        agent.load(model_path)
        queue_msg({
            "type": "info", 
            "message": f"Loaded trained weights from: {model_path} (Evaluating with Gravity: {gravity}m/s², Pole: {pole_length * 2:.1f}m)"
        })
    except Exception as e:
        queue_msg({"type": "info", "message": f"Error loading model: {str(e)}"})
        queue_msg({"type": "status", "status": "idle"})
        training_active = False
        env.close()
        return
        
    queue_msg({"type": "info", "message": "Starting agent visual evaluation runs..."})
    
    # Run 5 evaluation episodes
    for episode in range(1, 6):
        if not training_active:
            break
            
        state, info = env.reset(seed=99 + episode)
        episode_reward = 0
        
        for step in range(500):
            if not training_active:
                break
                
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            queue_msg({
                "type": "step",
                "episode": episode,
                "step": step + 1,
                "state": state.tolist(),
                "action": int(action),
                "reward": float(reward)
            })
            time.sleep(0.02)
            
            state = next_state
            episode_reward += reward
            if done:
                break
                
        if not training_active:
            break
            
        queue_msg({
            "type": "episode",
            "episode": episode,
            "reward": float(episode_reward),
            "avg_reward": float(episode_reward),
            "loss": 0.0,
            "epsilon": 0.0
        })
        
    env.close()
    queue_msg({"type": "status", "status": "idle", "message": "Visual evaluation completed."})
    training_active = False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global training_active, training_thread, loop
    await manager.connect(websocket)
    loop = asyncio.get_running_loop()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            command = message.get("command")
            
            if command == "start":
                if not training_active:
                    config = message.get("config", {})
                    training_active = True
                    training_thread = threading.Thread(
                        target=run_training_loop,
                        args=(config, loop),
                        daemon=True
                    )
                    training_thread.start()
                    await manager.broadcast({"type": "status", "status": "training"})
                else:
                    await websocket.send_json({"type": "info", "message": "An operation is already active."})
            
            elif command == "evaluate":
                if not training_active:
                    config = message.get("config", {})
                    training_active = True
                    training_thread = threading.Thread(
                        target=run_evaluation_loop,
                        args=(config, loop),
                        daemon=True
                    )
                    training_thread.start()
                    await manager.broadcast({"type": "status", "status": "training"})
                else:
                    await websocket.send_json({"type": "info", "message": "An operation is already active."})
                    
            elif command == "stop":
                if training_active:
                    training_active = False
                    await manager.broadcast({"type": "status", "status": "stopped", "message": "Stopping current operation..."})
                else:
                    await websocket.send_json({"type": "info", "message": "No operation is running."})
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
