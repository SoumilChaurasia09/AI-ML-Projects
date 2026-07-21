import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class ReplayBuffer:
    """Experience Replay Buffer to store and sample environment transitions."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of transitions uniformly at random."""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Deep Q-Network (Multi-Layer Perceptron) for estimating state-action values."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Deep Q-Network agent that interacts with the environment and learns."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,  # Soft update parameter
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.device = torch.device(device)

        # Initialize policy and target networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network doesn't compute gradients

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Loss function (Huber loss is more stable than MSE)
        self.loss_fn = nn.SmoothL1Loss()
        self.last_loss = 0.0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select an action using epsilon-greedy policy (or greedy if evaluating)."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add experience to memory and perform a learning step if enough samples exist."""
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn if we have enough experiences in buffer
        if len(self.memory) >= self.batch_size:
            self._learn()

    def _learn(self):
        """Perform one step of gradient descent update."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to PyTorch tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        # Get Q values for the selected actions
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q values
        # Max Q value for next states from target network
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (self.gamma * next_q * (1 - dones_t))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        self.last_loss = float(loss.item())

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Soft update target network: target = tau * policy + (1 - tau) * target
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save the policy network state dictionary."""
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str):
        """Load the policy network state dictionary."""
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
