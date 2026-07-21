import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Cart-Pole")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Ending epsilon for exploration")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--buffer_size", type=int, default=20000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update parameter (target network)")
    parser.add_argument("--save_path", type=str, default="cartpole_dqn.pth", help="Path to save trained model")
    parser.add_argument("--plot_path", type=str, default="training_rewards.png", help="Path to save training plot")
    return parser.parse_args()

def train():
    args = parse_args()

    # Create the environment
    # Using gymnasium CartPole-v1
    env = gym.make("CartPole-v1")
    
    # Set seeds for reproducibility
    seed = 42
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on device: {device}")

    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        device=device
    )

    rewards_history = []
    avg_rewards_history = []
    
    solved = False
    print("\n[+] Starting training...")
    print("-" * 65)
    print(f"{'Episode':<10} | {'Steps':<8} | {'Reward':<8} | {'Avg Reward (100)':<18} | {'Epsilon':<8}")
    print("-" * 65)

    for episode in range(1, args.episodes + 1):
        state, info = env.reset(seed=seed + episode)
        episode_reward = 0
        
        for step in range(args.max_steps):
            # Select action
            action = agent.select_action(state)

            # Step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record experience and learn
            agent.step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards_history.append(episode_reward)
        
        # Calculate running average reward (last 100 episodes)
        running_avg = np.mean(rewards_history[-100:])
        avg_rewards_history.append(running_avg)

        # Print progress every 10 episodes (or every episode if it solves it early)
        if episode % 10 == 0 or running_avg >= 475.0:
            print(f"{episode:<10} | {step+1:<8} | {episode_reward:<8.1f} | {running_avg:<18.1f} | {agent.epsilon:<8.3f}")

        # Check if environment is solved (Gymnasium CartPole-v1 solved threshold is typically 475+)
        if running_avg >= 475.0 and episode >= 100:
            print("-" * 65)
            print(f"[!] Environment solved in {episode} episodes! Average Reward: {running_avg:.1f}")
            solved = True
            break

    env.close()

    # Save trained model
    agent.save(args.save_path)
    print(f"\n[+] Saved model weights to {args.save_path}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label="Episode Reward", color="lightblue", alpha=0.8)
    plt.plot(avg_rewards_history, label="100-Episode Moving Avg", color="darkblue", linewidth=2)
    plt.axhline(y=475.0, color="red", linestyle="--", label="Solved Threshold (475)")
    plt.title("Cart-Pole DQN Agent Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path)
    plt.close()
    print(f"[+] Saved training plot to {args.plot_path}")

    if not solved:
        print(f"\n[!] Training finished. Agent did not officially solve the environment (reached Avg Reward: {running_avg:.1f}).")

if __name__ == "__main__":
    train()
