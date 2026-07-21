import argparse
import os
import gymnasium as gym
import numpy as np
import torch
from dqn_agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent on Cart-Pole")
    parser.add_argument("--model_path", type=str, default="cartpole_dqn.pth", help="Path to the trained model weights")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run evaluation")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array", "none"], help="Environment rendering mode")
    parser.add_argument("--record_video", action="store_true", help="Record video of evaluation episodes")
    parser.add_argument("--video_folder", type=str, default="videos", help="Directory where video files will be saved")
    return parser.parse_args()

def evaluate():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"[!] Error: Model weights file not found at '{args.model_path}'. Please train the agent first.")
        return

    # Determine render mode
    render_mode = None if args.render_mode == "none" else args.render_mode
    if args.record_video:
        # Gymnasium RecordVideo requires rgb_array rendering
        render_mode = "rgb_array"

    # Create the environment
    try:
        env = gym.make("CartPole-v1", render_mode=render_mode)
    except Exception as e:
        print(f"[*] Warning: Could not initialize environment with render_mode='{render_mode}'. Fallback to 'none'. Error: {e}")
        render_mode = None
        env = gym.make("CartPole-v1", render_mode=None)

    # Wrap for video recording if requested
    if args.record_video and render_mode == "rgb_array":
        print(f"[*] Video recording enabled. Output will be saved to: {args.video_folder}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder,
            episode_trigger=lambda ep: True,  # Record all episodes
            name_prefix="eval-episode"
        )

    # Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
    agent.load(args.model_path)
    print(f"[+] Loaded trained DQN agent from {args.model_path}")

    # Set seed for repeatability
    seed = 99
    
    print("\n[+] Starting evaluation...")
    print("-" * 40)
    
    eval_rewards = []
    
    for ep in range(1, args.episodes + 1):
        state, info = env.reset(seed=seed + ep)
        episode_reward = 0
        
        for step in range(args.max_steps):
            # Select action greedily (evaluate=True disables exploration)
            action = agent.select_action(state, evaluate=True)
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        eval_rewards.append(episode_reward)
        print(f"Episode {ep:<2} | Steps: {step+1:<3} | Total Reward: {episode_reward:.1f}")

    env.close()

    print("-" * 40)
    print(f"[+] Evaluation finished.")
    print(f"[*] Average Reward over {args.episodes} episodes: {np.mean(eval_rewards):.1f} / {args.max_steps:.1f}")
    print(f"[*] Reward standard deviation: {np.std(eval_rewards):.2f}")
    
    if args.record_video:
        print(f"[*] Check '{args.video_folder}' to watch the agent's performance video.")

if __name__ == "__main__":
    evaluate()
