"""
Evaluation Script for RoboTwin RL Models

This script evaluates trained RL models on RoboTwin tasks.
"""

import sys
import os

# Add RoboTwin root to path
current_file = os.path.abspath(__file__)
training_dir = os.path.dirname(current_file)    # rl/training/
rl_dir = os.path.dirname(training_dir)          # rl/
robotwi_dir = os.path.dirname(rl_dir)          # RoboTwin/
sys.path.insert(0, robotwin_dir)

import argparse
import numpy as np
import torch
from typing import Optional, List, Dict
import json
from datetime import datetime

from rl import RoboTwinGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL Model on RoboTwin")
    
    # Model settings
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="sb3",
                       choices=["sb3", "custom"],
                       help="Model type")
    
    # Environment settings
    parser.add_argument("--task", type=str, default="beat_block_hammer",
                       help="Task name")
    parser.add_argument("--task_config", type=str, default="demo_randomized",
                       help="Task configuration file")
    parser.add_argument("--max_episode_steps", type=int, default=300,
                       help="Maximum steps per episode")
    parser.add_argument("--action_type", type=str, default="qpos",
                       choices=["qpos", "ee", "delta_ee"],
                       help="Action type")
    
    # Evaluation settings
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic actions")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation")
    parser.add_argument("--save_video", action="store_true",
                       help="Save evaluation videos")
    parser.add_argument("--video_dir", type=str, default="./eval_videos",
                       help="Video save directory")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    
    return parser.parse_args()


def load_sb3_model(model_path: str):
    """Load Stable-Baselines3 model."""
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        return model
    except ImportError:
        print("Stable-Baselines3 not installed!")
        return None


def evaluate_model(
    model,
    env: RoboTwinGymEnv,
    num_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    save_video: bool = False,
    video_dir: str = None
) -> Dict:
    """
    Evaluate model on environment.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get('success', False):
            success_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"reward = {episode_reward:.2f}, "
                  f"length = {episode_length}, "
                  f"success = {info.get('success', False)}")
    
    # Compute statistics
    results = {
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': success_count / num_episodes,
        'success_count': success_count,
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
    }
    
    return results


def evaluate_random_policy(
    env: RoboTwinGymEnv,
    num_episodes: int = 100
) -> Dict:
    """Evaluate random policy as baseline."""
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get('success', False):
            success_count += 1
    
    return {
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'success_rate': success_count / num_episodes,
    }


def main():
    args = parse_args()
    
    # Setup multiprocessing
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    render_mode = 'human' if args.render else None
    env = RoboTwinGymEnv(
        task_name=args.task,
        task_config=args.task_config,
        action_type=args.action_type,
        obs_keys=['qpos', 'endpose'],
        max_episode_steps=args.max_episode_steps,
        reward_type='sparse',
        render_mode=render_mode,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("RoboTwin Model Evaluation")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.num_episodes}")
    print("=" * 60 + "\n")
    
    # Load model
    if args.model_type == "sb3":
        model = load_sb3_model(args.model_path)
        if model is None:
            print("Failed to load model!")
            return
    else:
        print("Custom model loading not implemented yet")
        return
    
    # Evaluate model
    print("Evaluating trained model...")
    results = evaluate_model(
        model=model,
        env=env,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        render=args.render,
        save_video=args.save_video,
        video_dir=args.video_dir
    )
    
    # Evaluate random baseline
    print("\nEvaluating random baseline...")
    env_baseline = RoboTwinGymEnv(
        task_name=args.task,
        task_config=args.task_config,
        action_type=args.action_type,
        obs_keys=['qpos', 'endpose'],
        max_episode_steps=args.max_episode_steps,
        seed=args.seed + 10000
    )
    random_results = evaluate_random_policy(env_baseline, num_episodes=min(50, args.num_episodes))
    env_baseline.close()
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nTrained Model:")
    print(f"  Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    print(f"  ({results['success_count']}/{results['num_episodes']} episodes)")
    
    print(f"\nRandom Baseline:")
    print(f"  Mean Reward: {random_results['mean_reward']:.4f} ± {random_results['std_reward']:.4f}")
    print(f"  Success Rate: {random_results['success_rate']*100:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        args.output_dir, 
        f"{args.task}_eval_{timestamp}.json"
    )
    
    full_results = {
        'args': vars(args),
        'trained_model': results,
        'random_baseline': random_results,
        'timestamp': timestamp
    }
    
    with open(result_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {result_path}")
    
    env.close()


if __name__ == "__main__":
    main()
