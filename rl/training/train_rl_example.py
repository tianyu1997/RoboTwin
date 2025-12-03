"""
Example script demonstrating how to use RoboTwin Gym environments for RL training.

This script shows:
1. Basic environment usage with Gymnasium API
2. Vectorized environment for parallel training
3. Integration with popular RL libraries (PPO example)
"""

import sys
import os

# Add RoboTwin root to path
current_file = os.path.abspath(__file__)
training_dir = os.path.dirname(current_file)    # rl/training/
rl_dir = os.path.dirname(training_dir)          # rl/
robotwi_dir = os.path.dirname(rl_dir)          # RoboTwin/
sys.path.insert(0, robotwin_dir)

import numpy as np
import gymnasium as gym
from typing import Optional


def example_basic_usage():
    """Basic example of using RoboTwin Gym environment."""
    print("=" * 60)
    print("Example 1: Basic Environment Usage")
    print("=" * 60)
    
    from rl import RoboTwinGymEnv, make_robotwin_env
    
    # Method 1: Direct instantiation
    env = RoboTwinGymEnv(
        task_name="beat_block_hammer",
        task_config="demo_randomized",
        action_type="qpos",  # or 'ee', 'delta_ee'
        obs_keys=['rgb', 'qpos', 'endpose'],
        max_episode_steps=300,
        reward_type="sparse",
        render_mode=None,  # or 'human' for visualization
        seed=42
    )
    
    # Method 2: Using factory function
    # env = make_robotwin_env("beat_block_hammer", task_config="demo_randomized")
    
    print(f"Task: {env.task_name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial observation keys: {obs.keys()}")
    print(f"Info: {info}")
    
    # Run a few random steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
    
    print(f"Total reward after {step + 1} steps: {total_reward}")
    
    env.close()
    print("Environment closed successfully.\n")


def example_vectorized_env():
    """Example of using vectorized environment for parallel training."""
    print("=" * 60)
    print("Example 2: Vectorized Environment")
    print("=" * 60)
    
    from rl import make_vec_env, RoboTwinVecEnv
    
    # Method 1: Using make_vec_env function
    # vec_env = make_vec_env(
    #     task_name="beat_block_hammer",
    #     num_envs=4,
    #     task_config="demo_randomized",
    #     seed=42,
    #     vec_env_type="sync"
    # )
    
    # Method 2: Using custom RoboTwinVecEnv
    vec_env = RoboTwinVecEnv(
        task_name="beat_block_hammer",
        num_envs=2,  # Use 2 parallel environments
        task_config="demo_randomized",
        action_type="qpos",
        obs_keys=['qpos', 'endpose'],
        max_episode_steps=100,
        seed=42
    )
    
    print(f"Number of environments: {len(vec_env)}")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # Reset all environments
    obs, infos = vec_env.reset()
    print(f"\nBatch observation shape example (qpos): {obs['qpos'].shape}")
    
    # Run a few steps with batch actions
    for step in range(5):
        # Sample batch of actions
        actions = np.array([vec_env.action_space.sample() for _ in range(len(vec_env))])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        print(f"Step {step + 1}: rewards = {rewards}, terminateds = {terminateds}")
    
    vec_env.close()
    print("Vectorized environment closed successfully.\n")


def example_gymnasium_registered():
    """Example using Gymnasium's registry."""
    print("=" * 60)
    print("Example 3: Using Gymnasium Registry")
    print("=" * 60)
    
    from rl import register_robotwin_envs
    
    # Register environments
    register_robotwin_envs()
    
    # Create environment using gym.make
    try:
        env = gym.make("RoboTwin-beat_block_hammer-v0")
        print(f"Created environment: RoboTwin-beat_block_hammer-v0")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"Registry example skipped: {e}")
    
    print()


def example_stable_baselines3():
    """Example integration with Stable-Baselines3 PPO."""
    print("=" * 60)
    print("Example 4: Stable-Baselines3 Integration (PPO)")
    print("=" * 60)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
        
        from rl import RoboTwinGymEnv
        
        # Create environment factory
        def make_env(seed: int):
            def _init():
                env = RoboTwinGymEnv(
                    task_name="beat_block_hammer",
                    task_config="demo_randomized",
                    action_type="qpos",
                    obs_keys=['qpos'],  # Simplified observation for this example
                    max_episode_steps=100,
                    seed=seed
                )
                return env
            return _init
        
        # Create vectorized environment
        num_envs = 2
        env = DummyVecEnv([make_env(seed=i) for i in range(num_envs)])
        
        # Create PPO agent
        model = PPO(
            "MultiInputPolicy",  # Use MultiInputPolicy for Dict observations
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="cuda",  # or "cpu"
        )
        
        print("PPO model created successfully!")
        print(f"Policy architecture: {model.policy}")
        
        # Training (commented out for demo - uncomment to train)
        # model.learn(total_timesteps=10000, progress_bar=True)
        # model.save("ppo_robotwin_beat_block_hammer")
        
        env.close()
        print("SB3 example completed.\n")
        
    except ImportError as e:
        print(f"Stable-Baselines3 not installed: {e}")
        print("Install with: pip install stable-baselines3")
        print()


def example_custom_reward():
    """Example of customizing reward function."""
    print("=" * 60)
    print("Example 5: Custom Reward Function")
    print("=" * 60)
    
    from envs.gym_wrapper import RoboTwinGymEnv
    
    class CustomRewardEnv(RoboTwinGymEnv):
        """Environment with custom dense reward."""
        
        def _compute_reward(self) -> float:
            """Custom dense reward based on task progress."""
            # Get success status
            success = self._task_env.check_success()
            
            if success:
                return 100.0  # Large reward for success
            
            # Dense reward example: distance-based reward
            reward = 0.0
            
            # Get current observation for reward computation
            try:
                obs = self._task_env.get_obs()
                
                # Example: Reward based on gripper state
                endpose = obs.get('endpose', {})
                left_gripper = endpose.get('left_gripper', 0.5)
                right_gripper = endpose.get('right_gripper', 0.5)
                
                # Penalize for open grippers when should be grasping
                # (this is just an example, actual reward depends on task)
                
                # Small penalty for each step to encourage efficiency
                reward -= 0.01
                
            except Exception:
                pass
            
            return reward
    
    # Use custom reward environment
    env = CustomRewardEnv(
        task_name="beat_block_hammer",
        task_config="demo_randomized",
        action_type="qpos",
        max_episode_steps=50,
        seed=42
    )
    
    obs, info = env.reset()
    
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: reward = {reward:.4f}")
        
        if terminated or truncated:
            break
    
    print(f"Total custom reward: {total_reward:.4f}")
    env.close()
    print()


def example_observation_wrappers():
    """Example of using observation wrappers."""
    print("=" * 60)
    print("Example 6: Observation Wrappers")
    print("=" * 60)
    
    from gymnasium.wrappers import FlattenObservation, FilterObservation
    from rl import RoboTwinGymEnv
    
    # Create base environment
    env = RoboTwinGymEnv(
        task_name="beat_block_hammer",
        task_config="demo_randomized",
        action_type="qpos",
        obs_keys=['qpos', 'endpose'],
        max_episode_steps=50,
        seed=42
    )
    
    print(f"Original observation space: {env.observation_space}")
    
    # Optionally filter observations
    # env = FilterObservation(env, filter_keys=['qpos'])
    
    # Flatten observation for MLP policies
    # Note: FlattenObservation works with Dict spaces
    # env = FlattenObservation(env)
    # print(f"Flattened observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    
    env.close()
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RoboTwin Gym Environment Examples")
    parser.add_argument(
        "--example",
        type=str,
        default="all",
        choices=["basic", "vec", "registry", "sb3", "reward", "wrapper", "all"],
        help="Which example to run"
    )
    args = parser.parse_args()
    
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    examples = {
        "basic": example_basic_usage,
        "vec": example_vectorized_env,
        "registry": example_gymnasium_registered,
        "sb3": example_stable_baselines3,
        "reward": example_custom_reward,
        "wrapper": example_observation_wrappers,
    }
    
    if args.example == "all":
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"Example '{name}' failed with error: {e}\n")
    else:
        examples[args.example]()
