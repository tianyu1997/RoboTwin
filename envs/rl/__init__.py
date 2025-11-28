"""
RoboTwin Reinforcement Learning Module

This module provides Gymnasium-compatible environment wrappers and training utilities
for reinforcement learning on RoboTwin manipulation tasks.

Components:
- gym_wrapper.py: Core Gym environment wrapper (RoboTwinGymEnv)
- vec_env.py: Vectorized environment for parallel training
- train_ppo.py: PPO training script
- train_rl_example.py: Usage examples
- evaluate_rl.py: Model evaluation script
"""

from .gym_wrapper import RoboTwinGymEnv, make_robotwin_env, register_robotwin_envs
from .vec_env import RoboTwinVecEnv, make_vec_env

__all__ = [
    "RoboTwinGymEnv",
    "make_robotwin_env", 
    "register_robotwin_envs",
    "RoboTwinVecEnv",
    "make_vec_env",
]
