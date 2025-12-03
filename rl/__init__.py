"""
RoboTwin Reinforcement Learning Module

This module provides Gymnasium-compatible environment wrappers and training utilities
for reinforcement learning on RoboTwin manipulation tasks.

Directory Structure:
    rl/
    ├── __init__.py          # This file
    ├── f1_rl_env.py         # F1-VLA RL environment
    ├── gym_wrapper.py       # Core Gym environment wrapper
    ├── vec_env.py           # Vectorized environment
    ├── normalizers.py       # Action/State normalization
    ├── suppress_logs.py     # Log suppression utilities
    ├── rl_config.yaml       # Training configuration
    └── training/            # Training scripts
        ├── __init__.py
        ├── rl_training_common.py  # Shared training utilities
        ├── train_teacher_rl.py    # Phase 1: World Model training
        ├── train_student_rl.py    # Phase 2: Explorer training
        ├── train_adversarial_rl.py # Phase 3: Adversarial training
        ├── train_rl.py            # Unified entry point
        └── train_rl.sh            # Shell script wrapper

Training Phases:
- Phase 1 (Teacher): Train LLM + World Model with random actions
- Phase 2 (Student): Train Explorer with frozen WM, reward = divergence reduction
- Phase 3 (Adversarial): WM vs Explorer min-max game
"""

from .gym_wrapper import RoboTwinGymEnv, make_robotwin_env, register_robotwin_envs
from .vec_env import RoboTwinVecEnv, make_vec_env
from .normalizers import ActionNormalizer, StateNormalizer, create_normalizer_from_env
from .f1_rl_env import F1RLEnv, TeacherEnv, StudentEnv

__all__ = [
    # Core Gym wrapper
    "RoboTwinGymEnv",
    "make_robotwin_env", 
    "register_robotwin_envs",
    # Vectorized environments
    "RoboTwinVecEnv",
    "make_vec_env",
    # Normalizers
    "ActionNormalizer",
    "StateNormalizer",
    "create_normalizer_from_env",
    # F1-VLA RL environments
    "F1RLEnv",
    "TeacherEnv", 
    "StudentEnv",
]
