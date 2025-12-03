"""
F1-VLA RL Training Module

This module contains all training-related code for the F1-VLA RL framework:

Training Scripts (Three-Phase Training):
- train_teacher_rl.py:     Phase 1 - Train world model with random exploration
- train_student_rl.py:     Phase 2 - Train policy using PPO with world model rewards
- train_adversarial_rl.py: Phase 3 - Adversarial training between WM and Explorer

Entry Points:
- train_rl.py:  Unified Python entry point for all phases
- train_rl.sh:  Shell script wrapper with environment setup

Shared Utilities:
- rl_training_common.py: Common classes and functions for all training phases
  - RLConfig: Training configuration dataclass
  - MemoryStateManager: Memory state tracking for F1-VLA
  - load_rl_config(): Load and merge configs
  - load_f1_policy(): Load F1-VLA policy with LoRA

Legacy/Examples:
- train_ppo.py:        Basic PPO training example
- train_rl_example.py: Simple RL training example
- evaluate_rl.py:      Policy evaluation utilities

Usage:
    # From command line (recommended)
    cd RoboTwin/rl/training
    python train_rl.py --phase teacher
    
    # Or use shell script
    ./train_rl.sh teacher
    ./train_rl.sh all  # Run all phases sequentially
"""

from .rl_training_common import (
    load_rl_config,
    get_training_config,
    get_environment_config,
    get_lora_config_from_dict,
    load_f1_policy,
    MemoryStateManager,
    clip_gradients,
)

__all__ = [
    "load_rl_config",
    "get_training_config",
    "get_environment_config",
    "get_lora_config_from_dict",
    "load_f1_policy",
    "MemoryStateManager",
    "clip_gradients",
]
