"""
RoboTwin Environments Package

This package provides manipulation task environments for dual-arm robots.

Structure:
- tasks/: All manipulation task definitions
- rl/: Reinforcement learning wrappers and training utilities  
- robot/: Robot control and planning
- camera/: Camera and observation utilities
- utils/: Common utilities
"""

from .utils import *
from ._GLOBAL_CONFIGS import *
from ._base_task import Base_Task

# Import task utilities
from .tasks import get_task_class, list_tasks, TASK_REGISTRY

# Import RL wrappers
from .rl import (
    RoboTwinGymEnv, 
    make_robotwin_env, 
    register_robotwin_envs,
    RoboTwinVecEnv, 
    make_vec_env
)

__all__ = [
    # Base
    "Base_Task",
    # Tasks
    "get_task_class",
    "list_tasks", 
    "TASK_REGISTRY",
    # RL
    "RoboTwinGymEnv",
    "make_robotwin_env",
    "register_robotwin_envs",
    "RoboTwinVecEnv",
    "make_vec_env",
]
