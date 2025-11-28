"""
Vectorized Environment Wrapper for RoboTwin

This module provides vectorized environment support for parallel RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv, AsyncVectorEnv, SyncVectorEnv
from typing import List, Optional, Callable, Dict, Any
import multiprocessing as mp

from .gym_wrapper import RoboTwinGymEnv, make_robotwin_env


def make_vec_env(
    task_name: str,
    num_envs: int = 4,
    task_config: str = "demo_randomized",
    seed: int = 0,
    vec_env_type: str = "sync",
    **env_kwargs
) -> VectorEnv:
    """
    Create a vectorized RoboTwin environment.
    
    Args:
        task_name: Name of the task
        num_envs: Number of parallel environments
        task_config: Configuration file name
        seed: Base random seed
        vec_env_type: Type of vectorized env ('sync' or 'async')
        **env_kwargs: Additional environment arguments
        
    Returns:
        Vectorized environment
    """
    def make_env(env_seed: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = RoboTwinGymEnv(
                task_name=task_name,
                task_config=task_config,
                seed=env_seed,
                **env_kwargs
            )
            return env
        return _init
    
    env_fns = [make_env(seed + i) for i in range(num_envs)]
    
    if vec_env_type == "async":
        return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)


class RoboTwinVecEnv:
    """
    Custom vectorized environment wrapper for RoboTwin.
    
    This provides more control over parallel environments compared to
    the standard Gymnasium vectorized environments.
    """
    
    def __init__(
        self,
        task_name: str,
        num_envs: int = 4,
        task_config: str = "demo_randomized",
        seed: int = 0,
        **env_kwargs
    ):
        self.task_name = task_name
        self.num_envs = num_envs
        self.task_config = task_config
        self.seed = seed
        self.env_kwargs = env_kwargs
        
        # Create environments
        self.envs: List[RoboTwinGymEnv] = []
        for i in range(num_envs):
            env = RoboTwinGymEnv(
                task_name=task_name,
                task_config=task_config,
                seed=seed + i,
                **env_kwargs
            )
            self.envs.append(env)
        
        # Get spaces from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """Reset all environments."""
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            observations.append(obs)
            infos.append(info)
        
        # Stack observations
        stacked_obs = self._stack_obs(observations)
        
        return stacked_obs, infos
    
    def step(self, actions: np.ndarray):
        """Step all environments with given actions."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            
            # Auto-reset if done
            if terminated or truncated:
                obs, info = env.reset()
                observations[-1] = obs
                infos[-1]['final_observation'] = observations[-1]
                infos[-1]['final_info'] = info
        
        stacked_obs = self._stack_obs(observations)
        rewards = np.array(rewards, dtype=np.float32)
        terminateds = np.array(terminateds, dtype=bool)
        truncateds = np.array(truncateds, dtype=bool)
        
        return stacked_obs, rewards, terminateds, truncateds, infos
    
    def _stack_obs(self, observations: List[Dict]) -> Dict[str, np.ndarray]:
        """Stack observations from all environments."""
        if not observations:
            return {}
        
        stacked = {}
        for key in observations[0].keys():
            stacked[key] = np.stack([obs[key] for obs in observations])
        
        return stacked
    
    def render(self):
        """Render all environments."""
        renders = []
        for env in self.envs:
            render = env.render()
            if render is not None:
                renders.append(render)
        return renders if renders else None
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
    
    def __len__(self):
        return self.num_envs
