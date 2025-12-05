#!/usr/bin/env python3
"""
Parallel Training Utilities for F1-VLA RL Training

Uses HuggingFace Accelerate library for DDP support (well-tested, production-ready).
Uses gymnasium's VectorEnv for multi-environment parallel data collection.

Features:
1. DDP via Accelerate - handles multi-GPU automatically
2. VectorEnv for parallel environments (gymnasium standard)
3. Sequential episode buffer for temporal ordering

Usage:
    # Launch with: accelerate launch --num_processes=8 train_teacher_rl.py
    # Or single GPU: python train_teacher_rl.py
"""

import os
import logging
import warnings

# Suppress the torch.distributed warning about device id not provided
# This warning appears even when we set the device correctly because
# Accelerate's internal barrier() calls happen before we can set the device.
warnings.filterwarnings(
    "ignore",
    message="No device id is provided via `init_process_group` or `barrier",
    category=UserWarning,
    module="torch.distributed.distributed_c10d"
)
from typing import Dict, Any, Optional, List, Callable
from collections import deque
import threading
import copy

import numpy as np
import torch
import torch.nn as nn

# Use Accelerate for DDP (HuggingFace's well-tested library)
from accelerate import Accelerator
from accelerate.utils import set_seed

# Use gymnasium's vector environments
from gymnasium.vector import SyncVectorEnv

logger = logging.getLogger(__name__)


# =============================================================================
# Accelerator Wrapper (uses HuggingFace Accelerate)
# =============================================================================

class AcceleratorWrapper:
    """
    Wrapper around HuggingFace Accelerate for easy DDP setup.
    
    Usage:
        wrapper = AcceleratorWrapper()
        model, optimizer, scheduler = wrapper.prepare(model, optimizer, scheduler)
        
        for batch in dataloader:
            loss = model(batch)
            wrapper.backward(loss)
            optimizer.step()
    """
    
    def __init__(
        self,
        mixed_precision: str = "no",  # "no", "fp16", "bf16"
        gradient_accumulation_steps: int = 1,
        log_with: Optional[str] = None,  # "tensorboard", "wandb", etc.
        project_dir: Optional[str] = None,
    ):
        """
        Initialize accelerator.
        
        Args:
            mixed_precision: Mixed precision mode
            gradient_accumulation_steps: Steps to accumulate gradients
            log_with: Logging backend
            project_dir: Directory for logs
        """
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
            project_dir=project_dir,
        )
        
        self._is_main = self.accelerator.is_main_process
        self._device = self.accelerator.device
        # Ensure the current CUDA device matches the accelerator's device.
        # This avoids warnings from torch.distributed when no device id is
        # explicitly provided to init_process_group/barrier — torch will
        # otherwise use whatever device the process happens to have set.
        # Use a guarded call in case CUDA isn't available or the device
        # is not a CUDA device.
        try:
            if torch.cuda.is_available() and hasattr(self._device, 'type') and self._device.type == 'cuda':
                torch.cuda.set_device(self._device)
        except Exception:
            # Don't fail initialization for device-setting issues; just warn.
            logger.debug("Failed to set current CUDA device to accelerator.device", exc_info=True)
        self._num_processes = self.accelerator.num_processes
        self._process_index = self.accelerator.process_index
        self._local_process_index = self.accelerator.local_process_index
    
    @property
    def is_main_process(self) -> bool:
        return self._is_main
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def num_processes(self) -> int:
        return self._num_processes
    
    @property
    def process_index(self) -> int:
        return self._process_index
    
    @property
    def local_process_index(self) -> int:
        """Local process index on this node (corresponds to GPU index)."""
        return self._local_process_index
    
    @property
    def is_distributed(self) -> bool:
        return self._num_processes > 1
    
    def prepare(self, *args):
        """Prepare model, optimizer, dataloader, etc. for distributed training."""
        return self.accelerator.prepare(*args)
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model only (without wrapping in DDP if not needed)."""
        return self.accelerator.prepare_model(model)
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient scaling if needed."""
        self.accelerator.backward(loss)
    
    def clip_grad_norm_(self, parameters, max_norm: float):
        """Clip gradients."""
        self.accelerator.clip_grad_norm_(parameters, max_norm)
    
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP model to get the underlying module."""
        return self.accelerator.unwrap_model(model)
    
    def save_state(self, output_dir: str):
        """Save training state."""
        self.accelerator.save_state(output_dir)
    
    def load_state(self, input_dir: str):
        """Load training state."""
        self.accelerator.load_state(input_dir)
    
    def wait_for_everyone(self):
        """Synchronization barrier."""
        self.accelerator.wait_for_everyone()
    
    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes."""
        return self.accelerator.gather(tensor)
    
    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduce tensor across processes."""
        return self.accelerator.reduce(tensor, reduction=reduction)
    
    def print(self, *args, **kwargs):
        """Print only on main process."""
        self.accelerator.print(*args, **kwargs)
    
    def log(self, values: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if hasattr(self.accelerator, 'log'):
            self.accelerator.log(values, step=step)
    
    def end_training(self):
        """Cleanup at end of training."""
        self.accelerator.end_training()


def create_accelerator(
    mixed_precision: str = "no",
    gradient_accumulation_steps: int = 1,
) -> AcceleratorWrapper:
    """
    Create an AcceleratorWrapper for distributed training.
    
    Args:
        mixed_precision: "no", "fp16", or "bf16"
        gradient_accumulation_steps: Number of gradient accumulation steps
        
    Returns:
        AcceleratorWrapper instance
    """
    return AcceleratorWrapper(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


# =============================================================================
# Sequential Episode Buffer
# =============================================================================

class SequentialEpisodeBuffer:
    """
    Buffer that stores complete episodes while maintaining sequential order.
    
    Each episode is stored as a list of transitions in temporal order.
    When sampling, returns consecutive transitions from the same episode
    to preserve the sequential nature of the data.
    """
    
    def __init__(
        self,
        max_episodes: int = 1000,
        max_transitions: int = 50000,
    ):
        self.max_episodes = max_episodes
        self.max_transitions = max_transitions
        
        # Store complete episodes
        self.episodes: deque = deque(maxlen=max_episodes)
        self.total_transitions = 0
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def add_episode(self, episode: List[Dict[str, Any]]):
        """Add a complete episode to the buffer."""
        if not episode:
            return
            
        with self._lock:
            # Remove old episodes if exceeding transition limit
            while (self.total_transitions + len(episode) > self.max_transitions 
                   and len(self.episodes) > 0):
                old_episode = self.episodes.popleft()
                self.total_transitions -= len(old_episode)
            
            self.episodes.append(episode)
            self.total_transitions += len(episode)
    
    def add_transitions(self, transitions: List[Dict[str, Any]]):
        """Add transitions as a single episode."""
        self.add_episode(transitions)
    
    def extend(self, transitions: List[Dict[str, Any]]):
        """Extend buffer with transitions (for compatibility with deque)."""
        # Group into single-transition "episodes" for random access
        for t in transitions:
            self.add_episode([t])
    
    def sample_sequential_batch(
        self,
        batch_size: int,
        sequence_length: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """
        Sample batch of sequential transitions.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence (1 = single transitions)
            
        Returns:
            List of sequences, each sequence is a list of consecutive transitions
        """
        with self._lock:
            if len(self.episodes) == 0:
                return []
            
            sequences = []
            attempts = 0
            max_attempts = batch_size * 10
            
            while len(sequences) < batch_size and attempts < max_attempts:
                attempts += 1
                
                # Random episode
                ep_idx = np.random.randint(len(self.episodes))
                episode = self.episodes[ep_idx]
                
                if len(episode) < sequence_length:
                    continue
                
                # Random start position
                start_idx = np.random.randint(len(episode) - sequence_length + 1)
                sequence = episode[start_idx:start_idx + sequence_length]
                sequences.append(sequence)
            
            return sequences
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample random single transitions."""
        sequences = self.sample_sequential_batch(batch_size, sequence_length=1)
        return [seq[0] for seq in sequences if seq]
    
    def __len__(self) -> int:
        return self.total_transitions
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index (for compatibility)."""
        with self._lock:
            cumsum = 0
            for episode in self.episodes:
                if idx < cumsum + len(episode):
                    return episode[idx - cumsum]
                cumsum += len(episode)
        raise IndexError(f"Index {idx} out of range")
    
    @property
    def num_episodes(self) -> int:
        return len(self.episodes)
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.episodes.clear()
            self.total_transitions = 0


# =============================================================================
# Parallel Environment Collector (using gymnasium VectorEnv)
# =============================================================================

class ParallelEnvCollector:
    """
    Parallel environment collector using gymnasium's SyncVectorEnv.
    
    Uses gymnasium's standard VectorEnv API for parallel data collection.
    Each environment maintains its own sequential episode.
    
    Example:
        >>> collector = ParallelEnvCollector(env_fn, num_envs=4)
        >>> episodes = collector.collect_steps(100)  # 100 steps per env
    """
    
    def __init__(
        self,
        env_fn: Callable,
        num_envs: int = 1,
        is_main_process: bool = True,
    ):
        """
        Args:
            env_fn: Function that creates a single environment instance
            num_envs: Number of parallel environments
            is_main_process: If True, print progress (False to suppress on non-main ranks)
        """
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.is_main_process = is_main_process
        
        # VectorEnv and state tracking
        self.vec_env: Optional[SyncVectorEnv] = None
        self.current_obs: Optional[Any] = None
        self.episode_buffers: List[List[Dict]] = [[] for _ in range(num_envs)]
        
        # Keep reference to individual envs for compatibility
        self.envs: List[Any] = []
        
        self._initialized = False
    
    def initialize(self):
        """Initialize the vectorized environment."""
        import sys
        import io
        
        if self._initialized:
            return
        
        if self.is_main_process:
            logger.info(f"Initializing {self.num_envs} parallel environments (SyncVectorEnv)...")
        
        # Suppress verbose environment output on non-main processes
        old_stdout = sys.stdout
        if not self.is_main_process:
            sys.stdout = io.StringIO()
        
        try:
            # Create VectorEnv using gymnasium's SyncVectorEnv
            env_fns = [self.env_fn for _ in range(self.num_envs)]
            self.vec_env = SyncVectorEnv(env_fns)
            
            # Keep references to individual environments
            self.envs = list(self.vec_env.envs)
            
            # Reset and get initial observations
            self.current_obs, _ = self.vec_env.reset()
            self.episode_buffers = [[] for _ in range(self.num_envs)]
        finally:
            sys.stdout = old_stdout
        
        self._initialized = True
        if self.is_main_process:
            logger.info(f"Initialized {self.num_envs} parallel environments")
    
    def collect_steps(
        self,
        num_steps: int,
        action_fn: Optional[Callable] = None,
        action_dim: int = 32,
    ) -> List[List[Dict[str, Any]]]:
        """
        Collect steps from all environments in parallel.
        
        Args:
            num_steps: Number of steps to collect (total across all envs)
            action_fn: Function (obs, env_idx) -> action. If None, random actions.
            action_dim: Action dimension for random actions
            
        Returns:
            List of completed episodes
        """
        if not self._initialized:
            self.initialize()
        
        completed_episodes = []
        steps_per_iter = self.num_envs
        num_iters = (num_steps + steps_per_iter - 1) // steps_per_iter
        
        for _ in range(num_iters):
            # Generate actions for all environments
            if action_fn is not None:
                actions = np.stack([
                    action_fn(self._get_single_obs(i), i) 
                    for i in range(self.num_envs)
                ])
            else:
                actions = np.random.uniform(
                    -1, 1, (self.num_envs, action_dim)
                ).astype(np.float32)
            
            # Step all environments
            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            
            # Process each environment
            for env_idx in range(self.num_envs):
                obs_i = self._get_single_obs(env_idx)
                next_obs_i = self._extract_single_obs(next_obs, env_idx)
                
                # Get action that was actually executed
                action_executed = actions[env_idx]
                if isinstance(infos, dict) and "action_executed" in infos:
                    action_executed = infos["action_executed"][env_idx]
                elif isinstance(infos, (list, tuple)) and env_idx < len(infos):
                    action_executed = infos[env_idx].get("action_executed", actions[env_idx])
                
                # Create transition
                transition = {
                    "obs": copy.deepcopy(obs_i),
                    "action": action_executed.copy() if hasattr(action_executed, 'copy') else action_executed,
                    "next_obs": copy.deepcopy(next_obs_i),
                    "reward": float(rewards[env_idx]),
                    "done": bool(dones[env_idx]),
                    "info": self._extract_single_info(infos, env_idx),
                }
                self.episode_buffers[env_idx].append(transition)
                
                # Check if episode is done
                if dones[env_idx]:
                    completed_episodes.append(self.episode_buffers[env_idx])
                    self.episode_buffers[env_idx] = []
            
            # Update current observation
            self.current_obs = next_obs
        
        return completed_episodes
    
    def _get_single_obs(self, env_idx: int) -> Dict[str, Any]:
        """Extract single environment observation from batched obs."""
        return self._extract_single_obs(self.current_obs, env_idx)
    
    def _extract_single_obs(self, obs: Any, env_idx: int) -> Dict[str, Any]:
        """Extract observation for a single environment."""
        if isinstance(obs, dict):
            return {k: v[env_idx] for k, v in obs.items()}
        elif isinstance(obs, np.ndarray):
            return obs[env_idx]
        else:
            return obs
    
    def _extract_single_info(self, infos: Any, env_idx: int) -> Dict[str, Any]:
        """Extract info for a single environment."""
        if isinstance(infos, dict):
            return {k: v[env_idx] if isinstance(v, (list, np.ndarray)) and len(v) > env_idx else v 
                   for k, v in infos.items()}
        elif isinstance(infos, (list, tuple)) and env_idx < len(infos):
            return infos[env_idx]
        return {}
    
    def collect_episode(
        self,
        env_idx: int = 0,
        action_fn: Optional[Callable] = None,
        action_dim: int = 32,
        max_steps: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Collect one complete episode from the first environment.
        
        Note: With VectorEnv, we collect from all envs and return the first completed.
        
        Args:
            env_idx: Environment index (for compatibility, uses first completed)
            action_fn: Action generation function
            action_dim: Action dimension
            max_steps: Maximum steps per episode
            
        Returns:
            List of transitions (complete episode)
        """
        if not self._initialized:
            self.initialize()
        
        # Collect until we get at least one episode
        total_steps = 0
        while total_steps < max_steps * self.num_envs:
            episodes = self.collect_steps(
                num_steps=self.num_envs,
                action_fn=action_fn,
                action_dim=action_dim,
            )
            if episodes:
                return episodes[0]
            total_steps += self.num_envs
        
        # Return partial episode if no complete episode
        return self.episode_buffers[0] if self.episode_buffers[0] else []
    
    def reset_all(self):
        """Reset all environments."""
        if self._initialized and self.vec_env is not None:
            self.current_obs, _ = self.vec_env.reset()
            self.episode_buffers = [[] for _ in range(self.num_envs)]
    
    def get_current_episode_buffer(self, env_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Get the current episode buffer for a specific environment.
        
        This returns transitions collected so far in the current (incomplete) episode.
        Useful for collecting video frames during training.
        
        Args:
            env_idx: Environment index (default 0 = first env)
            
        Returns:
            List of transitions in the current episode buffer
        """
        if not self._initialized or env_idx >= len(self.episode_buffers):
            return []
        return self.episode_buffers[env_idx]
    
    def close(self):
        """Close all environments."""
        if self._initialized and self.vec_env is not None:
            self.vec_env.close()
            self.vec_env = None
            self.envs = []
        self._initialized = False


# =============================================================================
# Utility Functions
# =============================================================================

def print_rank0(msg: str, accelerator: Optional[AcceleratorWrapper] = None):
    """Print only on main process."""
    if accelerator is None or accelerator.is_main_process:
        print(msg)


def set_random_seed(seed: int, accelerator: Optional[AcceleratorWrapper] = None):
    """Set random seed for reproducibility."""
    if accelerator is not None:
        # Accelerate handles seed setting with process offset
        set_seed(seed + accelerator.process_index)
    else:
        set_seed(seed)


def gather_dict_metrics(
    metrics: Dict[str, float],
    accelerator: AcceleratorWrapper,
) -> Dict[str, float]:
    """
    Gather and average metrics across all processes.
    
    Args:
        metrics: Local metrics dict
        accelerator: AcceleratorWrapper instance
        
    Returns:
        Averaged metrics
    """
    if not accelerator.is_distributed:
        return metrics
    
    gathered = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, device=accelerator.device)
        gathered_tensor = accelerator.gather(tensor)
        gathered[key] = gathered_tensor.mean().item()
    
    return gathered


# =============================================================================
# Trainer Mixin for Accelerate
# =============================================================================

class AccelerateTrainerMixin:
    """
    Mixin class that adds Accelerate support to trainers.
    
    Usage:
        class MyTrainer(AccelerateTrainerMixin):
            def __init__(self, ...):
                self.setup_accelerate()
                self.policy = self.accelerator.prepare_model(self.policy)
                self.optimizer = self.accelerator.prepare(self.optimizer)
    """
    
    accelerator: AcceleratorWrapper
    policy: nn.Module
    
    def setup_accelerate(
        self,
        mixed_precision: str = "no",
        gradient_accumulation_steps: int = 1,
    ):
        """Setup Accelerate for distributed training."""
        self.accelerator = create_accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        # Update device based on accelerator
        self.device = str(self.accelerator.device)
        
        self.accelerator.print(
            f"Accelerate initialized: {self.accelerator.num_processes} processes, "
            f"device={self.device}"
        )
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.accelerator.is_main_process
    
    def wait_for_everyone(self):
        """Synchronization barrier."""
        self.accelerator.wait_for_everyone()
    
    def print(self, *args, **kwargs):
        """Print only on main process."""
        self.accelerator.print(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor):
        """Backward with accelerator."""
        self.accelerator.backward(loss)
    
    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients."""
        params = self.policy.parameters()
        self.accelerator.clip_grad_norm_(params, max_norm)
    
    def unwrap_policy(self) -> nn.Module:
        """Get unwrapped policy for saving."""
        return self.accelerator.unwrap_model(self.policy)
    
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Gather and average metrics across processes."""
        return gather_dict_metrics(metrics, self.accelerator)
