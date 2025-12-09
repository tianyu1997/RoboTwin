#!/usr/bin/env python3
"""
Adversarial Training Script for F1-VLA (Phase 3)

Refactored version matching train_student_rl.py structure and improvements.

Phase 3 Training:
- World Model (WM): Tries to accurately predict the next frame (Teacher Policy in WM mode)
- Explorer (Policy): Tries to find actions that make WM's predictions fail (Student Policy)

The training alternates between:
1. WM update: Minimize prediction error on current explorer's actions
2. Explorer update: Maximize WM's prediction error (find novel actions)
"""

# ============== MUST BE FIRST: Set GPU device for SAPIEN Vulkan rendering ==============
import os
import sys

# Get LOCAL_RANK from environment (set by accelerate/torchrun)
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# Get the physical GPU ID from CUDA_VISIBLE_DEVICES mapping
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if cuda_visible:
    visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    if local_rank < len(visible_gpus):
        physical_gpu_id = visible_gpus[local_rank]
    else:
        physical_gpu_id = local_rank
else:
    physical_gpu_id = local_rank

# Set environment variables for SAPIEN/Vulkan BEFORE any imports
os.environ["VK_DEVICE_INDEX"] = str(physical_gpu_id)
os.environ["SAPIEN_DEVICE_INDEX"] = str(physical_gpu_id)
os.environ["EGL_DEVICE_ID"] = str(physical_gpu_id)

# ============== Setup paths BEFORE importing other modules ==============
script_dir = os.path.dirname(os.path.abspath(__file__))  # rl/training
rl_dir = os.path.dirname(script_dir)                      # rl
robotwin_dir = os.path.dirname(rl_dir)                    # RoboTwin
f1_vla_dir = os.path.dirname(robotwin_dir)                # F1-VLA
sys.path.insert(0, f1_vla_dir)
sys.path.insert(0, robotwin_dir)

# Import log suppression module (must be before any CuRobo imports)
from rl.suppress_logs import suppress_curobo_logs

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import deque
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf

# Import shared utilities
from rl.training.rl_training_common import (
    load_rl_config,
    get_training_config,
    get_environment_config,
    get_lora_config_from_dict,
    load_f1_policy,
    BatchBuilder,
    MemoryStateManager,
    BaseRLTrainer,
    setup_optimizer,
    setup_scheduler,
    clip_gradients,
    count_trainable_params,
    setup_logging_from_config,
    set_policy_requires_grad,
)

# Import parallel training utilities
from rl.training.parallel_utils import (
    AcceleratorWrapper,
    create_accelerator,
    SequentialEpisodeBuffer,
    ParallelEnvCollector,
    print_rank0,
)

# Default logging (will be overridden by config)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Training - Phase 3")
    
    # Config file
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to RL training config YAML file")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Override model config file path")
    
    # Checkpoints (required)
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                       help="Path to teacher checkpoint (acts as World Model)")
    parser.add_argument("--student_checkpoint", type=str, default=None,
                       help="Path to student (explorer) checkpoint from Phase 2")
    
    # Override parameters
    parser.add_argument("--total_iterations", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sequential_training", action="store_true", default=None)
    parser.add_argument("--no_sequential_training", action="store_false",
                       dest="sequential_training")
    
    # Device & debug
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume adversarial training from checkpoint")
    parser.add_argument("--auto_resume", action="store_true", default=False,
                       help="Automatically resume from latest checkpoint in output_dir")
    
    # DDP and multi-environment options
    parser.add_argument("--num_envs", type=int, default=1,
                       help="Number of parallel environments for data collection")
    parser.add_argument("--use_ddp", action="store_true",
                       help="Use distributed data parallel training")
    
    return parser.parse_args()


# =============================================================================
# World Model Wrapper
# =============================================================================

class WorldModelWrapper(nn.Module):
    """Wrapper around F1-VLA world model for adversarial training."""
    
    def __init__(self, f1_policy: nn.Module, policy_config, device: str = "cuda"):
        super().__init__()
        self.f1_policy = f1_policy
        self.device = device
        self.memory_manager = MemoryStateManager()
        
        # Memory configuration from model config (for GRU state)
        self.memory_enabled = policy_config.memory_enabled if hasattr(policy_config, 'memory_enabled') else True
        self.memory_hidden = policy_config.memory_hidden if hasattr(policy_config, 'memory_hidden') else 2048
        self.memory_num_layers = policy_config.memory_num_layers if hasattr(policy_config, 'memory_num_layers') else 4
        
        logger.debug(f"WorldModelWrapper: memory_enabled={self.memory_enabled}, hidden={self.memory_hidden}, layers={self.memory_num_layers}")
        
        # Use proper gradient configuration
        self._configure_for_wm_training()
    
    def _get_unwrapped_policy(self):
        """Get the unwrapped F1_VLA model from PEFT/DDP wrappers."""
        policy = self.f1_policy
        
        # Unwrap PEFT model if present
        if hasattr(policy, 'base_model'):
            # PeftModel -> base_model (LoraModel) -> model (F1_VLA)
            if hasattr(policy.base_model, 'model'):
                return policy.base_model.model
        return policy
    
    def _init_memory_state(self, batch_size: int) -> torch.Tensor:
        """Initialize memory state to zeros for first frame."""
        memory_state = torch.zeros(
            self.memory_num_layers,
            batch_size,
            self.memory_hidden,
            device=self.device,
            dtype=torch.float32
        )
        return memory_state
    
    def _configure_for_wm_training(self):
        """Configure model for world model training only."""
        # Get unwrapped model for proper gradient configuration
        unwrapped = self._get_unwrapped_policy()
        set_policy_requires_grad(
            unwrapped,
            freeze_vision_encoder=True,
            freeze_gen_expert=False,
            train_act_expert_only=False,
            train_gen_expert_only=True,
        )
    
    def unfreeze_wm_params(self):
        """Unfreeze world model parameters for training."""
        self._configure_for_wm_training()
    
    def reset_memory(self):
        """Reset memory state for new episode."""
        self.memory_manager.reset()
    
    def predict_next_frame(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict next frame using world model."""
        # Inject memory state - MUST initialize to zeros for first frame
        if self.memory_manager.current_memory is not None:
            batch["initial_memory_state"] = self.memory_manager.current_memory
        else:
            # First frame: initialize to zeros
            batch_size = batch["observation.state"].shape[0]
            batch["initial_memory_state"] = self._init_memory_state(batch_size)
        
        # Get unwrapped policy and call predict_images_only
        unwrapped = self._get_unwrapped_policy()
        output = unwrapped.predict_images_only(batch)
        
        # Update memory state
        if "memory_state" in output and output["memory_state"] is not None:
            self.memory_manager.update(output["memory_state"])
        
        return output["pred_imgs"]
    
    def compute_prediction_loss(
        self,
        pred_imgs: torch.Tensor,
        gt_imgs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prediction loss."""
        if pred_imgs.shape[-2:] != gt_imgs.shape[-2:]:
            pred_imgs = F.interpolate(pred_imgs, size=gt_imgs.shape[-2:], mode='bilinear', align_corners=False)
        return F.mse_loss(pred_imgs, gt_imgs)


# =============================================================================
# Adversarial Trainer
# =============================================================================

class AdversarialTrainer(BaseRLTrainer):
    """
    Adversarial trainer for WM vs Explorer.
    
    Supports DDP via HuggingFace Accelerate and multi-environment collection.
    """
    
    def __init__(
        self,
        rl_config: OmegaConf,
        env,
        student_policy: nn.Module,
        world_model: WorldModelWrapper,
        policy_config,
        model_config_file: str,
        device: str = "cuda",
        accelerator: Optional[AcceleratorWrapper] = None,
        num_envs: int = 1,
    ):
        # Load model config to get n_obs_img_steps and stride
        import yaml
        with open(model_config_file, 'r') as f:
            model_cfg = yaml.safe_load(f)
        
        train_datasets = model_cfg.get('dataset', {}).get('train_dir', {})
        if not train_datasets:
            raise ValueError("No train datasets found in model config")
        
        first_dataset = next(iter(train_datasets.values()))
        self.n_obs_img_steps = first_dataset.get('n_obs_img_steps', 4)
        self.obs_img_stride = first_dataset.get('obs_img_stride', 1)
        
        # Get configs
        train_config = get_training_config(rl_config)
        adv_config = rl_config.get("adversarial", {})
        output_dir = adv_config.get("output_dir", "outputs/adversarial_rl")
        
        # Override config with values from model config
        self.n_pred_img_steps = train_config.n_pred_img_steps
        train_config.history_length = self.n_obs_img_steps  # observation buffer
        
        super().__init__(
            policy=student_policy,
            config=train_config,
            output_dir=output_dir,
            device=device,
            accelerator=accelerator,
        )
        
        self.rl_config = rl_config
        self.env = env
        self.student_policy = student_policy
        self.world_model = world_model
        self.policy_config = policy_config
        self.num_envs = num_envs
        
        self.cur_n_obs_img_steps = self.n_obs_img_steps
        self.cur_n_pred_img_steps = train_config.n_pred_img_steps
        self.sequential_training = train_config.sequential_training
        
        # Training parameters
        self.wm_updates_per_iter = adv_config.get("wm_updates_per_iter", 5)
        self.explorer_updates_per_iter = adv_config.get("explorer_updates_per_iter", 1)
        self.warmup_steps = adv_config.get("warmup_steps", 1000)
        self.adversarial_weight = adv_config.get("adversarial_weight", 0.5)
        self.entropy_weight = adv_config.get("entropy_weight", 0.01)
        
        # PPO parameters for Explorer
        student_config = rl_config.get("student", {})
        ppo_config = student_config.get("ppo", {})
        self.ppo_config = ppo_config
        self.clip_epsilon = ppo_config.get("clip_epsilon", 0.2)
        self.entropy_coef = ppo_config.get("entropy_coef", 0.02)
        self.value_loss_coef = ppo_config.get("value_loss_coef", 0.5)
        self.gamma = ppo_config.get("gamma", 0.99)
        self.gae_lambda = ppo_config.get("gae_lambda", 0.95)
        
        # Value head for PPO
        proj_width = policy_config.proj_width if hasattr(policy_config, 'proj_width') else 1024
        self.value_head = nn.Linear(proj_width, 1).to(device)
        self.log_std = nn.Parameter(torch.zeros(train_config.action_dim, device=device))
        
        # Batch builder - adversarial uses wrist camera only for VLM (like student)
        self.batch_builder = BatchBuilder(
            device=device,
            image_keys=["head_rgb", "wrist_rgb"],
            use_head_camera=False,
        )
        
        # Optimizers
        # Explorer (Student) Optimizer
        param_groups = [
            {'params': self.student_policy.parameters()},
            {'params': self.value_head.parameters()},
            {'params': [self.log_std]},
        ]
        self.explorer_optimizer = AdamW(
            param_groups,
            lr=adv_config.get("explorer_lr", 3e-4),
            weight_decay=train_config.weight_decay,
        )
        
        # World Model Optimizer
        self.wm_optimizer = AdamW(
            filter(lambda p: p.requires_grad, world_model.parameters()),
            lr=adv_config.get("wm_lr", 1e-4),
            weight_decay=train_config.weight_decay,
        )
        
        # Schedulers
        total_iterations = adv_config.get("total_iterations", 100000)
        self.explorer_scheduler = CosineAnnealingLR(self.explorer_optimizer, T_max=total_iterations)
        self.wm_scheduler = CosineAnnealingLR(self.wm_optimizer, T_max=total_iterations)
        
        # Memory configuration
        self.memory_enabled = policy_config.memory_enabled if hasattr(policy_config, 'memory_enabled') else True
        self.memory_hidden = policy_config.memory_hidden if hasattr(policy_config, 'memory_hidden') else 2048
        self.memory_num_layers = policy_config.memory_num_layers if hasattr(policy_config, 'memory_num_layers') else 4
        
        self.student_memory = MemoryStateManager()
        
        # Multi-env collector
        self.env_collector = None  # Set in setup_multi_env if needed
        
        # Tracking
        self.global_step = 0
        self.episode_count = 0
        
        # Metrics
        self.metrics.update({
            "policy_loss": deque(maxlen=100),
            "value_loss": deque(maxlen=100),
            "entropy": deque(maxlen=100),
            "wm_loss": deque(maxlen=100),
            "adversarial_reward": deque(maxlen=100),
            "episode_reward": deque(maxlen=100),
        })
        
        # Prepare for DDP
        if self.accelerator is not None:
            self.student_policy, self.explorer_optimizer, self.explorer_scheduler = self.accelerator.prepare(
                self.student_policy, self.explorer_optimizer, self.explorer_scheduler
            )
            self.world_model, self.wm_optimizer, self.wm_scheduler = self.accelerator.prepare(
                self.world_model, self.wm_optimizer, self.wm_scheduler
            )
            self._print(f"DDP setup complete: {self.accelerator.num_processes} processes")
    
    def _print(self, msg: str):
        """Print only on main process."""
        if self._is_main_process():
            logger.info(msg)
    
    def _is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process
    
    def setup_multi_env(self, env_fn: Callable):
        """Setup multi-environment collector."""
        if self.num_envs > 1:
            self.env_collector = ParallelEnvCollector(
                env_fn=env_fn,
                num_envs=self.num_envs,
                is_main_process=self._is_main_process(),
            )
            self.env_collector.initialize()
            self._print(f"Parallel env collector setup: {self.num_envs} envs")
    
    def _init_memory_state(self, batch_size: int) -> torch.Tensor:
        """Initialize memory state to zeros for first frame."""
        memory_state = torch.zeros(
            self.memory_num_layers,
            batch_size,
            self.memory_hidden,
            device=self.device,
            dtype=torch.float32
        )
        return memory_state

    def _obs_to_batch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to batch tensor dict."""
        batch = {
            "observation.state": torch.from_numpy(obs["state"]).float().unsqueeze(0).to(self.device),
            "action_history": torch.from_numpy(obs["action_history"]).float().unsqueeze(0).to(self.device),
            "task": ["explore the environment\n"],
        }
        
        # Wrist camera for World Model history (always wrist_rgb -> image0_history)
        if "wrist_rgb" in obs:
            wrist_imgs = obs["wrist_rgb"]
            current_wrist = wrist_imgs[-1]  # Last frame for current observation
            
            # World Model uses wrist_rgb history
            batch["observation.images.image0_history"] = (
                torch.from_numpy(wrist_imgs).float().to(self.device) / 255.0 * 2.0 - 1.0
            ).unsqueeze(0)
            
            # Student uses wrist_rgb as image0 for Paligemma
            batch["observation.images.image0"] = (
                torch.from_numpy(current_wrist).float().to(self.device) / 255.0 * 2.0 - 1.0
            ).unsqueeze(0)
            batch["observation.images.image0_mask"] = torch.ones(1, dtype=torch.bool, device=self.device)
        
        return batch

    def _forward_explorer(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through explorer (student) model.
        """
        # Unwrap DDP model if needed
        policy = self.accelerator.unwrap_model(self.student_policy) if self.accelerator else self.student_policy
        
        # Access the underlying F1FlowMatching model through PEFT wrapper
        if hasattr(policy, 'base_model'):
            if hasattr(policy.base_model, 'model'):
                f1_model = policy.base_model.model
            else:
                f1_model = policy.base_model
        else:
            f1_model = policy
        
        # Use student model to generate actions
        noise = torch.randn(
            batch["observation.state"].shape[0], self.policy_config.chunk_size,
            self.policy_config.max_action_dim,
            device=self.device
        )
        
        # Prepare state with padding
        state = batch["observation.state"]
        if state.shape[-1] < self.policy_config.max_state_dim:
            padding = torch.zeros(
                state.shape[0], self.policy_config.max_state_dim - state.shape[-1],
                device=self.device
            )
            state = torch.cat([state, padding], dim=-1)
        
        # Sample actions from F1FlowMatching model
        image0 = batch.get("observation.images.image0")
        batch_size = image0.shape[0] if image0 is not None else state.shape[0]
        image0_mask = batch.get("observation.images.image0_mask", 
                               torch.ones(batch_size, dtype=torch.bool, device=self.device))
        
        # Inject memory state
        initial_memory_state = batch.get("initial_memory_state")
        
        # Forward pass
        output = f1_model(
            input_ids=None, # Not used for pure vision/state policy
            pixel_values=image0,
            pixel_mask=image0_mask,
            state=state,
            action=None, # We want to sample actions
            noise=noise,
            task=batch.get("task"),
            initial_memory_state=initial_memory_state,
            train_act_expert_only=True,
        )
        
        actions = output.get("actions") # [B, action_dim]
        
        # Compute log_probs
        action_mean = actions
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            sampled_action = action_mean
        else:
            sampled_action = action_mean + action_std * torch.randn_like(action_mean)
            
        log_probs = dist.log_prob(sampled_action).sum(-1)
        
        # Value estimate
        memory_state = output.get("memory_state")
        if memory_state is not None:
            value_input = memory_state[-1]
            values = self.value_head(value_input).squeeze(-1)
        else:
            values = torch.zeros(batch_size, device=self.device)
            
        return sampled_action, log_probs, values, memory_state

    def collect_episode(self, use_tqdm: bool = False) -> List[Dict[str, Any]]:
        """Collect one episode using explorer policy."""
        obs, info = self.env.reset()
        transitions = []
        
        # Reset memory states
        self.student_memory.reset()
        self.world_model.reset_memory()
        
        done = False
        step = 0
        
        while not done:
            # Build batch for explorer
            batch = self._obs_to_batch(obs)
            
            # Inject student memory state
            if self.student_memory.current_memory is not None:
                batch["initial_memory_state"] = self.student_memory.current_memory
            else:
                batch["initial_memory_state"] = self._init_memory_state(1)
            
            # Get action from explorer
            with torch.no_grad():
                actions, log_probs, values, student_memory_out = self._forward_explorer(batch)
            
            # Update student memory
            if student_memory_out is not None:
                self.student_memory.update(student_memory_out)
            
            action = actions[0].cpu().numpy()
            log_prob = log_probs[0].item()
            value = values[0].item()
            
            # Execute action
            next_obs, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Compute Adversarial Reward
            wm_batch = batch.copy()
            with torch.no_grad():
                pred_imgs = self.world_model.predict_next_frame(wm_batch)
            
            # Get ground truth next frame (wrist_rgb)
            if "wrist_rgb" in next_obs:
                gt_img = next_obs["wrist_rgb"][-1] # [C, H, W]
                gt_tensor = torch.from_numpy(gt_img).float().to(self.device).unsqueeze(0) / 255.0 * 2.0 - 1.0
                
                # Compute MSE
                mse = F.mse_loss(pred_imgs, gt_tensor)
                
                # Reward: Maximize MSE -> Reward = MSE
                reward = mse.item() * self.adversarial_weight
            else:
                reward = 0.0
            
            transitions.append({
                "obs": obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "info": {**info, "adversarial_reward": reward},
            })
            
            obs = next_obs
            step += 1
        
        # Compute advantages
        self._compute_advantages(transitions)
        
        return transitions

    def _compute_advantages(self, transitions: List[Dict[str, Any]]):
        """Compute GAE advantages for PPO."""
        rewards = [t["reward"] for t in transitions]
        values = [t["value"] for t in transitions]
        dones = [t["done"] for t in transitions]
        
        next_value = 0.0 if dones[-1] else values[-1]
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(transitions))):
            next_val = next_value if i == len(transitions) - 1 else values[i + 1]
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        for i, t in enumerate(transitions):
            t["advantage"] = advantages[i]
            t["return"] = advantages[i] + values[i]

    def train_step_explorer(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one PPO training step for Explorer."""
        batch_size = batch["observation.state"].shape[0]
        mini_batch_size = self.config.mini_batch_size
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_mini_batches = 0
        
        indices = torch.randperm(batch_size)
        
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            mb_indices = indices[start:end]
            
            mb_batch = {
                "observation.state": batch["observation.state"][mb_indices],
                "action_history": batch["action_history"][mb_indices],
                "task": [batch["task"][i] for i in mb_indices.tolist()],
            }
            if "observation.images.image0_history" in batch:
                mb_batch["observation.images.image0_history"] = batch["observation.images.image0_history"][mb_indices]
                mb_batch["observation.images.image0"] = batch["observation.images.image0"][mb_indices]
                mb_batch["observation.images.image0_mask"] = batch["observation.images.image0_mask"][mb_indices]
            
            self.explorer_optimizer.zero_grad()
            
            actions, log_probs, values, _ = self._forward_explorer(mb_batch)
            
            mb_old_log_probs = batch["old_log_probs"][mb_indices]
            mb_advantages = batch["advantages"][mb_indices]
            mb_returns = batch["returns"][mb_indices]
            
            # Normalize advantages
            if mb_advantages.std() > 1e-8:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Policy loss
            ratio = torch.exp(log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, mb_returns)
            
            # Entropy
            std = torch.exp(self.log_std)
            entropy = 0.5 * (1 + torch.log(2 * np.pi * std**2)).sum()
            
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            loss.backward()
            clip_gradients(self.student_policy, max_norm=self.config.max_grad_norm)
            self.explorer_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_mini_batches += 1
            
        return {
            "policy_loss": total_policy_loss / num_mini_batches,
            "value_loss": total_value_loss / num_mini_batches,
            "entropy": total_entropy / num_mini_batches,
        }

    def train_step_wm(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step for World Model."""
        self.wm_optimizer.zero_grad()
        
        target_imgs = batch.get("target_images")
        if target_imgs is None:
            return {"wm_loss": 0.0}
            
        pred_imgs = self.world_model.predict_next_frame(batch)
        loss = self.world_model.compute_prediction_loss(pred_imgs, target_imgs)
        
        loss.backward()
        self.wm_optimizer.step()
        
        return {"wm_loss": loss.item()}

    def _build_ppo_batch(self, transitions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert list of transitions to PPO batch."""
        batch_size = len(transitions)
        
        states = torch.stack([torch.from_numpy(t["obs"]["state"]).float() for t in transitions]).to(self.device)
        action_histories = torch.stack([torch.from_numpy(t["obs"]["action_history"]).float() for t in transitions]).to(self.device)
        
        batch = {
            "observation.state": states,
            "action_history": action_histories,
            "task": ["explore the environment\n"] * batch_size,
            "old_log_probs": torch.tensor([t["log_prob"] for t in transitions], device=self.device),
            "advantages": torch.tensor([t["advantage"] for t in transitions], device=self.device),
            "returns": torch.tensor([t["return"] for t in transitions], device=self.device),
        }
        
        if "wrist_rgb" in transitions[0]["obs"]:
            # History
            histories = [torch.from_numpy(t["obs"]["wrist_rgb"]).float() for t in transitions]
            batch["observation.images.image0_history"] = torch.stack(histories).to(self.device) / 255.0 * 2.0 - 1.0
            
            # Current
            currents = [torch.from_numpy(t["obs"]["wrist_rgb"][-1]).float() for t in transitions]
            batch["observation.images.image0"] = torch.stack(currents).to(self.device) / 255.0 * 2.0 - 1.0
            
            batch["observation.images.image0_mask"] = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            
            # Target images for WM (next_obs)
            targets = [torch.from_numpy(t["next_obs"]["wrist_rgb"][-1]).float() for t in transitions]
            batch["target_images"] = torch.stack(targets).to(self.device) / 255.0 * 2.0 - 1.0
            
        return batch

    def train(self, total_iterations: int, start_iteration: int = 0):
        """Main training loop."""
        logger.info(f"Starting adversarial training for {total_iterations} iterations")
        
        # Setup environment
        if self.num_envs > 1:
            def make_env():
                from rl.f1_rl_env import F1RLEnv
                return F1RLEnv(
                    task_config=self.env_config,
                    phase="student", # Use student phase for explorer
                    teacher_policy=self.accelerator.unwrap_model(self.world_model.f1_policy), # Pass teacher policy
                    history_length=self.n_obs_img_steps,
                    max_steps=self.config.steps_per_episode,
                    device=self.device,
                )
            self.setup_multi_env(make_env)
        
        iterator = range(start_iteration, total_iterations)
        if self._is_main_process():
            iterator = tqdm(iterator, desc="Adversarial Iterations")
            
        for iteration in iterator:
            # 1. Collect Data (Explorer interacts with Env)
            transitions = self.collect_episode()
            batch = self._build_ppo_batch(transitions)
            
            # 2. Update World Model
            wm_metrics = {}
            for _ in range(self.wm_updates_per_iter):
                m = self.train_step_wm(batch)
                wm_metrics.update(m)
            
            # 3. Update Explorer
            exp_metrics = {}
            for _ in range(self.explorer_updates_per_iter):
                m = self.train_step_explorer(batch)
                exp_metrics.update(m)
            
            # Log metrics
            if self._is_main_process():
                self.metrics["wm_loss"].append(wm_metrics.get("wm_loss", 0))
                self.metrics["policy_loss"].append(exp_metrics.get("policy_loss", 0))
                self.metrics["episode_reward"].append(np.sum([t["reward"] for t in transitions]))
                
                if iteration % self.config.log_every == 0:
                    logger.info(f"Iter {iteration}: WM Loss={np.mean(self.metrics['wm_loss']):.4f}, "
                                f"Exp Reward={np.mean(self.metrics['episode_reward']):.4f}")


def main():
    args = parse_args()
    
    # Load RL config
    rl_config = load_rl_config(args.rl_config)
    
    # Setup logging
    setup_logging_from_config(rl_config)
    
    # DDP setup
    accelerator = None
    if args.use_ddp:
        accelerator = create_accelerator(mixed_precision="no")
        device = str(accelerator.device)
    else:
        device = rl_config.get("device", "cuda:0")
        if device == "cuda": device = "cuda:0"
    
    debug = rl_config.get("debug", False)
    model_config_file = rl_config.get("model", {}).get("config_file", "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml")
    
    # Load Teacher (World Model)
    lora_config = get_lora_config_from_dict(rl_config)
    teacher_lora_config = copy.deepcopy(lora_config)
    teacher_lora_config.target_modules = ["q_proj", "v_proj"]
    
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"Loading teacher (WM) from: {args.teacher_checkpoint}")
        
    teacher_policy, policy_config, model_config = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=teacher_lora_config,
        checkpoint_path=args.teacher_checkpoint,
    )
    
    # Load Student (Explorer)
    # Use full LoRA config for explorer
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"Loading student (Explorer) from: {args.student_checkpoint}")
        
    student_policy, _, _ = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=lora_config,
        checkpoint_path=args.student_checkpoint, # Load from Phase 2 checkpoint
    )
    
    # Wrap WM
    world_model = WorldModelWrapper(teacher_policy, policy_config, device)
    world_model.unfreeze_wm_params()
    
    # Create Env
    from rl.f1_rl_env import F1RLEnv
    
    env_config = get_environment_config(rl_config)
    
    # Get GPU ID
    local_gpu_id = 0
    if accelerator is not None:
        local_gpu_id = accelerator.local_process_index
        os.environ["EGL_DEVICE_ID"] = str(local_gpu_id)
    
    def make_env():
        return F1RLEnv(
            task_config={**env_config, "render_device": local_gpu_id},
            phase="student", # Student phase implies wrist camera usage
            teacher_policy=None, # No teacher needed for env logic if we compute reward manually
            history_length=4, # Should match config
            max_steps=rl_config.training.steps_per_episode,
            device=device,
        )
    
    env = make_env()
    
    # Trainer
    trainer = AdversarialTrainer(
        rl_config=rl_config,
        env=env,
        student_policy=student_policy,
        world_model=world_model,
        policy_config=policy_config,
        model_config_file=model_config_file,
        device=device,
        accelerator=accelerator,
        num_envs=args.num_envs,
    )
    
    # Train
    total_iterations = rl_config.get("adversarial", {}).get("total_iterations", 10000)
    trainer.train(total_iterations)

if __name__ == "__main__":
    main()
