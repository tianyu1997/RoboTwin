#!/usr/bin/env python3
"""
Adversarial Training Script for F1-VLA (Phase 3)

Refactored version using shared rl_training_common module.

Phase 3 Training:
- World Model (WM): Tries to accurately predict the next frame
- Explorer (Policy): Tries to find actions that make WM's predictions fail

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
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
    clip_gradients,
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
                       help="Path to teacher checkpoint")
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
    
    # DDP and multi-environment options
    parser.add_argument("--num_envs", type=int, default=1,
                       help="Number of parallel environments for data collection")
    parser.add_argument("--use_ddp", action="store_true",
                       help="Use distributed data parallel training")
    
    return parser.parse_args()


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Replay buffer with memory state support for adversarial training."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(
        self,
        obs_before: Dict[str, np.ndarray],
        action: np.ndarray,
        obs_after: Dict[str, np.ndarray],
        state_before: np.ndarray,
        state_after: np.ndarray,
        memory_state: Optional[torch.Tensor] = None,
    ):
        """Store a transition with optional memory state."""
        data = {
            "obs_before": {k: v.copy() for k, v in obs_before.items()},
            "action": action.copy(),
            "obs_after": {k: v.copy() for k, v in obs_after.items()},
            "state_before": state_before.copy(),
            "state_after": state_after.copy(),
            "memory_state": memory_state,
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Explorer Policy Network
# =============================================================================

class ExplorerPolicy(nn.Module):
    """Explorer policy network for adversarial training."""
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 32,
        hidden_dim: int = 256,
        image_encoder_dim: int = 512,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Image encoder (simple CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, image_encoder_dim),
            nn.ReLU(),
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(image_encoder_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Action mean and std
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        img_features = self.image_encoder(image)
        x = torch.cat([img_features, state], dim=-1)
        x = self.policy_net(x)
        
        action_mean = torch.tanh(self.action_mean(x))
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        
        return action_mean, action_std
    
    def sample_action(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, action_std = self.forward(state, image)
        
        if deterministic:
            return action_mean, torch.zeros_like(action_mean)
        
        noise = torch.randn_like(action_mean) * action_std
        action = torch.clamp(action_mean + noise, -1.0, 1.0)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob


# =============================================================================
# World Model Wrapper
# =============================================================================

class WorldModelWrapper(nn.Module):
    """Wrapper around F1-VLA world model for adversarial training."""
    
    def __init__(self, f1_policy: nn.Module, device: str = "cuda"):
        super().__init__()
        self.f1_policy = f1_policy
        self.device = device
        self.memory_manager = MemoryStateManager()
        
        # Use proper gradient configuration
        self._configure_for_wm_training()
    
    def _configure_for_wm_training(self):
        """Configure model for world model training only."""
        # Use set_requires_grad for proper gradient configuration
        set_policy_requires_grad(
            self.f1_policy,
            freeze_vision_encoder=True,   # Freeze vision encoder
            freeze_gen_expert=False,      # Unfreeze world model
            train_act_expert_only=False,
            train_gen_expert_only=True,   # Only train world model
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
        # Inject memory state
        if self.memory_manager.current_memory is not None:
            batch["initial_memory_state"] = self.memory_manager.current_memory
        
        output = self.f1_policy.predict_images_only(batch)
        
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
            gt_imgs = F.interpolate(gt_imgs, size=pred_imgs.shape[-2:], mode="bilinear")
        return F.mse_loss(pred_imgs, gt_imgs)


# =============================================================================
# Adversarial Trainer
# =============================================================================

class AdversarialTrainer:
    """
    Adversarial trainer for WM vs Explorer.
    
    Supports DDP via HuggingFace Accelerate and multi-environment collection
    via gymnasium's SyncVectorEnv.
    """
    
    def __init__(
        self,
        rl_config: OmegaConf,
        env,
        explorer: ExplorerPolicy,
        world_model: WorldModelWrapper,
        model_config_file: str,
        device: str = "cuda",
        accelerator: Optional[AcceleratorWrapper] = None,
        num_envs: int = 1,
    ):
        self.rl_config = rl_config
        self.env = env
        self.explorer = explorer
        self.world_model = world_model
        self.device = device
        self.accelerator = accelerator
        self.num_envs = num_envs
        
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
        
        self.cur_n_obs_img_steps = self.n_obs_img_steps
        self.cur_n_pred_img_steps = train_config.n_pred_img_steps
        self.sequential_training = train_config.sequential_training
        
        # history_length is the observation buffer size
        self.history_length = self.n_obs_img_steps
        
        print(f"Adversarial training config from {model_config_file}:")
        print(f"  n_obs_img_steps: {self.n_obs_img_steps}")
        print(f"  n_pred_img_steps: {self.cur_n_pred_img_steps}")
        print(f"  history_length: {self.history_length}")
        
        # Training parameters
        self.batch_size = train_config.batch_size
        self.wm_updates_per_iter = adv_config.get("wm_updates_per_iter", 5)
        self.explorer_updates_per_iter = adv_config.get("explorer_updates_per_iter", 1)
        self.warmup_steps = adv_config.get("warmup_steps", 1000)
        self.adversarial_weight = adv_config.get("adversarial_weight", 0.5)
        self.entropy_weight = adv_config.get("entropy_weight", 0.01)
        
        # Batch builder - adversarial uses wrist camera only for VLM (like student)
        self.batch_builder = BatchBuilder(
            device=device,
            image_keys=["head_rgb", "wrist_rgb"],
            use_head_camera=False,  # Adversarial: wrist_rgb only (image0) for VLM
        )
        
        # Optimizers
        self.explorer_optimizer = AdamW(
            explorer.parameters(),
            lr=adv_config.get("explorer_lr", 3e-4),
            weight_decay=train_config.weight_decay,
        )
        self.wm_optimizer = AdamW(
            filter(lambda p: p.requires_grad, world_model.parameters()),
            lr=adv_config.get("wm_lr", 1e-4),
            weight_decay=train_config.weight_decay,
        )
        
        # Schedulers
        total_iterations = adv_config.get("total_iterations", 100000)
        self.explorer_scheduler = CosineAnnealingLR(self.explorer_optimizer, T_max=total_iterations)
        self.wm_scheduler = CosineAnnealingLR(self.wm_optimizer, T_max=total_iterations)
        
        # Prepare for DDP if accelerator is provided
        if self.accelerator is not None:
            # Prepare explorer and its optimizer
            self.explorer, self.explorer_optimizer, self.explorer_scheduler = self.accelerator.prepare(
                self.explorer, self.explorer_optimizer, self.explorer_scheduler
            )
            # Prepare WM and its optimizer
            self.world_model, self.wm_optimizer, self.wm_scheduler = self.accelerator.prepare(
                self.world_model, self.wm_optimizer, self.wm_scheduler
            )
            self._print(f"DDP setup complete: {self.accelerator.num_processes} processes")
        
        # Replay buffer (uses SequentialEpisodeBuffer for parallel collection)
        if self.num_envs > 1:
            self.replay_buffer = SequentialEpisodeBuffer(
                max_episodes=1000, max_transitions=100000
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=adv_config.get("buffer_size", 100000))
        
        # Memory manager for episode collection
        self.memory_manager = MemoryStateManager()
        
        # Multi-env collector
        self.env_collector = None  # Set in setup_multi_env if needed
        
        # Logging (only on main process)
        self.output_dir = Path(adv_config.get("output_dir", "outputs/adversarial_rl"))
        if self._is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
        
        # Tracking
        self.global_step = 0
        self.episode_count = 0
    
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
            )
            self.env_collector.initialize()
            self._print(f"Multi-env collector setup: {self.num_envs} envs")
        
        # Tracking
        self.global_step = 0
        self.episode_count = 0
    
    def collect_trajectory(self, num_steps: int = 50) -> Dict[str, float]:
        """Collect trajectory using current explorer policy."""
        obs, info = self.env.reset()
        
        # Reset memory states
        self.memory_manager.reset()
        self.world_model.reset_memory()
        
        episode_wm_loss = 0.0
        episode_steps = 0
        
        for _ in range(num_steps):
            # Get state and image (use wrist_rgb for explorer input)
            state = torch.from_numpy(obs["state"]).float().to(self.device).unsqueeze(0)
            
            # Get wrist camera image for explorer (VLM uses wrist only in adversarial phase)
            wrist_key = "wrist_rgb" if "wrist_rgb" in obs else "left_wrist_rgb"
            if wrist_key in obs:
                wrist_img = obs[wrist_key]
                # Handle both single frame and history
                if wrist_img.ndim == 4:
                    wrist_img = wrist_img[-1]  # Take last frame
                image = torch.from_numpy(wrist_img).float().to(self.device).unsqueeze(0)
                image = image / 255.0 * 2.0 - 1.0
            else:
                image = torch.zeros(1, 3, 224, 224, device=self.device)
            
            # Sample action from explorer
            with torch.no_grad():
                action, _ = self.explorer.sample_action(state, image)
                action_np = action.cpu().numpy().flatten()
            
            # Store observation before action (store both cameras for WM)
            raw_obs = self.env.task.get_observation() if hasattr(self.env.task, 'get_observation') else obs
            obs_before = {
                "head_rgb": raw_obs.get("head_rgb", np.zeros((3, 224, 224), dtype=np.uint8)),
                "wrist_rgb": raw_obs.get("wrist_rgb", raw_obs.get("left_wrist_rgb", np.zeros((3, 224, 224), dtype=np.uint8))),
                "wrist_rgb_history": obs.get("wrist_rgb", obs.get("left_wrist_rgb")),  # Full history
            }
            state_before = obs["state"].copy()
            memory_state_before = (
                self.memory_manager.current_memory.detach().clone()
                if self.memory_manager.current_memory is not None else None
            )
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Store observation after action
            raw_obs_after = self.env.task.get_observation() if hasattr(self.env.task, 'get_observation') else next_obs
            obs_after = {
                "head_rgb": raw_obs_after.get("head_rgb", np.zeros((3, 224, 224), dtype=np.uint8)),
                "wrist_rgb": raw_obs_after.get("wrist_rgb", raw_obs_after.get("left_wrist_rgb", np.zeros((3, 224, 224), dtype=np.uint8))),
            }
            state_after = next_obs["state"].copy()
            
            # Store in replay buffer
            self.replay_buffer.push(
                obs_before, action_np, obs_after, state_before, state_after,
                memory_state=memory_state_before,
            )
            
            episode_wm_loss += info.get("wm_uncertainty", 0.0)
            episode_steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        self.episode_count += 1
        
        return {
            "episode_steps": episode_steps,
            "episode_wm_loss": episode_wm_loss / max(episode_steps, 1),
        }
    
    def update_world_model(self) -> Dict[str, float]:
        """Update world model to minimize prediction error."""
        if len(self.replay_buffer) < self.batch_size:
            return {"wm_loss": 0.0}
        
        total_loss = 0.0
        
        for _ in range(self.wm_updates_per_iter):
            batch_data = self.replay_buffer.sample(self.batch_size)
            
            # Helper to process image: handle CHW (from env) or HWC formats
            def process_image(img):
                """Convert image to tensor, handling both CHW and HWC formats."""
                if img is None:
                    return torch.zeros(3, 224, 224, device=self.device)
                img_t = torch.from_numpy(img).float().to(self.device)
                # If HWC format, convert to CHW
                if img_t.ndim == 3 and img_t.shape[-1] == 3:
                    img_t = img_t.permute(2, 0, 1)
                return img_t / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
            
            # Use wrist_rgb for both VLM input and WM (adversarial phase)
            # Current wrist image for VLM (image0)
            obs_before_imgs = torch.stack([
                process_image(d["obs_before"].get("wrist_rgb"))
                for d in batch_data
            ])
            
            # Ground truth next frame
            obs_after_imgs = torch.stack([
                process_image(d["obs_after"].get("wrist_rgb"))
                for d in batch_data
            ])
            
            # Image history for world model (if available)
            wrist_histories = []
            for d in batch_data:
                hist = d["obs_before"].get("wrist_rgb_history")
                if hist is not None and hist.ndim == 4:  # [T, C, H, W]
                    hist_t = torch.from_numpy(hist).float().to(self.device) / 255.0 * 2.0 - 1.0
                    wrist_histories.append(hist_t)
                else:
                    # Use current image repeated if no history
                    wrist_histories.append(obs_before_imgs[len(wrist_histories)].unsqueeze(0).expand(self.n_obs_img_steps, -1, -1, -1))
            
            actions = torch.stack([
                torch.from_numpy(d["action"]).float()
                for d in batch_data
            ]).to(self.device)
            
            states = torch.stack([
                torch.from_numpy(d["state_before"]).float()
                for d in batch_data
            ]).to(self.device)
            
            # Memory states
            memory_states = [d.get("memory_state") for d in batch_data]
            
            # Build batch with proper keys
            batch = {
                "observation.images.image0": obs_before_imgs,  # Current wrist for VLM
                "observation.images.image0_mask": torch.ones(len(batch_data), dtype=torch.bool, device=self.device),
                "observation.images.image0_history": torch.stack(wrist_histories),  # Wrist history for WM
                "observation.state": states,
                "action": actions,
                "task": ["explore the environment\n"] * len(batch_data),
            }
            
            # Inject memory states if available
            if valid_states := [ms for ms in memory_states if ms is not None]:
                if len(valid_states) == len(batch_data):
                    batch["initial_memory_state"] = torch.stack(valid_states)
            
            # Forward pass
            self.wm_optimizer.zero_grad()
            pred_imgs = self.world_model.predict_next_frame(batch)
            
            # Compute loss
            loss = self.world_model.compute_prediction_loss(pred_imgs, obs_after_imgs)
            
            # Backward and update
            loss.backward()
            clip_gradients(self.world_model, max_norm=1.0)
            self.wm_optimizer.step()
            
            total_loss += loss.item()
        
        self.wm_scheduler.step()
        
        return {"wm_loss": total_loss / self.wm_updates_per_iter}
    
    def update_explorer(self) -> Dict[str, float]:
        """Update explorer to maximize WM prediction error."""
        if len(self.replay_buffer) < self.batch_size:
            return {"explorer_loss": 0.0}
        
        total_loss = 0.0
        total_adv_reward = 0.0
        total_entropy = 0.0
        
        for _ in range(self.explorer_updates_per_iter):
            batch_data = self.replay_buffer.sample(self.batch_size)
            
            # Helper to process image
            def process_image(img):
                """Convert image to tensor, handling both CHW and HWC formats."""
                if img is None:
                    return torch.zeros(3, 224, 224, device=self.device)
                img_t = torch.from_numpy(img).float().to(self.device)
                if img_t.ndim == 3 and img_t.shape[-1] == 3:
                    img_t = img_t.permute(2, 0, 1)
                return img_t / 255.0 * 2.0 - 1.0
            
            # Use wrist_rgb for explorer (adversarial phase)
            obs_before_imgs = torch.stack([
                process_image(d["obs_before"].get("wrist_rgb"))
                for d in batch_data
            ])
            
            obs_after_imgs = torch.stack([
                process_image(d["obs_after"].get("wrist_rgb"))
                for d in batch_data
            ])
            
            # Image history for world model
            wrist_histories = []
            for d in batch_data:
                hist = d["obs_before"].get("wrist_rgb_history")
                if hist is not None and hist.ndim == 4:
                    hist_t = torch.from_numpy(hist).float().to(self.device) / 255.0 * 2.0 - 1.0
                    wrist_histories.append(hist_t)
                else:
                    wrist_histories.append(obs_before_imgs[len(wrist_histories)].unsqueeze(0).expand(self.n_obs_img_steps, -1, -1, -1))
            
            states = torch.stack([
                torch.from_numpy(d["state_before"]).float()
                for d in batch_data
            ]).to(self.device)
            
            # Sample actions from explorer
            self.explorer_optimizer.zero_grad()
            action_mean, action_std = self.explorer(states, obs_before_imgs)
            
            noise = torch.randn_like(action_mean) * action_std
            actions = torch.clamp(action_mean + noise, -1.0, 1.0)
            
            # Compute entropy bonus
            dist = torch.distributions.Normal(action_mean, action_std)
            entropy = dist.entropy().mean()
            
            # Build batch for WM with proper keys
            batch = {
                "observation.images.image0": obs_before_imgs,
                "observation.images.image0_mask": torch.ones(len(batch_data), dtype=torch.bool, device=self.device),
                "observation.images.image0_history": torch.stack(wrist_histories),
                "observation.state": states,
                "action": actions,
                "task": ["explore the environment\n"] * len(batch_data),
            }
            
            # Get WM prediction
            with torch.no_grad():
                pred_imgs = self.world_model.predict_next_frame(batch)
            
            # Adversarial reward
            wm_loss = self.world_model.compute_prediction_loss(pred_imgs, obs_after_imgs)
            adversarial_reward = wm_loss.detach()
            
            # Explorer loss
            explorer_loss = -self.adversarial_weight * adversarial_reward - self.entropy_weight * entropy
            
            # Backward and update
            explorer_loss.backward()
            clip_gradients(self.explorer, max_norm=1.0)
            self.explorer_optimizer.step()
            
            total_loss += explorer_loss.item()
            total_adv_reward += adversarial_reward.item()
            total_entropy += entropy.item()
        
        self.explorer_scheduler.step()
        
        return {
            "explorer_loss": total_loss / self.explorer_updates_per_iter,
            "adversarial_reward": total_adv_reward / self.explorer_updates_per_iter,
            "entropy": total_entropy / self.explorer_updates_per_iter,
        }
    
    def train(self, total_iterations: int, start_iteration: int = 0):
        """Main training loop."""
        logger.info(f"Starting adversarial training for {total_iterations} iterations")
        logger.info(f"Sequential training: {self.sequential_training}")
        
        for iteration in range(start_iteration, total_iterations):
            self.global_step += 1
            
            # Collect trajectory
            collect_stats = self.collect_trajectory()
            
            # Warmup
            if self.global_step < self.warmup_steps:
                if self.global_step % 100 == 0:
                    logger.info(f"Warmup step {self.global_step}/{self.warmup_steps}, "
                              f"buffer size: {len(self.replay_buffer)}")
                continue
            
            # Update world model
            wm_stats = self.update_world_model()
            
            # Update explorer
            explorer_stats = self.update_explorer()
            
            # Logging
            if iteration % 100 == 0:
                logger.info(
                    f"Iter {iteration}: "
                    f"WM Loss: {wm_stats['wm_loss']:.4f}, "
                    f"Explorer Loss: {explorer_stats['explorer_loss']:.4f}, "
                    f"Adv Reward: {explorer_stats['adversarial_reward']:.4f}"
                )
                
                self.writer.add_scalar("train/wm_loss", wm_stats["wm_loss"], iteration)
                self.writer.add_scalar("train/explorer_loss", explorer_stats["explorer_loss"], iteration)
                self.writer.add_scalar("train/adversarial_reward", explorer_stats["adversarial_reward"], iteration)
                self.writer.add_scalar("train/entropy", explorer_stats["entropy"], iteration)
            
            # Save checkpoint
            if iteration % 5000 == 0 and iteration > 0:
                self.save_checkpoint(iteration)
        
        self.save_checkpoint(total_iterations)
        logger.info("Training complete!")
    
    def save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.explorer.state_dict(), checkpoint_dir / "explorer.pt")
        
        wm_state_dict = {
            k: v for k, v in self.world_model.state_dict().items()
            if "world_model" in k or "wm_" in k
        }
        torch.save(wm_state_dict, checkpoint_dir / "world_model.pt")
        
        torch.save({
            "explorer_optimizer": self.explorer_optimizer.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "explorer_scheduler": self.explorer_scheduler.state_dict(),
            "wm_scheduler": self.wm_scheduler.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "iteration": iteration,
        }, checkpoint_dir / "optimizer.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        
        explorer_path = checkpoint_path / "explorer.pt"
        if explorer_path.exists():
            self.explorer.load_state_dict(torch.load(explorer_path))
        
        wm_path = checkpoint_path / "world_model.pt"
        if wm_path.exists():
            self.world_model.load_state_dict(torch.load(wm_path), strict=False)
        
        opt_path = checkpoint_path / "optimizer.pt"
        if opt_path.exists():
            checkpoint = torch.load(opt_path)
            self.explorer_optimizer.load_state_dict(checkpoint["explorer_optimizer"])
            self.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
            self.explorer_scheduler.load_state_dict(checkpoint["explorer_scheduler"])
            self.wm_scheduler.load_state_dict(checkpoint["wm_scheduler"])
            self.global_step = checkpoint["global_step"]
            self.episode_count = checkpoint["episode_count"]
            return checkpoint.get("iteration", 0)
        
        return 0


def main():
    args = parse_args()
    
    # Load RL config
    rl_config = load_rl_config(args.rl_config)
    
    # Setup logging from config (must be done early)
    setup_logging_from_config(rl_config)
    
    # Apply overrides
    if args.model_config:
        rl_config.model.config_file = args.model_config
    if args.total_iterations is not None:
        rl_config.adversarial.total_iterations = args.total_iterations
    if args.output_dir is not None:
        rl_config.adversarial.output_dir = args.output_dir
    if args.sequential_training is not None:
        rl_config.training.sequential_training = args.sequential_training
    if args.device is not None:
        rl_config.device = args.device
    if args.debug is not None:
        rl_config.debug = args.debug
    
    # Create accelerator for DDP if requested
    accelerator = None
    if args.use_ddp:
        accelerator = create_accelerator(mixed_precision="no")
        device = str(accelerator.device)
        accelerator.print("=" * 70)
        accelerator.print("Adversarial Training (Phase 3) - DDP Mode")
        accelerator.print(f"Number of processes: {accelerator.num_processes}")
        accelerator.print("=" * 70)
    else:
        device = rl_config.get("device", "cuda")
        logger.info("=" * 70)
        logger.info("Adversarial Training (Phase 3)")
        logger.info("=" * 70)
    
    debug = rl_config.get("debug", False)
    
    # Load model config
    model_config_file = rl_config.get("model", {}).get(
        "config_file",
        "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml"
    )
    
    # Get LoRA config
    lora_config = get_lora_config_from_dict(rl_config)
    
    # Load teacher policy
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"Loading teacher policy from: {args.teacher_checkpoint}")
    teacher_policy, policy_config, model_config = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=lora_config,
        checkpoint_path=args.teacher_checkpoint,
    )
    
    # Create environment
    from rl.f1_rl_env import F1RLEnv
    
    env_config = get_environment_config(rl_config)
    train_config = get_training_config(rl_config)
    
    # Get GPU ID for render_device
    local_gpu_id = 0
    if accelerator is not None:
        local_process_idx = accelerator.local_process_index
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible:
            visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
            if local_process_idx < len(visible_gpus):
                local_gpu_id = visible_gpus[local_process_idx]
            else:
                local_gpu_id = local_process_idx
        else:
            local_gpu_id = local_process_idx
    
    def make_env():
        return F1RLEnv(
            task_config={
                **env_config,
                "render_device": local_gpu_id,
            },
            phase="student",
            teacher_policy=teacher_policy,
            device=device,
            max_steps=train_config.steps_per_episode,
        )
    
    env = make_env()
    
    # Create explorer
    adv_config = rl_config.get("adversarial", {})
    explorer_config = adv_config.get("explorer", {})
    
    explorer = ExplorerPolicy(
        state_dim=train_config.state_dim,
        action_dim=train_config.action_dim,
        hidden_dim=explorer_config.get("hidden_dim", 256),
        image_encoder_dim=explorer_config.get("image_encoder_dim", 512),
    ).to(device)
    
    # Load student checkpoint if provided
    if args.student_checkpoint:
        if accelerator is None or accelerator.is_main_process:
            logger.info(f"Loading student explorer from: {args.student_checkpoint}")
        explorer_path = Path(args.student_checkpoint) / "explorer.pt"
        if explorer_path.exists():
            explorer.load_state_dict(torch.load(explorer_path, map_location=device))
            if accelerator is None or accelerator.is_main_process:
                logger.info("Student explorer loaded successfully")
    
    # Wrap world model
    world_model = WorldModelWrapper(teacher_policy, device)
    world_model.unfreeze_wm_params()
    
    # Sync before creating trainer
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    # Create trainer with accelerator and num_envs
    trainer = AdversarialTrainer(
        rl_config=rl_config,
        env=env,
        explorer=explorer,
        world_model=world_model,
        model_config_file=model_config_file,
        device=device,
        accelerator=accelerator,
        num_envs=args.num_envs,
    )
    
    # Setup multi-env if needed
    if args.num_envs > 1:
        trainer.setup_multi_env(make_env)
    
    # Resume if specified
    start_iteration = 0
    if args.resume:
        if accelerator is None or accelerator.is_main_process:
            logger.info(f"Resuming from {args.resume}")
        start_iteration = trainer.load_checkpoint(args.resume)
    
    # Train
    total_iterations = adv_config.get("total_iterations", 100000)
    trainer.train(total_iterations, start_iteration=start_iteration)
    
    # Cleanup
    env.close()
    if trainer.env_collector is not None:
        trainer.env_collector.close()


if __name__ == "__main__":
    main()
