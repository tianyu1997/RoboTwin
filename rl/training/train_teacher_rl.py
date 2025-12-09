#!/usr/bin/env python3
"""
Teacher Policy Training Script for F1-VLA (Phase 1)

Phase 1 Training (Supervised Learning):
- Train World Model to predict next frame observation
- Input: history images, actions, states
- Output: predicted next frame image tokens
- Loss: cross-entropy on VQ-VAE image tokens

Note: This is supervised learning, not reinforcement learning.
Random actions are used only for data collection/exploration.

Distributed Training:
- Supports multi-GPU training via HuggingFace Accelerate
- Launch with: accelerate launch --num_processes=N train_teacher_rl.py
- Or single GPU: python train_teacher_rl.py
"""

# ============== MUST BE FIRST: Set GPU device for SAPIEN Vulkan rendering ==============
# This MUST happen before importing SAPIEN or any module that imports SAPIEN
import os
import sys

# Get LOCAL_RANK from environment (set by accelerate/torchrun)
# This tells us which GPU this process should use
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
# These control which GPU SAPIEN uses for rendering
os.environ["VK_DEVICE_INDEX"] = str(physical_gpu_id)
os.environ["SAPIEN_DEVICE_INDEX"] = str(physical_gpu_id)
os.environ["EGL_DEVICE_ID"] = str(physical_gpu_id)

# ============== Now continue with other imports and setup ==============
import warnings

# Set matplotlib backend BEFORE importing matplotlib (avoids IPython issues)
os.environ["MPLBACKEND"] = "Agg"

# Set environment variables BEFORE any torch/cuda imports
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")
os.environ.setdefault("CUROBO_LOG_LEVEL", "ERROR")

# Suppress warnings BEFORE importing packages that trigger them
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============== Setup paths BEFORE importing other modules ==============
script_dir = os.path.dirname(os.path.abspath(__file__))  # rl/training
rl_dir = os.path.dirname(script_dir)                      # rl
robotwin_dir = os.path.dirname(rl_dir)                    # RoboTwin
f1_vla_dir = os.path.dirname(robotwin_dir)                # F1-VLA
sys.path.insert(0, f1_vla_dir)
sys.path.insert(0, robotwin_dir)

# Import log suppression module (must be before any CuRobo imports)
from rl.suppress_logs import suppress_curobo_logs

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image
import cv2  # For video recording

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
    setup_optimizer,
    setup_scheduler,
    clip_gradients,
    count_trainable_params,
    setup_logging_from_config,
    set_policy_requires_grad,
)

# Import parallel training utilities (uses HuggingFace Accelerate)
from rl.training.parallel_utils import (
    AcceleratorWrapper,
    create_accelerator,
    SequentialEpisodeBuffer,
    ParallelEnvCollector,
    print_rank0,
    set_random_seed,
    gather_dict_metrics,
)

# Default logging (will be overridden by config)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train World Model (Phase 1) - Supervised Learning")
    
    # Config file
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to training config YAML file")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Override model config file path")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    
    # Distributed training (via Accelerate)
    parser.add_argument("--num_envs", type=int, default=1,
                       help="Number of parallel environments per GPU")
    parser.add_argument("--use_ddp", action="store_true",
                       help="Use distributed data parallel training (set automatically by accelerate launch)")
    parser.add_argument("--mixed_precision", type=str, default="no",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training mode")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    
    # Device & debug
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint directory to resume from")
    parser.add_argument("--auto_resume", action="store_true", default=False,
                       help="Automatically resume from latest checkpoint in output_dir")
    
    return parser.parse_args()


class WorldModelTrainer:
    """
    Trainer for World Model (supervised learning).
    
    Training loop:
    1. Collect trajectory with random actions (for exploration)
    2. Feed observation + action history to world model
    3. Predict next observation tokens
    4. Compute cross-entropy loss on predicted vs actual tokens
    5. Update model parameters via gradient descent
    
    Supports distributed training via HuggingFace Accelerate.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        policy_config,
        rl_config: OmegaConf,
        model_config_file: str,
        device: str = "cuda",
        accelerator: Optional[AcceleratorWrapper] = None,
        num_envs: int = 1,
    ):
        self.policy = policy
        self.policy_config = policy_config
        self.rl_config = rl_config
        self.accelerator = accelerator
        self.num_envs = num_envs
        
        # Use accelerator device if available
        if accelerator is not None:
            self.device = str(accelerator.device)
        else:
            self.device = device
        
        # Load model config (debug_test.yaml) to get n_obs_img_steps and obs_img_stride
        import yaml
        with open(model_config_file, 'r') as f:
            model_cfg = yaml.safe_load(f)
        
        # Extract n_obs_img_steps and obs_img_stride from first dataset
        train_datasets = model_cfg.get('dataset', {}).get('train_dir', {})
        if not train_datasets:
            raise ValueError("No train datasets found in model config")
        
        first_dataset = next(iter(train_datasets.values()))
        self.n_obs_img_steps = first_dataset.get('n_obs_img_steps', 4)
        self.obs_img_stride = first_dataset.get('obs_img_stride', 1)
        
        # Get training config
        self.config = get_training_config(rl_config)
        self.n_pred_img_steps = self.config.n_pred_img_steps
        
        # history_length is the observation buffer size (n_obs_img_steps)
        # The prediction target (next_obs) will be appended separately in batch building
        self.history_length = self.n_obs_img_steps
        
        # Memory configuration from model config (for GRU state)
        self.memory_enabled = policy_config.memory_enabled if hasattr(policy_config, 'memory_enabled') else True
        self.memory_hidden = policy_config.memory_hidden if hasattr(policy_config, 'memory_hidden') else 2048
        self.memory_num_layers = policy_config.memory_num_layers if hasattr(policy_config, 'memory_num_layers') else 4
        
        self._print(f"Memory config: enabled={self.memory_enabled}, hidden={self.memory_hidden}, layers={self.memory_num_layers}")
        
        self._print(f"World model config from {model_config_file}:")
        self._print(f"  n_obs_img_steps: {self.n_obs_img_steps} (input frames)")
        self._print(f"  n_pred_img_steps: {self.n_pred_img_steps} (prediction frames)")
        self._print(f"  history_length: {self.history_length} (observation buffer)")
        self._print(f"  obs_img_stride: {self.obs_img_stride}")
        if self.accelerator and self.accelerator.is_distributed:
            self._print(f"  num_processes: {self.accelerator.num_processes}")
            self._print(f"  num_envs_per_process: {num_envs}")
        
        teacher_config = rl_config.get("teacher", {})
        self.output_dir = Path(teacher_config.get("output_dir", "./outputs/teacher_rl"))
        if self._is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read video save frequency from shared `training` config
        train_cfg = rl_config.get("training", {})
        self.video_save_every = int(train_cfg.get("video_save_every", 1))
        
        # Setup policy for training - configure to train world model only
        self.policy.train()
        
        # Set gradient flags: train world model (gen expert) only
        # freeze_vision_encoder=True: saves ~30-40% VRAM, 2-3x faster, PaliGemma already strong
        self._print("\nConfiguring training mode: World Model only")
        # Suppress verbose model output on non-main processes
        if not self._is_main_process():
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        try:
            set_policy_requires_grad(
                self.policy,
                freeze_vision_encoder=True,   # Freeze vision encoder (recommended for all phases)
                freeze_gen_expert=False,
                train_act_expert_only=False,
                train_gen_expert_only=True,   # Only train world model
            )
        finally:
            if not self._is_main_process():
                sys.stdout = old_stdout
        
        # Setup optimizer and scheduler
        trainable, total = count_trainable_params(self.policy)
        self._print(f"Trainable parameters: {trainable:,} / {total:,}")
        
        self.optimizer = setup_optimizer(
            self.policy,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = setup_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            T_max=self.config.num_episodes * self.config.steps_per_episode,
            eta_min=1e-6,
        )
        
        # Prepare model and optimizer with accelerator (for DDP)
        if self.accelerator is not None:
            self.policy, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.policy, self.optimizer, self.scheduler
            )
        
        # Environment config
        self.env_config = get_environment_config(rl_config)
        
        # Batch size for training (larger batch = better GPU utilization)
        self.batch_size = self.config.batch_size
        
        # Sequential training configuration for BPTT (Backpropagation Through Time)
        train_cfg = rl_config.get("training", {})
        self.sequential_training = train_cfg.get("sequential_training", False)
        self.bptt_length = train_cfg.get("bptt_length", 8)  # Truncated BPTT sequence length
        self.memory_backprop = train_cfg.get("memory_backprop", False)
        
        # Enable memory_backprop in the model if configured
        if self.memory_backprop:
            policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
            if hasattr(policy, 'model') and hasattr(policy.model, 'memory_backprop'):
                policy.model.memory_backprop = True
                self._print(f"Enabled memory_backprop in model for BPTT")
        
        self._print(f"Sequential training config: enabled={self.sequential_training}, bptt_length={self.bptt_length}, memory_backprop={self.memory_backprop}")
        
        # Replay buffer for accumulating transitions (use episode-based buffer)
        self.replay_buffer = SequentialEpisodeBuffer(max_episodes=500, max_transitions=10000)
        
        # Batch builder - teacher uses head + wrist cameras
        self.batch_builder = BatchBuilder(
            device=self.device,
            image_keys=["head_rgb", "wrist_rgb"],
            use_head_camera=True,  # Teacher: head_rgb (image0) + wrist_rgb (image1)
        )
        
        # Memory state manager (for sequential processing)
        self.memory_manager = MemoryStateManager()
        
        # Training metrics
        self.metrics = {
            "wm_loss": deque(maxlen=100),
            "wm_accuracy": deque(maxlen=100),
            "total_steps": 0,
        }
        
        # Environment collector (multiple environments for faster data collection)
        self.env_collector = None
        self.env = None  # Single env fallback
        
        # Ensure output directories for logging and videos
        if self._is_main_process():
            self.metrics_log_path = self.output_dir / "episode_metrics.jsonl"
            (self.output_dir / "videos").mkdir(parents=True, exist_ok=True)
        else:
            self.metrics_log_path = None
        
        # Video recording - collect both head and wrist camera frames
        self.video_frames_head = []   # Head camera frames
        self.video_frames_wrist = []  # Wrist camera frames (GT for WM)
        self.video_transitions = []   # Store transitions for prediction comparison
    
    def _print(self, msg: str):
        """Print only on main process."""
        print_rank0(msg, self.accelerator)
    
    def _is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process
    
    def setup_environment(self):
        """Setup the simulation environment(s)."""
        self._print("Setting up environment...")
        
        # Set environment variables for distributed training BEFORE creating environments
        # These affect logging and GPU selection in the underlying task environment
        local_gpu_id = 0  # Default for single GPU
        render_device_id = 0 # Logical ID for sapien.Device (CUDA)
        
        if self.accelerator is not None:
            local_process_idx = self.accelerator.local_process_index
            render_device_id = local_process_idx # Always use logical index for CUDA
            
            # SAPIEN uses Vulkan rendering which doesn't respect CUDA_VISIBLE_DEVICES.
            # We need to map the local process index to the actual physical GPU ID.
            # CUDA_VISIBLE_DEVICES format: "0,1,2,3" means physical GPUs 0,1,2,3 are visible
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cuda_visible:
                # Parse the visible devices list
                visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                if local_process_idx < len(visible_gpus):
                    local_gpu_id = visible_gpus[local_process_idx]
                else:
                    # Fallback: use local_process_index directly
                    local_gpu_id = local_process_idx
                logger.debug(f"Process {self.accelerator.process_index}: GPU={local_gpu_id}")
            else:
                # No CUDA_VISIBLE_DEVICES set, use local_process_index directly
                local_gpu_id = local_process_idx
                logger.debug(f"Process {self.accelerator.process_index}: using GPU={local_process_idx}")
            
            # Set RL_MAIN_PROCESS to control logging in _base_task.py
            # Non-main processes will suppress INFO logs
            os.environ["RL_MAIN_PROCESS"] = "1" if self._is_main_process() else "0"
            
            # Set EGL device for SAPIEN rendering to use the correct GPU (fallback)
            os.environ["EGL_DEVICE_ID"] = str(local_gpu_id)
            os.environ["VK_DEVICE_INDEX"] = str(local_gpu_id)
            logger.debug(f"Process {self.accelerator.process_index}: GPU={local_gpu_id}, main={self._is_main_process()}")
        
        from rl.f1_rl_env import TeacherEnv
        
        # Get single_arm and scene_reset_interval from config
        single_arm = self.env_config.get("single_arm", False)
        scene_reset_interval = self.env_config.get("scene_reset_interval", 1)
        # Disable robot initial position randomization for debugging (use fixed home position)
        # This ensures wrist cameras can see the table properly
        randomize_robot_init = self.env_config.get("randomize_robot_init", False)
        
        # For RL training: disable motion planner (CuRobo) to save time and VRAM
        # We use delta action control which doesn't need trajectory planning
        need_planner = self.env_config.get("need_planner", False)
        need_topp = self.env_config.get("need_topp", False)
        
        # Log domain_randomization config for debugging
        domain_rand = self.env_config.get("domain_randomization", {})
        logger.debug(f"Environment config domain_randomization: {domain_rand}")
        
        def create_env():
            # Capture local_gpu_id in closure
            gpu_id = render_device_id
            logger.debug(f"create_env: gpu_id = {gpu_id}")
            return TeacherEnv(
                task_config={
                    **self.env_config,
                    "need_planner": need_planner,
                    "need_topp": need_topp,
                    "render_device": gpu_id,  # Specify GPU for SAPIEN rendering
                },
                history_length=self.history_length,
                max_steps=self.config.steps_per_episode,
                device=self.device,
                action_scale=self.config.action_scale,
                single_arm=single_arm,
                scene_reset_interval=scene_reset_interval,
                randomize_robot_init=randomize_robot_init,
            )
        
        if self.num_envs > 1:
            # Use parallel environment collector
            self.env_collector = ParallelEnvCollector(
                env_fn=create_env,
                num_envs=self.num_envs,
                is_main_process=self._is_main_process(),
            )
            self.env_collector.initialize()
            # Also keep single env reference for compatibility
            self.env = self.env_collector.envs[0]
            self._print(f"Environment ready! {self.num_envs} parallel envs, single_arm={single_arm}")
        else:
            # Single environment
            self.env = create_env()
            self._print(f"Environment ready! single_arm={single_arm}, scene_reset_interval={scene_reset_interval}")
        
        logger.debug(f"Environment ready (action_scale={self.config.action_scale}, single_arm={single_arm})")
    
    def collect_trajectory(self) -> List[Dict[str, Any]]:
        """
        Collect one trajectory with random actions.
        Returns list of (obs, action, next_obs) tuples for training.
        """
        obs, _ = self.env.reset()
        trajectory = []
        done = False
        
        while not done:
            # Random action for exploration
            action = np.random.uniform(-1, 1, self.config.action_dim).astype(np.float32)
            
            # Collect video frame from current observation (with action for prediction)
            self._collect_video_frame(obs, action)
            
            next_obs, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            trajectory.append({
                "obs": obs,
                "action": info.get("action_executed", action),
                "next_obs": next_obs,
            })
            obs = next_obs
        
        # Collect final frame (no action)
        self._collect_video_frame(obs)
        
        return trajectory
    
    def _init_memory_state(self, batch_size: int) -> torch.Tensor:
        """Initialize memory state to zeros for first frame.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Zero-initialized memory state tensor of shape (num_layers, batch_size, hidden_dim)
        """
        memory_state = torch.zeros(
            self.memory_num_layers,
            batch_size,
            self.memory_hidden,
            device=self.device,
            dtype=torch.float32
        )
        logger.debug(f"Initialized zero memory state: shape={memory_state.shape}")
        return memory_state
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step with memory state tracking."""
        self.optimizer.zero_grad()
        
        # Ensure model is in training mode (required for CuDNN RNN backward)
        # Unwrap if DDP wrapped
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        policy.train()
        
        # Get batch size
        batch_size = batch["observation.state"].shape[0]
        
        # Ensure memory state is present - CRITICAL: must not be None
        if "initial_memory_state" not in batch or batch["initial_memory_state"] is None:
            # Initialize to zeros for first frame
            batch["initial_memory_state"] = self._init_memory_state(batch_size)
            logger.debug(f"Memory state initialized to zeros for batch_size={batch_size}")
        
        # Validate memory state is not None
        if batch["initial_memory_state"] is None:
            raise ValueError("CRITICAL: initial_memory_state is None after initialization! This should never happen.")
        
        # Log memory state info (DEBUG level)
        mem_state = batch["initial_memory_state"]
        logger.debug(f"Memory state input: shape={mem_state.shape}, mean={mem_state.mean().item():.6f}, std={mem_state.std().item():.6f}")
        
        # Forward pass - predict next frame tokens
        loss_dict = policy.forward_with_world_model(
            batch,
            cur_n_obs_img_steps=self.n_obs_img_steps,
            cur_n_pred_img_steps=self.n_pred_img_steps,
            train_gen_expert_only=True,  # Only train world model
        )
        
        # Extract and validate output memory state (only relevant when memory_enabled=true)
        output_memory_state = loss_dict.get("memory_state")
        if output_memory_state is not None:
            logger.debug(f"Memory state output: shape={output_memory_state.shape}, mean={output_memory_state.mean().item():.6f}")
            # Update memory manager with new state (detached)
            self.memory_manager.update(output_memory_state.detach())
        # Note: None memory_state is expected when memory_enabled=false in config
        
        # Backward pass (use accelerator if available)
        loss = loss_dict["loss"]
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        # Gradient clipping
        if self.accelerator is not None:
            self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        else:
            clip_gradients(self.policy, max_norm=self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss": loss.item(),
            "accuracy": loss_dict.get("wm_acc_mean", torch.tensor(0.0)).item(),
        }
    
    def train_step_sequential(self, sequences: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Execute one training step with sequential transitions using truncated BPTT.
        
        This method processes transitions in temporal order within each sequence,
        propagating memory states through time and allowing gradients to flow
        through the memory RNN.
        
        Args:
            sequences: List of sequences, each sequence is a list of consecutive transitions
                       Shape: [batch_size, sequence_length, transition_dict]
        
        Returns:
            Dictionary with loss and accuracy
        """
        self.optimizer.zero_grad()
        
        # Unwrap if DDP wrapped
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        policy.train()
        
        batch_size = len(sequences)
        seq_length = len(sequences[0]) if sequences else 0
        
        if seq_length == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        
        # Initialize memory state for the beginning of sequences
        # Use the first transition's initial_memory_state if available, else zeros
        # CRITICAL: Always detach at sequence START to prevent infinite graph growth
        # Gradients only flow WITHIN a sequence (truncated BPTT), not across sequences
        memory_state = None
        first_transitions = [seq[0] for seq in sequences]
        for t in first_transitions:
            if t.get("initial_memory_state") is not None:
                # Found a valid initial memory state
                ms = t["initial_memory_state"]
                if isinstance(ms, torch.Tensor):
                    memory_state = ms.unsqueeze(0) if ms.dim() == 2 else ms
                    # Detach to start fresh computation graph for this sequence
                    memory_state = memory_state.detach()
                    break
        
        if memory_state is None:
            memory_state = self._init_memory_state(batch_size)
        else:
            memory_state = memory_state.detach()  # Ensure detached
        
        # Truncated BPTT: accumulate loss over the sequence, then backprop
        total_loss = 0.0
        total_acc = 0.0
        valid_steps = 0
        
        for step_idx in range(seq_length):
            # Gather transitions at this time step across all sequences
            step_transitions = [seq[step_idx] for seq in sequences]
            
            # Build batch for this step
            batch = self.batch_builder.build_batch(
                step_transitions, include_memory_states=True
            )
            
            # Override the initial_memory_state with our tracked memory
            batch["initial_memory_state"] = memory_state.to(self.device)
            
            # Forward pass
            loss_dict = policy.forward_with_world_model(
                batch,
                cur_n_obs_img_steps=self.n_obs_img_steps,
                cur_n_pred_img_steps=self.n_pred_img_steps,
                train_gen_expert_only=True,
            )
            
            # Update memory state for next step
            # When memory_backprop=True: keep gradients WITHIN this sequence for BPTT
            # The initial state is already detached, so graph won't grow infinitely
            output_memory_state = loss_dict.get("memory_state")
            if output_memory_state is not None:
                if self.memory_backprop and step_idx < seq_length - 1:
                    # Keep gradients for next step within sequence (truncated BPTT)
                    memory_state = output_memory_state
                else:
                    # Last step or memory_backprop=False: detach
                    memory_state = output_memory_state.detach()
            
            # Accumulate loss
            step_loss = loss_dict["loss"]
            total_loss = total_loss + step_loss
            total_acc += loss_dict.get("wm_acc_mean", torch.tensor(0.0)).item()
            valid_steps += 1
        
        # Average loss over sequence
        if valid_steps > 0:
            avg_loss = total_loss / valid_steps
        else:
            avg_loss = total_loss
        
        # Backward pass (use accelerator if available)
        if self.accelerator is not None:
            self.accelerator.backward(avg_loss)
        else:
            avg_loss.backward()
        
        # Gradient clipping (important for BPTT stability)
        if self.accelerator is not None:
            self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        else:
            clip_gradients(self.policy, max_norm=self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update memory manager with final state (detached)
        if output_memory_state is not None:
            self.memory_manager.update(output_memory_state.detach())
        
        return {
            "loss": avg_loss.item() if hasattr(avg_loss, 'item') else avg_loss,
            "accuracy": total_acc / valid_steps if valid_steps > 0 else 0.0,
        }
    
    def train(self, start_episode: int = 0):
        """Main training loop with tqdm progress bar."""
        from tqdm import tqdm
        import time
        
        num_episodes = self.config.num_episodes
        
        # Setup environment (suppress the log, tqdm will show progress)
        self.setup_environment()
        
        # Create progress bar (only on main process)
        disable_pbar = not self._is_main_process()
        pbar = tqdm(
            range(start_episode, num_episodes),
            desc="Training",
            unit="ep",
            ncols=120,
            initial=start_episode,
            total=num_episodes,
            disable=disable_pbar,
        )
        
        start_time = time.time()
        
        for episode in pbar:
            # Collect trajectory and add to replay buffer
            if self.env_collector is not None and self.num_envs > 1:
                # Collect from multiple environments
                # Note: collect_steps runs num_iters = ceil(num_steps / num_envs)
                # So to get each env to run steps_per_episode, we need:
                # total_steps = steps_per_episode * num_envs
                total_steps = self.config.steps_per_episode * self.num_envs
                completed_episodes = self.env_collector.collect_steps(
                    num_steps=total_steps,
                    action_fn=None,  # Random actions
                    action_dim=self.config.action_dim,
                )
                for ep in completed_episodes:
                    self.replay_buffer.add_episode(ep)
                
                # Collect video frames for this training episode (main process only)
                # Use completed episodes if available, otherwise use current buffer
                if self._is_main_process():
                    if completed_episodes:
                        # Use first completed episode
                        first_ep = completed_episodes[0]
                        for transition in first_ep:
                            self._collect_video_frame(
                                transition.get("obs", {}),
                                transition.get("action")
                            )
                    else:
                        # No completed episodes - use current buffer from first env
                        current_buffer = self.env_collector.get_current_episode_buffer(env_idx=0)
                        for transition in current_buffer:
                            self._collect_video_frame(
                                transition.get("obs", {}),
                                transition.get("action")
                            )
                
                # Use first episode's length for training steps calculation
                trajectory_len = sum(len(ep) for ep in completed_episodes) if completed_episodes else self.config.steps_per_episode
            else:
                # Single environment
                trajectory = self.collect_trajectory()
                self.replay_buffer.add_episode(trajectory)
                trajectory_len = len(trajectory)
            
            # Train on mini-batches from replay buffer
            ep_loss = []
            ep_acc = []
            
            # Number of training steps per episode
            num_train_steps = max(1, trajectory_len // self.batch_size)
            
            for _ in range(num_train_steps):
                if self.sequential_training:
                    # Sequential training with BPTT
                    # Sample sequences of consecutive transitions
                    if len(self.replay_buffer) >= self.batch_size * self.bptt_length:
                        sequences = self.replay_buffer.sample_sequential_batch(
                            batch_size=self.batch_size,
                            sequence_length=self.bptt_length
                        )
                    else:
                        # Not enough data for full sequences, use shorter ones
                        available_seq_len = max(1, len(self.replay_buffer) // max(1, self.batch_size))
                        sequences = self.replay_buffer.sample_sequential_batch(
                            batch_size=min(self.batch_size, len(self.replay_buffer)),
                            sequence_length=min(self.bptt_length, available_seq_len)
                        )
                    
                    if not sequences:
                        continue
                    
                    result = self.train_step_sequential(sequences)
                else:
                    # Random batch training (original behavior)
                    if len(self.replay_buffer) >= self.batch_size:
                        batch_transitions = self.replay_buffer.sample_batch(self.batch_size)
                    else:
                        batch_transitions = self.replay_buffer.sample_batch(len(self.replay_buffer))
                    
                    if not batch_transitions:
                        continue
                    
                    batch = self.batch_builder.build_batch(
                        batch_transitions, include_memory_states=True
                    )
                    
                    result = self.train_step(batch)
                
                ep_loss.append(result["loss"])
                ep_acc.append(result["accuracy"])
                self.metrics["wm_loss"].append(result["loss"])
                self.metrics["wm_accuracy"].append(result["accuracy"])
                self.metrics["total_steps"] += 1
            
            # Synchronize metrics across processes if distributed
            if self.accelerator and self.accelerator.is_distributed:
                self.accelerator.wait_for_everyone()
            
            # Update progress bar with current metrics
            avg_loss = np.mean(ep_loss) if ep_loss else 0
            avg_acc = np.mean(ep_acc) if ep_acc else 0
            lr = self.scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            fps = self.metrics["total_steps"] / elapsed if elapsed > 0 else 0
            
            if self._is_main_process():
                pbar.set_postfix({
                    "loss": f"{avg_loss:.3f}",
                    "acc": f"{avg_acc:.3f}",
                    "lr": f"{lr:.1e}",
                    "steps": self.metrics["total_steps"],
                    "fps": f"{fps:.1f}",
                })

            # Log metrics periodically (only on main process)
            log_every = getattr(self.config, "log_every", None) or self.rl_config.get("training", {}).get("log_every", None)
            if log_every is None:
                log_every = 10
            if (episode + 1) % int(log_every) == 0 and self._is_main_process():
                # Episode reward may be tracked by env
                episode_reward = getattr(self.env, "episode_reward", None)
                metrics_entry = {
                    "episode": int(episode),
                    "avg_loss": float(avg_loss),
                    "avg_acc": float(avg_acc),
                    "episode_reward": float(episode_reward) if episode_reward is not None else None,
                    "total_steps": int(self.metrics["total_steps"]),
                    "lr": float(lr),
                    "timestamp": time.time(),
                }
                try:
                    if self.metrics_log_path:
                        with open(self.metrics_log_path, "a") as fh:
                            fh.write(json.dumps(metrics_entry) + "\n")
                except Exception:
                    logger.exception("Failed to write metrics log")
            
            # Save video (every episode by default)
            if (episode + 1) % self.video_save_every == 0 and self._is_main_process():
                self._save_episode_video(episode + 1)
            else:
                # Clear frames if not saving to avoid memory buildup
                self.video_frames_head = []
                self.video_frames_wrist = []
                self.video_transitions = []

            # Save checkpoint periodically (only on main process)
            if (episode + 1) % self.config.save_every == 0:
                # Wait for all processes before saving
                if self.accelerator:
                    self.accelerator.wait_for_everyone()
                if self._is_main_process():
                    pbar.write(f"[Checkpoint] Saving episode {episode+1}...")
                    self.save_checkpoint(episode + 1)
        
        pbar.close()
        
        # Final save (only on main process)
        if self.accelerator:
            self.accelerator.wait_for_everyone()
        if self._is_main_process():
            self.save_checkpoint(num_episodes)
            print("\nTraining complete!")
        
        # Cleanup
        if self.accelerator:
            self.accelerator.end_training()
        if self.env_collector:
            self.env_collector.close()
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint (only on main process)."""
        if not self._is_main_process():
            return
            
        checkpoint_dir = self.output_dir / f"checkpoint-{episode}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Unwrap DDP model if needed
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        
        # Save model state dict
        torch.save(policy.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer & scheduler & training state
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "episode": episode,
            "total_steps": self.metrics["total_steps"],
            # Save training config for reference
            "config": {
                "sequential_training": self.sequential_training,
                "bptt_length": self.bptt_length,
                "memory_backprop": self.memory_backprop,
                "batch_size": self.batch_size,
                "n_obs_img_steps": self.n_obs_img_steps,
                "n_pred_img_steps": self.n_pred_img_steps,
            },
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # Save recent metrics for reference
        metrics_snapshot = {
            "wm_loss": list(self.metrics["wm_loss"]),
            "wm_accuracy": list(self.metrics["wm_accuracy"]),
        }
        torch.save(metrics_snapshot, checkpoint_dir / "metrics.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def _process_obs_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Process observation image to HWC uint8 RGB format.
        
        Handles various input formats:
        - (T, C, H, W): Stacked history, take last frame
        - (C, H, W): Single CHW frame
        - (H, W, C): Already HWC
        
        Returns:
            np.ndarray in HWC uint8 format, or None if processing fails
        """
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Handle stacked history images: (T, C, H, W)
        if img.ndim == 4:
            img = img[-1]  # Take last frame: (C, H, W)
        
        if img.ndim != 3:
            logger.warning(f"Unexpected image dimensions: {img.ndim}")
            return None
        
        # Determine format and convert to HWC
        if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
            # CHW format -> HWC
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[2] == 3:
            # Already HWC format
            pass
        else:
            logger.warning(f"Cannot determine image format: shape={img.shape}")
            return None
        
        # Ensure uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        return img
    
    def _collect_video_frame(self, obs: Dict[str, Any], action: Any = None):
        """Collect frames for video recording from environment observation.
        
        Collects both head and wrist camera images for comparison video.
        Also stores transition data for generating prediction comparison.
        """
        if not self._is_main_process():
            return
        
        # Collect head camera frame
        head_img = obs.get("head_rgb")
        if head_img is not None:
            if head_img.ndim == 4:
                head_img = head_img[-1]  # Take the most recent frame [C, H, W]
            frame = self._process_obs_image(head_img)
            if frame is not None:
                self.video_frames_head.append(frame.copy())
        
        # Collect wrist camera frame (GT for world model)
        wrist_img = obs.get("wrist_rgb")
        if wrist_img is not None:
            if wrist_img.ndim == 4:
                wrist_img = wrist_img[-1]
            frame = self._process_obs_image(wrist_img)
            if frame is not None:
                self.video_frames_wrist.append(frame.copy())
        
        # Store transition for prediction (limit to save memory)
        if len(self.video_transitions) < 200:  # Max 200 frames for prediction
            self.video_transitions.append({
                "obs": {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()},
                "action": action.copy() if action is not None and hasattr(action, 'copy') else action,
            })
    
    def _save_episode_video(self, episode: int):
        """Save combined video for the episode.
        
        Creates one video with three columns:
        [Head Camera | GT Wrist | Predicted Wrist]
        
        Starts from frame n_obs_img_steps (when we have enough history for prediction).
        """
        if not self._is_main_process():
            return
        
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined video (head + GT wrist + predicted wrist)
        self._save_combined_video(episode, video_dir)
        
        # Clear buffers
        self.video_frames_head = []
        self.video_frames_wrist = []
        self.video_transitions = []
    
    def _save_combined_video(self, episode: int, video_dir: Path):
        """Save combined video with Head, GT Wrist, and Predicted Wrist side by side.
        
        Layout: [Head Camera | GT Wrist | Predicted Wrist]
        Starts from frame n_obs_img_steps when we have enough history for prediction.
        """
        if not self.video_transitions or len(self.video_transitions) < self.n_obs_img_steps + 1:
            logger.warning(f"Not enough frames for video (need {self.n_obs_img_steps + 1}, got {len(self.video_transitions)})")
            return
        
        try:
            video_path = video_dir / f"episode_{episode:06d}.mp4"
            
            # Get policy for prediction
            policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
            policy.eval()
            
            # Collect frames starting from when we have enough history
            start_idx = self.n_obs_img_steps - 1  # Need n_obs_img_steps history frames
            
            head_frames = []
            gt_frames = []
            pred_frames = []
            
            for i in range(start_idx, len(self.video_transitions) - 1):
                trans = self.video_transitions[i]
                next_trans = self.video_transitions[i + 1]
                
                obs = trans["obs"]
                action = trans["action"]
                next_obs = next_trans["obs"]
                
                # Get head frame
                head_img = obs.get("head_rgb")
                if head_img is not None:
                    if head_img.ndim == 4:
                        head_img = head_img[-1]
                    head_frame = self._process_obs_image(head_img)
                    if head_frame is not None:
                        head_frames.append(head_frame)
                    else:
                        head_frames.append(None)
                else:
                    head_frames.append(None)
                
                # Get GT wrist frame (from next observation)
                gt_wrist = next_obs.get("wrist_rgb")
                if gt_wrist is None:
                    continue
                if gt_wrist.ndim == 4:
                    gt_wrist = gt_wrist[-1]
                gt_frame = self._process_obs_image(gt_wrist)
                if gt_frame is None:
                    continue
                gt_frames.append(gt_frame)
                
                # Get model prediction
                pred_frame = None
                try:
                    if action is not None:
                        # Build batch for prediction
                        batch = self.env._build_policy_batch(
                            obs, np.array(action, dtype=np.float32), use_head_camera=True
                        )
                        
                        with torch.no_grad():
                            pred_out = policy.predict_images_only(batch)
                        
                        pred_imgs = pred_out.get("pred_imgs")
                        if pred_imgs is not None:
                            pred = pred_imgs.detach().cpu()
                            if pred.ndim == 5:
                                pred = pred[:, -1]  # Last predicted frame
                            pred = pred[0]  # First batch item [C, H, W]
                            
                            # De-normalize: [-1, 1] -> [0, 255]
                            pred_np = ((pred + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
                            pred_frame = (np.transpose(pred_np, (1, 2, 0)) * 255.0).astype(np.uint8)
                            
                            # VAE outputs 256x256 images, resize to match GT frame size
                            gt_h, gt_w = gt_frame.shape[:2]
                            if pred_frame.shape[0] != gt_h or pred_frame.shape[1] != gt_w:
                                pred_frame = cv2.resize(pred_frame, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    logger.debug(f"Prediction failed for frame {i}: {e}")
                
                # Use GT as fallback if prediction failed
                if pred_frame is None:
                    pred_frame = gt_frame.copy()
                pred_frames.append(pred_frame)
            
            if not gt_frames:
                logger.warning(f"No valid frames to save for episode {episode}")
                return
            
            # Determine frame dimensions and round to 16 (libx264 requires even, 16 is safe)
            h, w = gt_frames[0].shape[:2]
            
            # Layout: [Head | GT Wrist | Predicted]
            # Each panel is w x h, with 5px gaps between panels
            gap = 5
            label_h = 25
            raw_combined_w = w * 3 + gap * 2
            raw_combined_h = h + label_h
            
            # Round dimensions up to nearest multiple of 16 for libx264 compatibility
            combined_w = ((raw_combined_w + 15) // 16) * 16
            combined_h = ((raw_combined_h + 15) // 16) * 16
            
            # Calculate padding to center content
            pad_x = (combined_w - raw_combined_w) // 2
            pad_y = (combined_h - raw_combined_h) // 2
            
            # Use imageio for better compatibility
            import imageio
            writer = imageio.get_writer(
                str(video_path), fps=10, codec='libx264',
                pixelformat='yuv420p', quality=8
            )
            
            num_frames = min(len(head_frames), len(gt_frames), len(pred_frames))
            for i in range(num_frames):
                # Create combined frame (white background, padded to 16-multiple size)
                combined = np.ones((combined_h, combined_w, 3), dtype=np.uint8) * 255
                
                # Start position with padding offset
                x_offset = pad_x
                y_offset = pad_y
                
                # Panel 1: Head Camera
                if head_frames[i] is not None:
                    head_frame = head_frames[i]
                    # Resize if needed
                    if head_frame.shape[:2] != (h, w):
                        head_frame = cv2.resize(head_frame, (w, h))
                    combined[y_offset+label_h:y_offset+label_h+h, x_offset:x_offset+w] = head_frame
                cv2.putText(combined, "Head", (x_offset + w//2 - 20, y_offset + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                x_offset += w + gap
                
                # Panel 2: GT Wrist
                gt_frame = gt_frames[i]
                if gt_frame.shape[:2] != (h, w):
                    gt_frame = cv2.resize(gt_frame, (w, h))
                combined[y_offset+label_h:y_offset+label_h+h, x_offset:x_offset+w] = gt_frame
                cv2.putText(combined, "GT Wrist", (x_offset + w//2 - 35, y_offset + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 1)
                x_offset += w + gap
                
                # Panel 3: Predicted Wrist
                pred_frame = pred_frames[i]
                if pred_frame.shape[:2] != (h, w):
                    pred_frame = cv2.resize(pred_frame, (w, h))
                combined[y_offset+label_h:y_offset+label_h+h, x_offset:x_offset+w] = pred_frame
                cv2.putText(combined, "Predicted", (x_offset + w//2 - 40, y_offset + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Write frame (imageio expects RGB)
                writer.append_data(combined)
            
            writer.close()
            logger.info(f"[Video] Saved: {video_path} ({num_frames} frames, start_idx={start_idx})")
            
        except Exception:
            logger.exception("Error saving combined video")
        finally:
            policy.train()  # Restore training mode

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in output_dir based on episode number.
        
        Returns:
            Path to latest checkpoint directory, or None if no checkpoints found.
        """
        if not self.output_dir.exists():
            return None
        
        checkpoints = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    episode = int(item.name.split("-")[1])
                    checkpoints.append((episode, item))
                except (ValueError, IndexError):
                    continue
        
        if not checkpoints:
            return None
        
        # Sort by episode number and return the latest
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest = checkpoints[0][1]
        logger.info(f"Found latest checkpoint: {latest}")
        return latest
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return starting episode.
        
        Works correctly with DDP - all processes load the checkpoint.
        """
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return 0
        
        # Unwrap DDP model if needed for loading
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        
        # Load model weights
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle potential key mismatches from DDP wrapping
            policy.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model weights from: {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
        
        # Load training state (optimizer, scheduler, etc.)
        state_path = checkpoint_dir / "training_state.pt"
        start_episode = 0
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            
            # Load optimizer state
            try:
                self.optimizer.load_state_dict(state["optimizer"])
                logger.info("Loaded optimizer state")
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")
            
            # Load scheduler state
            try:
                self.scheduler.load_state_dict(state["scheduler"])
                logger.info("Loaded scheduler state")
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")
            
            # Restore metrics counter
            self.metrics["total_steps"] = state.get("total_steps", 0)
            start_episode = state.get("episode", 0)
            
            # Log saved config for reference
            saved_config = state.get("config", {})
            if saved_config:
                logger.info(f"Checkpoint was saved with config: {saved_config}")
        else:
            logger.warning(f"Training state file not found: {state_path}")
        
        # Synchronize all processes after loading
        if self.accelerator and self.accelerator.is_distributed:
            self.accelerator.wait_for_everyone()
        
        logger.info(f"Resuming from episode {start_episode}")
        return start_episode


def main():
    args = parse_args()
    
    # Setup Accelerate first (before loading model to get correct device)
    accelerator = create_accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Load config
    rl_config = load_rl_config(args.rl_config)
    
    # Setup logging with DDP awareness (non-main processes only log WARNING+)
    setup_logging_from_config(rl_config, is_main_process=accelerator.is_main_process)
    
    # Apply command-line overrides
    if args.model_config:
        rl_config.model.config_file = args.model_config
    if args.num_episodes is not None:
        rl_config.training.num_episodes = args.num_episodes
    if args.steps_per_episode is not None:
        rl_config.training.steps_per_episode = args.steps_per_episode
    if args.learning_rate is not None:
        rl_config.training.learning_rate = args.learning_rate
    if args.output_dir is not None:
        rl_config.teacher.output_dir = args.output_dir
    if args.save_every is not None:
        rl_config.training.save_every = args.save_every
    if args.log_every is not None:
        rl_config.training.log_every = args.log_every
    if args.device is not None:
        rl_config.device = args.device
    if args.debug is not None:
        rl_config.debug = args.debug
    
    # Use accelerator device
    device = str(accelerator.device)
    debug = rl_config.get("debug", False)
    
    # Print startup info (only on main process)
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("World Model Training (Phase 1 - Supervised Learning)")
        print("=" * 60)
        if accelerator.is_distributed:
            print(f"Distributed training: {accelerator.num_processes} GPUs")
        print(f"Device: {device}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"Num envs per GPU: {args.num_envs}")
    
    # Load model config
    model_config_file = rl_config.get("model", {}).get(
        "config_file", 
        "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml"
    )
    
    if accelerator.is_main_process:
        print(f"Loading config from: {model_config_file}")
    
    # Get LoRA config
    lora_config = get_lora_config_from_dict(rl_config)
    
    if accelerator.is_main_process:
        print("Loading policy...")
    
    # Load policy on accelerator device
    policy, policy_config, model_config = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=lora_config,
        is_main_process=accelerator.is_main_process,
    )
    
    if accelerator.is_main_process:
        print("Model loaded successfully")
    
    # Create trainer with accelerator
    trainer = WorldModelTrainer(
        policy=policy,
        policy_config=policy_config,
        rl_config=rl_config,
        model_config_file=model_config_file,
        device=device,
        accelerator=accelerator,
        num_envs=args.num_envs,
    )
    
    # Resume if specified
    start_episode = 0
    resume_path = args.resume
    
    # Auto-resume: find latest checkpoint in output_dir
    if args.auto_resume and not resume_path:
        latest_ckpt = trainer.find_latest_checkpoint()
        if latest_ckpt:
            resume_path = str(latest_ckpt)
            if accelerator.is_main_process:
                print(f"Auto-resume: found latest checkpoint at {resume_path}")
    
    if resume_path:
        if accelerator.is_main_process:
            print(f"Resuming from: {resume_path}")
        start_episode = trainer.load_checkpoint(resume_path)
        if accelerator.is_main_process:
            print(f"Resumed from episode {start_episode}")
    
    if accelerator.is_main_process:
        print("")  # Empty line before training
    
    # Train
    trainer.train(start_episode=start_episode)


if __name__ == "__main__":
    main()
