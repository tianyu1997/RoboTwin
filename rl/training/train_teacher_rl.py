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

# Debug output (to stderr so it appears even with log suppression)
print(f"[PID {os.getpid()}] LOCAL_RANK={local_rank}, CUDA_VISIBLE_DEVICES={cuda_visible}, "
      f"physical_gpu_id={physical_gpu_id}, VK_DEVICE_INDEX={os.environ.get('VK_DEVICE_INDEX')}", 
      file=sys.stderr, flush=True)

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

        # Read video/sample save frequency from shared `training` config so it applies
        # to all phases (teacher/student/adversarial)
        train_cfg = rl_config.get("training", {})
        self.video_save_every = int(train_cfg.get("video_save_every", 1))
        self.sample_save_every = int(train_cfg.get("sample_save_every", 1))
        
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
        
        # Ensure output directories for logging and samples
        if self._is_main_process():
            self.metrics_log_path = self.output_dir / "episode_metrics.jsonl"
            (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "videos").mkdir(parents=True, exist_ok=True)
        else:
            self.metrics_log_path = None
        
        # Video recording - collect both head and wrist camera frames
        self.video_frames_head = []   # Head camera frames
        self.video_frames_wrist = []  # Wrist camera frames (GT for WM)
        self.video_transitions = []   # Store transitions for prediction comparison
        # `video_save_every` and `sample_save_every` are read from `training` config
    
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
        if self.accelerator is not None:
            local_process_idx = self.accelerator.local_process_index
            
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
                logger.info(f"Process {self.accelerator.process_index}: CUDA_VISIBLE_DEVICES={cuda_visible}, "
                           f"local_process_idx={local_process_idx}, physical_gpu_id={local_gpu_id}")
            else:
                # No CUDA_VISIBLE_DEVICES set, use local_process_index directly
                local_gpu_id = local_process_idx
                logger.info(f"Process {self.accelerator.process_index}: No CUDA_VISIBLE_DEVICES, "
                           f"using local_process_idx={local_process_idx} as GPU ID")
            
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
        logger.info(f"Environment config domain_randomization: {domain_rand}")
        
        # Debug: log GPU assignment for this process
        logger.info(f"create_env: local_gpu_id = {local_gpu_id} (will be passed as render_device)")
        
        def create_env():
            # Capture local_gpu_id in closure
            gpu_id = local_gpu_id
            logger.info(f"create_env closure: captured gpu_id = {gpu_id}")
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
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        # Ensure model is in training mode (required for CuDNN RNN backward)
        # Unwrap if DDP wrapped
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        policy.train()
        
        # Forward pass - predict next frame tokens
        loss_dict = policy.forward_with_world_model(
            batch,
            cur_n_obs_img_steps=self.n_obs_img_steps,
            cur_n_pred_img_steps=self.n_pred_img_steps,
            train_gen_expert_only=True,  # Only train world model
        )
        
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
                # Sample mini-batch from replay buffer
                if len(self.replay_buffer) >= self.batch_size:
                    # Random sample
                    batch_transitions = self.replay_buffer.sample_batch(self.batch_size)
                else:
                    # Use all available if buffer smaller than batch_size
                    batch_transitions = self.replay_buffer.sample_batch(len(self.replay_buffer))
                
                if not batch_transitions:
                    continue
                
                batch = self.batch_builder.build_batch(
                    batch_transitions, include_memory_states=False
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

            # Log metrics and save image samples periodically (only on main process)
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

            # Save sample image (every episode by default)
            if (episode + 1) % self.sample_save_every == 0 and self._is_main_process():
                try:
                    # Get last trajectory from replay buffer
                    if self.replay_buffer.num_episodes > 0:
                        last_episode = list(self.replay_buffer.episodes)[-1]
                        if last_episode:
                            last = last_episode[-1]
                            obs_before = last["obs"]
                            action = last.get("action")
                            obs_after = last.get("next_obs")
                            if obs_before is not None and obs_after is not None:
                                self._save_prediction_sample(episode + 1, obs_before, action, obs_after)
                except Exception:
                    logger.exception("Failed to save sample image")
            
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
        
        # Save model
        torch.save(policy.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer & scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "episode": episode,
            "total_steps": self.metrics["total_steps"],
        }, checkpoint_dir / "training_state.pt")
        
        logger.debug(f"Checkpoint saved: {checkpoint_dir}")

    def _save_prediction_sample(self, episode: int, obs_before: Dict[str, Any], action: Any, obs_after: Dict[str, Any]):
        """Run the world model to predict next image and save a side-by-side comparison with ground truth.

        Saves to `self.output_dir / 'samples'` with filename containing episode number.
        Only runs on main process.
        
        IMPORTANT: Images in obs are in uint8 [0, 255] format with shape [T, C, H, W].
        Model predicts in [-1, 1] range, so we need proper de-normalization.
        
        Layout: [head_rgb (context)] [GT wrist_rgb] [Predicted wrist_rgb]
        """
        if not self._is_main_process():
            return
        try:
            # Build batch using environment helper (use head camera)
            batch = self.env._build_policy_batch(obs_before, np.array(action, dtype=np.float32), use_head_camera=True)

            # Unwrap policy and use for prediction
            policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
            policy.eval()  # Set to eval mode for inference
            with torch.no_grad():
                pred_out = policy.predict_images_only(batch)
            pred_imgs = pred_out.get("pred_imgs")  # Tensor [B, C, H, W] or [B, T, C, H, W]
            if pred_imgs is None:
                return

            # Use first sample in batch
            pred = pred_imgs.detach().cpu()
            if pred.ndim == 5:
                # [B, T, C, H, W] -> take last predicted frame
                pred = pred[:, -1]
            pred = pred[0]  # [C, H, W]

            # Helper to extract last frame from observation
            def extract_frame(img_data):
                """Extract last frame from [T, C, H, W] or [C, H, W] image data."""
                if img_data.ndim == 4:
                    frame = img_data[-1]  # [C, H, W]
                else:
                    frame = img_data  # [C, H, W]
                # CHW -> HWC, ensure uint8
                frame = np.transpose(frame, (1, 2, 0))
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                return frame

            # Get head camera image (context) from obs_before
            head_img = None
            if "head_rgb" in obs_before:
                head_img = extract_frame(obs_before["head_rgb"])

            # Ground truth wrist image from obs_after
            gt_raw = obs_after.get("wrist_rgb")
            if gt_raw is None:
                logger.warning("No wrist_rgb in obs_after for GT comparison")
                return
            gt_img = extract_frame(gt_raw)
            
            # Prediction is in [-1, 1] range from world model
            # De-normalize: [-1, 1] -> [0, 1] -> [0, 255]
            pred_np = ((pred + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
            pred_img = (np.transpose(pred_np, (1, 2, 0)) * 255.0).astype(np.uint8)  # [H, W, C]

            # Log image stats for debugging
            if head_img is not None:
                logger.debug(f"Head image: range=[{head_img.min()}, {head_img.max()}], mean={head_img.mean():.1f}")
            logger.debug(f"GT wrist image: range=[{gt_img.min()}, {gt_img.max()}], mean={gt_img.mean():.1f}")
            logger.debug(f"Pred image: range=[{pred_img.min()}, {pred_img.max()}], mean={pred_img.mean():.1f}")

            # Compose comparison image using PIL
            from PIL import ImageDraw
            gt_pil = Image.fromarray(gt_img)
            pred_pil = Image.fromarray(pred_img)
            
            if head_img is not None:
                # 3-panel layout: [head] [GT wrist] [Pred wrist]
                head_pil = Image.fromarray(head_img)
                combined_w = head_pil.width + gt_pil.width + pred_pil.width + 60  # gaps
                combined_h = max(head_pil.height, gt_pil.height, pred_pil.height) + 30  # label space
                combined = Image.new('RGB', (combined_w, combined_h), color=(255, 255, 255))
                
                x_offset = 10
                combined.paste(head_pil, (x_offset, 25))
                x_offset += head_pil.width + 20
                combined.paste(gt_pil, (x_offset, 25))
                x_offset_gt = x_offset
                x_offset += gt_pil.width + 20
                combined.paste(pred_pil, (x_offset, 25))
                
                # Draw labels
                draw = ImageDraw.Draw(combined)
                draw.text((10, 5), "Head (context)", fill=(0, 0, 0))
                draw.text((x_offset_gt, 5), "GT (wrist)", fill=(0, 0, 0))
                draw.text((x_offset, 5), "Predicted", fill=(0, 0, 0))
            else:
                # 2-panel layout: [GT wrist] [Pred wrist]
                combined_w = gt_pil.width + pred_pil.width + 40
                combined_h = max(gt_pil.height, pred_pil.height) + 30
                combined = Image.new('RGB', (combined_w, combined_h), color=(255, 255, 255))
                combined.paste(gt_pil, (10, 25))
                combined.paste(pred_pil, (gt_pil.width + 30, 25))
                
                draw = ImageDraw.Draw(combined)
                draw.text((10, 5), "GT (wrist_rgb)", fill=(0, 0, 0))
                draw.text((gt_pil.width + 30, 5), "Predicted", fill=(0, 0, 0))
            
            out_path = self.output_dir / "samples" / f"episode_{episode:06d}.png"
            combined.save(out_path)

        except Exception:
            logger.exception("Error while saving prediction sample")
    
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
                except Exception as e:
                    logger.debug(f"Prediction failed for frame {i}: {e}")
                
                # Use GT as fallback if prediction failed
                if pred_frame is None:
                    pred_frame = gt_frame.copy()
                pred_frames.append(pred_frame)
            
            if not gt_frames:
                logger.warning(f"No valid frames to save for episode {episode}")
                return
            
            # Determine frame dimensions
            h, w = gt_frames[0].shape[:2]
            
            # Layout: [Head | GT Wrist | Predicted]
            # Each panel is w x h, with 5px gaps between panels
            gap = 5
            label_h = 25
            combined_w = w * 3 + gap * 2
            combined_h = h + label_h
            
            # Use imageio for better compatibility
            import imageio
            writer = imageio.get_writer(
                str(video_path), fps=10, codec='libx264',
                pixelformat='yuv420p', quality=8
            )
            
            num_frames = min(len(head_frames), len(gt_frames), len(pred_frames))
            for i in range(num_frames):
                # Create combined frame (white background)
                combined = np.ones((combined_h, combined_w, 3), dtype=np.uint8) * 255
                
                x_offset = 0
                
                # Panel 1: Head Camera
                if head_frames[i] is not None:
                    head_frame = head_frames[i]
                    # Resize if needed
                    if head_frame.shape[:2] != (h, w):
                        head_frame = cv2.resize(head_frame, (w, h))
                    combined[label_h:label_h+h, x_offset:x_offset+w] = head_frame
                cv2.putText(combined, "Head", (x_offset + w//2 - 20, 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                x_offset += w + gap
                
                # Panel 2: GT Wrist
                gt_frame = gt_frames[i]
                if gt_frame.shape[:2] != (h, w):
                    gt_frame = cv2.resize(gt_frame, (w, h))
                combined[label_h:label_h+h, x_offset:x_offset+w] = gt_frame
                cv2.putText(combined, "GT Wrist", (x_offset + w//2 - 35, 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 1)
                x_offset += w + gap
                
                # Panel 3: Predicted Wrist
                pred_frame = pred_frames[i]
                if pred_frame.shape[:2] != (h, w):
                    pred_frame = cv2.resize(pred_frame, (w, h))
                combined[label_h:label_h+h, x_offset:x_offset+w] = pred_frame
                cv2.putText(combined, "Predicted", (x_offset + w//2 - 40, 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Write frame (imageio expects RGB)
                writer.append_data(combined)
            
            writer.close()
            logger.info(f"[Video] Saved: {video_path} ({num_frames} frames, start_idx={start_idx})")
            
        except Exception:
            logger.exception("Error saving combined video")
        finally:
            policy.train()  # Restore training mode

    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return starting episode."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.metrics["total_steps"] = state.get("total_steps", 0)
            return state.get("episode", 0)
        
        return 0


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
    if args.resume:
        if accelerator.is_main_process:
            print(f"Resuming from: {args.resume}")
        start_episode = trainer.load_checkpoint(args.resume)
    
    if accelerator.is_main_process:
        print("")  # Empty line before training
    
    # Train
    trainer.train(start_episode=start_episode)


if __name__ == "__main__":
    main()
