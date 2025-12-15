#!/usr/bin/env python3
"""
Student Policy Training Script for F1-VLA (Phase 2)

Refactored version using shared rl_training_common module.

Phase 2 Training:
- Initialize new LLM but REUSE frozen World Model from teacher
- Use only wrist camera observations (no head camera)
- Explorer (F1-VLA actor) generates actions instead of random
- Reward based on:
  1. Memory RNN hidden state divergence between student and teacher
  2. Actions that make World Model unable to accurately predict next frame
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
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig

# Set CUDA device immediately to avoid NCCL warnings/hangs
if torch.cuda.is_available():
    # Try to get local rank from env vars (LOCAL_RANK is standard for torchrun/accelerate)
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        try:
            device_id = int(local_rank_env)
            torch.cuda.set_device(device_id)
            # print(f"Process {os.getpid()} set torch cuda device to {device_id}") 
        except Exception as e:
            print(f"Failed to set device: {e}")

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
    resolve_device_and_process,
    print_startup_header,
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
    parser = argparse.ArgumentParser(description="Train Student Policy (Explorer) - Phase 2")
    
    # Config file (recommended way)
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to RL training config YAML file")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Override model config file path")
    
    # Teacher checkpoint (required)
    parser.add_argument("--teacher_path", type=str, required=True,
                       help="Path to trained teacher policy checkpoint")
    
    # Override common training parameters
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    
    # Reward weights
    parser.add_argument("--memory_divergence_weight", type=float, default=None)
    parser.add_argument("--wm_uncertainty_weight", type=float, default=None)
    
    # Sequential training (can override config)
    parser.add_argument("--sequential_training", action="store_true", default=None)
    parser.add_argument("--no_sequential_training", action="store_false", 
                       dest="sequential_training")
    
    # Device & debug
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint directory to resume training from")
    parser.add_argument("--auto_resume", action="store_true", default=False,
                       help="Automatically resume from latest checkpoint in output_dir")
    
    # DDP and multi-environment options
    parser.add_argument("--num_envs", type=int, default=1,
                       help="Number of parallel environments for data collection")
    parser.add_argument("--use_ddp", action="store_true",
                       help="Use distributed data parallel training")
    
    return parser.parse_args()


class StudentTrainer(BaseRLTrainer):
    """
    Trainer for student policy (Explorer).
    
    Uses PPO-style policy gradient with custom rewards:
    1. Memory divergence: change in teacher's memory state
    2. WM uncertainty: prediction error (actions that surprise WM)
    
    Supports DDP via HuggingFace Accelerate and multi-environment collection
    via gymnasium's SyncVectorEnv.
    """
    
    def __init__(
        self,
        student_policy: nn.Module,
        teacher_policy: nn.Module,
        policy_config,
        rl_config: OmegaConf,
        model_config: Union[str, DictConfig],
        device: str = "cuda",
        accelerator: Optional[AcceleratorWrapper] = None,
        num_envs: int = 1,
    ):
        # Store accelerator for DDP support
        self.accelerator = accelerator
        self.num_envs = num_envs
        
        # Load model config to get n_obs_img_steps and stride
        if isinstance(model_config, (str, Path)):
            import yaml
            with open(model_config, 'r') as f:
                model_cfg = yaml.safe_load(f)
            model_config_desc = str(model_config)
        else:
            # Assume DictConfig or dict
            model_cfg = OmegaConf.to_container(model_config, resolve=True)
            model_config_desc = f"inline config ({type(model_config).__name__})"
        
        train_datasets = model_cfg.get('dataset', {}).get('train_dir', {})
        if not train_datasets:
            raise ValueError("No train datasets found in model config")
        
        first_dataset = next(iter(train_datasets.values()))
        self.n_obs_img_steps = first_dataset.get('n_obs_img_steps', 4)
        self.obs_img_stride = first_dataset.get('obs_img_stride', 1)
        
        # Get training config
        train_config = get_training_config(rl_config)
        student_config = rl_config.get("student", {})
        output_dir = student_config.get("output_dir", "./outputs/student_rl")
        
        # Override config with values from model config
        self.n_pred_img_steps = train_config.n_pred_img_steps
        train_config.history_length = self.n_obs_img_steps  # observation buffer
        
        super().__init__(
            policy=student_policy,
            config=train_config,
            output_dir=output_dir,
            device=device,
            accelerator=accelerator,  # Pass accelerator to base class
        )
        
        # Note: train_config.num_episodes is automatically adjusted for DDP in BaseRLTrainer.__init__
        
        self.student_policy = student_policy
        self.teacher_policy = teacher_policy
        self.policy_config = policy_config
        self.rl_config = rl_config
        
        print(f"Student policy config from {model_config_desc}:")
        print(f"  n_obs_img_steps: {self.n_obs_img_steps}")
        print(f"  n_pred_img_steps: {self.n_pred_img_steps}")
        print(f"  history_length: {train_config.history_length}")
        
        # Freeze teacher (unwrap DDP model first)
        teacher_policy_unwrapped = self.accelerator.unwrap_model(self.teacher_policy) if self.accelerator else self.teacher_policy
        teacher_policy_unwrapped.eval()
        for param in teacher_policy_unwrapped.parameters():
            param.requires_grad = False
        
        # World model image steps (for compatibility)
        self.cur_n_obs_img_steps = self.n_obs_img_steps
        self.cur_n_pred_img_steps = self.n_pred_img_steps
        
        # Reward weights
        rewards_config = student_config.get("rewards", {})
        # Increase memory divergence weight to prioritize teacher imitation
        self.memory_divergence_weight = rewards_config.get("memory_divergence_weight", 1.0)
        # Decrease WM uncertainty weight to prevent it from dominating the reward
        self.wm_uncertainty_weight = rewards_config.get("wm_uncertainty_weight", 0.01)
        
        # PPO parameters - adjusted for stability
        ppo_config = student_config.get("ppo", {})
        self.ppo_config = ppo_config  # Store for later use
        self.clip_epsilon = ppo_config.get("clip_epsilon", 0.1)  # Reduced from 0.2 for stability
        self.entropy_coef = ppo_config.get("entropy_coef", 0.05)  # Increased to 0.05 for more exploration
        self.value_loss_coef = ppo_config.get("value_loss_coef", 0.5)
        self.gamma = ppo_config.get("gamma", 0.99)
        self.gae_lambda = ppo_config.get("gae_lambda", 0.95)
        
        # Value head for PPO - MUST be initialized before optimizer
        # state_proj outputs [B, proj_width] where proj_width=1024 typically
        proj_width = policy_config.proj_width if hasattr(policy_config, 'proj_width') else 1024
        self.value_head = nn.Linear(proj_width, 1).to(device)
        # Initialize log std to a small value to stabilize early training
        self.log_std = nn.Parameter(torch.ones(train_config.action_dim, device=device) * -2.0)
        
        # Setup student policy for training - train action expert only
        # Note: unwrap DDP model for training setup, but use wrapped model for forward
        student_policy_unwrapped = self.accelerator.unwrap_model(self.student_policy) if self.accelerator else self.student_policy
        student_policy_unwrapped.train()
        
        # Set gradient flags: train action expert, freeze world model
        print("\nConfiguring student training: Action Expert only")
        set_policy_requires_grad(
            student_policy_unwrapped,
            freeze_vision_encoder=True,
            freeze_gen_expert=True,  # Freeze world model (use teacher's WM)
            train_act_expert_only=True,  # Only train action prediction
            train_gen_expert_only=False,
        )
        
        # Setup optimizer and scheduler
        trainable, total = count_trainable_params(self.student_policy)
        self._print(f"Student trainable parameters: {trainable:,} / {total:,}")
        
        # Collect all trainable parameters: student_policy + value_head + log_std
        param_groups = [
            {'params': self.student_policy.parameters()},
            {'params': self.value_head.parameters()},
            {'params': [self.log_std]},  # log_std is a single parameter
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            # Reduce LR for stability (scale down by 10x from training config)
            lr=max(1e-8, float(train_config.learning_rate) * 0.1),
            weight_decay=train_config.weight_decay,
        )
        self.scheduler = setup_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            T_max=train_config.num_episodes,
            eta_min=1e-6,
        )
        
        # Prepare model and optimizer for DDP if accelerator is provided
        if self.accelerator is not None:
            self.student_policy, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.student_policy, self.optimizer, self.scheduler
            )
            self._print(f"DDP setup complete: {self.accelerator.num_processes} processes")
            
        # Setup parallel environment collector if num_envs > 1
        self.env_collector = None  # Will be set in setup_environment
        
        # Environment config
        self.env_config = get_environment_config(rl_config)
        
        # Setup batch builder - student uses only wrist camera
        self.batch_builder = BatchBuilder(
            device=device,
            image_keys=["head_rgb", "wrist_rgb"],  # Need both for WM history
            use_head_camera=False,  # Student: wrist_rgb only (image0) for VLM
        )
        
        # Teacher uses head + wrist cameras
        self.teacher_batch_builder = BatchBuilder(
            device=device,
            image_keys=["head_rgb", "wrist_rgb"],
            use_head_camera=True,  # Teacher: head_rgb (image0) + wrist_rgb (image1)
        )
        
        # Memory configuration from model config (for GRU state)
        self.memory_enabled = policy_config.memory_enabled if hasattr(policy_config, 'memory_enabled') else True
        self.memory_hidden = policy_config.memory_hidden if hasattr(policy_config, 'memory_hidden') else 2048
        self.memory_num_layers = policy_config.memory_num_layers if hasattr(policy_config, 'memory_num_layers') else 4
        
        self._print(f"Memory config: enabled={self.memory_enabled}, hidden={self.memory_hidden}, layers={self.memory_num_layers}")
        
        # Memory managers for student and teacher
        self.student_memory = MemoryStateManager()
        self.teacher_memory = MemoryStateManager()
        # Track previous divergence for reward computation
        self.prev_memory_divergence: float = 0.0
        
        # Additional metrics
        self.metrics.update({
            "policy_loss": deque(maxlen=100),
            "value_loss": deque(maxlen=100),
            "entropy": deque(maxlen=100),
            "memory_divergence": deque(maxlen=100),
            "wm_uncertainty": deque(maxlen=100),
            "episode_reward": deque(maxlen=100),
        })
        # Video recording buffers (for on-disk MP4 saving and TB)
        self.video_frames_head = []   # Head camera frames (HWC uint8)
        self.video_frames_wrist = []  # Wrist camera frames (HWC uint8)
        self.video_transitions = []   # Store transitions for prediction

        # Video save frequency (episodes)
        train_cfg = rl_config.get("training", {})
        self.video_save_every = int(train_cfg.get("video_save_every", 1))
        if self._is_main_process():
            (Path(self.output_dir) / "videos").mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """
        Load checkpoint with Student-specific state (value_head, log_std).
        
        Extends base class to also restore PPO-specific components.
        """
        # Call base class to load model, optimizer, scheduler
        start_step = super().load_checkpoint(checkpoint_dir)
        
        # Load Student-specific state
        checkpoint_path = Path(checkpoint_dir)
        trainer_state_path = checkpoint_path / "trainer_state.pt"
        
        if trainer_state_path.exists():
            try:
                state = torch.load(trainer_state_path, map_location=self.device)
                
                # Restore value head
                if "value_head" in state:
                    self.value_head.load_state_dict(state["value_head"])
                    logger.info("Loaded value_head state")
                
                # Restore log_std
                if "log_std" in state:
                    self.log_std.data = state["log_std"].to(self.device)
                    logger.info("Loaded log_std state")
                    
            except Exception as e:
                logger.warning(f"Could not load Student-specific state: {e}")
        
        return start_step
    
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
        logger.debug(f"Student: initialized zero memory state: shape={memory_state.shape}")
        return memory_state
    
    def _print(self, msg: str):
        """Print only on main process."""
        if self._is_main_process():
            logger.info(msg)
    
    def _is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process
    
    def setup_environment(self):
        """Setup the RL environment with optional multi-env support."""
        from rl.f1_rl_env import StudentEnv
        
        # Get GPU ID for this process
        # IMPORTANT: When CUDA_VISIBLE_DEVICES is set, GPUs are remapped to 0,1,2,...
        # So we should use local_process_index directly as the render device
        local_gpu_id = 0
        render_device_id = 0
        
        if self.accelerator is not None:
            local_process_idx = self.accelerator.local_process_index
            render_device_id = local_process_idx
            
            # Calculate physical ID for Vulkan
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cuda_visible:
                visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                if local_process_idx < len(visible_gpus):
                    local_gpu_id = visible_gpus[local_process_idx]
                else:
                    local_gpu_id = local_process_idx
            else:
                local_gpu_id = local_process_idx
            
            # Set RL_MAIN_PROCESS to control logging in _base_task.py
            os.environ["RL_MAIN_PROCESS"] = "1" if self._is_main_process() else "0"
            
            # Set EGL device for SAPIEN rendering (use remapped ID)
            os.environ["EGL_DEVICE_ID"] = str(local_gpu_id)
            os.environ["VK_DEVICE_INDEX"] = str(local_gpu_id)
        
        # Get environment parameters (same as teacher)
        single_arm = self.env_config.get("single_arm", False)
        scene_reset_interval = self.env_config.get("scene_reset_interval", 1)
        randomize_robot_init = self.env_config.get("randomize_robot_init", False)
        need_planner = self.env_config.get("need_planner", False)
        need_topp = self.env_config.get("need_topp", False)
        
        # Unwrap teacher policy for environment (environment calls model directly)
        teacher_policy_unwrapped = self.accelerator.unwrap_model(self.teacher_policy) if self.accelerator else self.teacher_policy
        
        def make_env():
            # Capture local_gpu_id in closure
            gpu_id = render_device_id
            return StudentEnv(
                task_config={
                    **self.env_config,
                    "need_planner": need_planner,
                    "need_topp": need_topp,
                    "render_device": gpu_id,
                },
                teacher_policy=teacher_policy_unwrapped,
                history_length=self.config.history_length,
                max_steps=self.config.steps_per_episode,
                device=self.device,
                action_scale=self.config.action_scale,
                single_arm=single_arm,
                scene_reset_interval=scene_reset_interval,
                randomize_robot_init=randomize_robot_init,
            )
        
        # Single env for compatibility
        self.env = make_env()
        
        # Multi-env collector
        if self.num_envs > 1:
            self.env_collector = ParallelEnvCollector(
                env_fn=make_env,
                num_envs=self.num_envs,
                is_main_process=self._is_main_process(),
            )
            self.env_collector.initialize()
            self._print(f"Parallel env collector setup: {self.num_envs} envs")
        
        self._print(f"Student environment setup complete")
    
    def collect_episode(self, use_tqdm: bool = False) -> List[Dict[str, Any]]:
        """Collect one episode using student policy."""
        logger.debug(f"[collect_episode] Starting reset...")
        obs, info = self.env.reset()
        logger.debug(f"[collect_episode] Reset complete, starting rollout")
        transitions = []
        
        # Reset memory states (will be initialized to zeros on first step)
        self.student_memory.reset()
        self.teacher_memory.reset()
        self.prev_memory_divergence = 0.0
        
        done = False
        step = 0
        
        # Use tqdm for steps if requested
        pbar = None
        
        while not done:
            # Build batch for student policy
            batch = self._obs_to_batch(obs)
            
            # Inject student memory state - MUST initialize to zeros for first frame
            if self.student_memory.current_memory is not None:
                batch["initial_memory_state"] = self.student_memory.current_memory
            else:
                # First frame: initialize to zeros
                batch["initial_memory_state"] = self._init_memory_state(batch_size=1)
                logger.debug(f"Student episode: initialized zero memory for first frame")
            
            # Get action from student policy (also updates student memory)
            with torch.no_grad():
                actions, log_probs, values, student_memory_out, wm_logits = self._forward_student(batch)
            
            # Update student memory state
            if student_memory_out is not None:
                self.student_memory.update(student_memory_out)
            
            action = actions[0].cpu().numpy()
            log_prob = log_probs[0].item()
            value = values[0].item()
            
            # Random action for first step
            if step == 0:
                action = np.random.uniform(-1, 1, self.config.action_dim).astype(np.float32)
            
            # Execute action
            next_obs, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Collect frames for video recording
            try:
                if self._is_main_process():
                    self._collect_video_frame(obs, action)
            except Exception:
                logger.exception("Failed to collect video frame during episode collection")
            
            # Compute custom reward (pass wm_logits from student model)
            reward, reward_info = self._compute_custom_reward(obs, batch, wm_logits=wm_logits)
            
            # Update progress bar with metrics
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    "rew": f"{reward:.2f}",
                    "mem_div": f"{reward_info.get('memory_divergence_abs', 0):.2f}"
                })
            
    
            
            transitions.append({
                "obs": obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "info": {**info, **reward_info},
                # include initial memory state used for this step (may be None)
                "initial_memory_state": batch.get("initial_memory_state"),
            })
            
            obs = next_obs
            step += 1
        
        if pbar:
            pbar.close()
        
        # Compute advantages using GAE
        self._compute_advantages(transitions)
        
        return transitions

    def _process_obs_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Convert observation image to HWC uint8 format."""
        import numpy as _np
        if img is None:
            return None
        if not isinstance(img, _np.ndarray):
            img = _np.array(img)

        # If stacked history (T, C, H, W) take last
        if img.ndim == 4:
            img = img[-1]

        # If CHW -> HWC
        if img.ndim == 3 and img.shape[0] == 3:
            img = _np.transpose(img, (1, 2, 0))

        if img.ndim != 3:
            logger.warning(f"Unexpected image dims: {getattr(img, 'ndim', None)}")
            return None

        # Ensure uint8
        if img.dtype != _np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(_np.uint8)
            else:
                img = img.astype(_np.uint8)

        return img

    def _collect_video_frame(self, obs: Dict[str, np.ndarray], action: Any = None):
        """Collect frames and transition for later MP4 generation.

        Mirrors teacher implementation but simplified for student (head may be absent).
        """
        if not self._is_main_process():
            return

        # Head frame
        head_img = obs.get("head_rgb")
        head_frame = None
        if head_img is not None:
            head_frame = self._process_obs_image(head_img)
        if head_frame is not None:
            self.video_frames_head.append(head_frame.copy())
        else:
            # Keep alignment with wrist frames
            self.video_frames_head.append(None)

        # Wrist frame
        wrist_img = obs.get("wrist_rgb")
        wrist_frame = None
        if wrist_img is not None:
            wrist_frame = self._process_obs_image(wrist_img)
        if wrist_frame is not None:
            self.video_frames_wrist.append(wrist_frame.copy())
        else:
            self.video_frames_wrist.append(None)

        # Store transition (limit length to avoid memory blowup)
        if len(self.video_transitions) < 200:
            self.video_transitions.append({
                "obs": {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in obs.items()},
                "action": (action.copy() if hasattr(action, 'copy') else action),
            })

    def _save_episode_video(self, episode: int):
        """Save a combined MP4 for the episode: [Head | GT Wrist | Predicted Wrist]."""
        if not self._is_main_process():
            return

        if not self.video_transitions or len(self.video_transitions) < self.n_obs_img_steps + 1:
            logger.warning(f"Not enough frames for video (need {self.n_obs_img_steps + 1}, got {len(self.video_transitions)})")
            # Clear buffers
            self.video_frames_head = []
            self.video_frames_wrist = []
            self.video_transitions = []
            return

        try:
            video_path = Path(self.output_dir) / "videos" / f"episode_{episode:06d}.mp4"

            # Prepare writer
            import imageio
            import cv2
            import numpy as _np

            tb_frames = []

            start_idx = self.n_obs_img_steps - 1
            memory_state = None

            # Get unwrapped teacher policy if available for prediction
            teacher_policy = None
            try:
                teacher_policy = self.accelerator.unwrap_model(self.teacher_policy) if self.accelerator else self.teacher_policy
                teacher_policy.eval()
            except Exception:
                teacher_policy = None

            for i in range(start_idx, len(self.video_transitions) - 1):
                trans = self.video_transitions[i]
                next_trans = self.video_transitions[i + 1]

                obs = trans["obs"]
                next_obs = next_trans["obs"]

                # Head panel
                head_frame = None
                if i < len(self.video_frames_head):
                    head_frame = self.video_frames_head[i]
                if head_frame is None:
                    # white placeholder
                    if self.video_frames_wrist and self.video_frames_wrist[0] is not None:
                        h, w = self.video_frames_wrist[0].shape[:2]
                    else:
                        h, w = 256, 256
                    head_frame = _np.ones((h, w, 3), dtype=_np.uint8) * 255

                # GT wrist from next_obs
                gt_wrist = next_obs.get("wrist_rgb")
                if gt_wrist is None:
                    continue
                if getattr(gt_wrist, 'ndim', 3) == 4:
                    gt_wrist = gt_wrist[-1]
                gt_frame = self._process_obs_image(gt_wrist)
                if gt_frame is None:
                    continue

                # Prediction using teacher_policy if available
                pred_frame = None
                try:
                    if teacher_policy is not None:
                        # Build batch using env helper if present
                        if hasattr(self.env, '_build_policy_batch'):
                            batch = self.env._build_policy_batch(obs, np.array(trans.get('action', 0), dtype=np.float32), use_head_camera=True)
                        else:
                            # Fallback: create minimal batch
                            batch = {
                                "observation.state": torch.from_numpy(obs.get('state')).float().unsqueeze(0).to(self.device),
                            }

                        if memory_state is None:
                            memory_state = self._init_memory_state(batch_size=1)
                        batch["initial_memory_state"] = memory_state

                        with torch.no_grad():
                            pred_out = teacher_policy.predict_images_only(batch)
                        memory_state = pred_out.get("memory_state")
                        pred_imgs = pred_out.get("pred_imgs")
                        if pred_imgs is not None:
                            pred = pred_imgs.detach().cpu()
                            if pred.ndim == 5:
                                pred = pred[:, -1]
                            pred = pred[0]
                            pred_np = ((pred + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
                            pred_frame = (np.transpose(pred_np, (1, 2, 0)) * 255.0).astype(_np.uint8)
                            # Resize to GT size
                            if pred_frame.shape[:2] != gt_frame.shape[:2]:
                                pred_frame = cv2.resize(pred_frame, (gt_frame.shape[1], gt_frame.shape[0]))
                except Exception:
                    logger.debug("Prediction failed for video generation")

                if pred_frame is None:
                    pred_frame = gt_frame.copy()

                # Compose combined frame
                gap = 5
                label_h = 25
                h, w = gt_frame.shape[:2]
                raw_w = w * 3 + gap * 2
                raw_h = h + label_h
                combined_w = ((raw_w + 15) // 16) * 16
                combined_h = ((raw_h + 15) // 16) * 16
                pad_x = (combined_w - raw_w) // 2
                pad_y = (combined_h - raw_h) // 2

                combined = _np.ones((combined_h, combined_w, 3), dtype=_np.uint8) * 255
                x_offset = pad_x
                y_offset = pad_y

                # Head
                hf = head_frame
                if hf.shape[:2] != (h, w):
                    hf = cv2.resize(hf, (w, h))
                combined[y_offset+label_h:y_offset+label_h+h, x_offset:x_offset+w] = hf
                x_offset += w + gap

                # GT
                gf = gt_frame
                if gf.shape[:2] != (h, w):
                    gf = cv2.resize(gf, (w, h))
                combined[y_offset+label_h:y_offset+label_h+h, x_offset:x_offset+w] = gf
                x_offset += w + gap

                # Pred
                pf = pred_frame
                if pf.shape[:2] != (h, w):
                    pf = cv2.resize(pf, (w, h))
                combined[y_offset+label_h:y_offset+label_h+h, x_offset:x_offset+w] = pf

                tb_frames.append(combined)

            # Write mp4
            writer = imageio.get_writer(str(video_path), fps=10, codec='libx264', pixelformat='yuv420p', quality=8)
            for f in tb_frames:
                writer.append_data(f)
            writer.close()

            # Also add to TensorBoard
            try:
                if self.writer is not None and tb_frames:
                    from rl.training.rl_training_common import add_video_to_writer
                    add_video_to_writer(self.writer, f"video/episode_{episode:06d}", tb_frames, episode, fps=10)
            except Exception:
                logger.exception("Failed to add episode video to TensorBoard")

            logger.info(f"[Video] Saved student episode video: {video_path}")

        except Exception:
            logger.exception("Error saving combined video for student")
        finally:
            # Clear buffers
            self.video_frames_head = []
            self.video_frames_wrist = []
            self.video_transitions = []
    
    def _obs_to_batch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to batch tensor dict.
        
        Camera mapping:
        - Teacher Paligemma: head_rgb (image0) + wrist_rgb (image1)
        - Student Paligemma: wrist_rgb only (image0)
        - World Model: always uses wrist_rgb history (image0_history)
        """
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
        
        # For teacher batch (if needed), add head_rgb as image0, wrist_rgb as image1
        if "head_rgb" in obs:
            head_imgs = obs["head_rgb"]
            current_head = head_imgs[-1]
            # Teacher uses head_rgb (image0) + wrist_rgb (image1)
            # But we need separate teacher_batch for this
        
        return batch
    
    def _forward_student(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through student model.
        
        Returns:
            actions: Sampled actions [B, action_dim]
            log_probs: Log probabilities [B]
            values: Value estimates [B]
            memory_state: Updated memory state (or None)
            wm_logits: World model logits for uncertainty computation (or None)
        """
        # Unwrap DDP model if needed
        policy = self.accelerator.unwrap_model(self.student_policy) if self.accelerator else self.student_policy
        
        # Access the underlying F1FlowMatching model through PEFT wrapper
        # PEFT structure: PeftModel -> base_model (LoraModel) -> model (F1_VLA) -> model (F1FlowMatching)
        if hasattr(policy, 'base_model'):
            # PEFT wrapped model
            f1_vla = policy.base_model.model  # F1_VLA
            f1_flow = f1_vla.model  # F1FlowMatching
        else:
            # Direct F1_VLA model
            f1_vla = policy
            f1_flow = policy.model
        
        # Use student model to generate actions
        noise = torch.randn(
            batch["observation.state"].shape[0], self.policy_config.chunk_size,
            self.policy_config.max_action_dim,
            device=self.device
        )
        
        # ===== Prepare world_model_input_embs from image history =====
        # World Model needs VAE-encoded history images
        world_model_input_embs = None
        if "observation.images.image0_history" in batch:
            world_model_images = batch["observation.images.image0_history"]
            B, T, C, H, W = world_model_images.shape
            
            # VQ-VAE expects 256x256 images
            if H != 256 or W != 256:
                world_model_images = world_model_images.reshape(B * T, C, H, W)
                world_model_images = torch.nn.functional.interpolate(
                    world_model_images, size=(256, 256), mode='bilinear', align_corners=False
                )
                world_model_images = world_model_images.reshape(B, T, C, 256, 256)
            
            
            # Encode through VAE
            world_model_images_flat = world_model_images.reshape(B * T, C, 256, 256)
            world_model_indices_list = f1_flow.vae.img_to_idxBl(world_model_images_flat)
            world_model_input_embs = f1_flow.vae.quantize.idxBl_to_var_input(world_model_indices_list)
            world_model_input_embs = world_model_input_embs.reshape(B, T, *world_model_input_embs.shape[1:])
        
        # Prepare language tokens if needed
        lang_tokens = None
        lang_masks = None
        if "task" in batch:
            # Use f1_vla's language tokenizer
            tasks = batch["task"]
            tasks = [t if t.endswith("\n") else f"{t}\n" for t in tasks]
            tokenized = f1_vla.language_tokenizer(
                tasks,
                padding="max_length",
                padding_side="right",
                max_length=f1_vla.config.tokenizer_max_length,
                return_tensors="pt",
                truncation=True,
            )
            lang_tokens = tokenized["input_ids"].to(device=self.device)
            lang_masks = tokenized["attention_mask"].to(device=self.device, dtype=torch.bool)
        
        # Prepare state with padding
        state = batch["observation.state"]
        if state.shape[-1] < self.policy_config.max_state_dim:
            pad_size = self.policy_config.max_state_dim - state.shape[-1]
            state = torch.nn.functional.pad(state, (0, pad_size))
        
        # Sample actions from F1FlowMatching model
        # Student uses wrist_rgb as image0 (see _obs_to_batch)
        # Get image0 and create default mask with matching batch size
        image0 = batch.get("observation.images.image0")
        batch_size = image0.shape[0] if image0 is not None else state.shape[0]
        image0_mask = batch.get("observation.images.image0_mask", 
                               torch.ones(batch_size, dtype=torch.bool, device=self.device))
        
        action_output = f1_flow.sample_actions_with_world_model(
            images=[image0],
            image_masks=[image0_mask],
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            world_model_input_embs=world_model_input_embs,
            predict_action_only=True,
            noise=noise,
            action_history=batch.get("action_history"),
            initial_memory_state=batch.get("initial_memory_state"),
        )
        
        # Handle output format: (action, memory_state, wm_logits)
        wm_logits = None
        if isinstance(action_output, tuple):
            action_tensor = action_output[0]
            memory_state = action_output[1] if len(action_output) > 1 else None
            wm_logits = action_output[2] if len(action_output) > 2 else None
        else:
            action_tensor = action_output
            memory_state = None
        
        action_mean = action_tensor[:, 0, :]
        std = torch.exp(self.log_std)
        
        if actions is not None:
            # Evaluate provided actions
            dist = torch.distributions.Normal(action_mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
        elif deterministic:
            actions = action_mean
            log_probs = torch.zeros(1, device=self.device)
        else:
            dist = torch.distributions.Normal(action_mean, std)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Value estimate - use f1_flow.state_proj
        state_emb = f1_flow.state_proj(batch["observation.state"])
        values = self.value_head(state_emb).squeeze(-1)
        
        return actions, log_probs, values, memory_state, wm_logits
    
    def _compute_custom_reward(
        self,
        obs: Dict[str, np.ndarray],
        batch: Dict[str, torch.Tensor],
        wm_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute custom reward for student to imitate teacher.
        
        Memory divergence reward (negative to encourage convergence):
            r_mem = -(||h_student^{t+1} - h_teacher^{t+1}|| - ||h_student^t - h_teacher^t||)
        
        Positive reward when student gets closer to teacher (divergence decreases).
        Negative reward when student diverges from teacher.
        """
        # ===== Get student memory at t+1 =====
        # Student forward pass (already done in collect_episode, get memory from manager)
        student_memory_t1 = self.student_memory.current_memory
        
        # ===== Get teacher memory at t+1 =====
        # Inject teacher memory state - initialize to zeros if first frame
        teacher_batch = batch.copy()
        if self.teacher_memory.current_memory is not None:
            teacher_batch["initial_memory_state"] = self.teacher_memory.current_memory
        else:
            # First frame: initialize to zeros
            teacher_batch["initial_memory_state"] = self._init_memory_state(batch_size=1)
        
        # Get teacher's world model prediction (unwrap DDP model)
        teacher_policy = self.accelerator.unwrap_model(self.teacher_policy) if self.accelerator else self.teacher_policy
        with torch.no_grad():
            # Use forward_memory_only instead of forward_with_world_model
            # forward_with_world_model requires target images which we don't have in RL inference
            wm_output = teacher_policy.forward_memory_only(
                teacher_batch,
                use_student_mode=False,  # Teacher mode: head + wrist cameras
                prev_memory_state=teacher_batch.get("initial_memory_state"),
            )
        
        # Update teacher memory
        teacher_memory_t1 = wm_output.get("memory_state")
        if teacher_memory_t1 is not None:
            self.teacher_memory.update(teacher_memory_t1)
        else:
            logger.warning("Teacher WM returned None memory_state")
        
        # ===== Compute memory divergence reward =====
        # r_mem = -(||h_student^{t+1} - h_teacher^{t+1}|| - ||h_student^t - h_teacher^t||)
        # Negative sign: reward decreasing divergence (student gets closer to teacher)
        memory_divergence_reward = 0.0
        current_divergence = 0.0
        
        # Debug: log memory state availability
        if student_memory_t1 is None:
            logger.debug("Student memory_t1 is None")
        else:
            logger.debug(f"Student memory_t1 shape: {student_memory_t1.shape}, mean: {student_memory_t1.mean().item():.4f}, std: {student_memory_t1.std().item():.4f}")
        
        if teacher_memory_t1 is None:
            logger.debug("Teacher memory_t1 is None")
        else:
            logger.debug(f"Teacher memory_t1 shape: {teacher_memory_t1.shape}, mean: {teacher_memory_t1.mean().item():.4f}, std: {teacher_memory_t1.std().item():.4f}")
        
        if student_memory_t1 is not None and teacher_memory_t1 is not None:
            # Compute ||h_student^{t+1} - h_teacher^{t+1}||
            current_divergence = torch.norm(student_memory_t1 - teacher_memory_t1).item()
            
            # Handle initialization: if prev_divergence is 0 (first step), set reward to 0
            if self.prev_memory_divergence == 0.0:
                logger.info(f"[Step {getattr(self, 'global_step', 0)}] First memory divergence measurement: {current_divergence:.4f}, setting reward=0")
                memory_divergence_reward = 0.0
            else:
                # r_mem has two components:
                # 1. Progress reward: -(current - prev), positive when getting closer
                # 2. Proximity penalty: penalize being far away
                progress_reward = -(current_divergence - self.prev_memory_divergence)
                proximity_penalty = -0.07 * current_divergence  # Penalize absolute distance (increased for stronger guidance)
                memory_divergence_reward = progress_reward + proximity_penalty
                
                # Log detailed information every 50 steps to reduce verbosity
                if getattr(self, 'global_step', 0) % 50 == 0:
                    logger.debug(f"[Step {getattr(self, 'global_step', 0)}] Memory Divergence Details:")
                    logger.debug(f"  Current divergence: {current_divergence:.6f}")
                    logger.debug(f"  Previous divergence: {self.prev_memory_divergence:.6f}")
                    logger.debug(f"  Change (curr-prev): {current_divergence - self.prev_memory_divergence:+.6f}")
                    logger.debug(f"  Progress reward: {progress_reward:+.6f}")
                    logger.debug(f"  Proximity penalty: {proximity_penalty:+.6f}")
                    logger.debug(f"  Total mem_div reward: {memory_divergence_reward:+.6f}")
        else:
            logger.warning(f"Cannot compute memory divergence: student={student_memory_t1 is not None}, teacher={teacher_memory_t1 is not None}")
        
        # Update for next step (for tracking purposes)
        self.prev_memory_divergence = current_divergence
        
        # ===== WM uncertainty (entropy of prediction) =====
        # Use wm_logits from student model's forward pass
        wm_uncertainty = 0.0
        if wm_logits is not None and wm_logits.numel() > 0:
            # wm_logits shape: [B, seq_len, vocab_size]
            wm_probs = F.softmax(wm_logits, dim=-1)
            wm_entropy = -torch.sum(wm_probs * torch.log(wm_probs + 1e-8), dim=-1)
            wm_uncertainty = wm_entropy.mean().item()
            logger.debug(f"WM uncertainty from student: {wm_uncertainty:.4f}")
        else:
            logger.debug("WM logits is None or empty, wm_uncertainty=0")
        
        # ===== Combined reward =====
        reward = (
            self.memory_divergence_weight * memory_divergence_reward +
            self.wm_uncertainty_weight * wm_uncertainty
        )
        
        return reward, {
            "memory_divergence": memory_divergence_reward,
            "memory_divergence_abs": current_divergence,
            "wm_uncertainty": wm_uncertainty,
            "reward_total": reward,
        }
    
    def _compute_advantages(self, transitions: List[Dict[str, Any]]):
        """Compute GAE advantages for PPO."""
        rewards = [t["reward"] for t in transitions]
        values = [t["value"] for t in transitions]
        dones = [t["done"] for t in transitions]
        
        # Bootstrap value
        next_value = 0.0 if dones[-1] else values[-1]
        
        # GAE
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(transitions))):
            next_val = next_value if i == len(transitions) - 1 else values[i + 1]
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        # Store in transitions
        for i, t in enumerate(transitions):
            t["advantage"] = advantages[i]
            t["return"] = advantages[i] + values[i]
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one PPO training step with mini-batching to avoid OOM."""
        batch_size = batch["observation.state"].shape[0]
        mini_batch_size = self.config.mini_batch_size
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_mini_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            mb_indices = indices[start:end]
            
            # Create mini-batch
            mb_batch = {
                "observation.state": batch["observation.state"][mb_indices],
                "action_history": batch["action_history"][mb_indices],
                "task": [batch["task"][i] for i in mb_indices.tolist()],
                "actions": batch["actions"][mb_indices],
            }
            if "observation.images.image0_history" in batch:
                mb_batch["observation.images.image0_history"] = batch["observation.images.image0_history"][mb_indices]
                mb_batch["observation.images.image0"] = batch["observation.images.image0"][mb_indices]
                mb_batch["observation.images.image0_mask"] = batch["observation.images.image0_mask"][mb_indices]
            # Slice initial memory state for mini-batch if present
            if "initial_memory_state" in batch:
                try:
                    # batch["initial_memory_state"] shape: (num_layers, batch_size, hidden)
                    mb_batch["initial_memory_state"] = batch["initial_memory_state"][:, mb_indices, :]
                except Exception:
                    # Fallback: initialize zeros for mini-batch
                    mb_batch["initial_memory_state"] = self._init_memory_state(batch_size=mb_batch["observation.state"].shape[0])
            
            self.optimizer.zero_grad()
            
            # Forward pass (ignore memory_state and wm_logits during training)
            with torch.amp.autocast('cuda', enabled=False):  # Disable mixed precision for stability
                _, log_probs, values, _, _ = self._forward_student(mb_batch, actions=mb_batch["actions"])
            
            # PPO loss
            mb_old_log_probs = batch["old_log_probs"][mb_indices]
            mb_advantages = batch["advantages"][mb_indices]
            mb_returns = batch["returns"][mb_indices]
            
            # Normalize advantages for stability
            normalize_adv = getattr(self.ppo_config, 'normalize_advantages', True)
            if normalize_adv and mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Policy loss (clipped)
            ratio = torch.exp(log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with optional clipping — use Smooth L1 (Huber) for robustness
            clip_value = getattr(self.ppo_config, 'clip_value_loss', True)
            if clip_value:
                # Get old values from batch
                mb_old_values = batch["old_values"][mb_indices]
                values_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = F.smooth_l1_loss(values, mb_returns, reduction='none')
                value_loss_clipped = F.smooth_l1_loss(values_clipped, mb_returns, reduction='none')
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = F.smooth_l1_loss(values, mb_returns)
            
            # Entropy bonus
            std = torch.exp(self.log_std)
            entropy = 0.5 * (1 + torch.log(2 * np.pi * std**2)).sum()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Backward
            loss.backward()
            # Use unwrapped model for gradient clipping
            student_policy_unwrapped = self.accelerator.unwrap_model(self.student_policy) if self.accelerator else self.student_policy
            clip_gradients(student_policy_unwrapped, max_norm=self.config.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            num_mini_batches += 1
            # Diagnostic: warn on extremely large losses to help debugging
            if abs(policy_loss.item()) > 1e6 or abs(value_loss.item()) > 1e5:
                logger.warning(f"Large loss detected: policy_loss={policy_loss.item():.3e}, value_loss={value_loss.item():.3e}, episode={getattr(self, 'metrics',{}).get('episode', 'N/A')}")
        
        return {
            "policy_loss": total_policy_loss / num_mini_batches,
            "value_loss": total_value_loss / num_mini_batches,
            "entropy": total_entropy / num_mini_batches,
            "total_loss": total_loss / num_mini_batches,
        }
    
    def train(self, start_episode: int = 0):
        """Main training loop."""
        num_episodes = self.config.num_episodes
        
        logger.info(f"Starting student training for {num_episodes} episodes")
        logger.info(f"Starting from episode: {start_episode}")
        
        # Setup environment
        self.setup_environment()
        
        # Check if this is the main process
        is_main_process = (self.accelerator is None) or self.accelerator.is_main_process
        
        logger.info(f"[Rank {self.accelerator.local_process_index if self.accelerator else 0}] Starting training loop")
        
        # Use tqdm for progress bar on main process
        episode_iterator = range(start_episode, num_episodes)
        if is_main_process:
            episode_iterator = tqdm(episode_iterator, desc="Training Episodes", unit="ep")
        
        for episode in episode_iterator:
            self.metrics["episode"] = episode
            
            if not is_main_process:
                logger.debug(f"[Rank {self.accelerator.local_process_index}] Episode {episode}: collecting rollout")
            
            # Collect episode
            transitions = self.collect_episode(use_tqdm=is_main_process)

            # Log a sample wrist frame to TensorBoard (main process only)
            if is_main_process:
                try:
                    if self.writer is not None and transitions and "wrist_rgb" in transitions[0]["obs"]:
                        import numpy as _np
                        wrist = transitions[0]["obs"]["wrist_rgb"]
                        # wrist may be [T, C, H, W] or [C, H, W]
                        if hasattr(wrist, 'ndim') and wrist.ndim == 4:
                            frame = wrist[-1]
                        else:
                            frame = wrist
                        # Convert C,H,W -> H,W,C
                        frame = _np.transpose(frame, (1, 2, 0))
                        self.add_image("sample/wrist_gt", frame, episode)
                except Exception:
                    logger.exception("Failed to log sample wrist frame to TensorBoard")

            # Also add episode wrist video to TensorBoard (main process only)
            if is_main_process:
                try:
                    if self.writer is not None and transitions and "wrist_rgb" in transitions[0]["obs"]:
                        import numpy as _np
                        tb_frames = []
                        for t in transitions:
                            next_obs = t.get("next_obs", {})
                            wrist = next_obs.get("wrist_rgb")
                            if wrist is None:
                                wrist = t["obs"].get("wrist_rgb")
                            if wrist is None:
                                continue
                            if hasattr(wrist, 'ndim') and wrist.ndim == 4:
                                fr = wrist[-1]
                            else:
                                fr = wrist
                            fr = _np.transpose(fr, (1, 2, 0))
                            # Ensure uint8
                            if fr.dtype != _np.uint8:
                                if fr.max() <= 1.0:
                                    fr = (fr * 255.0).astype(_np.uint8)
                                else:
                                    fr = fr.astype(_np.uint8)
                            tb_frames.append(fr)
                        if tb_frames:
                            # Use trainer convenience which wraps add_video_to_writer
                            self.add_video(f"video/episode_{episode:06d}", tb_frames, episode, fps=10)
                except Exception:
                    logger.exception("Failed to add episode wrist video to TensorBoard")
            
            if not is_main_process:
                logger.debug(f"[Rank {self.accelerator.local_process_index}] Episode {episode}: collected {len(transitions)} transitions")
            
            # Track episode reward
            episode_reward = sum(t["reward"] for t in transitions)
            self.metrics["episode_reward"].append(episode_reward)
            
            # Track reward components
            avg_memory_div = np.mean([t["info"]["memory_divergence"] for t in transitions])
            avg_memory_div_abs = np.mean([t["info"]["memory_divergence_abs"] for t in transitions])
            avg_wm_unc = np.mean([t["info"]["wm_uncertainty"] for t in transitions])
            self.metrics["memory_divergence"].append(avg_memory_div)
            self.metrics["wm_uncertainty"].append(avg_wm_unc)
            
            # Log detailed episode statistics
            logger.info(f"[Episode {episode}] Completed - Detailed Statistics:")
            logger.info(f"  Total steps: {len(transitions)}")
            logger.info(f"  Episode reward: {episode_reward:.4f}")
            logger.info(f"  Avg memory div reward: {avg_memory_div:.6f}")
            logger.info(f"  Avg memory div abs: {avg_memory_div_abs:.6f}")
            logger.info(f"  Avg WM uncertainty: {avg_wm_unc:.6f}")
            logger.info(f"  Min/Max reward: {min(t['reward'] for t in transitions):.4f} / {max(t['reward'] for t in transitions):.4f}")
            
            # Build PPO batch
            batch = self._build_ppo_batch(transitions)
            
            # Multiple PPO epochs
            for ppo_epoch in range(4):  # PPO epochs
                loss_dict = self.train_step(batch)
                self.metrics["policy_loss"].append(loss_dict["policy_loss"])
                self.metrics["value_loss"].append(loss_dict["value_loss"])
                self.metrics["entropy"].append(loss_dict["entropy"])
                
                # Log training metrics for each PPO epoch
                if ppo_epoch == 0:  # Log first epoch for detailed tracking
                    logger.info(f"[Episode {episode}] PPO Epoch {ppo_epoch} - Training Losses:")
                    logger.info(f"  Policy loss: {loss_dict['policy_loss']:.6f}")
                    logger.info(f"  Value loss: {loss_dict['value_loss']:.6f}")
                    logger.info(f"  Entropy: {loss_dict['entropy']:.6f}")
                    logger.info(f"  Total loss: {loss_dict['total_loss']:.6f}")
            
            self.metrics["total_steps"] += len(transitions)
            self.scheduler.step()
            
            # Logging (only on main process)
            if (episode + 1) % self.config.log_every == 0 and is_main_process:
                self._log_episode_metrics(episode)
            
            # Save checkpoint (only on main process)
            if (episode + 1) % self.config.save_every == 0 and is_main_process:
                self.save_checkpoint(
                    episode,
                    extra_state={
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "value_head": self.value_head.state_dict(),
                        "log_std": self.log_std.data,
                    }
                )
            # Save episode video (MP4) to disk and TensorBoard
            if (episode + 1) % self.video_save_every == 0 and is_main_process:
                try:
                    self._save_episode_video(episode + 1)
                except Exception:
                    logger.exception("Failed to save episode video for student")
            else:
                # Clear buffers to avoid memory growth
                self.video_frames_head = []
                self.video_frames_wrist = []
                self.video_transitions = []
        
        # Final save
        self.save_checkpoint(num_episodes)
        logger.info("Training complete!")
    
    def _build_ppo_batch(self, transitions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Build PPO training batch."""
        batch = {
            "observation.state": torch.stack([
                torch.from_numpy(t["obs"]["state"]).float() 
                for t in transitions
            ]).to(self.device),
            "actions": torch.stack([
                torch.from_numpy(t["action"]).float()
                for t in transitions
            ]).to(self.device),
            "old_log_probs": torch.tensor(
                [t["log_prob"] for t in transitions],
                device=self.device
            ),
            "old_values": torch.tensor(
                [t["value"] for t in transitions],
                device=self.device
            ),
            "advantages": torch.tensor(
                [t["advantage"] for t in transitions],
                device=self.device
            ),
            "returns": torch.tensor(
                [t["return"] for t in transitions],
                device=self.device
            ),
        }
        
        # Add action history
        batch["action_history"] = torch.stack([
            torch.from_numpy(t["obs"]["action_history"]).float()
            for t in transitions
        ]).to(self.device)
        
        # Add task prompt
        batch["task"] = ["explore the environment\n"] * len(transitions)
        
        # Add image data for student model
        # Student uses wrist_rgb as image0, and wrist_rgb history for world model
        if "wrist_rgb" in transitions[0]["obs"]:
            # World Model history (image0_history)
            wrist_histories = []
            current_wrists = []
            for t in transitions:
                wrist_imgs = t["obs"]["wrist_rgb"]  # [T, C, H, W]
                wrist_histories.append(torch.from_numpy(wrist_imgs).float() / 255.0 * 2.0 - 1.0)
                current_wrists.append(torch.from_numpy(wrist_imgs[-1]).float() / 255.0 * 2.0 - 1.0)
            
            batch["observation.images.image0_history"] = torch.stack(wrist_histories).to(self.device)
            batch["observation.images.image0"] = torch.stack(current_wrists).to(self.device)
            batch["observation.images.image0_mask"] = torch.ones(len(transitions), dtype=torch.bool, device=self.device)
        
        # Note: Advantages are normalized in train_step (mini-batch level)
        # We skip full-batch normalization here to avoid double normalization

        # Handle per-transition initial memory states (if present)
        # Expect per-transition entry: transitions[i]["initial_memory_state"]
        memory_list = []
        for t in transitions:
            ms = t.get("initial_memory_state")
            if ms is None:
                # initialize zeros of shape (num_layers, hidden)
                tmp = self._init_memory_state(batch_size=1).squeeze(1).to(self.device)
                memory_list.append(tmp)
            else:
                if isinstance(ms, torch.Tensor):
                    m = ms.detach().to(self.device)
                else:
                    # assume numpy
                    m = torch.from_numpy(ms).float().to(self.device)
                # Normalize shape to (num_layers, hidden)
                if m.dim() == 3 and m.shape[1] == 1:
                    m = m.squeeze(1)
                elif m.dim() == 3 and m.shape[1] != 1:
                    # take first batch entry
                    m = m[:, 0, :]
                elif m.dim() == 2:
                    pass
                else:
                    # try to reshape
                    m = m.reshape(m.shape[0], -1)
                memory_list.append(m)

        if memory_list:
            # stack into shape (num_layers, batch_size, hidden)
            try:
                stacked = torch.stack(memory_list, dim=1).to(self.device)
                batch["initial_memory_state"] = stacked
            except Exception:
                logger.exception("Failed to stack initial memory states for PPO batch")
        
        return batch
    
    def _log_episode_metrics(self, episode: int):
        """Log training metrics."""
        metrics = {
            "policy_loss": np.mean(self.metrics["policy_loss"]) if self.metrics["policy_loss"] else 0,
            "value_loss": np.mean(self.metrics["value_loss"]) if self.metrics["value_loss"] else 0,
            "entropy": np.mean(self.metrics["entropy"]) if self.metrics["entropy"] else 0,
            "memory_divergence": np.mean(self.metrics["memory_divergence"]) if self.metrics["memory_divergence"] else 0,
            "wm_uncertainty": np.mean(self.metrics["wm_uncertainty"]) if self.metrics["wm_uncertainty"] else 0,
            "episode_reward": np.mean(self.metrics["episode_reward"]) if self.metrics["episode_reward"] else 0,
            "lr": self.scheduler.get_last_lr()[0],
        }
        
        # Calculate statistics over recent episodes
        recent_window = 10
        recent_rewards = list(self.metrics["episode_reward"])[-recent_window:]
        recent_mem_div = list(self.metrics["memory_divergence"])[-recent_window:]
        recent_p_loss = list(self.metrics["policy_loss"])[-recent_window:]
        
        # Log to tensorboard
        self.log_metrics(episode, metrics)
        
        # Log detailed console output
        logger.info("=" * 80)
        logger.info(f"Episode {episode} - Aggregated Metrics Summary:")
        logger.info(f"  Reward: {metrics['episode_reward']:.4f} (recent avg: {np.mean(recent_rewards):.4f}, std: {np.std(recent_rewards):.4f})")
        logger.info(f"  Policy Loss: {metrics['policy_loss']:.6f} (recent avg: {np.mean(recent_p_loss):.6f})")
        logger.info(f"  Value Loss: {metrics['value_loss']:.6f}")
        logger.info(f"  Entropy: {metrics['entropy']:.6f}")
        logger.info(f"  Memory Div Reward: {metrics['memory_divergence']:.6f} (recent avg: {np.mean(recent_mem_div):.6f})")
        logger.info(f"  WM Uncertainty: {metrics['wm_uncertainty']:.6f}")
        logger.info(f"  Learning Rate: {metrics['lr']:.2e}")
        logger.info(f"  Total Steps: {self.metrics['total_steps']}")
        logger.info(f"  Log Std: mean={self.log_std.mean().item():.4f}, std={self.log_std.std().item():.4f}")
        logger.info("=" * 80)
        
        # Legacy format for compatibility
        logger.info(
            f"Episode {episode}: "
            f"reward={metrics['episode_reward']:.2f}, "
            f"p_loss={metrics['policy_loss']:.4f}, "
            f"v_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, "
            f"mem_div={metrics['memory_divergence']:.4f}, "
            f"wm_unc={metrics['wm_uncertainty']:.4f}, "
            f"lr={metrics['lr']:.2e}"
        )


def main():
    args = parse_args()
    
    # Load RL config
    rl_config = load_rl_config(args.rl_config)
    
    # Apply command-line overrides
    if args.model_config:
        rl_config.model.config_file = args.model_config
    if args.num_episodes is not None:
        rl_config.training.num_episodes = args.num_episodes
    if args.output_dir is not None:
        rl_config.student.output_dir = args.output_dir
    if args.memory_divergence_weight is not None:
        rl_config.student.rewards.memory_divergence_weight = args.memory_divergence_weight
    if args.wm_uncertainty_weight is not None:
        rl_config.student.rewards.wm_uncertainty_weight = args.wm_uncertainty_weight
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

    # Resolve device and main-process flag via shared helper
    device, is_main_process = resolve_device_and_process(accelerator, rl_config, args)

    # Setup logging from config (after accelerator is created so we know if main process)
    setup_logging_from_config(rl_config, is_main_process=is_main_process)

    # Print startup header via shared helper
    extra_info = {"Number of episodes": (args.num_episodes or rl_config.training.num_episodes)}
    print_startup_header("Student Policy Training (Phase 2)", device, is_main_process, use_ddp=args.use_ddp, num_processes=(getattr(accelerator, 'num_processes', None) if accelerator is not None else None), extra=extra_info)
    
    debug = rl_config.get("debug", False)
    
    # Determine model configuration source
    if hasattr(rl_config, "f1_vla") and rl_config.f1_vla is not None:
        if is_main_process:
            logger.info("Using embedded F1-VLA configuration from RL config")
        model_config_source = rl_config.f1_vla
    else:
        # Fallback to external file
        model_config_file = rl_config.get("model", {}).get(
            "config_file",
            "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml"
        )
        if model_config_file is None:
             # If explicitly null in config but no f1_vla section, fallback to default
             model_config_file = "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml"
             
        if is_main_process:
            logger.info(f"Using external F1-VLA configuration file: {model_config_file}")
        model_config_source = model_config_file
    
    # Get LoRA config
    lora_config = get_lora_config_from_dict(rl_config)
    
    # Create specific LoRA config for Teacher (matching the checkpoint training settings)
    # Teacher was trained with only ["q_proj", "v_proj"]
    teacher_lora_config = copy.deepcopy(lora_config)
    # teacher_lora_config.target_modules = ["q_proj", "v_proj"]
    
    # Load teacher policy (frozen)
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"Loading teacher policy from: {args.teacher_path}")
    teacher_policy, _, _ = load_f1_policy(
        config_file=model_config_source,
        device=device,
        debug=debug,
        lora_config=teacher_lora_config,
        checkpoint_path=args.teacher_path,
    )
    
    # Load student policy (trainable)
    student_policy, policy_config, model_config = load_f1_policy(
        config_file=model_config_source,
        device=device,
        debug=debug,
        lora_config=lora_config,
    )
    
    if accelerator is None or accelerator.is_main_process:
        logger.info("Models loaded successfully")
    
    # Sync before creating trainer
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    # Create trainer with accelerator and num_envs
    trainer = StudentTrainer(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        policy_config=policy_config,
        rl_config=rl_config,
        model_config=model_config_source,
        device=device,
        accelerator=accelerator,
        num_envs=args.num_envs,
    )
    
    # Resume training if specified
    start_episode = 0
    resume_path = args.resume
    
    # Auto-resume: find latest checkpoint in output_dir
    if args.auto_resume and not resume_path:
        latest_ckpt = trainer.find_latest_checkpoint()
        if latest_ckpt:
            resume_path = str(latest_ckpt)
            logger.info(f"Auto-resume: found latest checkpoint at {resume_path}")
    
    if resume_path:
        logger.info(f"Resuming training from {resume_path}")
        start_episode = trainer.load_checkpoint(resume_path)
        logger.info(f"Resumed from episode {start_episode}")
    
    # Train
    trainer.train(start_episode=start_episode)


if __name__ == "__main__":
    main()
