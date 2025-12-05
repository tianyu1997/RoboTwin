#!/usr/bin/env python3
"""
Common utilities for F1-VLA RL Training

This module provides shared functionality across all three training phases:
- Phase 1: Teacher (World Model) Training
- Phase 2: Student (Explorer) Training  
- Phase 3: Adversarial Training

Key components:
- Config loading and validation
- Model loading with LoRA/PEFT
- Batch building and preprocessing
- Memory state management for sequential processing
- Checkpoint saving/loading
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# GPU Assignment for SAPIEN Rendering
# =============================================================================

def get_physical_gpu_id(accelerator=None) -> int:
    """
    Get the physical GPU ID for this process.
    
    Maps LOCAL_RANK to actual GPU ID when using CUDA_VISIBLE_DEVICES.
    This is needed because SAPIEN/Vulkan uses physical GPU indices,
    not the remapped indices from CUDA_VISIBLE_DEVICES.
    
    Args:
        accelerator: Optional AcceleratorWrapper for DDP training
        
    Returns:
        Physical GPU ID for SAPIEN rendering
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if accelerator is not None:
        local_rank = accelerator.local_process_index
    
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        if local_rank < len(visible_gpus):
            return visible_gpus[local_rank]
    
    return local_rank


def setup_sapien_gpu(gpu_id: Optional[int] = None):
    """
    Set environment variables for SAPIEN/Vulkan GPU selection.
    
    Must be called BEFORE importing SAPIEN or any module that imports it.
    
    Args:
        gpu_id: Physical GPU ID. If None, auto-detect from LOCAL_RANK.
    """
    if gpu_id is None:
        gpu_id = get_physical_gpu_id()
    
    os.environ["VK_DEVICE_INDEX"] = str(gpu_id)
    os.environ["SAPIEN_DEVICE_INDEX"] = str(gpu_id)
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    
    logger.debug(f"SAPIEN GPU set to: {gpu_id}")


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging_from_config(config: DictConfig, is_main_process: bool = True) -> None:
    """
    Setup logging based on config file settings.
    
    Args:
        config: Full RL config with logging section
        is_main_process: Whether this is the main process (for DDP).
                        Non-main processes only log WARNING+ to reduce noise.
    """
    log_cfg = config.get("logging", {})
    console_level = log_cfg.get("console_level", "WARNING").upper()
    file_level = log_cfg.get("file_level", "DEBUG").upper()
    enable_file = log_cfg.get("enable_file_logging", True)
    
    # Map string to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    # For non-main processes in DDP, only show warnings and above
    if not is_main_process:
        console_lvl = logging.WARNING
        enable_file = False  # Only main process writes to file
    else:
        console_lvl = level_map.get(console_level, logging.WARNING)
    file_lvl = level_map.get(file_level, logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all, let handlers filter
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with process rank prefix for non-main processes
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_lvl)
    if is_main_process:
        fmt = '%(asctime)s [%(levelname)s] %(message)s'
    else:
        rank = os.environ.get('LOCAL_RANK', os.environ.get('RANK', '?'))
        fmt = f'%(asctime)s [RANK {rank}] [%(levelname)s] %(message)s'
    console_handler.setFormatter(logging.Formatter(fmt, datefmt='%H:%M:%S'))
    root_logger.addHandler(console_handler)
    
    # File handler (optional, only on main process)
    if enable_file and is_main_process:
        os.makedirs("logs", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"logs/rl_training_{timestamp}.log")
        file_handler.setLevel(file_lvl)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(file_handler)
    
    # Suppress verbose loggers from third-party libraries
    for lib_name in ["curobo", "sapien", "warp", "nvdiffrast", "trimesh", 
                     "PIL", "matplotlib", "urllib3"]:
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    
    # Use DEBUG level for this message since we're in setup
    if is_main_process:
        logger.debug(f"Logging configured: console={console_level}, file={file_level}")


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class LoRAConfig:
    """LoRA configuration."""
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration shared across phases."""
    # Dimensions
    action_dim: int = 32
    state_dim: int = 32
    
    # World model steps (n_obs_img_steps and history_length will be loaded from model config)
    n_pred_img_steps: int = 3
    
    # Training params
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Episode settings
    num_episodes: int = 10000
    steps_per_episode: int = 50
    batch_size: int = 8
    
    # Logging
    log_every: int = 10
    save_every: int = 1000
    
    # Memory/Sequential processing
    sequential_training: bool = True
    
    # Action bounds
    action_scale: float = 1.0
    action_bounds_low: float = -1.0
    action_bounds_high: float = 1.0


@dataclass 
class EnvironmentConfig:
    """Environment configuration."""
    task_name: str = "random_exploration"
    control_mode: str = "delta_qpos"
    delta_qpos_scale: float = 0.05
    render_mode: str = "rasterize"
    num_objects: int = 5
    embodiment: List = field(default_factory=lambda: ["franka-panda"])  # Single arm
    
    # Camera config - collect both cameras
    camera: Dict[str, Any] = field(default_factory=lambda: {
        "head_camera_type": "D435",
        "wrist_camera_type": "D435",
        "collect_head_camera": True,
        "collect_wrist_camera": True,
    })
    
    # Domain randomization
    domain_randomization: Dict[str, bool] = field(default_factory=lambda: {
        "random_appearance": False,
        "random_background": True,
        "random_light": True,
        "cluttered_table": False,
    })
    
    # Data type
    data_type: Dict[str, bool] = field(default_factory=lambda: {
        "collect_rgb": True,
        "collect_depth": False,
        "collect_qpos": True,
        "collect_endpose": True,
    })


# =============================================================================
# Config Loading
# =============================================================================

def load_rl_config(config_path: str) -> DictConfig:
    """
    Load RL training configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        OmegaConf DictConfig object
    """
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded RL config from: {config_path}")
    return config


def get_training_config(config: DictConfig) -> TrainingConfig:
    """Extract training config from full config."""
    train_cfg = config.get("training", {})
    return TrainingConfig(
        action_dim=train_cfg.get("action_dim", 32),
        state_dim=train_cfg.get("state_dim", 32),
        n_pred_img_steps=train_cfg.get("n_pred_img_steps", 3),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        num_episodes=train_cfg.get("num_episodes", 10000),
        steps_per_episode=train_cfg.get("steps_per_episode", 50),
        batch_size=train_cfg.get("batch_size", 8),
        log_every=train_cfg.get("log_every", 10),
        save_every=train_cfg.get("save_every", 1000),
        sequential_training=train_cfg.get("sequential_training", True),
        action_scale=train_cfg.get("action_scale", 1.0),
        action_bounds_low=train_cfg.get("action_bounds", {}).get("low", -1.0),
        action_bounds_high=train_cfg.get("action_bounds", {}).get("high", 1.0),
    )


def get_f1_flow_matching_model(policy):
    """Get the F1FlowMatching model from policy, handling PEFT wrapping.
    
    When using PEFT/LoRA:
    - policy is PeftModel
    - policy.model is F1_VLA  
    - policy.model.model is F1FlowMatching
    
    Without PEFT:
    - policy is F1_VLA
    - policy.model is F1FlowMatching
    
    Returns:
        F1FlowMatching model instance
    """
    # Check if wrapped by PEFT
    if hasattr(policy, 'model') and hasattr(policy.model, 'model'):
        # PEFT wrapped: policy.model is F1_VLA, policy.model.model is F1FlowMatching
        if hasattr(policy.model.model, 'set_requires_grad'):
            return policy.model.model
    
    # Direct F1_VLA: policy.model is F1FlowMatching
    if hasattr(policy, 'model') and hasattr(policy.model, 'set_requires_grad'):
        return policy.model
    
    raise AttributeError(f"Cannot find F1FlowMatching model in {type(policy)}")


class TrainingArgsAdapter:
    """Adapter class to match F1FlowMatching.set_requires_grad expected interface."""
    def __init__(
        self,
        freeze_vision_encoder: bool = True,
        freeze_gen_expert: bool = False,
        train_act_expert_only: bool = False,
        train_gen_expert_only: bool = False,
        train_state_proj: bool = True,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_gen_expert = freeze_gen_expert
        self.train_act_expert_only = train_act_expert_only
        self.train_gen_expert_only = train_gen_expert_only
        self.train_state_proj = train_state_proj


def set_policy_requires_grad(
    policy,
    freeze_vision_encoder: bool = True,
    freeze_gen_expert: bool = False,
    train_act_expert_only: bool = False,
    train_gen_expert_only: bool = False,
):
    """Set requires_grad flags on F1 policy, handling PEFT wrapping.
    
    Args:
        policy: F1_VLA or PeftModel wrapping F1_VLA
        freeze_vision_encoder: Freeze vision encoder (saves VRAM, faster training)
        freeze_gen_expert: Freeze world model expert
        train_act_expert_only: Only train action expert
        train_gen_expert_only: Only train world model expert
    """
    model = get_f1_flow_matching_model(policy)
    
    # Create adapter object that matches F1FlowMatching.set_requires_grad expected interface
    training_args = TrainingArgsAdapter(
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_gen_expert=freeze_gen_expert,
        train_act_expert_only=train_act_expert_only,
        train_gen_expert_only=train_gen_expert_only,
        train_state_proj=not train_gen_expert_only,  # Freeze state_proj when training gen expert only
    )
    model.set_requires_grad(training_args)


def get_environment_config(config: DictConfig, num_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    Build environment config dict from RL config.
    
    Args:
        config: Full RL config
        num_steps: Override for num_random_steps
        
    Returns:
        Dict suitable for passing to environment
    """
    env_cfg = config.get("environment", {})
    train_cfg = config.get("training", {})
    
    # Convert OmegaConf DictConfig to regular Python dict if needed
    def to_dict(obj):
        if hasattr(obj, '_iter_ex_keys'):  # OmegaConf DictConfig
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_dict(v) for v in obj]
        else:
            return obj
    
    # Get domain_randomization - ensure it's converted to regular dict
    domain_rand_cfg = env_cfg.get("domain_randomization", None)
    if domain_rand_cfg is not None:
        domain_randomization = to_dict(domain_rand_cfg)
    else:
        domain_randomization = {
            "random_appearance": False,
            "random_background": True,
            "random_light": True,
            "cluttered_table": False,
        }
    
    # Log for debugging
    logger.info(f"Loading domain_randomization from config: {domain_randomization}")
    
    env_config = {
        "task_name": env_cfg.get("task_name", "random_exploration"),
        "control_mode": env_cfg.get("control_mode", "delta_qpos"),
        "num_random_steps": num_steps or train_cfg.get("steps_per_episode", 50),
        "num_objects": env_cfg.get("num_objects", 5),
        "delta_qpos_scale": env_cfg.get("delta_qpos_scale", 0.05),
        "render_mode": env_cfg.get("render_mode", "rasterize"),
        "embodiment": env_cfg.get("embodiment", ["franka-panda", "franka-panda", 0.6]),
        # Single arm mode and scene reset interval
        "single_arm": env_cfg.get("single_arm", False),
        "scene_reset_interval": env_cfg.get("scene_reset_interval", 1),
        "camera": to_dict(env_cfg.get("camera", {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        })),
        "domain_randomization": domain_randomization,
        "data_type": to_dict(env_cfg.get("data_type", {
            "collect_rgb": True,
            "collect_depth": False,
            "collect_qpos": True,
            "collect_endpose": True,
        })),
    }
    
    return env_config


# =============================================================================
# Model Loading
# =============================================================================

def load_f1_policy(
    config_file: str,
    device: str = "cuda",
    debug: bool = False,
    lora_config: Optional[LoRAConfig] = None,
    checkpoint_path: Optional[str] = None,
    is_main_process: bool = True,
):
    """
    Load F1-VLA policy with optional LoRA and checkpoint.
    
    Args:
        config_file: Path to model YAML config file
        device: Device to use
        debug: If True, create model from scratch
        lora_config: LoRA configuration (None to skip LoRA)
        checkpoint_path: Path to additional checkpoint to load
        is_main_process: If True, print progress messages (set False for non-main DDP ranks)
        
    Returns:
        Tuple of (policy, policy_config, full_config)
    """
    import sys
    import io
    
    # Helper to print only on main process
    def _print(msg):
        if is_main_process:
            print(msg)
    
    # Context manager to suppress stdout on non-main processes
    # This catches all prints from F1_VLA and other internal code
    class SuppressStdout:
        def __enter__(self):
            if not is_main_process:
                self._stdout = sys.stdout
                sys.stdout = io.StringIO()
            return self
        def __exit__(self, *args):
            if not is_main_process:
                sys.stdout = self._stdout
    
    # Ensure current CUDA device is set BEFORE loading model
    # This prevents from_pretrained from loading to default GPU (cuda:0)
    if device.startswith("cuda"):
        import torch
        torch.cuda.set_device(device)
    
    from f1_vla.src.models.configuration_f1 import F1Config
    from f1_vla.src.policies.f1_policy import F1_VLA
    from f1_vla.src.utils.utils import load_ckpt, set_policy_config, unfreeze_memory_params
    
    # Load config from YAML
    _print("  Loading config file...")
    config = OmegaConf.load(Path(config_file))
    
    # Load F1Config and set policy config
    _print(f"  Loading F1Config from: {config.policy.ckpt_path}")
    policy_config = F1Config.from_pretrained(config.policy.ckpt_path)
    policy_config = set_policy_config(policy_config, config.policy)
    
    logger.debug(f"Policy config loaded from: {config.policy.ckpt_path}")
    logger.debug(f"Pretrained path: {policy_config.pretrained_path}")
    logger.debug(f"Use world model: {policy_config.use_world_model}")
    
    # Create model (suppress verbose output on non-main processes)
    kwargs = {
        "config": policy_config,
        "pretrained_name_or_path": policy_config.pretrained_path,
    }
    
    with SuppressStdout():
        if policy_config.pretrained_path and not debug:
            _print(f"  Loading pretrained model from: {policy_config.pretrained_path}")
            policy = F1_VLA.from_pretrained(**kwargs)
            _print("  Loading additional checkpoint...")
            # Load additional checkpoint (e.g., world model weights)
            policy = load_ckpt(policy, config)
            _print("  Pretrained weights loaded successfully")
        else:
            _print("  Creating model from scratch (debug mode)")
            policy = F1_VLA(**kwargs)
        
        # Load additional checkpoint if specified
        if checkpoint_path and os.path.exists(checkpoint_path):
            _print(f"  Loading checkpoint from: {checkpoint_path}")
            load_config = OmegaConf.create({"exp": {"load_ckpt": checkpoint_path}})
            policy = load_ckpt(policy, load_config)
        
        # Apply LoRA if configured
        if lora_config is not None:
            _print("  Applying LoRA configuration...")
            from peft import get_peft_model, LoraConfig as PeftLoraConfig
            
            peft_config = PeftLoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=lora_config.task_type,
            )
            policy = get_peft_model(policy, peft_config)
            logger.debug("Applied LoRA configuration")
        
        # Unfreeze memory-related parameters
        unfrozen = unfreeze_memory_params(policy)
        _print(f"  Unfrozen {len(unfrozen)} memory parameters")
    
    # Move to device
    _print(f"  Moving model to {device}...")
    policy = policy.to(device)
    _print("  Model ready!")
    
    return policy, policy_config, config


def get_lora_config_from_dict(config: DictConfig) -> LoRAConfig:
    """Extract LoRA config from config dict."""
    lora_cfg = config.get("model", {}).get("lora", {})
    return LoRAConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )


# =============================================================================
# Batch Building
# =============================================================================

class BatchBuilder:
    """
    Builds training batches from transitions.
    
    Handles:
    - Image preprocessing (normalization to [-1, 1])
    - State/action history packaging
    - Memory state tracking for sequential processing
    """
    
    def __init__(
        self,
        device: str = "cuda",
        image_keys: List[str] = None,
        normalize_images: bool = True,
        use_head_camera: bool = True,
    ):
        self.device = device
        self.image_keys = image_keys or ["head_rgb"]
        self.normalize_images = normalize_images
        self.use_head_camera = use_head_camera
    
    def normalize_image(self, img: np.ndarray) -> torch.Tensor:
        """Normalize image from [0, 255] to [-1, 1]."""
        img = torch.from_numpy(img).float().to(self.device)
        if self.normalize_images:
            img = img / 255.0 * 2.0 - 1.0
        return img
    
    def build_batch(
        self,
        transitions: List[Dict[str, Any]],
        include_memory_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Build training batch from transitions.
        
        Args:
            transitions: List of transition dicts
            include_memory_states: Whether to include memory states for sequential processing
            
        Returns:
            Batch dict suitable for F1-VLA forward
        """
        batch_size = len(transitions)
        
        # Collect data
        states = []
        action_histories = []
        actions = []
        memory_states = []
        
        # Image containers
        images = {key: [] for key in self.image_keys}
        image_histories = {key: [] for key in self.image_keys}
        next_images = {key: [] for key in self.image_keys}  # For world model prediction target
        
        for t in transitions:
            obs = t["obs"]
            next_obs = t.get("next_obs", obs)  # Get next observation for prediction target
            
            # State
            states.append(obs["state"])
            
            # Action history
            if "action_history" in obs:
                action_histories.append(obs["action_history"])
            
            # Images
            for key in self.image_keys:
                if key in obs:
                    # Take last frame as current observation
                    images[key].append(obs[key][-1])
                    # All frames as history
                    image_histories[key].append(obs[key])
                    # Next frame as prediction target (from next_obs)
                    if key in next_obs:
                        next_images[key].append(next_obs[key][-1])
            
            # Action taken
            if "action" in t:
                actions.append(t["action"])
            
            # Memory state
            if include_memory_states and "initial_memory_state" in t:
                memory_states.append(t.get("initial_memory_state"))
        
        # Build batch dict
        batch = {
            "observation.state": torch.from_numpy(np.stack(states)).float().to(self.device),
        }
        
        if actions:
            batch["action"] = torch.from_numpy(np.stack(actions)).float().to(self.device)
        
        if action_histories:
            batch["action_history"] = torch.from_numpy(np.stack(action_histories)).float().to(self.device)
        
        # Process images
        for key in self.image_keys:
            if images[key]:
                # Current images for VLM
                imgs = np.stack(images[key])
                batch_key = f"observation.images.{self._get_image_index(key)}"
                # Skip if mapped to "image_unused" (head camera in student mode)
                if "unused" not in batch_key:
                    batch[batch_key] = self.normalize_image(imgs)
                    batch[f"{batch_key}_mask"] = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                
                # History images + next frame for world model
                # World model ALWAYS needs wrist_rgb history -> image0_history
                # This is independent of use_head_camera setting
                if key in ["wrist_rgb", "left_wrist_rgb"]:
                    if image_histories[key] and next_images[key]:
                        hist_imgs = np.stack(image_histories[key])  # [B, T, C, H, W]
                        next_imgs = np.stack(next_images[key])      # [B, C, H, W]
                        next_imgs = next_imgs[:, np.newaxis, :, :, :]  # [B, 1, C, H, W]
                        # Concatenate: history + next frame
                        combined_imgs = np.concatenate([hist_imgs, next_imgs], axis=1)  # [B, T+1, C, H, W]
                        batch["observation.images.image0_history"] = self.normalize_image(combined_imgs)
                    elif image_histories[key]:
                        hist_imgs = np.stack(image_histories[key])
                        batch["observation.images.image0_history"] = self.normalize_image(hist_imgs)
        
        # Task description
        batch["task"] = ["explore the environment\n"] * batch_size
        
        # Memory states for sequential processing
        # Note: If no memory states are provided, the trainer should initialize zeros
        # We don't set initial_memory_state here - let the trainer handle initialization
        if include_memory_states and memory_states:
            valid_states = [s for s in memory_states if s is not None]
            if valid_states:
                # Stack valid memory states into a tensor
                try:
                    batch["initial_memory_state"] = torch.stack(valid_states)
                    logger.debug(f"BatchBuilder: stacked {len(valid_states)} memory states")
                except Exception as e:
                    logger.warning(f"Failed to stack memory states: {e}")
                    # Leave initial_memory_state unset, trainer will initialize
        
        return batch
    
    def _get_image_index(self, key: str) -> str:
        """Map image key to F1-VLA naming convention.
        
        For Paligemma (VLM) input:
        - Teacher (use_head_camera=True): head_rgb -> image0, wrist_rgb -> image1
        - Student (use_head_camera=False): wrist_rgb -> image0
        
        For World Model:
        - Always uses wrist_rgb history -> image0_history
        """
        if self.use_head_camera:
            # Teacher: head_rgb (image0) + wrist_rgb (image1)
            key_map = {
                "head_rgb": "image0",           # Head camera -> image0 for VLM
                "wrist_rgb": "image1",          # Wrist camera -> image1 for VLM
                "left_wrist_rgb": "image1",
                "right_wrist_rgb": "image2",
            }
        else:
            # Student: wrist_rgb only (image0)
            key_map = {
                "head_rgb": "image_unused",     # Head camera not used
                "wrist_rgb": "image0",          # Wrist camera -> image0 for VLM
                "left_wrist_rgb": "image0",
                "right_wrist_rgb": "image1",
            }
        return key_map.get(key, key)


# =============================================================================
# Memory State Manager
# =============================================================================

class MemoryStateManager:
    """
    Manages memory states for sequential processing.
    
    F1-VLA has a memory_rnn (GRU) that requires sequential state propagation:
    - Each frame's output memory becomes the next frame's input
    - Memory resets at episode boundaries
    - Supports batch processing with episode tracking
    """
    
    def __init__(self, max_cache_size: int = 10000):
        """
        Args:
            max_cache_size: Maximum number of memory states to cache
        """
        self.max_cache_size = max_cache_size
        
        # Current memory states (for online processing)
        self.current_memory: Optional[torch.Tensor] = None
        
        # Memory cache for batch training
        # Key: (episode_idx, frame_idx) -> memory_state tensor
        self.memory_cache: OrderedDict = OrderedDict()
    
    def reset(self):
        """Reset current memory state (call at episode start)."""
        self.current_memory = None
    
    def update(self, memory_state: Optional[torch.Tensor]):
        """Update current memory state from model output."""
        if memory_state is not None:
            self.current_memory = memory_state.detach()
    
    def get_current(self) -> Optional[torch.Tensor]:
        """Get current memory state for injection into batch."""
        return self.current_memory
    
    def cache_memory(
        self,
        episode_idx: int,
        frame_idx: int,
        memory_state: Optional[torch.Tensor],
    ):
        """
        Cache memory state for batch training.
        
        Args:
            episode_idx: Episode index
            frame_idx: Frame index within episode
            memory_state: Memory state tensor to cache
        """
        key = (episode_idx, frame_idx)
        
        # Evict oldest if at capacity
        while len(self.memory_cache) >= self.max_cache_size:
            self.memory_cache.popitem(last=False)
        
        self.memory_cache[key] = memory_state.detach() if memory_state is not None else None
    
    def get_cached(
        self,
        episode_idx: int,
        frame_idx: int,
    ) -> Optional[torch.Tensor]:
        """Retrieve cached memory state."""
        return self.memory_cache.get((episode_idx, frame_idx))
    
    def clear_episode(self, episode_idx: int):
        """Clear all cached memories for an episode."""
        keys_to_remove = [k for k in self.memory_cache if k[0] == episode_idx]
        for key in keys_to_remove:
            del self.memory_cache[key]
    
    def clear_cache(self):
        """Clear entire memory cache."""
        self.memory_cache.clear()


# =============================================================================
# Base Trainer Class
# =============================================================================

class BaseRLTrainer(ABC):
    """
    Base class for RL trainers.
    
    Provides common functionality:
    - Checkpoint saving/loading
    - Metrics tracking
    - Tensorboard logging
    - Memory state management
    """
    
    def __init__(
        self,
        policy: nn.Module,
        config: TrainingConfig,
        output_dir: str,
        device: str = "cuda",
    ):
        self.policy = policy
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory state manager
        self.memory_manager = MemoryStateManager()
        
        # Batch builder
        self.batch_builder = BatchBuilder(device=device)
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # Metrics
        self.metrics: Dict[str, Any] = {
            "episode": 0,
            "total_steps": 0,
        }
        
        # Environment (to be set by subclass)
        self.env = None
    
    @abstractmethod
    def setup_environment(self):
        """Setup the RL environment. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def collect_episode(self) -> List[Dict[str, Any]]:
        """Collect one episode of experience. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step. Must be implemented by subclass."""
        pass
    
    def inject_memory_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Inject initial memory state into batch if available.
        
        Handles both single memory state and list of memory states.
        """
        # Check for memory states list (from build_batch)
        if "initial_memory_states_list" in batch:
            memory_states = batch.pop("initial_memory_states_list")
            # Use first valid memory state for batch
            for ms in memory_states:
                if ms is not None:
                    batch["initial_memory_state"] = ms
                    break
        # Check for current memory state
        elif self.memory_manager.current_memory is not None:
            batch["initial_memory_state"] = self.memory_manager.current_memory
        
        return batch
    
    def update_memory_from_output(self, output: Dict[str, Any]):
        """Update memory state from model output."""
        if "memory_state" in output:
            self.memory_manager.update(output["memory_state"])
    
    def save_checkpoint(self, step: int, extra_state: Optional[Dict] = None):
        """
        Save training checkpoint.
        
        Args:
            step: Current training step/episode
            extra_state: Additional state to save
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PEFT model (adapter weights)
        self.policy.save_pretrained(checkpoint_dir)
        
        # Save trainer state
        state = {
            "step": step,
            "total_steps": self.metrics["total_steps"],
            "metrics": {
                k: list(v) if isinstance(v, deque) else v
                for k, v in self.metrics.items()
            },
        }
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, checkpoint_dir / "trainer_state.pt")
        logger.debug(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            
        Returns:
            Step/episode to resume from
        """
        checkpoint_path = Path(checkpoint_dir)
        
        # Load PEFT adapter weights
        logger.debug(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            adapter_config_path = checkpoint_path / "adapter_config.json"
            if adapter_config_path.exists():
                self.policy.load_adapter(str(checkpoint_path), "default")
                logger.debug("Loaded PEFT adapter weights")
        except Exception as e:
            logger.warning(f"Could not load PEFT adapter: {e}")
        
        # Load trainer state
        trainer_state_path = checkpoint_path / "trainer_state.pt"
        if trainer_state_path.exists():
            state = torch.load(trainer_state_path, map_location=self.device)
            self.metrics["total_steps"] = state.get("total_steps", 0)
            
            # Restore metrics
            if "metrics" in state:
                for key, value in state["metrics"].items():
                    if key in self.metrics and isinstance(self.metrics[key], deque):
                        self.metrics[key] = deque(value, maxlen=self.metrics[key].maxlen)
                    elif key in self.metrics:
                        self.metrics[key] = value
            
            return state.get("step", 0)
        
        return 0
    
    def log_metrics(self, step: int, metrics_dict: Dict[str, float]):
        """Log metrics to tensorboard and console."""
        # Tensorboard
        for key, value in metrics_dict.items():
            self.writer.add_scalar(f"train/{key}", value, step)
        
        # Console (only at debug level - use tqdm in training loop for console output)
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics_dict.items())
        logger.debug(f"Step {step}: {metrics_str}")


# =============================================================================
# Utility Functions
# =============================================================================

def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def setup_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Setup AdamW optimizer with trainable parameters."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    T_max: int = 10000,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Setup learning rate scheduler."""
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """Clip gradients to prevent explosion."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
