"""
F1-VLA RL Environment for Teacher-Student Policy Training

This environment wraps the random_exploration task for RL training of F1-VLA policy.

Phase 1 (Teacher): Train LLM + World Model
    - Input: history actions, states, head observation
    - Output: random action, next frame prediction
    - Reward: prediction accuracy of next frame

Phase 2 (Student): Train new LLM + Explorer
    - Reuse frozen world model from teacher
    - Only use wrist camera observations
    - Explorer generates actions
    - Reward: |h_student_t - h_teacher_t| - |h_student_t+1 - h_teacher_t+1|
"""

import os
import warnings

# Suppress CuRobo verbose logging before importing
os.environ.setdefault("CUROBO_LOG_LEVEL", "WARNING")
warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import copy
import logging
import yaml
from datetime import datetime

from .normalizers import ActionNormalizer, StateNormalizer

# Suppress verbose logging from CuRobo and related libraries
for lib_name in ["curobo", "sapien", "warp", "nvdiffrast", "trimesh"]:
    logging.getLogger(lib_name).setLevel(logging.WARNING)


# ==================== Embodiment Config Helpers ====================

def get_embodiment_file(embodiment_name: str, base_dir: str = None) -> str:
    """Get the file path for a given embodiment name."""
    if base_dir is None:
        # Default to RoboTwin directory relative to this file (rl/f1_rl_env.py -> RoboTwin)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embodiment_config_path = os.path.join(base_dir, "task_config", "_embodiment_config.yml")
    with open(embodiment_config_path, "r") as f:
        embodiment_config = yaml.safe_load(f)

    # Defensive: callers sometimes pass a list like [left, right, offset].
    # Ensure we have a single embodiment name (string) for lookup.
    if isinstance(embodiment_name, (list, tuple)):
        if len(embodiment_name) == 0:
            raise ValueError("Empty embodiment list provided")
        embodiment_name_lookup = embodiment_name[0]
    else:
        embodiment_name_lookup = embodiment_name

    if embodiment_name_lookup not in embodiment_config:
        raise ValueError(f"Unknown embodiment: {embodiment_name_lookup}. Available: {list(embodiment_config.keys())}")

    file_path = embodiment_config[embodiment_name_lookup]["file_path"]
    # Resolve relative path
    if file_path.startswith("./"):
        file_path = os.path.join(base_dir, file_path[2:])
    return file_path


def get_embodiment_config(embodiment_path: str) -> dict:
    """Load the configuration for an embodiment from its config.yml file."""
    config_path = os.path.join(embodiment_path, "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_embodiment_config(task_config: dict, base_dir: str = None) -> dict:
    """
    Load embodiment configurations and add them to task_config.
    
    The task_config should contain 'embodiment' which can be:
    - A list: [left_embodiment, right_embodiment, y_offset] 
    - A string: same embodiment for both arms
    
    Returns updated config with:
    - left_embodiment_config, right_embodiment_config
    - left_robot_file, right_robot_file
    - camera, domain_randomization, data_type (defaults if not provided)
    """
    config = task_config.copy()
    
    embodiment = config.get("embodiment", "franka-panda")

    # Accept several formats for 'embodiment':
    # - string: "franka-panda"
    # - list: [left_name, right_name] or [left_name, right_name, y_offset]
    # - single-element list: ["franka-panda"] meaning dual-arm same
    # Note: OmegaConf returns ListConfig which is not isinstance of list/tuple
    # Use hasattr to check for list-like behavior
    is_list_like = isinstance(embodiment, (list, tuple)) or hasattr(embodiment, '__iter__') and not isinstance(embodiment, str)
    
    if is_list_like:
        # Convert to regular list for consistent handling
        embodiment = list(embodiment)
        
        # Normalize entries to strings where possible
        if len(embodiment) == 0:
            left_embodiment_name = "franka-panda"
            right_embodiment_name = "franka-panda"
            embodiment_dis = 0.6
        else:
            left_embodiment_name = str(embodiment[0])
            right_embodiment_name = str(embodiment[1]) if len(embodiment) > 1 else left_embodiment_name
            embodiment_dis = float(embodiment[2]) if len(embodiment) > 2 else 0.6

        # If names were accidentally provided as lists, coerce to first element
        if isinstance(left_embodiment_name, (list, tuple)) or (hasattr(left_embodiment_name, '__iter__') and not isinstance(left_embodiment_name, str)):
            left_embodiment_name = str(list(left_embodiment_name)[0])
        if isinstance(right_embodiment_name, (list, tuple)) or (hasattr(right_embodiment_name, '__iter__') and not isinstance(right_embodiment_name, str)):
            right_embodiment_name = str(list(right_embodiment_name)[0])

        is_dual_arm = (left_embodiment_name == right_embodiment_name)
    else:
        left_embodiment_name = str(embodiment)
        right_embodiment_name = str(embodiment)
        embodiment_dis = 0.6
        is_dual_arm = True  # Single embodiment name means dual arm robot
    
    # Get file paths for embodiments
    left_robot_file = get_embodiment_file(left_embodiment_name, base_dir)
    right_robot_file = get_embodiment_file(right_embodiment_name, base_dir)
    
    # Load embodiment configs
    left_embodiment_config = get_embodiment_config(left_robot_file)
    right_embodiment_config = get_embodiment_config(right_robot_file)
    
    # Check if the embodiment itself declares dual_arm
    if left_embodiment_config.get("dual_arm", False):
        is_dual_arm = True
    else:
        # Two separate robots with same or different embodiment
        is_dual_arm = False
    
    # Add to config
    config["left_embodiment_config"] = left_embodiment_config
    config["right_embodiment_config"] = right_embodiment_config
    config["left_robot_file"] = left_robot_file
    config["right_robot_file"] = right_robot_file
    config["dual_arm_embodied"] = is_dual_arm
    config["embodiment_dis"] = embodiment_dis
    
    # Add default camera config if not provided
    if "camera" not in config:
        config["camera"] = {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        }
    
    # Add default domain_randomization config if not provided
    if "domain_randomization" not in config:
        config["domain_randomization"] = {
            "random_appearance": False,
            "collect_any_randomization": False,
            "random_desk_color": False,
            "random_light_position": False,
            "random_background": False,
            "random_object_texture": False,
        }
    
    # Add default data_type config if not provided
    if "data_type" not in config:
        config["data_type"] = {
            "collect_rgb": True,
            "collect_depth": False,
            "collect_qpos": True,
            "collect_endpose": True,
        }
    
    # Verify all required configs are present
    assert "camera" in config, f"camera config missing! Config keys: {list(config.keys())}"
    assert "left_embodiment_config" in config, "left_embodiment_config missing!"
    assert "right_embodiment_config" in config, "right_embodiment_config missing!"
    
    return config


# ==================== Environment Logger Setup ====================

def setup_env_logger(log_dir: str = "logs/env") -> logging.Logger:
    """Setup logger for F1RLEnv module."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("f1_rl_env")
    if logger.handlers:  # Already configured
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"f1_rl_env_{timestamp}.log")
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (WARNING level only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Module-level logger
_env_logger = None

def get_env_logger() -> logging.Logger:
    """Get or create the environment logger."""
    global _env_logger
    if _env_logger is None:
        _env_logger = setup_env_logger()
    return _env_logger


class F1RLEnv(gym.Env):
    """
    Gym environment for F1-VLA RL training.
    
    Supports two training phases:
    1. Teacher training: random actions, world model prediction reward
    2. Student training: explorer actions, memory divergence + uncertainty reward
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        task_config: Dict[str, Any],
        phase: str = "teacher",  # "teacher" or "student"
        history_length: int = 12,  # Auto-calculated from n_obs_img_steps + (n_obs_img_steps-1)*stride
        max_steps: int = 50,
        device: str = "cuda",
        teacher_policy: Optional[Any] = None,  # For student phase
        image_size: Tuple[int, int] = (224, 224),
        action_dim: int = 32,
        state_dim: int = 32,
        render_mode: str = "rasterize",
        max_reset_retries: int = 10,  # Max retries for UnstableError during reset
        action_scale: float = 1.0,  # Scale factor for action bounds (0-1), lower = smaller actions
        action_bounds: Optional[Dict[str, Tuple[float, float]]] = None,  # Custom action bounds override
        single_arm: bool = False,  # Single arm mode: only use left arm
        scene_reset_interval: int = 1,  # Regenerate scene every N episodes
    ):
        """
        Initialize F1 RL Environment.
        
        Args:
            task_config: Configuration for random_exploration task
            phase: Training phase - "teacher" or "student"
            history_length: Number of history frames (auto-calculated from model config)
            max_steps: Maximum steps per episode
            device: Torch device
            teacher_policy: Pre-trained teacher policy (required for student phase)
            image_size: Size of observation images
            action_dim: Unified action dimension (32)
            state_dim: Unified state dimension (32)
            render_mode: Render mode for simulation
            action_scale: Scale factor for action bounds (0-1). Use smaller values for safer exploration.
            action_bounds: Custom action bounds dict to override defaults. Keys depend on control_mode:
                - delta_qpos: {'joint': (low, high), 'gripper': (low, high)}
                - delta_ee_pos: {'position': (low, high), 'gripper': (low, high)}
                - delta_ee: {'position': (low, high), 'rotation': (low, high), 'gripper': (low, high)}
            single_arm: If True, only use left arm, right arm actions are zeroed
            scene_reset_interval: Regenerate scene every N episodes (1 = every episode, higher = faster)
        """
        super().__init__()
        
        self.phase = phase
        self.history_length = history_length
        self.max_steps = max_steps
        self.device = device
        self.teacher_policy = teacher_policy
        self.image_size = image_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.render_mode_str = render_mode
        self.max_reset_retries = max_reset_retries
        self.action_scale = np.clip(action_scale, 0.01, 1.0)  # Clamp to valid range
        self.custom_action_bounds = action_bounds
        self.single_arm = single_arm
        self.scene_reset_interval = max(1, scene_reset_interval)  # At least 1
        
        # Logger
        self.logger = get_env_logger()
        
        # Task configuration
        self.task_config = task_config
        self.task = None
        self._setup_task()
        
        # Action and state normalizers (will be setup in reset)
        self.action_normalizer = None
        self.state_normalizer = None
        
        # Define spaces
        self._define_spaces()
        
        # History buffers (will be initialized in reset)
        self.action_history = None
        self.state_history = None
        self.image_history = None
        
        # Memory states for student phase
        self.teacher_memory_state = None
        self.student_memory_state = None
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_count = 0
        
        # Initialized successfully
        
    def _setup_task(self):
        """Setup the underlying task environment."""
        import sys
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        robotwin_dir = os.path.dirname(os.path.dirname(script_dir))
        sys.path.insert(0, robotwin_dir)
        
        from envs.tasks import random_exploration
        self.task_class = random_exploration
    
    def _compute_scaled_action_bounds(self, control_mode: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Compute scaled action bounds based on action_scale and custom_action_bounds.
        
        Args:
            control_mode: 'delta_qpos', 'delta_ee_pos', or 'delta_ee'
            
        Returns:
            Dictionary of scaled action bounds, or None to use defaults
        """
        # Default bounds (from ActionNormalizer.DEFAULT_BOUNDS)
        default_bounds = {
            'delta_qpos': {
                'joint': (-0.1, 0.1),      # radians
                'gripper': (-1.0, 1.0),    # normalized gripper
            },
            'delta_ee_pos': {
                'position': (-0.05, 0.05),  # meters
                'gripper': (-1.0, 1.0),
            },
            'delta_ee': {
                'position': (-0.05, 0.05),  # meters  
                'rotation': (-0.1, 0.1),    # quaternion delta (small)
                'gripper': (-1.0, 1.0),
            },
        }
        
        # Start with defaults for this control mode
        base_bounds = default_bounds.get(control_mode, {}).copy()
        
        # Apply custom bounds if provided (override defaults)
        if self.custom_action_bounds:
            base_bounds.update(self.custom_action_bounds)
        
        # Apply action_scale to non-gripper bounds
        scaled_bounds = {}
        for key, (low, high) in base_bounds.items():
            if key == 'gripper':
                # Don't scale gripper - keep full range
                scaled_bounds[key] = (low, high)
            else:
                # Scale other action bounds
                scaled_bounds[key] = (low * self.action_scale, high * self.action_scale)
        
        self.logger.debug(
            f"Action bounds (scale={self.action_scale}): {scaled_bounds}"
        )
        
        return scaled_bounds
    
    def set_action_scale(self, scale: float):
        """
        Dynamically update action scale. Useful for curriculum learning.
        
        Args:
            scale: New scale factor (0.01-1.0)
        """
        old_scale = self.action_scale
        self.action_scale = np.clip(scale, 0.01, 1.0)
        
        # Update normalizer bounds if normalizer exists
        if self.action_normalizer is not None:
            control_mode = self.action_normalizer.control_mode
            new_bounds = self._compute_scaled_action_bounds(control_mode)
            self.action_normalizer.update_bounds(new_bounds)
            
        self.logger.info(f"Action scale changed: {old_scale} -> {self.action_scale}")
    
    def set_action_bounds(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Dynamically update custom action bounds.
        
        Args:
            bounds: Dictionary of action bounds, e.g. {'joint': (-0.05, 0.05)}
        """
        self.custom_action_bounds = bounds
        
        # Update normalizer if exists
        if self.action_normalizer is not None:
            control_mode = self.action_normalizer.control_mode
            new_bounds = self._compute_scaled_action_bounds(control_mode)
            self.action_normalizer.update_bounds(new_bounds)
            
        self.logger.info(f"Action bounds updated: {bounds}")
        
    def _define_spaces(self):
        """Define observation and action spaces."""
        # Action space: normalized 32-dim continuous
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        # Observation space
        obs_dict = {
            # State: 32-dim normalized
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32
            ),
            # Action history: (history_length, action_dim)
            "action_history": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.history_length, self.action_dim),
                dtype=np.float32
            ),
        }
        
        # Single arm: teacher uses head + wrist, student only uses wrist
        if self.phase == "teacher":
            obs_dict["head_rgb"] = spaces.Box(
                low=0, high=255,
                shape=(self.history_length, 3, *self.image_size),
                dtype=np.uint8
            )
        obs_dict["wrist_rgb"] = spaces.Box(
            low=0, high=255,
            shape=(self.history_length, 3, *self.image_size),
            dtype=np.uint8
        )
            
        self.observation_space = spaces.Dict(obs_dict)
        
    def _init_history_buffers(self):
        """Initialize history buffers with zeros/copies."""
        # Action history: filled with zeros
        self.action_history = deque(
            [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.history_length)],
            maxlen=self.history_length
        )
        
        # State history
        self.state_history = deque(maxlen=self.history_length)
        
        # Image history - head and wrist cameras
        self.image_history = {
            "head_rgb": deque(maxlen=self.history_length),
            "wrist_rgb": deque(maxlen=self.history_length),
        }
        
    def _fill_history_with_initial_obs(self, initial_obs: Dict[str, Any]):
        """Fill history buffers by repeating initial observation."""
        # Get current state
        state_dict = self.task._get_current_state()
        state_vec = self.state_normalizer.normalize_state(state_dict)
        
        # Fill state history
        for _ in range(self.history_length):
            self.state_history.append(state_vec.copy())
            
        # Fill image history by repeating initial image
        for key in self.image_history:
            img = initial_obs.get(key)
            if img is not None:
                for _ in range(self.history_length):
                    self.image_history[key].append(img.copy())
            else:
                # Create placeholder
                placeholder = np.zeros((3, *self.image_size), dtype=np.uint8)
                for _ in range(self.history_length):
                    self.image_history[key].append(placeholder)
                    
    def _get_raw_observation(self) -> Dict[str, np.ndarray]:
        """Get raw observation from task."""
        obs = self.task.get_observation()
        
        # Process images
        result = {}
        
        # Head camera
        if "head_rgb" in obs:
            img = obs["head_rgb"]
            if img.shape[:2] != self.image_size:
                img = self._resize_image(img, self.image_size)
            result["head_rgb"] = img.transpose(2, 0, 1)  # HWC -> CHW
        
        # Wrist camera - single arm uses "wrist_rgb" (from left_wrist_rgb or right_wrist_rgb)
        wrist_key = None
        for key in ["left_wrist_rgb", "right_wrist_rgb", "wrist_rgb"]:
            if key in obs:
                wrist_key = key
                break
        
        if wrist_key:
            img = obs[wrist_key]
            if img.shape[:2] != self.image_size:
                img = self._resize_image(img, self.image_size)
            result["wrist_rgb"] = img.transpose(2, 0, 1)  # HWC -> CHW
                
        return result
    
    def _resize_image(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        import cv2
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    
    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build observation dict from history buffers."""
        obs = {}
        
        # Current state
        state_dict = self.task._get_current_state()
        obs["state"] = self.state_normalizer.normalize_state(state_dict)
        
        # Action history
        obs["action_history"] = np.stack(list(self.action_history), axis=0)
        
        # Image history based on phase
        # Teacher gets head + wrist, student only gets wrist
        if self.phase == "teacher":
            obs["head_rgb"] = np.stack(list(self.image_history["head_rgb"]), axis=0)
        obs["wrist_rgb"] = np.stack(list(self.image_history["wrist_rgb"]), axis=0)
            
        return obs
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment with retry logic for UnstableError.
        
        This follows the collect_data.py pattern:
        - Robot is loaded ONCE via setup_demo (first reset)
        - Subsequent resets only reload scene objects via reset_for_new_episode
        - On UnStableError, just try a new seed without destroying the task
        """
        super().reset(seed=seed)
        
        # Import UnStableError
        from envs.utils.create_actor import UnStableError
        
        # Initialize seed counter
        current_seed = seed if seed is not None else np.random.randint(0, 2**31)
        
        # First, initialize task if not yet created (load robot ONCE)
        need_initial_setup = self.task is None
        
        if need_initial_setup:
            config = self.task_config.copy()
            config["seed"] = current_seed
            config["render_mode"] = self.render_mode_str
            # Disable viewer for headless training (no display)
            config["render_freq"] = 0
            
            # Load embodiment configurations (required for robot setup)
            # rl/f1_rl_env.py -> RoboTwin
            robotwin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config = load_embodiment_config(config, base_dir=robotwin_dir)
            
            # First reset: load robot
            if self.episode_count == 0:
                self.logger.info("Loading robot (first reset)...")
            
            # Save config for later use in reset_for_new_episode
            self._saved_config = config
            
            # Create task and load robot - this may fail with UnStableError on first setup
            for retry in range(self.max_reset_retries):
                try:
                    self.task = self.task_class()
                    self.task.setup_demo(**config)
                    
                    # Setup normalizers with optional action bounds
                    control_mode = config.get("control_mode", "delta_qpos")
                    custom_bounds = self._compute_scaled_action_bounds(control_mode)
                    
                    self.action_normalizer = ActionNormalizer(
                        control_mode=control_mode,
                        left_arm_dof=self.task.robot_info["left_arm_dof"],
                        right_arm_dof=self.task.robot_info["right_arm_dof"],
                        custom_bounds=custom_bounds,
                        single_arm=self.single_arm,
                    )
                    self.state_normalizer = StateNormalizer(
                        control_mode=config.get("control_mode", "delta_qpos"),
                        left_arm_dof=self.task.robot_info["left_arm_dof"],
                        right_arm_dof=self.task.robot_info["right_arm_dof"],
                        joint_limits=self.task.robot_info["arm_joint_limits"],
                        single_arm=self.single_arm,
                    )
                    
                    # Robot loaded
                    break  # Success
                    
                except UnStableError as e:
                    self.logger.debug(
                        f"UnStableError on initial setup attempt {retry + 1}/{self.max_reset_retries}: {e}"
                    )
                    # On initial setup failure, need to destroy and recreate
                    if self.task is not None:
                        try:
                            self.task.close_env()
                        except:
                            pass
                        self.task = None
                    
                    # Try with a different seed
                    current_seed = np.random.randint(0, 2**31)
                    config["seed"] = current_seed
                    
                    if retry == self.max_reset_retries - 1:
                        self.logger.error(
                            f"Failed to setup environment after {self.max_reset_retries} attempts "
                            f"due to UnStableError"
                        )
                        raise RuntimeError(
                            f"Environment setup failed after {self.max_reset_retries} attempts "
                            f"due to unstable objects: {e}"
                        )
        else:
            # Task already exists - decide whether to reset scene or just robot state
            # Based on scene_reset_interval setting
            should_reset_scene = (self.episode_count % self.scene_reset_interval == 0)
            
            if should_reset_scene:
                # Full scene reset (regenerate objects)
                # Full scene reset
                for retry in range(self.max_reset_retries):
                    try:
                        self.task.reset_for_new_episode(seed=current_seed)
                        break  # Success
                        
                    except UnStableError as e:
                        self.logger.debug(
                            f"UnStableError on episode reset attempt {retry + 1}/{self.max_reset_retries}: {e}"
                        )
                        # Just try a new seed, DO NOT destroy the task
                        current_seed = np.random.randint(0, 2**31)
                        
                        if retry == self.max_reset_retries - 1:
                            self.logger.error(
                                f"Failed to reset episode after {self.max_reset_retries} attempts "
                                f"due to UnStableError"
                            )
                            raise RuntimeError(
                                f"Episode reset failed after {self.max_reset_retries} attempts "
                                f"due to unstable objects: {e}"
                            )
            else:
                # Fast reset: only reset robot to home state, keep scene unchanged
                # Fast reset (robot only)
                self.task._reset_robot_only()
                    
        # Initialize history buffers
        self._init_history_buffers()
        
        # Get initial observation
        raw_obs = self._get_raw_observation()
        self._fill_history_with_initial_obs(raw_obs)
        
        # Reset memory states
        self.teacher_memory_state = None
        self.student_memory_state = None
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_count += 1
        
        obs = self._build_observation()
        info = {
            "embodiment": self.task.robot_info.get("embodiment", "unknown"),
            "control_mode": self.task.control_mode,
        }
        
        # Only log episode reset at INFO level for every 100th episode
        if self.episode_count % 100 == 1:
            self.logger.info(
                f"Episode {self.episode_count}: phase={self.phase}, "
                f"embodiment={info['embodiment']}, control_mode={info['control_mode']}"
            )
        
        return obs, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        For teacher phase: action is ignored, random action is used
        For student phase: action is from explorer policy
        """
        self.current_step += 1
        
        # Get observation before action
        obs_before = self._get_raw_observation()
        state_before = self.task._get_current_state()
        state_vec_before = self.state_normalizer.normalize_state(state_before)
        
        # Execute action
        if self.phase == "teacher":
            # Teacher phase: use random action
            raw_action = self.action_normalizer.sample_random_raw_action()
            normalized_action = self.action_normalizer.normalize(raw_action)
        else:
            # Student phase: use provided action from explorer
            normalized_action = np.clip(action, -1.0, 1.0)
            raw_action = self.action_normalizer.denormalize(normalized_action)
            
        # Convert to action dict and execute
        action_dict = self.action_normalizer.vector_to_action_dict(raw_action)
        self.task._execute_delta_action(action_dict)
        
        # Get observation after action
        obs_after = self._get_raw_observation()
        state_after = self.task._get_current_state()
        state_vec_after = self.state_normalizer.normalize_state(state_after)
        
        # Update history
        self.action_history.append(normalized_action.copy())
        self.state_history.append(state_vec_after.copy())
        for key in self.image_history:
            if key in obs_after:
                self.image_history[key].append(obs_after[key].copy())
                
        # Compute reward based on phase
        if self.phase == "teacher":
            reward, reward_info = self._compute_teacher_reward(
                obs_before, obs_after, normalized_action
            )
        else:
            reward, reward_info = self._compute_student_reward(
                obs_before, obs_after, normalized_action
            )
            
        self.episode_reward += reward
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Build observation
        obs = self._build_observation()
        
        info = {
            "embodiment": self.task.robot_info.get("embodiment", "unknown"),
            "control_mode": self.task.control_mode,
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "action_executed": normalized_action.tolist(),
            **reward_info
        }
        
        # Log step info periodically (only at DEBUG level to reduce noise)
        if self.current_step % 10 == 0 or truncated:
            self.logger.debug(
                f"Step {self.current_step}: reward={reward:.4f}, "
                f"episode_reward={self.episode_reward:.4f}"
            )
        
        # Episode finish log at DEBUG level - main training script will log summary
        if truncated:
            self.logger.debug(
                f"Episode {self.episode_count} finished: "
                f"steps={self.current_step}, total_reward={self.episode_reward:.4f}"
            )
        
        return obs, reward, terminated, truncated, info
    
    def _compute_teacher_reward(
        self,
        obs_before: Dict[str, np.ndarray],
        obs_after: Dict[str, np.ndarray],
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward for teacher phase.
        
        Reward = negative prediction error of world model
        """
        if self.teacher_policy is None:
            # No policy to evaluate, return dummy reward
            return 0.0, {"reward_type": "dummy"}
            
        # Build batch for world model prediction
        batch = self._build_policy_batch(obs_before, action, use_head_camera=True)
        
        with torch.no_grad():
            # Get world model prediction
            pred_output = self.teacher_policy.predict_images_only(batch)
            pred_imgs = pred_output["pred_imgs"]  # [B, C, H, W]
            
            # Get ground truth next image
            gt_img = torch.from_numpy(obs_after["head_rgb"]).float().to(self.device)
            gt_img = gt_img.unsqueeze(0) / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
            
            # Resize if needed
            if pred_imgs.shape[-2:] != gt_img.shape[-2:]:
                gt_img = F.interpolate(gt_img, size=pred_imgs.shape[-2:], mode="bilinear")
                
            # Compute prediction error (MSE)
            pred_error = F.mse_loss(pred_imgs, gt_img).item()
            
            # Reward is negative error (higher accuracy = higher reward)
            reward = -pred_error
            
            # Update memory state
            self.teacher_memory_state = pred_output.get("memory_state")
            
        reward_info = {
            "reward_type": "teacher_wm",
            "prediction_error": pred_error,
            "wm_accuracy": 1.0 - min(pred_error, 1.0),
        }
        
        self.logger.debug(
            f"Teacher reward: pred_error={pred_error:.4f}, "
            f"wm_accuracy={reward_info['wm_accuracy']:.4f}"
        )
        
        return reward, reward_info
    
    def _compute_student_reward(
        self,
        obs_before: Dict[str, np.ndarray],
        obs_after: Dict[str, np.ndarray],
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward for student phase.
        
        Reward = |h_student_t - h_teacher_t| - |h_student_t+1 - h_teacher_t+1|
        
        This reward is positive when student's memory state moves closer to teacher's memory state.
        The student is encouraged to take actions that reduce the divergence between its 
        memory state and the teacher's memory state over time.
        """
        if self.teacher_policy is None:
            return 0.0, {"reward_type": "dummy"}
            
        # Build batch for prediction
        batch_teacher = self._build_policy_batch(obs_before, action, use_head_camera=True)
        batch_student = self._build_policy_batch(obs_before, action, use_head_camera=False)
        
        with torch.no_grad():
            # Get teacher's memory state at time t (before action) and t+1 (after action)
            # First, get teacher memory at t (before action was applied)
            teacher_output_t = self.teacher_policy.forward_memory_only(batch_teacher)
            teacher_memory_t = teacher_output_t.get("memory_state")
            
            # Get student memory at t
            student_output_t = self.teacher_policy.forward_memory_only(
                batch_student, 
                use_student_mode=True  # Use student's view (wrist cameras only)
            )
            student_memory_t = student_output_t.get("memory_state")
            
            # Compute divergence at time t: |h_student_t - h_teacher_t|
            divergence_t = 0.0
            if teacher_memory_t is not None and student_memory_t is not None:
                divergence_t = torch.norm(student_memory_t - teacher_memory_t).item()
            
            # Now simulate what happens after the action
            # Update batches with obs_after
            batch_teacher_after = self._build_policy_batch(obs_after, action, use_head_camera=True)
            batch_student_after = self._build_policy_batch(obs_after, action, use_head_camera=False)
            
            # Get teacher memory at t+1
            teacher_output_t1 = self.teacher_policy.forward_memory_only(
                batch_teacher_after,
                prev_memory_state=teacher_memory_t
            )
            teacher_memory_t1 = teacher_output_t1.get("memory_state")
            
            # Get student memory at t+1
            student_output_t1 = self.teacher_policy.forward_memory_only(
                batch_student_after,
                prev_memory_state=student_memory_t,
                use_student_mode=True
            )
            student_memory_t1 = student_output_t1.get("memory_state")
            
            # Compute divergence at time t+1: |h_student_t+1 - h_teacher_t+1|
            divergence_t1 = 0.0
            if teacher_memory_t1 is not None and student_memory_t1 is not None:
                divergence_t1 = torch.norm(student_memory_t1 - teacher_memory_t1).item()
            
            # Reward = divergence_t - divergence_t+1
            # Positive when divergence decreases (student moves closer to teacher)
            reward = divergence_t - divergence_t1
            
            # Update memory states for next step
            self.teacher_memory_state = teacher_memory_t1
            self.student_memory_state = student_memory_t1
            
            # Optional: Add small bonus for exploration (WM uncertainty)
            # This encourages student to explore diverse states
            if "pred_imgs" in teacher_output_t1:
                pred_imgs = teacher_output_t1["pred_imgs"]
                gt_img = torch.from_numpy(
                    obs_after.get("head_rgb", np.zeros((3, *self.image_size), dtype=np.uint8))
                ).float().to(self.device)
                gt_img = gt_img.unsqueeze(0) / 255.0 * 2.0 - 1.0
                
                if pred_imgs.shape[-2:] != gt_img.shape[-2:]:
                    gt_img = F.interpolate(gt_img, size=pred_imgs.shape[-2:], mode="bilinear")
                    
                wm_uncertainty = F.mse_loss(pred_imgs, gt_img).item()
            else:
                wm_uncertainty = 0.0
                
            # Optional: Add small exploration bonus
            exploration_bonus_weight = 0.1
            reward += exploration_bonus_weight * wm_uncertainty
            
        reward_info = {
            "reward_type": "student_memory_alignment",
            "divergence_t": divergence_t,
            "divergence_t1": divergence_t1,
            "divergence_reduction": divergence_t - divergence_t1,
            "wm_uncertainty": wm_uncertainty,
            "exploration_bonus_weight": exploration_bonus_weight,
        }
        
        self.logger.debug(
            f"Student reward: div_t={divergence_t:.4f}, div_t1={divergence_t1:.4f}, "
            f"reduction={divergence_t - divergence_t1:.4f}, wm_unc={wm_uncertainty:.4f}"
        )
        
        return reward, reward_info
    
    def _build_policy_batch(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        use_head_camera: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Build batch dict for F1 policy forward pass.
        
        According to model requirements:
        - LLM input: head_rgb (image0) + wrist_rgb (image1) single frames + state
        - WM input: wrist_rgb history (n_obs_img_steps frames) + action history
        
        Args:
            obs: Observation dict. Images can be either:
                - Single frame [C, H, W] (from raw observation)
                - Stacked history [T, C, H, W] (from _build_observation)
            action: Action to take
            use_head_camera: Whether to use head camera (True) or wrist cameras (False)
        """
        batch = {}
        
        # World model needs n_obs_img_steps frames
        # From debug_test.yaml: n_obs_img_steps=4, obs_img_stride=1
        # So we need exactly 4 frames (no sampling needed when stride=1)
        n_obs_frames = self.history_length  # Should be 4 (or 5 if including pred target)
        
        # State
        state_dict = self.task._get_current_state()
        state_vec = self.state_normalizer.normalize_state(state_dict)
        batch["observation.state"] = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        
        # Action
        batch["action"] = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
        
        # Action history - take the last n_obs_frames actions
        action_hist = np.stack(list(self.action_history), axis=0)  # [history_length, action_dim]
        # Take the most recent n_obs_frames (or all if less)
        sampled_action_hist = action_hist[-n_obs_frames:]  # [n_obs_frames, action_dim]
        batch["action_history"] = torch.from_numpy(sampled_action_hist).float().unsqueeze(0).to(self.device)
        
        # Images
        def process_image(img, normalize=True):
            """Process image to tensor.
            
            Args:
                img: Either [C, H, W] or [T, C, H, W]
                normalize: Whether to normalize to [-1, 1] range. Set False if 
                          the policy will normalize it again.
            Returns:
                Tensor with batch dim: [1, C, H, W] or [1, T, C, H, W]
            """
            img_tensor = torch.from_numpy(img).float().to(self.device)
            if normalize:
                img_tensor = img_tensor / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
            else:
                img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
            return img_tensor.unsqueeze(0)
        
        def get_current_and_history(img_data, history_deque):
            """Extract current frame and history from image data.
            
            Returns:
                current_img: [C, H, W]
                hist_imgs: [n_obs_frames, C, H, W]
            """
            if img_data.ndim == 4:
                # Stacked history [T, C, H, W]: use last frame for current
                current_img = img_data[-1]  # [C, H, W]
                # Take the most recent n_obs_frames
                hist_imgs = img_data[-n_obs_frames:]  # [n_obs_frames, C, H, W]
            else:
                # Single image [C, H, W]
                current_img = img_data
                full_hist = np.stack(list(history_deque), axis=0)  # [history_length, C, H, W]
                hist_imgs = full_hist[-n_obs_frames:]  # [n_obs_frames, C, H, W]
            return current_img, hist_imgs
        
        # === LLM Inputs: images for Paligemma ===
        # === WM Input: wrist_rgb history (image0_history for world model) ===
        
        # Get wrist camera data (always needed for WM)
        if "wrist_rgb" in obs:
            wrist_img = obs["wrist_rgb"]
            current_wrist, hist_wrist = get_current_and_history(wrist_img, self.image_history["wrist_rgb"])
            
            # World model always needs wrist history
            batch["observation.images.image0_history"] = process_image(hist_wrist, normalize=True)
            
            if use_head_camera:
                # Teacher: head_rgb (image0) + wrist_rgb (image1)
                # Head camera -> image0
                if "head_rgb" in obs:
                    head_img = obs["head_rgb"]
                    if head_img.ndim == 4:
                        current_head = head_img[-1]  # Last frame
                    else:
                        current_head = head_img
                    batch["observation.images.image0"] = process_image(current_head, normalize=False)
                    batch["observation.images.image0_mask"] = torch.ones(1, dtype=torch.bool, device=self.device)
                
                # Wrist camera -> image1
                batch["observation.images.image1"] = process_image(current_wrist, normalize=False)
                batch["observation.images.image1_mask"] = torch.ones(1, dtype=torch.bool, device=self.device)
            else:
                # Student: wrist_rgb only (image0)
                batch["observation.images.image0"] = process_image(current_wrist, normalize=False)
                batch["observation.images.image0_mask"] = torch.ones(1, dtype=torch.bool, device=self.device)
                
        # Task description (placeholder)
        batch["task"] = ["explore the environment"]
        
        return batch
    
    def close(self):
        """Close the environment."""
        if self.task is not None:
            try:
                self.task.close_env()
            except:
                pass
            self.task = None
            
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.task is None:
            return None
        obs = self.task.get_observation()
        return obs.get("head_rgb")


class TeacherEnv(F1RLEnv):
    """Convenience class for teacher phase training."""
    
    def __init__(self, task_config: Dict[str, Any], **kwargs):
        kwargs.pop("phase", None)
        super().__init__(task_config, phase="teacher", **kwargs)


class StudentEnv(F1RLEnv):
    """Convenience class for student phase training."""
    
    def __init__(self, task_config: Dict[str, Any], teacher_policy: Any, **kwargs):
        kwargs.pop("phase", None)
        if teacher_policy is None:
            raise ValueError("teacher_policy is required for student phase")
        super().__init__(task_config, phase="student", teacher_policy=teacher_policy, **kwargs)
