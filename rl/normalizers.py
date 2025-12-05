"""
Action Normalizer for Cross-Embodiment, Cross-Control-Mode RL Training

This module handles:
1. Normalization of raw actions to [-1, 1] range for unified 32-dim action space
2. Denormalization from normalized actions back to raw actions
3. Zero-padding for different DOF configurations
4. Control mode specific transformations

Action Dimension Layout (32-dim):
    [left_arm (max 8), left_gripper (1), right_arm (max 8), right_gripper (1), padding (14)]
    
For delta_qpos mode with 7-DOF robot:
    [left_qpos (7), 0, left_gripper (1), right_qpos (7), 0, right_gripper (1), padding (14)]
    
For delta_ee_pos mode:
    [left_pos (3), 0,0,0,0,0, left_gripper (1), right_pos (3), 0,0,0,0,0, right_gripper (1), padding (14)]
    
For delta_ee mode:
    [left_pos (3), left_quat (4), 0, left_gripper (1), right_pos (3), right_quat (4), 0, right_gripper (1), padding (14)]
"""

import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, Any


def setup_normalizer_logger(log_dir: str = "logs/normalizer") -> logging.Logger:
    """Setup logger for normalizer module."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("normalizer")
    if logger.handlers:  # Already configured
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"normalizer_{timestamp}.log")
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (INFO level only)
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
_logger = None

def get_logger() -> logging.Logger:
    """Get or create the normalizer logger."""
    global _logger
    if _logger is None:
        _logger = setup_normalizer_logger()
    return _logger


class ActionNormalizer:
    """
    Normalizes and denormalizes actions between raw space and unified 32-dim normalized space.
    """
    
    # Default action bounds for different control modes
    DEFAULT_BOUNDS = {
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
    
    MAX_ACTION_DIM = 32
    MAX_ARM_DIM = 8  # max DOF per arm (including padding)
    
    def __init__(
        self,
        control_mode: str,
        left_arm_dof: int,
        right_arm_dof: int,
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        single_arm: bool = False,  # Single arm mode: only use left arm
    ):
        """
        Initialize the action normalizer.
        
        Args:
            control_mode: 'delta_qpos', 'delta_ee', or 'delta_ee_pos'
            left_arm_dof: Degrees of freedom for left arm
            right_arm_dof: Degrees of freedom for right arm
            custom_bounds: Optional custom action bounds
            single_arm: If True, only use left arm, right arm is zeroed
        """
        self.control_mode = control_mode
        self.left_arm_dof = left_arm_dof
        self.right_arm_dof = right_arm_dof
        self.single_arm = single_arm
        
        # Compute raw action dimension based on control mode
        self.raw_action_dim = self._compute_raw_action_dim()
        
        # Setup bounds - IMPORTANT: copy to avoid modifying class defaults
        self.bounds = self.DEFAULT_BOUNDS.get(control_mode, {}).copy()
        if custom_bounds:
            self.bounds.update(custom_bounds)
            
        # Precompute normalization parameters
        self._setup_normalization_params()
        
        # Logger
        self.logger = get_logger()
        self.logger.info(
            f"ActionNormalizer initialized: control_mode={control_mode}, "
            f"left_dof={left_arm_dof}, right_dof={right_arm_dof}, "
            f"raw_dim={self.raw_action_dim}, single_arm={single_arm}"
        )
        self.logger.info(f"Action bounds: {self.bounds}")
        
    def _compute_raw_action_dim(self) -> int:
        """Compute raw action dimension based on control mode and DOFs."""
        if self.control_mode == 'delta_qpos':
            # [left_qpos, left_gripper, right_qpos, right_gripper]
            return self.left_arm_dof + 1 + self.right_arm_dof + 1
        elif self.control_mode == 'delta_ee_pos':
            # [left_pos(3), left_gripper, right_pos(3), right_gripper]
            return 3 + 1 + 3 + 1
        elif self.control_mode == 'delta_ee':
            # [left_pos(3), left_quat(4), left_gripper, right_pos(3), right_quat(4), right_gripper]
            return 3 + 4 + 1 + 3 + 4 + 1
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
            
    def _setup_normalization_params(self):
        """Setup normalization parameters for each action component."""
        if self.control_mode == 'delta_qpos':
            joint_low, joint_high = self.bounds['joint']
            grip_low, grip_high = self.bounds['gripper']
            
            # Create per-element bounds
            self.raw_low = np.concatenate([
                np.full(self.left_arm_dof, joint_low),
                [grip_low],
                np.full(self.right_arm_dof, joint_low),
                [grip_low]
            ])
            self.raw_high = np.concatenate([
                np.full(self.left_arm_dof, joint_high),
                [grip_high],
                np.full(self.right_arm_dof, joint_high),
                [grip_high]
            ])
            
        elif self.control_mode == 'delta_ee_pos':
            pos_low, pos_high = self.bounds['position']
            grip_low, grip_high = self.bounds['gripper']
            
            self.raw_low = np.array([
                pos_low, pos_low, pos_low, grip_low,
                pos_low, pos_low, pos_low, grip_low
            ])
            self.raw_high = np.array([
                pos_high, pos_high, pos_high, grip_high,
                pos_high, pos_high, pos_high, grip_high
            ])
            
        elif self.control_mode == 'delta_ee':
            pos_low, pos_high = self.bounds['position']
            rot_low, rot_high = self.bounds['rotation']
            grip_low, grip_high = self.bounds['gripper']
            
            self.raw_low = np.array([
                # Left arm
                pos_low, pos_low, pos_low,  # position
                rot_low, rot_low, rot_low, rot_low,  # quaternion
                grip_low,
                # Right arm
                pos_low, pos_low, pos_low,
                rot_low, rot_low, rot_low, rot_low,
                grip_low
            ])
            self.raw_high = np.array([
                # Left arm
                pos_high, pos_high, pos_high,
                rot_high, rot_high, rot_high, rot_high,
                grip_high,
                # Right arm
                pos_high, pos_high, pos_high,
                rot_high, rot_high, rot_high, rot_high,
                grip_high
            ])

    def normalize(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Normalize raw action to [-1, 1] range and pad to 32-dim.
        
        Args:
            raw_action: Raw action from environment (variable dimension)
            
        Returns:
            Normalized 32-dim action in [-1, 1] range
        """
        # Check for out-of-bound values
        oob_low = raw_action < self.raw_low
        oob_high = raw_action > self.raw_high
        if np.any(oob_low) or np.any(oob_high):
            self.logger.warning(
                f"Action out of bounds: raw={raw_action}, "
                f"bounds=[{self.raw_low}, {self.raw_high}]"
            )
        
        # Clip to bounds
        clipped = np.clip(raw_action, self.raw_low, self.raw_high)
        
        # Normalize to [-1, 1]
        normalized = 2.0 * (clipped - self.raw_low) / (self.raw_high - self.raw_low + 1e-8) - 1.0
        
        # Pad to 32 dimensions
        padded = self._pad_to_32dim(normalized)
        
        # Only log if DEBUG is enabled (avoids overhead)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Normalized action: raw_shape={raw_action.shape}, "
                f"norm_range=[{normalized.min():.3f}, {normalized.max():.3f}]"
            )
        
        return padded.astype(np.float32)
    
    def denormalize(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Denormalize from [-1, 1] range back to raw action space.
        
        Args:
            normalized_action: 32-dim normalized action
            
        Returns:
            Raw action for environment (variable dimension)
        """
        # Extract only the relevant dimensions
        raw_normalized = self._extract_from_32dim(normalized_action)
        
        # Check for out-of-range normalized values
        if np.any(np.abs(raw_normalized) > 1.0 + 1e-6):
            self.logger.warning(
                f"Normalized action out of [-1,1] range: "
                f"range=[{raw_normalized.min():.3f}, {raw_normalized.max():.3f}]"
            )
        
        # Clip to [-1, 1]
        raw_normalized = np.clip(raw_normalized, -1.0, 1.0)
        
        # Denormalize to raw space
        raw_action = (raw_normalized + 1.0) / 2.0 * (self.raw_high - self.raw_low) + self.raw_low
        
        self.logger.debug(
            f"Denormalized action: output_shape={raw_action.shape}, "
            f"raw_range=[{raw_action.min():.4f}, {raw_action.max():.4f}]"
        )
        
        return raw_action.astype(np.float32)
    
    def _pad_to_32dim(self, normalized: np.ndarray) -> np.ndarray:
        """Pad normalized action to 32 dimensions with structured layout."""
        result = np.zeros(self.MAX_ACTION_DIM, dtype=np.float32)
        
        if self.control_mode == 'delta_qpos':
            # Layout: [left_arm(8), left_grip(1), right_arm(8), right_grip(1), padding(14)]
            # Left arm
            result[:self.left_arm_dof] = normalized[:self.left_arm_dof]
            result[self.MAX_ARM_DIM] = normalized[self.left_arm_dof]  # left gripper at index 8
            # Right arm
            right_start = self.left_arm_dof + 1
            result[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 1 + self.right_arm_dof] = normalized[right_start:right_start + self.right_arm_dof]
            result[2 * self.MAX_ARM_DIM + 1] = normalized[-1]  # right gripper at index 17
            
        elif self.control_mode == 'delta_ee_pos':
            # Layout: [left_pos(3), padding(5), left_grip(1), right_pos(3), padding(5), right_grip(1), padding(14)]
            result[0:3] = normalized[0:3]  # left position
            result[self.MAX_ARM_DIM] = normalized[3]  # left gripper
            result[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4] = normalized[4:7]  # right position
            result[2 * self.MAX_ARM_DIM + 1] = normalized[7]  # right gripper
            
        elif self.control_mode == 'delta_ee':
            # Layout: [left_pos(3), left_quat(4), padding(1), left_grip(1), ...]
            result[0:3] = normalized[0:3]  # left position
            result[3:7] = normalized[3:7]  # left quaternion
            result[self.MAX_ARM_DIM] = normalized[7]  # left gripper
            result[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4] = normalized[8:11]  # right position
            result[self.MAX_ARM_DIM + 4:self.MAX_ARM_DIM + 8] = normalized[11:15]  # right quaternion
            result[2 * self.MAX_ARM_DIM + 1] = normalized[15]  # right gripper
            
        return result
    
    def _extract_from_32dim(self, padded: np.ndarray) -> np.ndarray:
        """Extract raw action dimensions from 32-dim padded action."""
        if self.control_mode == 'delta_qpos':
            # Extract left arm, left gripper, right arm, right gripper
            left_arm = padded[:self.left_arm_dof]
            left_grip = padded[self.MAX_ARM_DIM:self.MAX_ARM_DIM + 1]
            right_arm = padded[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 1 + self.right_arm_dof]
            right_grip = padded[2 * self.MAX_ARM_DIM + 1:2 * self.MAX_ARM_DIM + 2]
            return np.concatenate([left_arm, left_grip, right_arm, right_grip])
            
        elif self.control_mode == 'delta_ee_pos':
            left_pos = padded[0:3]
            left_grip = padded[self.MAX_ARM_DIM:self.MAX_ARM_DIM + 1]
            right_pos = padded[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4]
            right_grip = padded[2 * self.MAX_ARM_DIM + 1:2 * self.MAX_ARM_DIM + 2]
            return np.concatenate([left_pos, left_grip, right_pos, right_grip])
            
        elif self.control_mode == 'delta_ee':
            left_pos = padded[0:3]
            left_quat = padded[3:7]
            left_grip = padded[self.MAX_ARM_DIM:self.MAX_ARM_DIM + 1]
            right_pos = padded[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4]
            right_quat = padded[self.MAX_ARM_DIM + 4:self.MAX_ARM_DIM + 8]
            right_grip = padded[2 * self.MAX_ARM_DIM + 1:2 * self.MAX_ARM_DIM + 2]
            return np.concatenate([left_pos, left_quat, left_grip, right_pos, right_quat, right_grip])
            
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action bounds for RL algorithms (in normalized [-1, 1] space).
        
        Returns:
            (low, high) bounds as 32-dim arrays
        """
        return np.full(self.MAX_ACTION_DIM, -1.0, dtype=np.float32), np.full(self.MAX_ACTION_DIM, 1.0, dtype=np.float32)
    
    def get_raw_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get raw action bounds."""
        return self.raw_low.copy(), self.raw_high.copy()
    
    def update_bounds(self, new_bounds: Dict[str, Tuple[float, float]]):
        """
        Update action bounds dynamically. Useful for curriculum learning.
        
        Args:
            new_bounds: Dictionary of new bounds, e.g. {'joint': (-0.05, 0.05)}
        """
        self.bounds.update(new_bounds)
        self._setup_normalization_params()
        self.logger.info(f"Action bounds updated: {self.bounds}")
    
    def sample_random_action(self) -> np.ndarray:
        """Sample a random normalized action."""
        action = np.random.uniform(-1.0, 1.0, self.MAX_ACTION_DIM).astype(np.float32)
        if self.single_arm:
            # Zero out right arm portion (indices 9-17 for delta_qpos layout)
            action[self.MAX_ARM_DIM + 1:2 * self.MAX_ARM_DIM + 2] = 0.0
        return action
    
    def sample_random_raw_action(self) -> np.ndarray:
        """Sample a random raw action within bounds."""
        action = np.random.uniform(self.raw_low, self.raw_high).astype(np.float32)
        if self.single_arm:
            # Zero out right arm portion in raw action
            # Layout: [left_qpos(N), left_grip, right_qpos(N), right_grip]
            right_start = self.left_arm_dof + 1
            action[right_start:] = 0.0
        return action
    
    def action_dict_to_vector(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """Convert action dict from environment to raw action vector."""
        if self.control_mode == 'delta_qpos':
            return np.concatenate([
                action_dict['left_delta_qpos'],
                [action_dict['left_delta_gripper']],
                action_dict['right_delta_qpos'],
                [action_dict['right_delta_gripper']]
            ]).astype(np.float32)
        elif self.control_mode == 'delta_ee_pos':
            return np.concatenate([
                action_dict['left_delta_pos'],
                [action_dict['left_delta_gripper']],
                action_dict['right_delta_pos'],
                [action_dict['right_delta_gripper']]
            ]).astype(np.float32)
        elif self.control_mode == 'delta_ee':
            return np.concatenate([
                action_dict['left_delta_pos'],
                action_dict['left_delta_quat'],
                [action_dict['left_delta_gripper']],
                action_dict['right_delta_pos'],
                action_dict['right_delta_quat'],
                [action_dict['right_delta_gripper']]
            ]).astype(np.float32)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
    
    def vector_to_action_dict(self, raw_action: np.ndarray) -> Dict[str, Any]:
        """Convert raw action vector to action dict for environment."""
        if self.control_mode == 'delta_qpos':
            idx = 0
            left_qpos = raw_action[idx:idx + self.left_arm_dof]
            idx += self.left_arm_dof
            left_grip = raw_action[idx]
            idx += 1
            right_qpos = raw_action[idx:idx + self.right_arm_dof]
            idx += self.right_arm_dof
            right_grip = raw_action[idx]
            
            # In single arm mode, zero out right arm
            if self.single_arm:
                right_qpos = np.zeros_like(right_qpos)
                right_grip = 0.0
            
            return {
                'left_delta_qpos': left_qpos,
                'left_delta_gripper': float(left_grip),
                'right_delta_qpos': right_qpos,
                'right_delta_gripper': float(right_grip),
            }
        elif self.control_mode == 'delta_ee_pos':
            right_pos = raw_action[4:7]
            right_grip = float(raw_action[7])
            
            # In single arm mode, zero out right arm
            if self.single_arm:
                right_pos = np.zeros(3, dtype=np.float32)
                right_grip = 0.0
            
            return {
                'left_delta_pos': raw_action[0:3],
                'left_delta_gripper': float(raw_action[3]),
                'right_delta_pos': right_pos,
                'right_delta_gripper': right_grip,
                'left_delta_quat': np.array([1, 0, 0, 0], dtype=np.float32),
                'right_delta_quat': np.array([1, 0, 0, 0], dtype=np.float32),
            }
        elif self.control_mode == 'delta_ee':
            right_pos = raw_action[8:11]
            right_quat = raw_action[11:15]
            right_grip = float(raw_action[15])
            
            # In single arm mode, zero out right arm
            if self.single_arm:
                right_pos = np.zeros(3, dtype=np.float32)
                right_quat = np.array([1, 0, 0, 0], dtype=np.float32)  # identity quat
                right_grip = 0.0
            
            return {
                'left_delta_pos': raw_action[0:3],
                'left_delta_quat': raw_action[3:7],
                'left_delta_gripper': float(raw_action[7]),
                'right_delta_pos': right_pos,
                'right_delta_quat': right_quat,
                'right_delta_gripper': right_grip,
            }
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")


class StateNormalizer:
    """
    Normalizes robot states to a unified 32-dim representation.
    
    State Layout (32-dim):
        [left_arm(8), left_gripper(1), right_arm(8), right_gripper(1), ee_info(6), padding(8)]
        
    For delta_qpos mode:
        [left_qpos(N), padding, left_gripper, right_qpos(N), padding, right_gripper, ...]
        
    For delta_ee modes:
        [left_ee_pos(3), left_ee_quat(4), padding, left_gripper, right_ee_pos(3), right_ee_quat(4), padding, right_gripper, ...]
    """
    
    MAX_STATE_DIM = 32
    MAX_ARM_DIM = 8
    
    # Default workspace bounds for EE position normalization
    DEFAULT_WORKSPACE_BOUNDS = {
        'x': (-0.5, 0.5),
        'y': (-0.5, 0.5),
        'z': (0.5, 1.5),
    }
    
    def __init__(
        self,
        control_mode: str,
        left_arm_dof: int,
        right_arm_dof: int,
        joint_limits: Optional[Dict[str, np.ndarray]] = None,
        workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        single_arm: bool = False,  # Single arm mode: right arm state is zeroed
    ):
        """
        Initialize state normalizer.
        
        Args:
            control_mode: Control mode ('delta_qpos', 'delta_ee', 'delta_ee_pos')
            left_arm_dof: Left arm degrees of freedom
            right_arm_dof: Right arm degrees of freedom
            joint_limits: Joint position limits for normalization
            workspace_bounds: Workspace bounds for EE position normalization
            single_arm: If True, right arm state is zeroed
        """
        self.control_mode = control_mode
        self.left_arm_dof = left_arm_dof
        self.right_arm_dof = right_arm_dof
        self.joint_limits = joint_limits
        self.workspace_bounds = workspace_bounds or self.DEFAULT_WORKSPACE_BOUNDS
        self.single_arm = single_arm
        
        # Compute raw state dimension
        self.raw_state_dim = self._compute_raw_state_dim()
        
        # Logger
        self.logger = get_logger()
        self.logger.info(
            f"StateNormalizer initialized: control_mode={control_mode}, "
            f"left_dof={left_arm_dof}, right_dof={right_arm_dof}, "
            f"raw_dim={self.raw_state_dim}, single_arm={single_arm}"
        )
        if joint_limits:
            self.logger.info(f"Joint limits provided: {list(joint_limits.keys())}")
        self.logger.info(f"Workspace bounds: {self.workspace_bounds}")
        
    def _compute_raw_state_dim(self) -> int:
        """Compute raw state dimension based on control mode."""
        if self.control_mode == 'delta_qpos':
            return self.left_arm_dof + 1 + self.right_arm_dof + 1
        elif self.control_mode == 'delta_ee_pos':
            return 3 + 1 + 3 + 1  # pos + grip for each arm
        elif self.control_mode == 'delta_ee':
            return 3 + 4 + 1 + 3 + 4 + 1  # pos + quat + grip for each arm
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
        
    def normalize_state(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """
        Normalize state dict to unified 32-dim representation.
        
        Args:
            state_dict: State dictionary from environment containing:
                - left_qpos, right_qpos (for delta_qpos)
                - left_ee_pos, right_ee_pos (for EE modes)
                - left_ee_quat, right_ee_quat (for delta_ee)
                - left_gripper, right_gripper
            
        Returns:
            Normalized 32-dim state vector in [-1, 1] range
        """
        result = np.zeros(self.MAX_STATE_DIM, dtype=np.float32)
        
        if self.control_mode == 'delta_qpos':
            # Normalize joint positions to [-1, 1] using joint limits
            left_qpos = np.asarray(state_dict['left_qpos']).flatten()
            right_qpos = np.asarray(state_dict['right_qpos']).flatten()
            
            if self.joint_limits is not None:
                left_qpos = self._normalize_joints(
                    left_qpos, 
                    self.joint_limits['left_lower'], 
                    self.joint_limits['left_upper']
                )
                right_qpos = self._normalize_joints(
                    right_qpos,
                    self.joint_limits['right_lower'],
                    self.joint_limits['right_upper']
                )
            
            # Layout: [left_qpos(N), padding, left_grip, right_qpos(N), padding, right_grip, ...]
            result[:self.left_arm_dof] = left_qpos[:self.left_arm_dof]
            result[self.MAX_ARM_DIM] = float(state_dict['left_gripper'])
            result[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 1 + self.right_arm_dof] = right_qpos[:self.right_arm_dof]
            result[2 * self.MAX_ARM_DIM + 1] = float(state_dict['right_gripper'])
            
        elif self.control_mode == 'delta_ee_pos':
            # Normalize EE positions
            left_pos = self._normalize_ee_pos(np.asarray(state_dict['left_ee_pos']).flatten())
            right_pos = self._normalize_ee_pos(np.asarray(state_dict['right_ee_pos']).flatten())
            
            result[0:3] = left_pos
            result[self.MAX_ARM_DIM] = float(state_dict['left_gripper'])
            result[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4] = right_pos
            result[2 * self.MAX_ARM_DIM + 1] = float(state_dict['right_gripper'])
            
        elif self.control_mode == 'delta_ee':
            # Normalize EE positions and include quaternions
            left_pos = self._normalize_ee_pos(np.asarray(state_dict['left_ee_pos']).flatten())
            right_pos = self._normalize_ee_pos(np.asarray(state_dict['right_ee_pos']).flatten())
            left_quat = np.asarray(state_dict['left_ee_quat']).flatten()
            right_quat = np.asarray(state_dict['right_ee_quat']).flatten()
            
            result[0:3] = left_pos
            result[3:7] = left_quat  # quaternions are already normalized
            result[self.MAX_ARM_DIM] = float(state_dict['left_gripper'])
            result[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4] = right_pos
            result[self.MAX_ARM_DIM + 4:self.MAX_ARM_DIM + 8] = right_quat
            result[2 * self.MAX_ARM_DIM + 1] = float(state_dict['right_gripper'])
        
        # In single arm mode, zero out right arm portion
        if self.single_arm:
            result[self.MAX_ARM_DIM + 1:2 * self.MAX_ARM_DIM + 2] = 0.0
        
        # Only log if DEBUG is enabled (avoids overhead)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Normalized state: mode={self.control_mode}, "
                f"output_range=[{result.min():.3f}, {result.max():.3f}]"
            )
            
        return result
    
    def denormalize_state(self, normalized_state: np.ndarray) -> Dict[str, Any]:
        """
        Denormalize state vector back to state dict.
        
        Args:
            normalized_state: 32-dim normalized state vector
            
        Returns:
            State dictionary with original scale values
        """
        state_dict = {}
        
        if self.control_mode == 'delta_qpos':
            left_qpos_norm = normalized_state[:self.left_arm_dof]
            left_grip = normalized_state[self.MAX_ARM_DIM]
            right_qpos_norm = normalized_state[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 1 + self.right_arm_dof]
            right_grip = normalized_state[2 * self.MAX_ARM_DIM + 1]
            
            if self.joint_limits is not None:
                state_dict['left_qpos'] = self._denormalize_joints(
                    left_qpos_norm,
                    self.joint_limits['left_lower'],
                    self.joint_limits['left_upper']
                )
                state_dict['right_qpos'] = self._denormalize_joints(
                    right_qpos_norm,
                    self.joint_limits['right_lower'],
                    self.joint_limits['right_upper']
                )
            else:
                state_dict['left_qpos'] = left_qpos_norm
                state_dict['right_qpos'] = right_qpos_norm
                
            state_dict['left_gripper'] = float(left_grip)
            state_dict['right_gripper'] = float(right_grip)
            
        elif self.control_mode == 'delta_ee_pos':
            state_dict['left_ee_pos'] = self._denormalize_ee_pos(normalized_state[0:3])
            state_dict['left_gripper'] = float(normalized_state[self.MAX_ARM_DIM])
            state_dict['right_ee_pos'] = self._denormalize_ee_pos(
                normalized_state[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4]
            )
            state_dict['right_gripper'] = float(normalized_state[2 * self.MAX_ARM_DIM + 1])
            
        elif self.control_mode == 'delta_ee':
            state_dict['left_ee_pos'] = self._denormalize_ee_pos(normalized_state[0:3])
            state_dict['left_ee_quat'] = normalized_state[3:7]
            state_dict['left_gripper'] = float(normalized_state[self.MAX_ARM_DIM])
            state_dict['right_ee_pos'] = self._denormalize_ee_pos(
                normalized_state[self.MAX_ARM_DIM + 1:self.MAX_ARM_DIM + 4]
            )
            state_dict['right_ee_quat'] = normalized_state[self.MAX_ARM_DIM + 4:self.MAX_ARM_DIM + 8]
            state_dict['right_gripper'] = float(normalized_state[2 * self.MAX_ARM_DIM + 1])
        
        self.logger.debug(
            f"Denormalized state: mode={self.control_mode}, "
            f"keys={list(state_dict.keys())}"
        )
            
        return state_dict
    
    def _normalize_joints(self, qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Normalize joint positions to [-1, 1]."""
        return 2.0 * (qpos - lower) / (upper - lower + 1e-8) - 1.0
    
    def _denormalize_joints(self, qpos_norm: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Denormalize joint positions from [-1, 1]."""
        return (qpos_norm + 1.0) / 2.0 * (upper - lower) + lower
    
    def _normalize_ee_pos(self, pos: np.ndarray) -> np.ndarray:
        """Normalize EE position to [-1, 1] using workspace bounds."""
        result = np.zeros(3, dtype=np.float32)
        bounds = self.workspace_bounds
        
        # X
        x_min, x_max = bounds['x']
        result[0] = 2.0 * (pos[0] - x_min) / (x_max - x_min + 1e-8) - 1.0
        
        # Y
        y_min, y_max = bounds['y']
        result[1] = 2.0 * (pos[1] - y_min) / (y_max - y_min + 1e-8) - 1.0
        
        # Z
        z_min, z_max = bounds['z']
        result[2] = 2.0 * (pos[2] - z_min) / (z_max - z_min + 1e-8) - 1.0
        
        return np.clip(result, -1.0, 1.0)
    
    def _denormalize_ee_pos(self, pos_norm: np.ndarray) -> np.ndarray:
        """Denormalize EE position from [-1, 1]."""
        result = np.zeros(3, dtype=np.float32)
        bounds = self.workspace_bounds
        
        x_min, x_max = bounds['x']
        result[0] = (pos_norm[0] + 1.0) / 2.0 * (x_max - x_min) + x_min
        
        y_min, y_max = bounds['y']
        result[1] = (pos_norm[1] + 1.0) / 2.0 * (y_max - y_min) + y_min
        
        z_min, z_max = bounds['z']
        result[2] = (pos_norm[2] + 1.0) / 2.0 * (z_max - z_min) + z_min
        
        return result
    
    def state_dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Convert raw state dict to raw state vector (without normalization)."""
        if self.control_mode == 'delta_qpos':
            return np.concatenate([
                state_dict['left_qpos'],
                [state_dict['left_gripper']],
                state_dict['right_qpos'],
                [state_dict['right_gripper']]
            ]).astype(np.float32)
        elif self.control_mode == 'delta_ee_pos':
            return np.concatenate([
                state_dict['left_ee_pos'],
                [state_dict['left_gripper']],
                state_dict['right_ee_pos'],
                [state_dict['right_gripper']]
            ]).astype(np.float32)
        elif self.control_mode == 'delta_ee':
            return np.concatenate([
                state_dict['left_ee_pos'],
                state_dict['left_ee_quat'],
                [state_dict['left_gripper']],
                state_dict['right_ee_pos'],
                state_dict['right_ee_quat'],
                [state_dict['right_gripper']]
            ]).astype(np.float32)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
    
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state bounds in normalized space."""
        return (
            np.full(self.MAX_STATE_DIM, -1.0, dtype=np.float32),
            np.full(self.MAX_STATE_DIM, 1.0, dtype=np.float32)
        )


def create_normalizer_from_env(env_task, control_mode: str) -> ActionNormalizer:
    """
    Create an ActionNormalizer from an environment task instance.
    
    Args:
        env_task: random_exploration task instance
        control_mode: Control mode string
        
    Returns:
        Configured ActionNormalizer
    """
    robot_info = env_task.robot_info
    return ActionNormalizer(
        control_mode=control_mode,
        left_arm_dof=robot_info['left_arm_dof'],
        right_arm_dof=robot_info['right_arm_dof'],
    )
