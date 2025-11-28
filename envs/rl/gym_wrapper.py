"""
RoboTwin Gym Environment Wrapper

This module provides a standard Gymnasium-compatible wrapper for RoboTwin tasks,
enabling reinforcement learning training with observation and action spaces.
"""

import os
import sys
import importlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List, Literal
import yaml
from copy import deepcopy

# Ensure path is set correctly
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
envs_directory = os.path.dirname(parent_directory)
root_directory = os.path.dirname(envs_directory)
if root_directory not in sys.path:
    sys.path.insert(0, root_directory)

from envs._GLOBAL_CONFIGS import *


class RoboTwinGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for RoboTwin manipulation tasks.
    
    This wrapper provides:
    - Standard observation_space and action_space definitions
    - reset() and step() methods compatible with Gym API
    - Support for different action types (qpos, ee, delta_ee)
    - Configurable observation modalities (rgb, depth, pointcloud, etc.)
    
    Args:
        task_name: Name of the task (e.g., 'beat_block_hammer', 'place_bread_basket')
        task_config: Configuration file name (default: 'demo_randomized')
        action_type: Type of action space ('qpos', 'ee', 'delta_ee')
        obs_keys: List of observation keys to include
        max_episode_steps: Maximum steps per episode
        reward_type: Reward function type ('sparse', 'dense')
        render_mode: Rendering mode ('human', 'rgb_array', None)
        seed: Random seed for reproducibility
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        task_name: str = "beat_block_hammer",
        task_config: str = "demo_randomized",
        action_type: Literal['qpos', 'ee', 'delta_ee'] = 'qpos',
        obs_keys: List[str] = None,
        max_episode_steps: int = 300,
        reward_type: Literal['sparse', 'dense'] = 'sparse',
        render_mode: Optional[str] = None,
        seed: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.task_name = task_name
        self.task_config = task_config
        self.action_type = action_type
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.render_mode = render_mode
        self._seed = seed
        self._episode_count = 0
        self._step_count = 0
        
        # Default observation keys
        if obs_keys is None:
            obs_keys = ['rgb', 'qpos', 'endpose']
        self.obs_keys = obs_keys
        
        # Load task configuration
        self._load_config(task_config)
        self._kwargs = kwargs
        
        # Create underlying task environment
        self._task_env = self._create_task_env()
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Initialize environment
        self._initialized = False
        
    def _load_config(self, task_config: str):
        """Load task configuration from YAML file."""
        config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Update config for RL training
        self.config['task_name'] = self.task_name
        self.config['render_freq'] = 10 if self.render_mode == 'human' else 0
        self.config['save_data'] = False
        self.config['eval_mode'] = True
        self.config['need_plan'] = False
        
        # Load embodiment configuration
        self._load_embodiment_config()
        
    def _load_embodiment_config(self):
        """Load robot embodiment configuration."""
        embodiment_type = self.config.get("embodiment", ["franka-panda", "franka-panda", 0.6])
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise FileNotFoundError("Missing embodiment files")
            return robot_file
        
        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)
        
        if len(embodiment_type) == 1:
            self.config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.config["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.config["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            self.config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.config["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            self.config["embodiment_dis"] = embodiment_type[2]
            self.config["dual_arm_embodied"] = False
        else:
            raise ValueError("Number of embodiment config parameters should be 1 or 3")
        
        self.config["left_embodiment_config"] = get_embodiment_config(self.config["left_robot_file"])
        self.config["right_embodiment_config"] = get_embodiment_config(self.config["right_robot_file"])
        
        # Get arm dimensions from embodiment config
        left_config = self.config["left_embodiment_config"]
        right_config = self.config["right_embodiment_config"]
        self._left_arm_dim = len(left_config["arm_joints_name"][0])
        self._right_arm_dim = len(right_config["arm_joints_name"][1])
        
    def _create_task_env(self):
        """Create the underlying task environment."""
        envs_module = importlib.import_module(f"envs.tasks.{self.task_name}")
        try:
            env_class = getattr(envs_module, self.task_name)
            return env_class()
        except AttributeError:
            raise ValueError(f"Task '{self.task_name}' not found in envs.tasks module")
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        # Action space depends on action_type
        if self.action_type == 'qpos':
            # Joint positions for both arms + grippers
            # left_arm (7) + left_gripper (1) + right_arm (7) + right_gripper (1) = 16
            action_dim = self._left_arm_dim + 1 + self._right_arm_dim + 1
            self.action_space = spaces.Box(
                low=-np.pi, high=np.pi, shape=(action_dim,), dtype=np.float32
            )
        elif self.action_type in ['ee', 'delta_ee']:
            # End-effector pose: position (3) + quaternion (4) + gripper (1) for each arm
            action_dim = (3 + 4 + 1) * 2  # 16
            if self.action_type == 'delta_ee':
                self.action_space = spaces.Box(
                    low=-0.1, high=0.1, shape=(action_dim,), dtype=np.float32
                )
            else:
                self.action_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
                )
        
        # Observation space - define based on obs_keys
        obs_dict = {}
        
        # Image observations
        img_height, img_width = 480, 640  # Default camera resolution
        
        if 'rgb' in self.obs_keys:
            obs_dict['head_camera_rgb'] = spaces.Box(
                low=0, high=255, shape=(img_height, img_width, 3), dtype=np.uint8
            )
            obs_dict['left_wrist_camera_rgb'] = spaces.Box(
                low=0, high=255, shape=(img_height, img_width, 3), dtype=np.uint8
            )
            obs_dict['right_wrist_camera_rgb'] = spaces.Box(
                low=0, high=255, shape=(img_height, img_width, 3), dtype=np.uint8
            )
        
        if 'depth' in self.obs_keys:
            obs_dict['head_camera_depth'] = spaces.Box(
                low=0, high=10, shape=(img_height, img_width), dtype=np.float32
            )
        
        if 'pointcloud' in self.obs_keys:
            pcd_num = self.config.get('pcd_down_sample_num', 1024)
            obs_dict['pointcloud'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(pcd_num, 3), dtype=np.float32
            )
            obs_dict['pointcloud_color'] = spaces.Box(
                low=0, high=1, shape=(pcd_num, 3), dtype=np.float32
            )
        
        if 'qpos' in self.obs_keys:
            # Joint positions: left_arm + left_gripper + right_arm + right_gripper
            qpos_dim = self._left_arm_dim + 1 + self._right_arm_dim + 1
            obs_dict['qpos'] = spaces.Box(
                low=-np.pi, high=np.pi, shape=(qpos_dim,), dtype=np.float32
            )
        
        if 'endpose' in self.obs_keys:
            # End-effector pose: position (3) + quaternion (4) + gripper (1) for each arm
            obs_dict['left_endpose'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
            obs_dict['right_endpose'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
            obs_dict['left_gripper'] = spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
            obs_dict['right_gripper'] = spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        
        self.observation_space = spaces.Dict(obs_dict)
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation from the environment."""
        # Get raw observation from task environment
        raw_obs = self._task_env.get_obs()
        
        obs = {}
        
        # Process RGB observations
        if 'rgb' in self.obs_keys:
            obs_data = raw_obs.get('observation', {})
            if 'head_camera' in obs_data and 'rgb' in obs_data['head_camera']:
                obs['head_camera_rgb'] = obs_data['head_camera']['rgb'].astype(np.uint8)
            if 'left_wrist_camera' in obs_data and 'rgb' in obs_data['left_wrist_camera']:
                obs['left_wrist_camera_rgb'] = obs_data['left_wrist_camera']['rgb'].astype(np.uint8)
            if 'right_wrist_camera' in obs_data and 'rgb' in obs_data['right_wrist_camera']:
                obs['right_wrist_camera_rgb'] = obs_data['right_wrist_camera']['rgb'].astype(np.uint8)
        
        # Process depth observations
        if 'depth' in self.obs_keys:
            obs_data = raw_obs.get('observation', {})
            if 'head_camera' in obs_data and 'depth' in obs_data['head_camera']:
                obs['head_camera_depth'] = obs_data['head_camera']['depth'].astype(np.float32)
        
        # Process pointcloud observations
        if 'pointcloud' in self.obs_keys:
            pcd_data = raw_obs.get('pointcloud', [])
            if len(pcd_data) > 0:
                obs['pointcloud'] = pcd_data[0].astype(np.float32) if isinstance(pcd_data[0], np.ndarray) else np.array(pcd_data[0], dtype=np.float32)
                if len(pcd_data) > 1:
                    obs['pointcloud_color'] = pcd_data[1].astype(np.float32) if isinstance(pcd_data[1], np.ndarray) else np.array(pcd_data[1], dtype=np.float32)
        
        # Process joint position observations
        if 'qpos' in self.obs_keys:
            joint_data = raw_obs.get('joint_action', {})
            if 'vector' in joint_data:
                obs['qpos'] = joint_data['vector'].astype(np.float32)
        
        # Process end-effector observations
        if 'endpose' in self.obs_keys:
            endpose_data = raw_obs.get('endpose', {})
            if 'left_endpose' in endpose_data:
                obs['left_endpose'] = np.array(endpose_data['left_endpose'], dtype=np.float32)
            if 'right_endpose' in endpose_data:
                obs['right_endpose'] = np.array(endpose_data['right_endpose'], dtype=np.float32)
            if 'left_gripper' in endpose_data:
                obs['left_gripper'] = np.array([endpose_data['left_gripper']], dtype=np.float32)
            if 'right_gripper' in endpose_data:
                obs['right_gripper'] = np.array([endpose_data['right_gripper']], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        success = self._task_env.check_success()
        
        if self.reward_type == 'sparse':
            return 1.0 if success else 0.0
        else:
            # Dense reward can be task-specific
            # Basic dense reward: distance-based + success bonus
            reward = 0.0
            if success:
                reward += 10.0
            # Add task-specific dense rewards here
            return reward
    
    def _check_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)."""
        return self._step_count >= self.max_episode_steps
    
    def _check_terminated(self) -> bool:
        """Check if episode has terminated (success)."""
        return self._task_env.check_success()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for this episode
            options: Additional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._seed = seed
        
        # Close previous environment if exists
        if self._initialized:
            try:
                self._task_env.close_env(clear_cache=True)
                if hasattr(self._task_env, 'viewer') and self._task_env.viewer:
                    self._task_env.viewer.close()
            except:
                pass
            # Recreate task environment
            self._task_env = self._create_task_env()
        
        # Setup the task environment
        self.config['seed'] = self._seed + self._episode_count
        self.config['now_ep_num'] = self._episode_count
        
        self._task_env.setup_demo(**self.config)
        self._task_env.step_lim = self.max_episode_steps
        
        self._initialized = True
        self._step_count = 0
        self._episode_count += 1
        
        obs = self._get_obs()
        info = {
            'episode': self._episode_count,
            'seed': self._seed + self._episode_count - 1,
            'task_name': self.task_name,
        }
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: New observation after action
            reward: Reward received
            terminated: Whether episode has terminated (success)
            truncated: Whether episode was truncated (time limit)
            info: Additional information
        """
        self._step_count += 1
        
        # Execute action
        self._task_env.take_action(action, action_type=self.action_type)
        
        # Get new observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        info = {
            'step': self._step_count,
            'success': terminated,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            self._task_env._update_render()
            obs = self._task_env.get_obs()
            if 'observation' in obs and 'head_camera' in obs['observation']:
                return obs['observation']['head_camera'].get('rgb', None)
        elif self.render_mode == 'human':
            if hasattr(self._task_env, 'viewer') and self._task_env.viewer:
                self._task_env.viewer.render()
        return None
    
    def close(self):
        """Clean up the environment."""
        if self._initialized:
            try:
                self._task_env.close_env(clear_cache=True)
                if hasattr(self._task_env, 'viewer') and self._task_env.viewer:
                    self._task_env.viewer.close()
            except:
                pass
    
    def get_task_instruction(self) -> Optional[str]:
        """Get the natural language instruction for the current task."""
        return self._task_env.get_instruction()
    
    def set_task_instruction(self, instruction: str):
        """Set the natural language instruction for the task."""
        self._task_env.set_instruction(instruction)


def make_robotwin_env(
    task_name: str,
    task_config: str = "demo_randomized",
    **kwargs
) -> RoboTwinGymEnv:
    """
    Factory function to create RoboTwin Gym environment.
    
    Args:
        task_name: Name of the task
        task_config: Configuration file name
        **kwargs: Additional environment arguments
        
    Returns:
        RoboTwinGymEnv instance
    """
    return RoboTwinGymEnv(
        task_name=task_name,
        task_config=task_config,
        **kwargs
    )


# Register environments with Gymnasium
def register_robotwin_envs():
    """Register all RoboTwin tasks as Gymnasium environments."""
    # List of available tasks
    tasks = [
        "beat_block_hammer",
        "place_bread_basket",
        "stack_blocks_two",
        "stack_blocks_three",
        "pick_dual_bottles",
        "place_can_basket",
        "handover_block",
        "lift_pot",
        "open_laptop",
        "press_stapler",
        # Add more tasks as needed
    ]
    
    for task in tasks:
        env_id = f"RoboTwin-{task}-v0"
        try:
            gym.register(
                id=env_id,
                entry_point="envs.gym_wrapper:RoboTwinGymEnv",
                kwargs={"task_name": task},
                max_episode_steps=300,
            )
        except gym.error.Error:
            # Already registered
            pass


# Auto-register when module is imported
try:
    register_robotwin_envs()
except:
    pass
