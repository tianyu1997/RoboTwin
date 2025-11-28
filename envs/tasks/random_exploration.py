"""
Random Exploration Task for Visual-Action Jacobian Learning

Designed for training cross-embodiment, cross-scene visual-action Jacobian models.
Data format ensures: s(t+1) = s(t) + a(t), where a(t) is the delta/incremental action.

Features:
- Delta action format for both joint space and end-effector space
- Cross-robot embodiment support with unified interface
- Random scene generation (objects, textures, lighting)
- Precise state-action recording for Jacobian estimation
"""

from .._base_task import Base_Task
from ..utils import *
import sapien
import numpy as np
import glob
import os
import transforms3d as t3d
from typing import List, Tuple, Optional, Dict, Any


class random_exploration(Base_Task):
    """
    Random exploration task for collecting visual-action Jacobian training data.
    
    Key Design Principles:
    1. Action is always delta/incremental: a(t) = s(t+1) - s(t)
    2. Records: s(t), a(t), s(t+1), images, and computed delta
    3. Supports multiple control modes with consistent interface
    4. Cross-embodiment compatible with normalized state representation
    """

    # Supported control modes
    CONTROL_MODES = {
        'delta_qpos': 'Joint space incremental control',
        'delta_ee': 'End-effector space incremental control (position + orientation)',
        'delta_ee_pos': 'End-effector position only incremental control',
    }

    def setup_demo(self, **kwargs):
        """Initialize the task environment with randomization."""
        # Control mode: delta_qpos, delta_ee, delta_ee_pos
        self.control_mode = kwargs.get("control_mode", "delta_qpos")
        if self.control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Unknown control_mode: {self.control_mode}. "
                           f"Supported: {list(self.CONTROL_MODES.keys())}")
        
        # Exploration parameters
        self.num_random_steps = kwargs.get("num_random_steps", 100)
        self.num_objects = kwargs.get("num_objects", 5)
        self.object_types = kwargs.get("object_types", None)
        
        # Action generation parameters
        self.delta_qpos_scale = kwargs.get("delta_qpos_scale", 0.05)  # radians for joints
        self.delta_ee_pos_scale = kwargs.get("delta_ee_pos_scale", 0.02)  # meters
        self.delta_ee_rot_scale = kwargs.get("delta_ee_rot_scale", 0.05)  # radians
        self.gripper_action_prob = kwargs.get("gripper_action_prob", 0.1)  # probability of gripper action
        
        # Enable domain randomization
        if "domain_randomization" not in kwargs:
            kwargs["domain_randomization"] = {}
        kwargs["domain_randomization"]["random_background"] = kwargs["domain_randomization"].get("random_background", True)
        kwargs["domain_randomization"]["random_light"] = kwargs["domain_randomization"].get("random_light", True)
        
        # Initialize base environment (will check stability and may raise UnStableError)
        super()._init_task_env_(**kwargs)
        
        # State tracking
        self.executed_steps = 0
        self.recorded_data = []
        
        # Get robot info for cross-embodiment
        self._setup_robot_info()
        
        # Randomize initial state
        self._randomize_robot_initial_state()

    def _setup_robot_info(self):
        """Extract robot information for cross-embodiment support."""
        self.robot_info = {
            "embodiment": self.robot.name if hasattr(self.robot, 'name') else "unknown",
            "left_arm_dof": len(self.robot.left_arm_joints),
            "right_arm_dof": len(self.robot.right_arm_joints),
            "has_gripper": True,
            "arm_joint_limits": self._get_joint_limits(),
        }

    def _get_joint_limits(self) -> Dict[str, np.ndarray]:
        """Get joint limits for normalization."""
        left_lower, left_upper = [], []
        right_lower, right_upper = [], []
        
        for joint in self.robot.left_arm_joints:
            limits = joint.get_limit()
            left_lower.append(limits[0][0])
            left_upper.append(limits[0][1])
        
        for joint in self.robot.right_arm_joints:
            limits = joint.get_limit()
            right_lower.append(limits[0][0])
            right_upper.append(limits[0][1])
        
        return {
            "left_lower": np.array(left_lower),
            "left_upper": np.array(left_upper),
            "right_lower": np.array(right_lower),
            "right_upper": np.array(right_upper),
        }

    def load_actors(self):
        """Load random objects onto the table without collision."""
        self.placed_objects: List = []
        
        available_objects = self._get_available_objects()
        if len(available_objects) == 0:
            print("Warning: No available objects found, using default set")
            available_objects = ["071_can", "021_cup", "002_bowl", "003_plate", "010_pen"]
        
        num_to_place = min(self.num_objects, len(available_objects))
        if num_to_place > 0:
            selected_objects = np.random.choice(available_objects, size=num_to_place, replace=False)
        else:
            selected_objects = []
        
        # Table bounds
        x_range = [-0.35, 0.35]
        y_range = [-0.25, 0.15]
        z_base = 0.76 + self.table_z_bias
        
        placed_positions = []
        min_distance = 0.12
        
        for obj_name in selected_objects:
            max_attempts = 50
            for attempt in range(max_attempts):
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                
                if abs(x) < 0.1 and y < -0.15:
                    continue
                
                valid = True
                for px, py in placed_positions:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                        valid = False
                        break
                
                for area in self.prohibited_area:
                    if area[0] <= x <= area[2] and area[1] <= y <= area[3]:
                        valid = False
                        break
                
                if valid:
                    placed_positions.append((x, y))
                    break
            else:
                continue
            
            z_rotation = np.random.uniform(-np.pi, np.pi)
            quat = t3d.euler.euler2quat(0, 0, z_rotation)
            model_id = self._get_random_model_id(obj_name)
            
            try:
                obj_pose = sapien.Pose([x, y, z_base], quat)
                obj = create_actor(
                    scene=self,
                    pose=obj_pose,
                    modelname=obj_name,
                    convex=True,
                    model_id=model_id,
                )
                self.placed_objects.append(obj)
                self.add_prohibit_area(obj, padding=0.02)
            except Exception as e:
                print(f"Warning: Failed to create object {obj_name}: {e}")

        print(f"Placed {len(self.placed_objects)} objects on the table")

    def _get_available_objects(self) -> List[str]:
        """Get list of available object types, using only known stable objects."""
        if self.object_types is not None:
            return self.object_types
        
        # Use only verified stable objects with complete model data (scale, stable:true)
        # These objects have been verified to have proper collision meshes and loading data
        stable_objects = [
            "002_bowl",       # 7/7 stable
            "003_plate",      # 1/1 stable
            "004_fluted-block", # 2/2 stable
            "007_shoe-box",   # 1/1 stable
            "008_tray",       # 4/4 stable
            "019_coaster",    # 1/1 stable
            "021_cup",        # 13/13 stable
            "023_tissue-box", # 7/7 stable
            "039_mug",        # 13/13 stable
            "047_mouse",      # 3/3 stable
            "048_stapler",    # 7/7 stable
            "057_toycar",     # 6/6 stable
            "059_pencup",     # 7/7 stable
            "062_plasticbox", # 11/11 stable
            "071_can",        # 6/6 stable
            "073_rubikscube", # 3/3 stable
            "075_bread",      # 7/7 stable
            "077_phone",      # 5/5 stable
            "079_remotecontrol", # 7/7 stable
        ]
        
        objects_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "..", "assets", "objects")
        
        available = []
        try:
            for item in os.listdir(objects_dir):
                item_path = os.path.join(objects_dir, item)
                if os.path.isdir(item_path) and item in stable_objects:
                    # Check if this is in our stable list
                    json_files = glob.glob(os.path.join(item_path, "model_data*.json"))
                    if json_files:
                        available.append(item)
        except Exception as e:
            print(f"Error scanning objects directory: {e}")
        
        if not available:
            # Fallback if no stable objects found
            print("Warning: No stable objects found, using fallback list")
            available = ["002_bowl", "003_plate", "037_box"]
        
        print(f"Found {len(available)} stable objects for exploration")
        return available

    def _get_random_model_id(self, obj_name: str) -> int:
        """Get a random model ID for the given object."""
        objects_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "..", "assets", "objects", obj_name)
        try:
            json_files = glob.glob(os.path.join(objects_dir, "model_data*.json"))
            if not json_files:
                return 0
            model_ids = []
            for f in json_files:
                basename = os.path.basename(f)
                try:
                    id_str = basename.replace("model_data", "").replace(".json", "")
                    model_ids.append(int(id_str) if id_str else 0)
                except ValueError:
                    continue
            return np.random.choice(model_ids) if model_ids else 0
        except Exception:
            return 0

    def _randomize_robot_initial_state(self):
        """Randomize robot initial joint positions while keeping wrist cameras facing the table."""
        left_home = np.array(self.robot.left_homestate, dtype=np.float32)
        right_home = np.array(self.robot.right_homestate, dtype=np.float32)
        
        perturbation_scale = 0.15
        num_arm_joints = min(len(left_home), 6)
        
        left_perturb = np.random.uniform(-perturbation_scale, perturbation_scale, num_arm_joints)
        right_perturb = np.random.uniform(-perturbation_scale, perturbation_scale, num_arm_joints)
        
        left_home[:num_arm_joints] += left_perturb
        right_home[:num_arm_joints] += right_perturb
        
        for i, joint in enumerate(self.robot.left_arm_joints):
            if i < len(left_home):
                joint.set_drive_target(left_home[i])
        
        for i, joint in enumerate(self.robot.right_arm_joints):
            if i < len(right_home):
                joint.set_drive_target(right_home[i])
        
        for _ in range(50):
            self.scene.step()
        self.scene.update_render()

    # ==================== State Representation ====================
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get current robot state in a unified format.
        
        Returns a dict with:
        - qpos: joint positions for both arms
        - ee_pos: end-effector positions (x, y, z) for both arms
        - ee_quat: end-effector orientations (quaternion) for both arms
        - gripper: gripper states
        """
        # Joint positions - ensure 1D arrays
        left_qpos = np.array([j.get_drive_target() for j in self.robot.left_arm_joints], dtype=np.float32).flatten()
        right_qpos = np.array([j.get_drive_target() for j in self.robot.right_arm_joints], dtype=np.float32).flatten()
        
        # End-effector poses
        left_ee_pose = self.get_arm_pose("left")  # [x, y, z, qw, qx, qy, qz]
        right_ee_pose = self.get_arm_pose("right")
        
        # Gripper states
        left_gripper = self.robot.get_left_gripper_val()
        right_gripper = self.robot.get_right_gripper_val()
        
        return {
            # Joint space state
            "left_qpos": left_qpos,
            "right_qpos": right_qpos,
            # End-effector state
            "left_ee_pos": np.array(left_ee_pose[:3], dtype=np.float32).flatten(),
            "left_ee_quat": np.array(left_ee_pose[3:7], dtype=np.float32).flatten(),
            "right_ee_pos": np.array(right_ee_pose[:3], dtype=np.float32).flatten(),
            "right_ee_quat": np.array(right_ee_pose[3:7], dtype=np.float32).flatten(),
            # Gripper state
            "left_gripper": float(left_gripper),
            "right_gripper": float(right_gripper),
        }

    def _state_to_vector(self, state: Dict[str, Any], mode: str) -> np.ndarray:
        """
        Convert state dict to vector representation based on control mode.
        
        This provides a unified state vector format for different control modes.
        """
        if mode == 'delta_qpos':
            # State = [left_qpos, left_gripper, right_qpos, right_gripper]
            return np.concatenate([
                state["left_qpos"], [state["left_gripper"]],
                state["right_qpos"], [state["right_gripper"]]
            ]).astype(np.float32)
        
        elif mode == 'delta_ee':
            # State = [left_ee_pos, left_ee_quat, left_gripper, right_ee_pos, right_ee_quat, right_gripper]
            return np.concatenate([
                state["left_ee_pos"], state["left_ee_quat"], [state["left_gripper"]],
                state["right_ee_pos"], state["right_ee_quat"], [state["right_gripper"]]
            ]).astype(np.float32)
        
        elif mode == 'delta_ee_pos':
            # State = [left_ee_pos, left_gripper, right_ee_pos, right_gripper]
            return np.concatenate([
                state["left_ee_pos"], [state["left_gripper"]],
                state["right_ee_pos"], [state["right_gripper"]]
            ]).astype(np.float32)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ==================== Action Generation ====================

    def _generate_delta_action(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate random delta action based on control mode.
        
        Returns:
            action_vector: The action vector to be recorded
            action_dict: Detailed action components for execution
        """
        if self.control_mode == 'delta_qpos':
            return self._generate_delta_qpos_action()
        elif self.control_mode == 'delta_ee':
            return self._generate_delta_ee_action(include_rotation=True)
        elif self.control_mode == 'delta_ee_pos':
            return self._generate_delta_ee_action(include_rotation=False)
        else:
            raise ValueError(f"Unknown control_mode: {self.control_mode}")

    def _generate_delta_qpos_action(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate delta joint position action."""
        left_dof = len(self.robot.left_arm_joints)
        right_dof = len(self.robot.right_arm_joints)
        
        # Random delta joint positions
        left_delta = np.random.uniform(-self.delta_qpos_scale, self.delta_qpos_scale, left_dof).astype(np.float32)
        right_delta = np.random.uniform(-self.delta_qpos_scale, self.delta_qpos_scale, right_dof).astype(np.float32)
        
        # Gripper delta (sparse - only occasionally change)
        left_gripper_delta = 0.0
        right_gripper_delta = 0.0
        if np.random.random() < self.gripper_action_prob:
            left_gripper_delta = np.random.uniform(-0.5, 0.5)
        if np.random.random() < self.gripper_action_prob:
            right_gripper_delta = np.random.uniform(-0.5, 0.5)
        
        # Action vector: [left_delta_qpos, left_delta_gripper, right_delta_qpos, right_delta_gripper]
        action_vector = np.concatenate([
            left_delta, [left_gripper_delta],
            right_delta, [right_gripper_delta]
        ]).astype(np.float32)
        
        action_dict = {
            "left_delta_qpos": left_delta,
            "right_delta_qpos": right_delta,
            "left_delta_gripper": left_gripper_delta,
            "right_delta_gripper": right_gripper_delta,
        }
        
        return action_vector, action_dict

    def _generate_delta_ee_action(self, include_rotation: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate delta end-effector action."""
        # Position deltas
        left_delta_pos = np.random.uniform(-self.delta_ee_pos_scale, self.delta_ee_pos_scale, 3).astype(np.float32)
        right_delta_pos = np.random.uniform(-self.delta_ee_pos_scale, self.delta_ee_pos_scale, 3).astype(np.float32)
        
        # Rotation deltas (as euler angles, then convert to quaternion delta)
        if include_rotation:
            left_delta_euler = np.random.uniform(-self.delta_ee_rot_scale, self.delta_ee_rot_scale, 3)
            right_delta_euler = np.random.uniform(-self.delta_ee_rot_scale, self.delta_ee_rot_scale, 3)
            left_delta_quat = np.array(t3d.euler.euler2quat(*left_delta_euler), dtype=np.float32)
            right_delta_quat = np.array(t3d.euler.euler2quat(*right_delta_euler), dtype=np.float32)
        else:
            left_delta_quat = np.array([1, 0, 0, 0], dtype=np.float32)  # identity
            right_delta_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        
        # Gripper delta
        left_gripper_delta = 0.0
        right_gripper_delta = 0.0
        if np.random.random() < self.gripper_action_prob:
            left_gripper_delta = np.random.uniform(-0.5, 0.5)
        if np.random.random() < self.gripper_action_prob:
            right_gripper_delta = np.random.uniform(-0.5, 0.5)
        
        if include_rotation:
            action_vector = np.concatenate([
                left_delta_pos, left_delta_quat, [left_gripper_delta],
                right_delta_pos, right_delta_quat, [right_gripper_delta]
            ]).astype(np.float32)
        else:
            action_vector = np.concatenate([
                left_delta_pos, [left_gripper_delta],
                right_delta_pos, [right_gripper_delta]
            ]).astype(np.float32)
        
        action_dict = {
            "left_delta_pos": left_delta_pos,
            "right_delta_pos": right_delta_pos,
            "left_delta_quat": left_delta_quat,
            "right_delta_quat": right_delta_quat,
            "left_delta_gripper": left_gripper_delta,
            "right_delta_gripper": right_gripper_delta,
        }
        
        return action_vector, action_dict

    # ==================== Action Execution ====================

    def _execute_delta_action(self, action_dict: Dict[str, np.ndarray]):
        """
        Execute delta action by computing target state and applying it.
        
        For delta actions, target = current + delta
        """
        if self.control_mode == 'delta_qpos':
            self._execute_delta_qpos(action_dict)
        elif self.control_mode in ['delta_ee', 'delta_ee_pos']:
            self._execute_delta_ee(action_dict)

    def _execute_delta_qpos(self, action_dict: Dict[str, np.ndarray]):
        """Execute delta joint position action."""
        # Get current positions - ensure 1D
        left_current = np.array([j.get_drive_target() for j in self.robot.left_arm_joints]).flatten()
        right_current = np.array([j.get_drive_target() for j in self.robot.right_arm_joints]).flatten()
        
        # Ensure action deltas are 1D
        left_delta = np.asarray(action_dict["left_delta_qpos"]).flatten()
        right_delta = np.asarray(action_dict["right_delta_qpos"]).flatten()
        
        # Compute targets: target = current + delta
        left_target = left_current + left_delta
        right_target = right_current + right_delta
        
        # Clip to joint limits
        limits = self.robot_info["arm_joint_limits"]
        left_target = np.clip(left_target, limits["left_lower"], limits["left_upper"])
        right_target = np.clip(right_target, limits["right_lower"], limits["right_upper"])
        
        # Apply joint targets
        for i, joint in enumerate(self.robot.left_arm_joints):
            joint.set_drive_target(float(left_target[i]))
        for i, joint in enumerate(self.robot.right_arm_joints):
            joint.set_drive_target(float(right_target[i]))
        
        # Gripper
        left_gripper_current = float(self.robot.get_left_gripper_val())
        right_gripper_current = float(self.robot.get_right_gripper_val())
        left_gripper_target = np.clip(left_gripper_current + float(action_dict["left_delta_gripper"]), 0, 1)
        right_gripper_target = np.clip(right_gripper_current + float(action_dict["right_delta_gripper"]), 0, 1)
        
        self.robot.set_gripper(float(left_gripper_target), "left")
        self.robot.set_gripper(float(right_gripper_target), "right")
        
        # Step simulation
        for _ in range(self.frame_skip):
            self.scene.step()
        self.scene.update_render()

    def _execute_delta_ee(self, action_dict: Dict[str, np.ndarray]):
        """Execute delta end-effector action using IK."""
        # Get current EE poses
        left_ee_pose = self.get_arm_pose("left")
        right_ee_pose = self.get_arm_pose("right")
        
        # Compute target positions
        left_target_pos = np.array(left_ee_pose[:3]) + action_dict["left_delta_pos"]
        right_target_pos = np.array(right_ee_pose[:3]) + action_dict["right_delta_pos"]
        
        # Compute target orientations (quaternion multiplication for rotation composition)
        left_current_quat = np.array(left_ee_pose[3:7])
        right_current_quat = np.array(right_ee_pose[3:7])
        
        # Apply rotation delta: q_new = q_delta * q_current
        left_target_quat = t3d.quaternions.qmult(action_dict["left_delta_quat"], left_current_quat)
        right_target_quat = t3d.quaternions.qmult(action_dict["right_delta_quat"], right_current_quat)
        
        # Normalize quaternions
        left_target_quat = left_target_quat / np.linalg.norm(left_target_quat)
        right_target_quat = right_target_quat / np.linalg.norm(right_target_quat)
        
        # Gripper
        left_gripper_current = float(self.robot.get_left_gripper_val())
        right_gripper_current = float(self.robot.get_right_gripper_val())
        left_gripper_target = float(np.clip(left_gripper_current + float(action_dict["left_delta_gripper"]), 0, 1))
        right_gripper_target = float(np.clip(right_gripper_current + float(action_dict["right_delta_gripper"]), 0, 1))
        
        # Build action for take_action with ee mode
        action = np.concatenate([
            left_target_pos, left_target_quat, [left_gripper_target],
            right_target_pos, right_target_quat, [right_gripper_target]
        ])
        
        try:
            self.take_action(action, action_type='ee')
        except Exception as e:
            # If IK fails, just step the simulation
            print(f"Warning: EE action failed: {e}")
            for _ in range(self.frame_skip):
                self.scene.step()
            self.scene.update_render()

    # ==================== Data Recording ====================

    def _record_transition(self, 
                          state_before: Dict[str, Any],
                          action_vector: np.ndarray,
                          action_dict: Dict[str, np.ndarray],
                          state_after: Dict[str, Any],
                          obs_before: Dict[str, Any]):
        """
        Record a complete transition for Jacobian learning.
        
        Records: (s_t, a_t, s_{t+1}) where a_t = s_{t+1} - s_t (approximately)
        """
        # Convert states to vectors for the current control mode
        s_t = self._state_to_vector(state_before, self.control_mode)
        s_t1 = self._state_to_vector(state_after, self.control_mode)
        
        # Compute actual delta (what actually happened)
        actual_delta = s_t1 - s_t
        
        transition = {
            "step": self.executed_steps,
            "control_mode": self.control_mode,
            
            # State vectors (for s(t+1) = s(t) + a(t) verification)
            "state_t": s_t,
            "action": action_vector,  # commanded action (delta)
            "state_t1": s_t1,
            "actual_delta": actual_delta,  # what actually happened
            
            # Full state dicts (for flexibility)
            "state_before": state_before,
            "state_after": state_after,
            
            # Action components
            "action_dict": action_dict,
            
            # Visual observations (images)
            "observation": obs_before,
            
            # Robot info for cross-embodiment
            "robot_info": self.robot_info,
        }
        
        self.recorded_data.append(transition)

    # ==================== Main Loop ====================

    def play_once(self):
        """Execute random exploration by performing random delta actions."""
        print(f"Starting random exploration:")
        print(f"  Control mode: {self.control_mode}")
        print(f"  Steps: {self.num_random_steps}")
        print(f"  Robot: {self.robot_info['embodiment']}")
        
        for step in range(self.num_random_steps):
            # 1. Get current state BEFORE action
            state_before = self._get_current_state()
            
            # 2. Get visual observation BEFORE action
            obs_before = self.get_obs()
            
            # 3. Generate random delta action
            action_vector, action_dict = self._generate_delta_action()
            
            # 4. Execute action
            try:
                self._execute_delta_action(action_dict)
            except Exception as e:
                print(f"Warning: Action execution failed at step {step}: {e}")
            
            # 5. Get state AFTER action
            state_after = self._get_current_state()
            
            # 6. Record transition (s_t, a_t, s_{t+1})
            self._record_transition(state_before, action_vector, action_dict, state_after, obs_before)
            
            self.executed_steps += 1
            
            if step % 20 == 0:
                print(f"  Step {step}/{self.num_random_steps}")
        
        print(f"Random exploration completed. Recorded {len(self.recorded_data)} transitions.")
        
        self.info["info"] = {
            "num_objects": len(self.placed_objects),
            "num_steps": self.executed_steps,
            "control_mode": self.control_mode,
            "robot_info": self.robot_info,
        }
        
        return self.info

    def check_success(self) -> bool:
        """Success is determined by completing the specified number of random steps."""
        return self.executed_steps >= self.num_random_steps

    def get_recorded_data(self) -> List[Dict[str, Any]]:
        """Return the recorded exploration data for Jacobian learning."""
        return self.recorded_data

    def get_jacobian_training_data(self) -> Dict[str, np.ndarray]:
        """
        Format recorded data specifically for Jacobian model training.
        
        Returns arrays suitable for training: s(t+1) = s(t) + J(s_t, I_t) @ a(t)
        where J is the visual-action Jacobian to be learned.
        """
        if not self.recorded_data:
            return {}
        
        states_t = np.stack([d["state_t"] for d in self.recorded_data])
        actions = np.stack([d["action"] for d in self.recorded_data])
        states_t1 = np.stack([d["state_t1"] for d in self.recorded_data])
        actual_deltas = np.stack([d["actual_delta"] for d in self.recorded_data])
        
        # Images - extract relevant camera views
        images = {}
        sample_obs = self.recorded_data[0]["observation"]
        for key in sample_obs:
            if "rgb" in key or "head" in key or "wrist" in key:
                try:
                    images[key] = np.stack([d["observation"][key] for d in self.recorded_data])
                except:
                    pass
        
        return {
            "states_t": states_t,
            "actions": actions,
            "states_t1": states_t1,
            "actual_deltas": actual_deltas,
            "images": images,
            "control_mode": self.control_mode,
            "robot_info": self.robot_info,
        }
