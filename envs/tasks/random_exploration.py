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
from ..utils.rand_create_cluttered_actor import get_all_cluttered_objects, rand_create_cluttered_actor
import sapien
import numpy as np
import glob
import os
import transforms3d as t3d
from typing import List, Tuple, Optional, Dict, Any

# Pre-load object info at module level for speed
_CLUTTERED_OBJECTS_INFO, _CLUTTERED_OBJECTS_LIST, _SAME_OBJ = None, None, None

# Verified stable objects that won't cause UnStableError
# These are flat-bottomed or have stable bases
_VERIFIED_STABLE_OBJECTS = [
    # From assets/objects - verified stable with "stable": true in json
    "002_bowl", "003_plate", "007_shoe-box", "008_tray",
    "021_cup", "023_tissue-box", "039_mug", "047_mouse",
    "048_stapler", "059_pencup", "062_plasticbox", "071_can",
    "073_rubikscube", "077_phone",
]

def _init_object_cache():
    """Initialize object cache once at module load."""
    global _CLUTTERED_OBJECTS_INFO, _CLUTTERED_OBJECTS_LIST, _SAME_OBJ
    if _CLUTTERED_OBJECTS_INFO is None:
        try:
            _CLUTTERED_OBJECTS_INFO, _CLUTTERED_OBJECTS_LIST, _SAME_OBJ = get_all_cluttered_objects()
            # Filter to only verified stable objects
            _CLUTTERED_OBJECTS_LIST = [obj for obj in _CLUTTERED_OBJECTS_LIST 
                                        if obj in _VERIFIED_STABLE_OBJECTS]
        except Exception as e:
            _CLUTTERED_OBJECTS_INFO, _CLUTTERED_OBJECTS_LIST, _SAME_OBJ = {}, [], {}

_init_object_cache()


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
        
        # Store embodiment config for robot info
        self._embodiment_config = kwargs.get("embodiment", None)
        self.left_robot_file = kwargs.get("left_robot_file", None)
        
        # Initialize base environment (will check stability and may raise UnStableError)
        super()._init_task_env_(**kwargs)
        
        # Simulation parameters
        self.frame_skip = kwargs.get("frame_skip", 10)  # Number of sim steps per action
        
        # State tracking
        self.executed_steps = 0
        self.recorded_data = []
        
        # Get robot info for cross-embodiment
        self._setup_robot_info()
        
        # Randomize initial state
        self._randomize_robot_initial_state()

    def reset_for_new_episode(self, seed: int = None):
        """
        Reset scene for a new episode WITHOUT reloading the robot.
        This is much faster than calling setup_demo again.
        
        Only resets:
        - Objects on the table
        - Robot position to home state
        - Recording data
        
        Does NOT reload:
        - Robot model
        - Cameras
        - Scene/renderer
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Remove existing placed objects
        for obj in self.placed_objects:
            try:
                self.scene.remove_actor(obj.entity if hasattr(obj, 'entity') else obj)
            except:
                pass
        self.placed_objects = []
        
        # Reset prohibited area and size_dict (keep robot area)
        self.prohibited_area = []
        self.size_dict = []
        
        # Add robot prohibited area back
        self._add_robot_prohibited_area()
        
        # Load new objects
        self.load_actors()
        
        # Check stability
        from envs.utils.create_actor import UnStableError
        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError(f'Objects unstable: {", ".join(unstable_list)}')
        
        # Reset robot to home state
        self.robot.move_to_homestate()
        
        # Open grippers
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=-1)
        self.render_freq = render_freq
        
        # Randomize initial robot position
        self._randomize_robot_initial_state()
        
        # Reset tracking
        self.executed_steps = 0
        self.recorded_data = []
        self.take_action_cnt = 0
        
        return True
    
    def _add_robot_prohibited_area(self):
        """Add robot base area to prohibited area."""
        # Robot base is roughly at center, add safety margin
        robot_radius = 0.3
        self.prohibited_area.append([-robot_radius, -robot_radius, robot_radius, robot_radius])

    def _setup_robot_info(self):
        """Extract robot information for cross-embodiment support."""
        # Get embodiment name from config (passed in setup_demo)
        embodiment_config = getattr(self, '_embodiment_config', None)
        if embodiment_config:
            embodiment_name = embodiment_config[0] if isinstance(embodiment_config, list) else str(embodiment_config)
        elif hasattr(self.robot, 'name'):
            embodiment_name = self.robot.name
        else:
            # Try to extract from left_robot_file path
            left_robot_file = getattr(self, 'left_robot_file', None)
            if left_robot_file:
                # Extract embodiment name from path like ".../assets/embodiments/franka-panda"
                embodiment_name = os.path.basename(left_robot_file.rstrip('/'))
            else:
                embodiment_name = "unknown"
        
        self.robot_info = {
            "embodiment": embodiment_name,
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
        """
        Load random objects onto the table using rand_create_cluttered_actor.
        This method follows the same approach as get_cluttered_table for stability.
        """
        self.placed_objects: List = []
        
        # Use cached object info
        global _CLUTTERED_OBJECTS_INFO, _CLUTTERED_OBJECTS_LIST
        
        # Get available objects with their info
        if _CLUTTERED_OBJECTS_INFO:
            available_objects = list(_CLUTTERED_OBJECTS_INFO.keys())
        else:
            available_objects = _VERIFIED_STABLE_OBJECTS.copy()
        
        if len(available_objects) == 0:
            return
        
        # Table bounds (same as get_cluttered_table)
        xlim = [-0.35, 0.35]
        ylim = [-0.25, 0.15]
        zlim = [0.76 + self.table_z_bias]
        
        # Initialize size_dict for collision tracking (same format as get_cluttered_table)
        size_dict = list(self.size_dict) if self.size_dict else []
        
        success_count = 0
        max_try = 30  # Limit attempts to avoid infinite loops
        trys = 0
        
        while success_count < self.num_objects and trys < max_try:
            trys += 1
            
            # Randomly select an object
            obj_name = np.random.choice(available_objects)
            obj_info = _CLUTTERED_OBJECTS_INFO.get(obj_name, {})
            
            # Get object parameters
            if "ids" in obj_info and "params" in obj_info:
                # Use cluttered object format
                obj_ids = obj_info["ids"]
                obj_idx = np.random.choice(obj_ids)
                obj_params = obj_info["params"].get(str(obj_idx), obj_info["params"].get(obj_idx, {}))
                obj_radius = obj_params.get("radius", 0.05)
                obj_offset = obj_params.get("z_offset", 0.0)
                obj_maxz = obj_params.get("z_max", 0.1)
                obj_type = obj_info.get("type", "glb")
            else:
                # Fallback for simple objects
                obj_idx = self._get_random_model_id(obj_name)
                obj_radius = obj_info.get("radius", 0.05)
                obj_offset = obj_info.get("z_offset", 0.0)
                obj_maxz = 0.1
                obj_type = "glb"
            
            # Use rand_create_cluttered_actor (same as get_cluttered_table)
            success, obj = rand_create_cluttered_actor(
                self.scene,
                xlim=xlim,
                ylim=ylim,
                zlim=np.array(zlim),
                modelname=obj_name,
                modelid=str(obj_idx) if obj_type == "urdf" else obj_idx,
                modeltype=obj_type,
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi],
                size_dict=size_dict,
                obj_radius=obj_radius,
                z_offset=obj_offset,
                z_max=obj_maxz,
                prohibited_area=self.prohibited_area,
            )
            
            if not success or obj is None:
                continue
            
            # Successfully placed object
            obj.set_name(f"{obj_name}")
            self.placed_objects.append(obj)
            
            # Update size_dict for next placement
            pose = obj.get_pose().p.tolist()
            pose.append(obj_radius)
            size_dict.append(pose)
            
            # Add to prohibited area directly using obj_radius (avoids actor.config issues)
            padding = 0.02
            x, y, z = pose[:3]
            self.prohibited_area.append([
                x - obj_radius - padding,
                y - obj_radius - padding,
                x + obj_radius + padding,
                y + obj_radius + padding,
            ])
            
            success_count += 1

    def _get_available_objects(self) -> List[str]:
        """Get list of available object types, using verified stable objects."""
        if self.object_types is not None:
            return self.object_types
        
        # First try to use cached cluttered objects (much faster)
        global _CLUTTERED_OBJECTS_LIST
        if _CLUTTERED_OBJECTS_LIST:
            return _CLUTTERED_OBJECTS_LIST.copy()
        
        # Return verified stable objects directly
        return _VERIFIED_STABLE_OBJECTS.copy()

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
        """
        Randomize robot initial joint positions while ensuring:
        1. End-effectors are above the table in reachable workspace
        2. Wrist cameras can see the table (EE pointing downward)
        3. Arms don't collide with each other
        """
        # Table surface height
        table_z = 0.74 + self.table_z_bias
        
        # Relaxed workspace bounds - will validate EE is above table and pointing down
        # Different robots have different workspace, so we use relative checks
        min_height_above_table = 0.10  # minimum 10cm above table
        max_height_above_table = 0.50  # maximum 50cm above table
        
        max_attempts = 30
        success = False
        
        for attempt in range(max_attempts):
            # Start from home position
            left_qpos = np.array(self.robot.left_homestate, dtype=np.float32).copy()
            right_qpos = np.array(self.robot.right_homestate, dtype=np.float32).copy()
            
            # Perturbation scale - start smaller, increase if failing
            base_scale = 0.15 + 0.15 * (attempt / max_attempts)
            
            # Perturb joints with decreasing scale towards wrist
            num_left = len(left_qpos)
            num_right = len(right_qpos)
            
            # Joint-specific scales: more perturbation on base joints, less on wrist
            def get_joint_scales(n):
                if n >= 7:  # 7 DOF (franka)
                    return np.array([1.0, 0.8, 0.8, 0.6, 0.5, 0.4, 0.3])
                else:  # 6 DOF
                    return np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.3])
            
            left_scales = get_joint_scales(num_left)[:num_left]
            right_scales = get_joint_scales(num_right)[:num_right]
            
            left_perturb = np.random.uniform(-1, 1, num_left) * base_scale * left_scales
            right_perturb = np.random.uniform(-1, 1, num_right) * base_scale * right_scales
            
            left_qpos += left_perturb
            right_qpos += right_perturb
            
            # Clip to joint limits
            limits = self.robot_info["arm_joint_limits"]
            left_qpos = np.clip(left_qpos, limits["left_lower"], limits["left_upper"])
            right_qpos = np.clip(right_qpos, limits["right_lower"], limits["right_upper"])
            
            # Apply joint positions
            for i, joint in enumerate(self.robot.left_arm_joints):
                joint.set_drive_target(float(left_qpos[i]))
            for i, joint in enumerate(self.robot.right_arm_joints):
                joint.set_drive_target(float(right_qpos[i]))
            
            # Step simulation to update poses
            for _ in range(30):
                self.scene.step()
            self.scene.update_render()
            
            # Check end-effector positions
            left_ee = self.get_arm_pose("left")
            right_ee = self.get_arm_pose("right")
            
            # Validate height above table
            left_height_ok = min_height_above_table <= (left_ee[2] - table_z) <= max_height_above_table
            right_height_ok = min_height_above_table <= (right_ee[2] - table_z) <= max_height_above_table
            
            # Check EEs are within reasonable horizontal range (not too far from table center)
            left_horiz_ok = abs(left_ee[0]) < 0.6 and abs(left_ee[1]) < 0.5
            right_horiz_ok = abs(right_ee[0]) < 0.6 and abs(right_ee[1]) < 0.5
            
            # Check arms don't collide (minimum distance between EEs)
            ee_distance = np.linalg.norm(np.array(left_ee[:3]) - np.array(right_ee[:3]))
            arms_safe = ee_distance > 0.12
            
            # Check EE orientation - should be roughly pointing down for wrist camera to see table
            left_z_down = self._check_ee_pointing_down("left")
            right_z_down = self._check_ee_pointing_down("right")
            
            all_valid = (left_height_ok and right_height_ok and 
                        left_horiz_ok and right_horiz_ok and 
                        arms_safe and left_z_down and right_z_down)
            
            if all_valid:
                success = True
                break
        
        if not success:
            # Fallback to home position with very small perturbation
            left_qpos = np.array(self.robot.left_homestate, dtype=np.float32)
            right_qpos = np.array(self.robot.right_homestate, dtype=np.float32)
            
            # Very small safe perturbation on first few joints only
            n_perturb = min(3, len(left_qpos))
            left_qpos[:n_perturb] += np.random.uniform(-0.08, 0.08, n_perturb)
            right_qpos[:n_perturb] += np.random.uniform(-0.08, 0.08, n_perturb)
            
            for i, joint in enumerate(self.robot.left_arm_joints):
                joint.set_drive_target(float(left_qpos[i]))
            for i, joint in enumerate(self.robot.right_arm_joints):
                joint.set_drive_target(float(right_qpos[i]))
            
            for _ in range(30):
                self.scene.step()
            self.scene.update_render()
    
    def _check_ee_pointing_down(self, arm: str) -> bool:
        """
        Check if end-effector is roughly pointing downward (for wrist camera to see table).
        
        Returns True if the EE z-axis has a significant downward component.
        """
        ee_pose = self.get_arm_pose(arm)  # [x, y, z, qw, qx, qy, qz]
        quat = ee_pose[3:7]  # quaternion
        
        # Convert quaternion to rotation matrix and check z-axis direction
        # The EE z-axis in world frame should have negative z component (pointing down)
        import transforms3d as t3d
        
        try:
            # quat format: [w, x, y, z]
            rot_matrix = t3d.quaternions.quat2mat(quat)
            # z-axis of EE in world frame
            ee_z_axis = rot_matrix[:, 2]
            
            # Check if EE z-axis points downward (negative world z)
            # Allow some tolerance - z component should be < -0.3 (roughly pointing down)
            return ee_z_axis[2] < -0.3
        except:
            # If conversion fails, assume it's okay
            return True

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
        # Single line header with all info
        header = f"[{self.robot_info['embodiment']}|{self.control_mode}] "
        
        for step in range(self.num_random_steps):
            # Progress bar
            progress = (step + 1) / self.num_random_steps
            bar_len = 20
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r{header}[{bar}] {step+1}/{self.num_random_steps}", end='', flush=True)
            
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
                print(f"\n  Warning: Action execution failed at step {step}: {e}")
            
            # 5. Get state AFTER action
            state_after = self._get_current_state()
            
            # 6. Record transition (s_t, a_t, s_{t+1})
            self._record_transition(state_before, action_vector, action_dict, state_after, obs_before)
            
            self.executed_steps += 1
        
        # Complete the progress bar
        bar = '█' * bar_len
        print(f"\r{header}[{bar}] {self.num_random_steps}/{self.num_random_steps} ✓")
        
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
        
        # Images - extract from observation structure
        # The observation structure is: {"observation": {camera_name: {"rgb": array, ...}, ...}}
        images = {}
        sample_obs = self.recorded_data[0]["observation"]
        
        # Check if it's the nested format (from get_obs)
        if "observation" in sample_obs:
            # Nested format: {"observation": {camera_name: {"rgb": ...}}}
            for camera_name, camera_data in sample_obs["observation"].items():
                if isinstance(camera_data, dict) and "rgb" in camera_data:
                    key = f"{camera_name}_rgb"
                    try:
                        images[key] = np.stack([
                            d["observation"]["observation"][camera_name]["rgb"] 
                            for d in self.recorded_data
                        ])
                    except Exception as e:
                        print(f"Warning: Could not stack images for {key}: {e}")
        else:
            # Flat format: look for rgb keys directly
            for key in sample_obs:
                if "rgb" in key.lower() or "head" in key.lower() or "wrist" in key.lower():
                    try:
                        if isinstance(sample_obs[key], np.ndarray):
                            images[key] = np.stack([d["observation"][key] for d in self.recorded_data])
                    except Exception as e:
                        print(f"Warning: Could not stack images for {key}: {e}")
        
        return {
            "states_t": states_t,
            "actions": actions,
            "states_t1": states_t1,
            "actual_deltas": actual_deltas,
            "images": images,
            "control_mode": self.control_mode,
            "robot_info": self.robot_info,
        }
