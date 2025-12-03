#!/usr/bin/env python3
"""
Random Exploration Data Collection for Visual-Action Jacobian Learning

Fast and efficient collection with:
- Random embodiment switching
- Random control mode switching  
- No pre-search for stable seeds (retry on-the-fly)
- Images + states + actions saved together

Data Format:
    Each transition contains:
    - state_t: robot state at time t
    - action: delta action a(t)
    - state_t1: robot state at time t+1
    - images: visual observations at time t (head_rgb, left_wrist_rgb, right_wrist_rgb)
    - robot_info: embodiment information

Usage:
    # Collect with random embodiment and control mode
    python collect_random_exploration.py --random_embodiment --random_control_mode
    
    # Fast collection with no objects (for testing)
    python collect_random_exploration.py --num_objects 0 --num_episodes 10
"""

import sys
import os
import warnings
import logging
import traceback

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress CuRobo JIT compilation warnings
logging.getLogger("curobo").setLevel(logging.ERROR)
logging.getLogger("torch.utils.cpp_extension").setLevel(logging.ERROR)

# Suppress some warnings before importing torch
warnings.filterwarnings('ignore', message='.*TORCH_CUDA_ARCH_LIST.*')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.cpp_extension')

# Add RoboTwin root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
robotwin_dir = os.path.dirname(script_dir)
sys.path.insert(0, robotwin_dir)

import argparse
import yaml
import json
import time
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Visual-Action Jacobian Training Data")
    
    # Basic settings
    parser.add_argument("--config", type=str, default="random_exploration")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--steps_per_episode", type=int, default=50)
    parser.add_argument("--num_objects", type=int, default=3)
    
    # Control mode settings
    parser.add_argument("--control_mode", type=str, default="delta_qpos",
                       choices=["delta_qpos", "delta_ee", "delta_ee_pos"])
    parser.add_argument("--random_control_mode", action="store_true",
                       help="Randomly select control mode each episode")
    parser.add_argument("--delta_qpos_scale", type=float, default=0.05)
    parser.add_argument("--delta_ee_pos_scale", type=float, default=0.02)
    parser.add_argument("--delta_ee_rot_scale", type=float, default=0.05)
    parser.add_argument("--gripper_action_prob", type=float, default=0.1)
    
    # Embodiment settings
    parser.add_argument("--random_embodiment", action="store_true",
                       help="Randomly select robot embodiment each episode")
    parser.add_argument("--embodiment", type=str, default="franka-panda",
                       choices=["franka-panda", "aloha-agilex", "ARX-X5", "piper", "ur5-wsg"],
                       help="Fixed embodiment type (if not random)")
    parser.add_argument("--embodiment_list", type=str, nargs="+", default=None,
                       help="List of embodiments to randomly sample from (overrides --embodiment)")
    parser.add_argument("--uniform_embodiment", action="store_true",
                       help="Use uniform sampling instead of weighted for random embodiment")
    parser.add_argument("--episodes_per_robot", type=int, default=5,
                       help="Number of episodes to collect per robot before switching (reduces robot loading overhead)")
    
    # Retry settings
    parser.add_argument("--max_retries", type=int, default=10,
                       help="Max retries per episode on error")
    
    # Output
    parser.add_argument("--save_path", type=str, default="./data/jacobian_data")
    parser.add_argument("--render", action="store_true")
    
    # Rendering optimization options
    parser.add_argument("--render_mode", type=str, default="rt",
                       choices=["rt", "rasterize"],
                       help="Render mode: 'rt' for ray-tracing (quality), 'rasterize' for speed")
    parser.add_argument("--rt_samples", type=int, default=32,
                       help="Ray tracing samples per pixel (lower=faster but noisier)")
    parser.add_argument("--rt_denoiser", type=str, default="oidn",
                       choices=["oidn", "none"],
                       help="Denoiser: 'oidn' for quality, 'none' for speed")
    
    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging (show setup details)")
    
    return parser.parse_args()


# Available embodiments with their arm distance configurations
# Format: [left_robot, right_robot, arm_distance]
EMBODIMENTS = [
    ["franka-panda", "franka-panda", 0.6],      # 7 DOF, Franka Emika Panda
    ["aloha-agilex", "aloha-agilex", 0.5],      # 6 DOF, ALOHA AgileX
    ["ARX-X5", "ARX-X5", 0.5],                  # 6 DOF, ARX X5
    ["piper", "piper", 0.5],                    # 6 DOF, Piper
    ["ur5-wsg", "ur5-wsg", 0.6],                # 6 DOF, UR5 with WSG gripper
]

# Embodiments that support EE (end-effector) control via IK
# Others only support delta_qpos mode
EMBODIMENTS_WITH_EE_SUPPORT = {"franka-panda", "ARX-X5", "piper", "ur5-wsg"}

# Embodiment weights (can adjust based on data balance needs)
EMBODIMENT_WEIGHTS = [0.25, 0.15, 0.2, 0.2, 0.2]  # Slightly less aloha since it's qpos-only

CONTROL_MODES = ["delta_qpos", "delta_ee", "delta_ee_pos"]
CONTROL_MODES_QPOS_ONLY = ["delta_qpos"]  # For embodiments without IK support


def get_random_embodiment(weighted: bool = True):
    """Randomly select a robot embodiment.
    
    Args:
        weighted: If True, use weighted sampling; otherwise uniform.
    """
    if weighted:
        idx = np.random.choice(len(EMBODIMENTS), p=EMBODIMENT_WEIGHTS)
    else:
        idx = np.random.randint(len(EMBODIMENTS))
    return EMBODIMENTS[idx]


def get_random_control_mode(embodiment_name: str = None):
    """Randomly select a control mode compatible with the embodiment.
    
    Args:
        embodiment_name: If provided, select mode compatible with this embodiment.
    """
    if embodiment_name and embodiment_name not in EMBODIMENTS_WITH_EE_SUPPORT:
        # This embodiment only supports qpos control
        return np.random.choice(CONTROL_MODES_QPOS_ONLY)
    else:
        return np.random.choice(CONTROL_MODES)


def load_config(config_name: str, args) -> dict:
    """Load and update configuration."""
    config_path = os.path.join(robotwin_dir, "task_config", f"{config_name}.yml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # Override with command line arguments
    config["episode_num"] = args.num_episodes
    config["num_random_steps"] = args.steps_per_episode
    config["num_objects"] = args.num_objects
    config["delta_qpos_scale"] = args.delta_qpos_scale
    config["delta_ee_pos_scale"] = args.delta_ee_pos_scale
    config["delta_ee_rot_scale"] = args.delta_ee_rot_scale
    config["gripper_action_prob"] = args.gripper_action_prob
    config["save_path"] = args.save_path
    config["render_freq"] = 10 if args.render else 0
    
    # Rendering optimization
    config["render_mode"] = args.render_mode
    config["rt_samples"] = args.rt_samples
    config["rt_denoiser"] = args.rt_denoiser
    
    return config


def setup_embodiment(config: dict) -> dict:
    """Setup embodiment configuration."""
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH
    
    embodiment_type = config.get("embodiment", ["franka-panda", "franka-panda", 0.6])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise FileNotFoundError(f"Missing embodiment files for {embodiment_type}")
        return robot_file
    
    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)
    
    if len(embodiment_type) == 1:
        config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        config["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        config["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        config["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        config["embodiment_dis"] = embodiment_type[2]
        config["dual_arm_embodied"] = False
    else:
        raise ValueError("Number of embodiment config parameters should be 1 or 3")
    
    config["left_embodiment_config"] = get_embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = get_embodiment_config(config["right_robot_file"])
    
    return config


def save_episode_hdf5(save_path: Path, episode_idx: int, 
                      jacobian_data: Dict[str, Any], 
                      embodiment_name: str, control_mode: str):
    """Save episode data to HDF5 with images and complete dimension info."""
    filename = save_path / f"episode_{episode_idx:06d}.h5"
    
    with h5py.File(filename, 'w') as f:
        # States and actions
        f.create_dataset('states_t', data=jacobian_data['states_t'], compression='gzip')
        f.create_dataset('states_t1', data=jacobian_data['states_t1'], compression='gzip')
        f.create_dataset('actions', data=jacobian_data['actions'], compression='gzip')
        f.create_dataset('actual_deltas', data=jacobian_data['actual_deltas'], compression='gzip')
        
        # Images - save each camera view
        images_grp = f.create_group('images')
        for img_key, img_data in jacobian_data.get('images', {}).items():
            if img_data is not None and len(img_data) > 0:
                # Use chunked storage for images (better for random access)
                images_grp.create_dataset(
                    img_key, data=img_data, 
                    compression='gzip', compression_opts=4,
                    chunks=(1,) + img_data.shape[1:]  # chunk per frame
                )
        
        # Metadata - basic info (encode strings for HDF5 compatibility)
        f.attrs['control_mode'] = np.bytes_(control_mode)
        f.attrs['embodiment'] = np.bytes_(embodiment_name)
        f.attrs['episode_idx'] = episode_idx
        f.attrs['num_transitions'] = len(jacobian_data['states_t'])
        
        # Metadata - dimension info (important for cross-embodiment, cross-mode training)
        robot_info = jacobian_data.get('robot_info', {})
        f.attrs['left_arm_dof'] = robot_info.get('left_arm_dof', 7)
        f.attrs['right_arm_dof'] = robot_info.get('right_arm_dof', 7)
        f.attrs['state_dim'] = jacobian_data['states_t'].shape[1]
        f.attrs['action_dim'] = jacobian_data['actions'].shape[1]
        
        # Save dimension breakdown for easier parsing
        # Format depends on control_mode:
        # - delta_qpos: [left_qpos(N), left_grip(1), right_qpos(N), right_grip(1)]
        # - delta_ee_pos: [left_pos(3), left_grip(1), right_pos(3), right_grip(1)]
        # - delta_ee: [left_pos(3), left_quat(4), left_grip(1), right_pos(3), right_quat(4), right_grip(1)]
        left_dof = robot_info.get('left_arm_dof', 7)
        right_dof = robot_info.get('right_arm_dof', 7)
        
        if control_mode == 'delta_qpos':
            f.attrs['dim_breakdown'] = np.bytes_(f"left_qpos:{left_dof},left_grip:1,right_qpos:{right_dof},right_grip:1")
        elif control_mode == 'delta_ee_pos':
            f.attrs['dim_breakdown'] = np.bytes_("left_pos:3,left_grip:1,right_pos:3,right_grip:1")
        elif control_mode == 'delta_ee':
            f.attrs['dim_breakdown'] = np.bytes_("left_pos:3,left_quat:4,left_grip:1,right_pos:3,right_quat:4,right_grip:1")
    
    return filename


def main():
    args = parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("envs._base_task").setLevel(logging.DEBUG)
    
    # Set random seed
    np.random.seed(int(time.time()) % 2**32)
    
    # Setup multiprocessing
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    # Create save directory
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Determine embodiment list for logging
    if args.random_embodiment:
        if args.embodiment_list:
            embodiment_list = args.embodiment_list
        else:
            embodiment_list = [e[0] for e in EMBODIMENTS]
    else:
        embodiment_list = [args.embodiment]
    
    # Save collection config
    collection_config = {
        "num_episodes": args.num_episodes,
        "steps_per_episode": args.steps_per_episode,
        "num_objects": args.num_objects,
        "random_embodiment": args.random_embodiment,
        "embodiment_list": embodiment_list,
        "uniform_embodiment": args.uniform_embodiment if args.random_embodiment else None,
        "random_control_mode": args.random_control_mode,
        "control_mode": args.control_mode if not args.random_control_mode else "random",
        "timestamp": datetime.now().isoformat(),
    }
    with open(save_path / "collection_config.json", "w") as f:
        json.dump(collection_config, f, indent=2)
    
    logger.info(f"Episodes: {args.num_episodes}")
    logger.info(f"Steps per episode: {args.steps_per_episode}")
    logger.info(f"Objects per scene: {args.num_objects}")
    logger.info(f"Random embodiment: {args.random_embodiment}")
    if args.random_embodiment:
        logger.info(f"  Embodiments: {embodiment_list}")
        logger.info(f"  Uniform sampling: {args.uniform_embodiment}")
        logger.info(f"  Episodes per robot: {args.episodes_per_robot}")
    else:
        logger.info(f"  Fixed embodiment: {args.embodiment}")
    logger.info(f"Random control mode: {args.random_control_mode}")
    logger.info(f"Save path: {save_path}")
    print("=" * 70 + "\n")
    
    # Import task
    from envs.tasks import random_exploration
    from envs.utils.create_actor import UnStableError
    
    # Statistics
    successful_episodes = 0
    failed_episodes = 0
    total_transitions = 0
    embodiment_counts = {}
    control_mode_counts = {}
    
    episode_idx = 0
    seed_counter = int(time.time()) % 2**32
    episodes_with_current_robot = 0
    current_embodiment = None
    current_embodiment_name = None
    current_task = None  # Reusable task object
    
    while successful_episodes < args.num_episodes:
        # Select embodiment - only change when needed
        need_new_robot = (current_embodiment is None or 
                         (args.random_embodiment and episodes_with_current_robot >= args.episodes_per_robot))
        
        if need_new_robot:
            # Close previous task if exists
            if current_task is not None:
                try:
                    current_task.close_env()
                except:
                    pass
                current_task = None
            
            episodes_with_current_robot = 0
            if args.random_embodiment:
                if args.embodiment_list:
                    emb_name = np.random.choice(args.embodiment_list)
                    emb_configs = {e[0]: e for e in EMBODIMENTS}
                    if emb_name in emb_configs:
                        current_embodiment = emb_configs[emb_name]
                    else:
                        current_embodiment = [emb_name, emb_name, 0.5]
                else:
                    current_embodiment = get_random_embodiment(weighted=not args.uniform_embodiment)
            else:
                current_embodiment = [args.embodiment, args.embodiment, 0.6]
            current_embodiment_name = current_embodiment[0]
            logger.info(f"--- Switching to robot: {current_embodiment_name} (will collect {args.episodes_per_robot} episodes) ---")
        
        embodiment = current_embodiment
        embodiment_name = current_embodiment_name
        
        # Select control mode for this episode (must be after embodiment selection)
        if args.random_control_mode:
            control_mode = get_random_control_mode(embodiment_name)
        else:
            # Check if fixed control mode is compatible with this embodiment
            if args.control_mode in ['delta_ee', 'delta_ee_pos'] and embodiment_name not in EMBODIMENTS_WITH_EE_SUPPORT:
                logger.warning(f"  {embodiment_name} doesn't support {args.control_mode}, using delta_qpos")
                control_mode = 'delta_qpos'
            else:
                control_mode = args.control_mode
        
        retry_count = 0
        
        while retry_count < args.max_retries:
            current_stage = "init"
            try:
                # First episode with this robot OR task was closed due to error
                if current_task is None:
                    # Load fresh config
                    config = load_config(args.config, args)
                    config["embodiment"] = embodiment
                    config["control_mode"] = control_mode
                    config = setup_embodiment(config)
                    config["task_name"] = "random_exploration"
                    config["seed"] = seed_counter
                    config["now_ep_num"] = episode_idx
                    config["save_data"] = False
                    
                    seed_counter += 1
                    
                    # Create new task with full setup
                    current_stage = "create_task"
                    logger.debug(f"  [{embodiment_name}] Creating task...")
                    current_task = random_exploration()
                    
                    current_stage = "setup_demo"
                    logger.info(f"  [{embodiment_name}] Setting up scene with {args.num_objects} objects...")
                    current_task.setup_demo(**config)
                else:
                    # Reuse existing task - just reset scene
                    current_stage = "reset_scene"
                    logger.info(f"  [{embodiment_name}] Resetting scene (reusing robot)...")
                    current_task.control_mode = control_mode  # Update control mode
                    current_task.reset_for_new_episode(seed=seed_counter)
                    seed_counter += 1
                
                # Run exploration
                current_stage = "play_once"
                info = current_task.play_once()
                
                # Check success
                current_stage = "check_success"
                if current_task.check_success():
                    # Get data with images
                    current_stage = "get_jacobian_data"
                    jacobian_data = current_task.get_jacobian_training_data()
                    
                    if jacobian_data is None:
                        logger.warning(f"  get_jacobian_training_data() returned None")
                        retry_count += 1
                        continue
                    
                    num_transitions = len(jacobian_data.get('states_t', []))
                    
                    if num_transitions > 0:
                        # Save episode
                        save_episode_hdf5(save_path, successful_episodes, 
                                         jacobian_data, embodiment_name, control_mode)
                        
                        successful_episodes += 1
                        total_transitions += num_transitions
                        embodiment_counts[embodiment_name] = embodiment_counts.get(embodiment_name, 0) + 1
                        control_mode_counts[control_mode] = control_mode_counts.get(control_mode, 0) + 1
                        episodes_with_current_robot += 1
                        
                        # Verify Jacobian relationship
                        recon_err = np.mean(np.abs(
                            jacobian_data['states_t'] + jacobian_data['actions'] - jacobian_data['states_t1']
                        ))
                        
                        # Count images
                        num_images = sum(len(v) for v in jacobian_data.get('images', {}).values())
                        
                        logger.info(f"Ep {successful_episodes:4d}/{args.num_episodes}: "
                              f"{embodiment_name[:12]:12s} | {control_mode:12s} | "
                              f"{num_transitions} trans | {num_images} imgs | "
                              f"recon_err={recon_err:.4f}")
                        
                        # Don't close task - keep for reuse!
                        break
                    else:
                        logger.warning(f"  No transitions recorded, retrying (retry {retry_count+1}/{args.max_retries})...")
                        retry_count += 1
                else:
                    logger.warning(f"  Episode failed success check, retrying (retry {retry_count+1}/{args.max_retries})...")
                    retry_count += 1
                
                seed_counter += 1
                
            except UnStableError as e:
                # Scene unstable, try different seed
                logger.debug(f"  UnStableError at stage '{current_stage}': {e}")
                seed_counter += 1
                retry_count += 1
                continue
                
            except Exception as e:
                # Log detailed error with traceback
                error_msg = f"Episode {episode_idx}, retry {retry_count+1}/{args.max_retries}"
                logger.error(f"  [{error_msg}] Error at stage '{current_stage}': {type(e).__name__}: {e}")
                
                # Print traceback for debugging (only first occurrence per episode)
                if retry_count == 0:
                    logger.error(f"  Traceback:\n{traceback.format_exc()}")
                
                # On error, close and recreate task
                if current_task is not None:
                    try:
                        current_task.close_env()
                    except:
                        pass
                    current_task = None
                
                retry_count += 1
                seed_counter += 1
                continue
        
        if retry_count >= args.max_retries:
            failed_episodes += 1
            logger.warning(f"  FAILED episode {episode_idx} after {args.max_retries} retries (last stage: {current_stage})")
        
        episode_idx += 1
        
        # Progress update every 10 episodes
        if episode_idx % 10 == 0:
            logger.info(f"--- Progress: {successful_episodes}/{args.num_episodes} episodes, "
                  f"{total_transitions} total transitions ---")
    
    # Summary
    print("\n" + "=" * 70)
    print("Collection Complete")
    print("=" * 70)
    logger.info(f"Successful episodes: {successful_episodes}/{args.num_episodes}")
    logger.info(f"Failed episodes: {failed_episodes}")
    logger.info(f"Total transitions: {total_transitions}")
    logger.info(f"Avg transitions/episode: {total_transitions / max(1, successful_episodes):.1f}")
    logger.info("Embodiment distribution:")
    for emb, count in sorted(embodiment_counts.items()):
        logger.info(f"  {emb}: {count} episodes")
    logger.info("Control mode distribution:")
    for mode, count in sorted(control_mode_counts.items()):
        logger.info(f"  {mode}: {count} episodes")
    logger.info(f"Data saved to: {save_path}")
    
    # Save statistics
    stats = {
        "successful_episodes": successful_episodes,
        "failed_episodes": failed_episodes,
        "total_transitions": total_transitions,
        "embodiment_distribution": embodiment_counts,
        "control_mode_distribution": control_mode_counts,
    }
    with open(save_path / "collection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
