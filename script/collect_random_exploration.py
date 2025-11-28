#!/usr/bin/env python3
"""
Random Exploration Data Collection for Visual-Action Jacobian Learning

This script collects training data for visual-action Jacobian models:
- Ensures s(t+1) = s(t) + a(t) relationship
- Supports multiple control modes (delta_qpos, delta_ee, delta_ee_pos)
- Cross-embodiment data collection
- Saves data in formats suitable for Jacobian training

Data Format:
    Each transition contains:
    - state_t: robot state at time t
    - action: delta action a(t)
    - state_t1: robot state at time t+1
    - actual_delta: s(t+1) - s(t)
    - images: visual observations at time t
    - robot_info: embodiment information

Usage:
    # Collect with joint space control
    python collect_random_exploration.py --control_mode delta_qpos
    
    # Collect with end-effector control
    python collect_random_exploration.py --control_mode delta_ee
    
    # Multi-embodiment collection
    python collect_random_exploration.py --random_embodiment --num_episodes 1000
"""

import sys
import os

# Add RoboTwin root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
robotwin_dir = os.path.dirname(script_dir)
sys.path.insert(0, robotwin_dir)

import argparse
import yaml
import json
import time
import h5py
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Visual-Action Jacobian Training Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic collection with joint space control
    python collect_random_exploration.py --num_episodes 100
    
    # End-effector control mode
    python collect_random_exploration.py --control_mode delta_ee
    
    # Multi-embodiment with larger action deltas
    python collect_random_exploration.py --random_embodiment --delta_qpos_scale 0.1
        """
    )
    
    # Basic settings
    parser.add_argument("--config", type=str, default="random_exploration",
                       help="Config file name (without .yml)")
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--steps_per_episode", type=int, default=50,
                       help="Random steps per episode")
    parser.add_argument("--num_objects", type=int, default=5,
                       help="Number of random objects per scene")
    
    # Control mode settings (for Jacobian learning)
    parser.add_argument("--control_mode", type=str, default="delta_qpos",
                       choices=["delta_qpos", "delta_ee", "delta_ee_pos"],
                       help="Control mode: delta_qpos (joint), delta_ee (full EE), delta_ee_pos (EE position only)")
    parser.add_argument("--delta_qpos_scale", type=float, default=0.05,
                       help="Scale for joint position deltas (radians)")
    parser.add_argument("--delta_ee_pos_scale", type=float, default=0.02,
                       help="Scale for EE position deltas (meters)")
    parser.add_argument("--delta_ee_rot_scale", type=float, default=0.05,
                       help="Scale for EE rotation deltas (radians)")
    parser.add_argument("--gripper_action_prob", type=float, default=0.1,
                       help="Probability of gripper action per step")
    
    # Embodiment settings
    parser.add_argument("--random_embodiment", action="store_true",
                       help="Randomly select robot embodiment each episode")
    parser.add_argument("--embodiment", type=str, default="franka-panda",
                       help="Fixed embodiment type (if not random)")
    
    # Randomization and retry settings
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: use time-based)")
    parser.add_argument("--max_seed_retries", type=int, default=50,
                       help="Maximum seed retries per episode for stability")
    
    # Output
    parser.add_argument("--save_path", type=str, default="./data/jacobian_data",
                       help="Path to save collected data")
    parser.add_argument("--save_format", type=str, default="hdf5",
                       choices=["hdf5", "pkl", "both"],
                       help="Format to save data")
    parser.add_argument("--render", action="store_true",
                       help="Enable rendering for visualization")
    
    return parser.parse_args()


def get_random_embodiment():
    """Randomly select a robot embodiment."""
    embodiments = [
        ["franka-panda", "franka-panda", 0.6],
        # Uncomment as more embodiments become available:
        # ["aloha-agilex", "aloha-agilex", 0.5],
        # ["ARX-X5", "ARX-X5", 0.5],
        # ["piper", "piper", 0.5],
    ]
    return embodiments[np.random.randint(len(embodiments))]


def load_config(config_name: str, args) -> dict:
    """Load and update configuration."""
    config_path = os.path.join(robotwin_dir, "task_config", f"{config_name}.yml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # Override with command line arguments
    config["episode_num"] = args.num_episodes
    config["num_random_steps"] = args.steps_per_episode
    config["num_objects"] = args.num_objects
    config["control_mode"] = args.control_mode
    config["delta_qpos_scale"] = args.delta_qpos_scale
    config["delta_ee_pos_scale"] = args.delta_ee_pos_scale
    config["delta_ee_rot_scale"] = args.delta_ee_rot_scale
    config["gripper_action_prob"] = args.gripper_action_prob
    config["save_path"] = args.save_path
    config["render_freq"] = 10 if args.render else 0
    
    if args.random_embodiment:
        config["embodiment"] = get_random_embodiment()
    elif args.embodiment:
        config["embodiment"] = [args.embodiment, args.embodiment, 0.6]
    
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


def save_jacobian_data_hdf5(save_path: Path, episode_idx: int, 
                            jacobian_data: Dict[str, Any], config: dict):
    """Save Jacobian training data to HDF5 format."""
    filename = save_path / f"episode_{episode_idx:06d}.h5"
    
    with h5py.File(filename, 'w') as f:
        # Create groups
        states_grp = f.create_group('states')
        actions_grp = f.create_group('actions')
        images_grp = f.create_group('images')
        meta_grp = f.create_group('metadata')
        
        # Save state arrays
        states_grp.create_dataset('state_t', data=jacobian_data['states_t'], 
                                   compression='gzip')
        states_grp.create_dataset('state_t1', data=jacobian_data['states_t1'],
                                   compression='gzip')
        states_grp.create_dataset('actual_delta', data=jacobian_data['actual_deltas'],
                                   compression='gzip')
        
        # Save actions
        actions_grp.create_dataset('delta_action', data=jacobian_data['actions'],
                                    compression='gzip')
        
        # Save images
        for img_key, img_data in jacobian_data.get('images', {}).items():
            if img_data is not None and len(img_data) > 0:
                images_grp.create_dataset(img_key, data=img_data, compression='gzip')
        
        # Save metadata
        meta_grp.attrs['control_mode'] = jacobian_data['control_mode']
        meta_grp.attrs['episode_idx'] = episode_idx
        meta_grp.attrs['num_transitions'] = len(jacobian_data['states_t'])
        
        robot_info = jacobian_data['robot_info']
        meta_grp.attrs['embodiment'] = robot_info.get('embodiment', 'unknown')
        meta_grp.attrs['left_arm_dof'] = robot_info.get('left_arm_dof', 7)
        meta_grp.attrs['right_arm_dof'] = robot_info.get('right_arm_dof', 7)
        
        # Save joint limits
        limits = robot_info.get('arm_joint_limits', {})
        if limits:
            limits_grp = meta_grp.create_group('joint_limits')
            for k, v in limits.items():
                limits_grp.create_dataset(k, data=v)
    
    return filename


def save_jacobian_data_pkl(save_path: Path, episode_idx: int,
                           jacobian_data: Dict[str, Any], config: dict):
    """Save Jacobian training data to pickle format."""
    filename = save_path / f"episode_{episode_idx:06d}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(jacobian_data, f)
    
    return filename


def main():
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed(int(time.time()) % 2**32)
    
    # Setup multiprocessing
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    # Load configuration
    config = load_config(args.config, args)
    config = setup_embodiment(config)
    config["task_name"] = "random_exploration"
    
    # Create save directory
    save_path = Path(config["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save collection config
    collection_config = {
        "control_mode": args.control_mode,
        "num_episodes": args.num_episodes,
        "steps_per_episode": args.steps_per_episode,
        "delta_qpos_scale": args.delta_qpos_scale,
        "delta_ee_pos_scale": args.delta_ee_pos_scale,
        "delta_ee_rot_scale": args.delta_ee_rot_scale,
        "gripper_action_prob": args.gripper_action_prob,
        "random_embodiment": args.random_embodiment,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(save_path / "collection_config.json", "w") as f:
        json.dump(collection_config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Visual-Action Jacobian Data Collection")
    print("=" * 70)
    print(f"Control Mode: {args.control_mode}")
    print(f"  -> Data format: s(t+1) = s(t) + a(t)")
    print(f"Episodes: {args.num_episodes}")
    print(f"Steps per episode: {args.steps_per_episode}")
    print(f"Objects per scene: {args.num_objects}")
    print(f"Random embodiment: {args.random_embodiment}")
    print(f"Save path: {save_path}")
    print(f"Save format: {args.save_format}")
    print(f"Max seed retries: {args.max_seed_retries}")
    print("=" * 70 + "\n")
    
    # Import task and UnStableError
    from envs.tasks import random_exploration
    from envs.utils.create_actor import UnStableError
    
    # Statistics
    successful_episodes = 0
    failed_episodes = 0
    total_transitions = 0
    embodiment_counts = {}
    
    # Phase 1: Collect stable seeds
    print("Phase 1: Finding stable seeds...")
    stable_seeds = []
    seed_attempts = 0
    base_seed = args.seed if args.seed else int(time.time()) % 2**32
    
    while len(stable_seeds) < args.num_episodes and seed_attempts < args.num_episodes * args.max_seed_retries:
        test_seed = base_seed + seed_attempts
        seed_attempts += 1
        
        # Optionally randomize embodiment
        if args.random_embodiment:
            config["embodiment"] = get_random_embodiment()
            config = setup_embodiment(config)
        
        task = None
        try:
            task = random_exploration()
            config["seed"] = test_seed
            config["now_ep_num"] = len(stable_seeds)
            config["save_data"] = False
            
            task.setup_demo(**config)
            
            # If we get here, the scene is stable
            stable_seeds.append(test_seed)
            print(f"  Found stable seed {len(stable_seeds)}/{args.num_episodes}: {test_seed}")
            
            task.close_env()
            if config["render_freq"] > 0 and hasattr(task, 'viewer') and task.viewer:
                task.viewer.close()
                
        except UnStableError as e:
            # Unstable - try next seed
            if task is not None:
                try:
                    task.close_env()
                    if config["render_freq"] > 0 and hasattr(task, 'viewer') and task.viewer:
                        task.viewer.close()
                except:
                    pass
            continue
        except Exception as e:
            print(f"  Error testing seed {test_seed}: {e}")
            if task is not None:
                try:
                    task.close_env()
                except:
                    pass
            continue
    
    print(f"Found {len(stable_seeds)} stable seeds after {seed_attempts} attempts\n")
    
    if len(stable_seeds) == 0:
        print("ERROR: Could not find any stable seeds!")
        return
    
    # Phase 2: Collect data with stable seeds
    print("Phase 2: Collecting data...")
    
    for episode_idx, episode_seed in enumerate(stable_seeds):
        task = None
        try:
            # Optionally randomize embodiment each episode
            if args.random_embodiment:
                config["embodiment"] = get_random_embodiment()
                config = setup_embodiment(config)
            
            task = random_exploration()
            config["seed"] = episode_seed
            config["now_ep_num"] = episode_idx
            config["save_data"] = False
            
            task.setup_demo(**config)
            
            # Run exploration
            info = task.play_once()
            
            # Check success and get data
            if task.check_success():
                successful_episodes += 1
                
                # Get Jacobian training data
                jacobian_data = task.get_jacobian_training_data()
                num_transitions = len(jacobian_data.get('states_t', []))
                total_transitions += num_transitions
                
                # Track embodiment distribution
                emb = jacobian_data['robot_info'].get('embodiment', 'unknown')
                embodiment_counts[emb] = embodiment_counts.get(emb, 0) + 1
                
                # Save data
                if args.save_format in ['hdf5', 'both']:
                    save_jacobian_data_hdf5(save_path, episode_idx, jacobian_data, config)
                if args.save_format in ['pkl', 'both']:
                    save_jacobian_data_pkl(save_path, episode_idx, jacobian_data, config)
                
                # Verify s(t+1) = s(t) + a(t) relationship
                reconstruction_error = np.mean(np.abs(
                    jacobian_data['states_t'] + jacobian_data['actions'] - jacobian_data['states_t1']
                ))
                actual_delta_vs_action = np.mean(np.abs(
                    jacobian_data['actual_deltas'] - jacobian_data['actions']
                ))
                
                print(f"Ep {episode_idx + 1:4d}/{len(stable_seeds)}: "
                      f"OK - {num_transitions} trans, "
                      f"recon_err={reconstruction_error:.6f}, "
                      f"delta_diff={actual_delta_vs_action:.6f}")
            else:
                failed_episodes += 1
                print(f"Ep {episode_idx + 1:4d}/{len(stable_seeds)}: FAILED (exploration incomplete)")
            
            # Cleanup
            task.close_env(clear_cache=(episode_idx % 10 == 0))
            if config["render_freq"] > 0 and hasattr(task, 'viewer') and task.viewer:
                task.viewer.close()
                
        except Exception as e:
            failed_episodes += 1
            print(f"Ep {episode_idx + 1:4d}/{len(stable_seeds)}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            
            if task is not None:
                try:
                    task.close_env()
                    if config["render_freq"] > 0 and hasattr(task, 'viewer') and task.viewer:
                        task.viewer.close()
                except:
                    pass
    
    # Summary statistics
    avg_seed_attempts = seed_attempts / max(1, len(stable_seeds))
    print("\n" + "=" * 70)
    print("Collection Complete")
    print("=" * 70)
    print(f"Stable seeds found: {len(stable_seeds)}/{args.num_episodes}")
    print(f"Seed attempts: {seed_attempts} (avg {avg_seed_attempts:.1f} per episode)")
    print(f"Successful episodes: {successful_episodes}/{len(stable_seeds)}")
    print(f"Failed episodes: {failed_episodes}")
    print(f"Total transitions: {total_transitions}")
    print(f"Avg transitions/episode: {total_transitions / max(1, successful_episodes):.1f}")
    print(f"\nEmbodiment distribution:")
    for emb, count in sorted(embodiment_counts.items()):
        print(f"  {emb}: {count} episodes")
    print(f"\nData saved to: {save_path}")
    print("=" * 70)
    
    # Save final statistics
    stats = {
        "successful_episodes": successful_episodes,
        "failed_episodes": failed_episodes,
        "total_transitions": total_transitions,
        "embodiment_distribution": embodiment_counts,
        "control_mode": args.control_mode,
    }
    with open(save_path / "collection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
