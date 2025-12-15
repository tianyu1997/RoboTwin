#!/usr/bin/env python3
"""
Data Collection Script for F1-VLA Teacher Training (Phase 1)

This script collects trajectories using random actions and saves them to disk
for offline training of the World Model.

Usage:
    python collect_data_teacher.py --num_episodes 100 --output_dir ./data/teacher_offline
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from omegaconf import OmegaConf

# ============== Setup paths ==============
script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.dirname(script_dir)
robotwin_dir = os.path.dirname(rl_dir)
f1_vla_dir = os.path.dirname(robotwin_dir)
sys.path.insert(0, f1_vla_dir)
sys.path.insert(0, robotwin_dir)

from rl.suppress_logs import suppress_curobo_logs
from rl.training.rl_training_common import (
    load_rl_config,
    get_environment_config,
    get_training_config,
)
from rl.training.parallel_utils import ParallelEnvCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Collect Data for Teacher Training")
    
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to RL training config YAML file")
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--output_dir", type=str, default="./rl/data/teacher_offline",
                       help="Directory to save collected data")
    parser.add_argument("--num_envs", type=int, default=16,
                       help="Number of parallel environments")
    parser.add_argument("--steps_per_episode", type=int, default=None,
                       help="Override steps per episode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()

class DataCollector:
    def __init__(self, rl_config: OmegaConf, output_dir: str, num_envs: int = 1):
        self.rl_config = rl_config
        self.output_dir = Path(output_dir)
        self.num_envs = num_envs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.env_config = get_environment_config(rl_config)
        self.train_config = get_training_config(rl_config)
        
        # Determine history length from model config (default to 4 if not found)
        # We don't load the full model config here to avoid dependency on model files
        # Just use a reasonable default or config value
        self.history_length = 4 
        
        self.steps_per_episode = self.train_config.steps_per_episode
        self.action_dim = self.train_config.action_dim
        
        self.env = None
        self.env_collector = None
        
    def setup_environment(self):
        """Setup the simulation environment(s)."""
        logger.info("Setting up environment...")
        
        # Set environment variables for SAPIEN
        # For data collection, we can just use the first GPU or CPU
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
        # Basic GPU setup for SAPIEN
        os.environ["VK_DEVICE_INDEX"] = "0"
        os.environ["SAPIEN_DEVICE_INDEX"] = "0"
        os.environ["EGL_DEVICE_ID"] = "0"

        from rl.f1_rl_env import TeacherEnv
        
        single_arm = self.env_config.get("single_arm", False)
        scene_reset_interval = self.env_config.get("scene_reset_interval", 1)
        randomize_robot_init = self.env_config.get("randomize_robot_init", False)
        need_planner = self.env_config.get("need_planner", False)
        need_topp = self.env_config.get("need_topp", False)
        
        def create_env():
            return TeacherEnv(
                task_config={
                    **self.env_config,
                    "need_planner": need_planner,
                    "need_topp": need_topp,
                    "render_device": 0,
                },
                history_length=self.history_length,
                max_steps=self.steps_per_episode,
                device="cuda", # Use CUDA for tensors
                action_scale=self.train_config.action_scale,
                single_arm=single_arm,
                scene_reset_interval=scene_reset_interval,
                randomize_robot_init=randomize_robot_init,
            )
        
        if self.num_envs > 1:
            self.env_collector = ParallelEnvCollector(
                env_fn=create_env,
                num_envs=self.num_envs,
                is_main_process=True,
            )
            self.env_collector.initialize()
            self.env = self.env_collector.envs[0]
            logger.info(f"Environment ready! {self.num_envs} parallel envs")
        else:
            self.env = create_env()
            logger.info("Environment ready! Single env")

    def collect(self, num_episodes: int):
        self.setup_environment()
        
        # Check existing files to continue numbering
        existing_files = list(self.output_dir.glob("episode_*.pt"))
        start_idx = 0
        if existing_files:
            indices = []
            for f in existing_files:
                try:
                    # Assumes format episode_XXXXXX.pt
                    idx = int(f.stem.split("_")[-1])
                    indices.append(idx)
                except (IndexError, ValueError):
                    pass
            if indices:
                start_idx = max(indices) + 1
        
        logger.info(f"Found {len(existing_files)} existing episodes. Starting numbering from {start_idx}.")
        logger.info(f"Starting collection of {num_episodes} new episodes...")
        pbar = tqdm(total=num_episodes, desc="Collecting")
        
        collected_count = 0
        current_idx = start_idx
        
        while collected_count < num_episodes:
            episodes_to_save = []
            
            if self.env_collector is not None and self.num_envs > 1:
                # Calculate steps needed
                remaining = num_episodes - collected_count
                # We collect in batches of num_envs
                
                # Collect steps
                total_steps = self.steps_per_episode * self.num_envs
                completed_episodes = self.env_collector.collect_steps(
                    num_steps=total_steps,
                    action_fn=None, # Random actions
                    action_dim=self.action_dim,
                )
                episodes_to_save.extend(completed_episodes)
            else:
                # Single env collection
                obs, _ = self.env.reset()
                trajectory = []
                done = False
                while not done:
                    action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
                    next_obs, _, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    trajectory.append({
                        "obs": obs,
                        "action": info.get("action_executed", action),
                        "next_obs": next_obs,
                    })
                    obs = next_obs
                episodes_to_save.append(trajectory)
            
            # Save episodes
            for ep in episodes_to_save:
                if collected_count >= num_episodes:
                    break
                
                save_path = self.output_dir / f"episode_{current_idx:06d}.pt"
                torch.save(ep, save_path)
                collected_count += 1
                current_idx += 1
                pbar.update(1)
        
        pbar.close()
        logger.info(f"Collection complete. Saved to {self.output_dir}")
        
        # Cleanup
        if self.env_collector:
            self.env_collector.close()

def main():
    args = parse_args()
    
    rl_config = load_rl_config(args.rl_config)
    
    # Overrides
    if args.steps_per_episode:
        rl_config.training.steps_per_episode = args.steps_per_episode
    
    collector = DataCollector(
        rl_config=rl_config,
        output_dir=args.output_dir,
        num_envs=args.num_envs
    )
    
    collector.collect(args.num_episodes)

if __name__ == "__main__":
    main()
