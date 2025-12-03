#!/usr/bin/env python3
"""
Teacher Policy Training Script for F1-VLA (Phase 1)

Refactored version using shared rl_training_common module.

Phase 1 Training:
- Train LLM + World Model using random exploration
- Input: history actions, states, head observation
- Action: randomly generated (for exploration)
- Reward: prediction accuracy of next frame observation
"""

import os
import sys

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
from typing import Dict, Any, Optional, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

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
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher Policy (World Model) - Phase 1")
    
    # Config file (recommended way)
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to RL training config YAML file")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Override model config file path")
    
    # Override common training parameters
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    
    # Sequential training (can override config)
    parser.add_argument("--sequential_training", action="store_true", default=None,
                       help="Enable sequential training for memory state propagation")
    parser.add_argument("--no_sequential_training", action="store_false", 
                       dest="sequential_training")
    
    # Device & debug
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint directory to resume training from")
    
    return parser.parse_args()


class TeacherTrainer(BaseRLTrainer):
    """
    Trainer for teacher policy (LLM + World Model).
    
    Training loop:
    1. Collect trajectory with random actions
    2. Feed observation + action history to world model
    3. Predict next observation
    4. Compute loss = prediction error
    5. Update model parameters
    """
    
    def __init__(
        self,
        policy: nn.Module,
        policy_config,
        rl_config: OmegaConf,
        device: str = "cuda",
    ):
        # Get training config
        train_config = get_training_config(rl_config)
        teacher_config = rl_config.get("teacher", {})
        output_dir = teacher_config.get("output_dir", "./outputs/teacher_rl")
        
        super().__init__(
            policy=policy,
            config=train_config,
            output_dir=output_dir,
            device=device,
        )
        
        self.policy_config = policy_config
        self.rl_config = rl_config
        
        # World model image steps
        self.cur_n_obs_img_steps = train_config.n_obs_img_steps
        self.cur_n_pred_img_steps = train_config.n_pred_img_steps
        logger.info(f"World model config: obs_img_steps={self.cur_n_obs_img_steps}, "
                   f"pred_img_steps={self.cur_n_pred_img_steps}")
        
        # Setup policy for training
        self.policy.train()
        
        # Setup optimizer and scheduler
        trainable, total = count_trainable_params(self.policy)
        logger.info(f"Trainable parameters: {trainable:,} / {total:,}")
        
        self.optimizer = setup_optimizer(
            self.policy,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.scheduler = setup_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            T_max=train_config.num_episodes,
            eta_min=1e-6,
        )
        
        # Environment config
        self.env_config = get_environment_config(rl_config)
        
        # Setup batch builder with head camera
        self.batch_builder = BatchBuilder(
            device=device,
            image_keys=["head_rgb"],
        )
        
        # Additional metrics for teacher training
        self.metrics.update({
            "wm_loss": deque(maxlen=100),
            "wm_accuracy": deque(maxlen=100),
            "episode_reward": deque(maxlen=100),
        })
    
    def setup_environment(self):
        """Setup the RL environment."""
        from rl.f1_rl_env import TeacherEnv
        
        self.env = TeacherEnv(
            task_config=self.env_config,
            history_length=self.config.history_length,
            max_steps=self.config.steps_per_episode,
            device=self.device,
            action_scale=self.config.action_scale,
        )
        logger.info(f"Environment setup complete (action_scale={self.config.action_scale})")
    
    def collect_episode(self) -> List[Dict[str, Any]]:
        """
        Collect one episode of experience with random actions.
        
        Memory state is tracked sequentially: each frame uses the previous
        frame's output memory as input.
        """
        obs, info = self.env.reset()
        transitions = []
        
        # Reset memory state for new episode
        self.memory_manager.reset()
        
        done = False
        frame_idx = 0
        
        while not done:
            # Random action for teacher phase
            action = np.random.uniform(-1, 1, self.config.action_dim).astype(np.float32)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            transitions.append({
                "obs": obs,
                "action": info.get("action_executed", action),
                "next_obs": next_obs,
                "reward": reward,
                "done": done,
                "info": info,
                "frame_idx": frame_idx,
                "initial_memory_state": (
                    self.memory_manager.current_memory.detach().clone() 
                    if self.memory_manager.current_memory is not None else None
                ),
            })
            
            obs = next_obs
            frame_idx += 1
        
        return transitions
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        # Inject memory state if available
        batch = self.inject_memory_state(batch)
        
        # Forward pass with world model only (Phase 1)
        loss_dict = self.policy.forward_with_world_model(
            batch,
            cur_n_obs_img_steps=self.cur_n_obs_img_steps,
            cur_n_pred_img_steps=self.cur_n_pred_img_steps,
            train_gen_expert_only=True,  # Only train WM, not action
        )
        
        # Update memory state from output
        self.update_memory_from_output(loss_dict)
        
        # Total loss
        loss = loss_dict["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        clip_gradients(self.policy, max_norm=self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            "wm_loss": loss_dict.get("wm_loss", torch.tensor(0.0)).item(),
            "wm_acc_mean": loss_dict.get("wm_acc_mean", torch.tensor(0.0)).item(),
            "wm_acc_tail": loss_dict.get("wm_acc_tail", torch.tensor(0.0)).item(),
        }
    
    def train(self, start_episode: int = 0):
        """Main training loop."""
        num_episodes = self.config.num_episodes
        sequential = self.config.sequential_training
        
        logger.info(f"Starting teacher training for {num_episodes} episodes")
        logger.info(f"Starting from episode: {start_episode}")
        logger.info(f"Sequential training: {sequential}")
        
        # Setup environment
        self.setup_environment()
        
        # Training buffer (for non-sequential mode)
        episode_buffer = []
        accumulated_steps = 0
        
        for episode in range(start_episode, num_episodes):
            self.metrics["episode"] = episode
            
            # Collect episode
            transitions = self.collect_episode()
            
            # Track episode reward
            episode_reward = sum(t["reward"] for t in transitions)
            self.metrics["episode_reward"].append(episode_reward)
            
            if sequential:
                # Sequential training: process each frame in order
                self.memory_manager.reset()
                
                for transition in transitions:
                    # Build single-frame batch
                    batch = self.batch_builder.build_batch(
                        [transition], include_memory_states=True
                    )
                    
                    # Inject current memory state
                    if self.memory_manager.current_memory is not None:
                        batch["initial_memory_state"] = self.memory_manager.current_memory
                    
                    # Training step (updates memory state)
                    loss_dict = self.train_step(batch)
                    
                    accumulated_steps += 1
                    self.metrics["wm_loss"].append(loss_dict["wm_loss"])
                    self.metrics["wm_accuracy"].append(loss_dict["wm_acc_mean"])
                    self.metrics["total_steps"] += 1
                    
                    # Gradient accumulation update
                    if accumulated_steps >= self.config.gradient_accumulation_steps:
                        self.scheduler.step()
                        accumulated_steps = 0
            else:
                # Non-sequential: batch random transitions
                episode_buffer.extend(transitions)
                
                if len(episode_buffer) >= self.config.batch_size:
                    # Sample batch
                    batch_indices = np.random.choice(
                        len(episode_buffer),
                        size=self.config.batch_size,
                        replace=False
                    )
                    batch_transitions = [episode_buffer[i] for i in batch_indices]
                    
                    # Build and train
                    batch = self.batch_builder.build_batch(
                        batch_transitions, include_memory_states=False
                    )
                    loss_dict = self.train_step(batch)
                    
                    accumulated_steps += 1
                    self.metrics["wm_loss"].append(loss_dict["wm_loss"])
                    self.metrics["wm_accuracy"].append(loss_dict["wm_acc_mean"])
                    self.metrics["total_steps"] += self.config.batch_size
                    
                    # Clear buffer periodically
                    if len(episode_buffer) > self.config.batch_size * 10:
                        episode_buffer = episode_buffer[-self.config.batch_size * 5:]
                    
                    # Gradient accumulation update
                    if accumulated_steps >= self.config.gradient_accumulation_steps:
                        self.scheduler.step()
                        accumulated_steps = 0
            
            # Logging
            if (episode + 1) % self.config.log_every == 0:
                self._log_episode_metrics(episode)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_every == 0:
                self.save_checkpoint(
                    episode,
                    extra_state={
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    }
                )
        
        # Final save
        self.save_checkpoint(num_episodes)
        logger.info("Training complete!")
    
    def _log_episode_metrics(self, episode: int):
        """Log training metrics."""
        avg_wm_loss = np.mean(self.metrics["wm_loss"]) if self.metrics["wm_loss"] else 0
        avg_wm_acc = np.mean(self.metrics["wm_accuracy"]) if self.metrics["wm_accuracy"] else 0
        avg_reward = np.mean(self.metrics["episode_reward"]) if self.metrics["episode_reward"] else 0
        
        self.log_metrics(episode, {
            "wm_loss": avg_wm_loss,
            "wm_accuracy": avg_wm_acc,
            "episode_reward": avg_reward,
            "lr": self.scheduler.get_last_lr()[0],
        })


def main():
    args = parse_args()
    
    # Load RL config
    rl_config = load_rl_config(args.rl_config)
    
    # Apply command-line overrides
    if args.model_config:
        rl_config.model.config_file = args.model_config
    if args.num_episodes is not None:
        rl_config.training.num_episodes = args.num_episodes
    if args.steps_per_episode is not None:
        rl_config.training.steps_per_episode = args.steps_per_episode
    if args.batch_size is not None:
        rl_config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        rl_config.training.learning_rate = args.learning_rate
    if args.output_dir is not None:
        rl_config.teacher.output_dir = args.output_dir
    if args.save_every is not None:
        rl_config.training.save_every = args.save_every
    if args.log_every is not None:
        rl_config.training.log_every = args.log_every
    if args.sequential_training is not None:
        rl_config.training.sequential_training = args.sequential_training
    if args.device is not None:
        rl_config.device = args.device
    if args.debug is not None:
        rl_config.debug = args.debug
    
    device = rl_config.get("device", "cuda")
    debug = rl_config.get("debug", False)
    
    logger.info("=" * 70)
    logger.info("Teacher Policy Training (Phase 1)")
    logger.info("=" * 70)
    
    # Load model config file path
    model_config_file = rl_config.get("model", {}).get(
        "config_file", 
        "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml"
    )
    
    # Get LoRA config
    lora_config = get_lora_config_from_dict(rl_config)
    
    # Load policy
    policy, policy_config, model_config = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=lora_config,
    )
    
    logger.info("Model loaded successfully")
    logger.info(f"Model config: use_world_model={policy_config.use_world_model}")
    
    # Create trainer
    trainer = TeacherTrainer(
        policy=policy,
        policy_config=policy_config,
        rl_config=rl_config,
        device=device,
    )
    
    # Resume training if specified
    start_episode = 0
    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        start_episode = trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(start_episode=start_episode)


if __name__ == "__main__":
    main()
