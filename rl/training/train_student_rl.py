#!/usr/bin/env python3
"""
Student Policy Training Script for F1-VLA (Phase 2)

Refactored version using shared rl_training_common module.

Phase 2 Training:
- Initialize new LLM but REUSE frozen World Model from teacher
- Use only wrist camera observations (no head camera)
- Explorer (F1-VLA actor) generates actions instead of random
- Reward based on:
  1. Memory RNN hidden state divergence between student and teacher
  2. Actions that make World Model unable to accurately predict next frame
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
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="Train Student Policy (Explorer) - Phase 2")
    
    # Config file (recommended way)
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to RL training config YAML file")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Override model config file path")
    
    # Teacher checkpoint (required)
    parser.add_argument("--teacher_path", type=str, required=True,
                       help="Path to trained teacher policy checkpoint")
    
    # Override common training parameters
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    
    # Reward weights
    parser.add_argument("--memory_divergence_weight", type=float, default=None)
    parser.add_argument("--wm_uncertainty_weight", type=float, default=None)
    
    # Sequential training (can override config)
    parser.add_argument("--sequential_training", action="store_true", default=None)
    parser.add_argument("--no_sequential_training", action="store_false", 
                       dest="sequential_training")
    
    # Device & debug
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint directory to resume training from")
    
    return parser.parse_args()


class StudentTrainer(BaseRLTrainer):
    """
    Trainer for student policy (Explorer).
    
    Uses PPO-style policy gradient with custom rewards:
    1. Memory divergence: change in teacher's memory state
    2. WM uncertainty: prediction error (actions that surprise WM)
    """
    
    def __init__(
        self,
        student_policy: nn.Module,
        teacher_policy: nn.Module,
        policy_config,
        rl_config: OmegaConf,
        device: str = "cuda",
    ):
        # Get training config
        train_config = get_training_config(rl_config)
        student_config = rl_config.get("student", {})
        output_dir = student_config.get("output_dir", "./outputs/student_rl")
        
        super().__init__(
            policy=student_policy,
            config=train_config,
            output_dir=output_dir,
            device=device,
        )
        
        self.student_policy = student_policy
        self.teacher_policy = teacher_policy
        self.policy_config = policy_config
        self.rl_config = rl_config
        
        # Freeze teacher
        self.teacher_policy.eval()
        for param in self.teacher_policy.parameters():
            param.requires_grad = False
        
        # World model image steps
        self.cur_n_obs_img_steps = train_config.n_obs_img_steps
        self.cur_n_pred_img_steps = train_config.n_pred_img_steps
        
        # Reward weights
        rewards_config = student_config.get("rewards", {})
        self.memory_divergence_weight = rewards_config.get("memory_divergence_weight", 0.5)
        self.wm_uncertainty_weight = rewards_config.get("wm_uncertainty_weight", 0.5)
        
        # PPO parameters
        ppo_config = student_config.get("ppo", {})
        self.clip_epsilon = ppo_config.get("clip_epsilon", 0.2)
        self.entropy_coef = ppo_config.get("entropy_coef", 0.01)
        self.value_loss_coef = ppo_config.get("value_loss_coef", 0.5)
        self.gamma = ppo_config.get("gamma", 0.99)
        self.gae_lambda = ppo_config.get("gae_lambda", 0.95)
        
        # Setup policy for training
        self.student_policy.train()
        
        # Setup optimizer and scheduler
        trainable, total = count_trainable_params(self.student_policy)
        logger.info(f"Student trainable parameters: {trainable:,} / {total:,}")
        
        self.optimizer = setup_optimizer(
            self.student_policy,
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
        
        # Setup batch builder with wrist cameras
        self.batch_builder = BatchBuilder(
            device=device,
            image_keys=["left_wrist_rgb", "right_wrist_rgb"],
        )
        
        # Additional batch builder for teacher (head camera)
        self.teacher_batch_builder = BatchBuilder(
            device=device,
            image_keys=["head_rgb"],
        )
        
        # Memory managers for student and teacher
        self.student_memory = MemoryStateManager()
        self.teacher_memory = MemoryStateManager()
        self.prev_teacher_memory: Optional[torch.Tensor] = None
        
        # Value head for PPO
        self.value_head = nn.Linear(policy_config.state_dim, 1).to(device)
        self.log_std = nn.Parameter(torch.zeros(train_config.action_dim, device=device))
        
        # Additional metrics
        self.metrics.update({
            "policy_loss": deque(maxlen=100),
            "value_loss": deque(maxlen=100),
            "entropy": deque(maxlen=100),
            "memory_divergence": deque(maxlen=100),
            "wm_uncertainty": deque(maxlen=100),
            "episode_reward": deque(maxlen=100),
        })
    
    def setup_environment(self):
        """Setup the RL environment."""
        from rl.f1_rl_env import StudentEnv
        
        self.env = StudentEnv(
            task_config=self.env_config,
            teacher_policy=self.teacher_policy,
            history_length=self.config.history_length,
            max_steps=self.config.steps_per_episode,
            device=self.device,
            action_scale=self.config.action_scale,
        )
        logger.info(f"Student environment setup complete")
    
    def collect_episode(self) -> List[Dict[str, Any]]:
        """Collect one episode using student policy."""
        obs, info = self.env.reset()
        transitions = []
        
        # Reset memory states
        self.student_memory.reset()
        self.teacher_memory.reset()
        self.prev_teacher_memory = None
        
        done = False
        step = 0
        
        while not done:
            # Build batch for student policy
            batch = self._obs_to_batch(obs)
            
            # Inject student memory state
            if self.student_memory.current_memory is not None:
                batch["initial_memory_state"] = self.student_memory.current_memory
            
            # Get action from student policy
            with torch.no_grad():
                actions, log_probs, values = self._forward_student(batch)
            
            action = actions[0].cpu().numpy()
            log_prob = log_probs[0].item()
            value = values[0].item()
            
            # Random action for first step
            if step == 0:
                action = np.random.uniform(-1, 1, self.config.action_dim).astype(np.float32)
            
            # Execute action
            next_obs, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Compute custom reward
            reward, reward_info = self._compute_custom_reward(obs, batch)
            
            transitions.append({
                "obs": obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "info": {**info, **reward_info},
            })
            
            obs = next_obs
            step += 1
        
        # Compute advantages using GAE
        self._compute_advantages(transitions)
        
        return transitions
    
    def _obs_to_batch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to batch tensor dict."""
        batch = {
            "observation.state": torch.from_numpy(obs["state"]).float().unsqueeze(0).to(self.device),
            "action_history": torch.from_numpy(obs["action_history"]).float().unsqueeze(0).to(self.device),
            "task": ["explore the environment\n"],
        }
        
        # Wrist images (student observation)
        if "left_wrist_rgb" in obs:
            imgs = obs["left_wrist_rgb"][-1]
            imgs = torch.from_numpy(imgs).float().to(self.device) / 255.0 * 2.0 - 1.0
            batch["observation.images.image1"] = imgs.unsqueeze(0)
            batch["observation.images.image1_mask"] = torch.ones(1, dtype=torch.bool, device=self.device)
        
        if "right_wrist_rgb" in obs:
            imgs = obs["right_wrist_rgb"][-1]
            imgs = torch.from_numpy(imgs).float().to(self.device) / 255.0 * 2.0 - 1.0
            batch["observation.images.image2"] = imgs.unsqueeze(0)
            batch["observation.images.image2_mask"] = torch.ones(1, dtype=torch.bool, device=self.device)
        
        # Head images (for teacher WM)
        if "head_rgb" in obs:
            imgs = obs["head_rgb"][-1]
            imgs = torch.from_numpy(imgs).float().to(self.device) / 255.0 * 2.0 - 1.0
            batch["observation.images.image0"] = imgs.unsqueeze(0)
            batch["observation.images.image0_mask"] = torch.ones(1, dtype=torch.bool, device=self.device)
            batch["observation.images.image0_history"] = (
                torch.from_numpy(obs["head_rgb"]).float().to(self.device) / 255.0 * 2.0 - 1.0
            ).unsqueeze(0)
        
        return batch
    
    def _forward_student(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through student model."""
        # Use student model to generate actions
        noise = torch.randn(
            1, self.policy_config.chunk_size,
            self.policy_config.max_action_dim,
            device=self.device
        )
        
        # Sample actions from student model
        action_output = self.student_policy.sample_actions_with_world_model(
            images=[batch.get("observation.images.image1")],
            image_masks=[batch.get("observation.images.image1_mask", 
                        torch.ones(1, dtype=torch.bool, device=self.device))],
            lang_tokens=None,
            lang_masks=None,
            state=batch["observation.state"],
            world_model_input_embs=None,
            predict_action_only=True,
            noise=noise,
            action_history=batch.get("action_history"),
            initial_memory_state=batch.get("initial_memory_state"),
        )
        
        action_mean = action_output[:, 0, :]
        std = torch.exp(self.log_std)
        
        if deterministic:
            actions = action_mean
            log_probs = torch.zeros(1, device=self.device)
        else:
            dist = torch.distributions.Normal(action_mean, std)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Value estimate
        state_emb = self.student_policy.state_proj(batch["observation.state"])
        values = self.value_head(state_emb).squeeze(-1)
        
        return actions, log_probs, values
    
    def _compute_custom_reward(
        self,
        obs: Dict[str, np.ndarray],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute custom reward for student exploration."""
        # Inject teacher memory state
        teacher_batch = batch.copy()
        if self.teacher_memory.current_memory is not None:
            teacher_batch["initial_memory_state"] = self.teacher_memory.current_memory
        
        # Get teacher's world model prediction
        with torch.no_grad():
            wm_output = self.teacher_policy.forward_with_world_model(
                teacher_batch,
                cur_n_obs_img_steps=self.cur_n_obs_img_steps,
                cur_n_pred_img_steps=self.cur_n_pred_img_steps,
                train_gen_expert_only=True,
            )
        
        # Update teacher memory
        teacher_memory = wm_output.get("memory_state")
        if teacher_memory is not None:
            self.teacher_memory.update(teacher_memory)
        
        # Memory divergence reward
        memory_divergence = 0.0
        if self.prev_teacher_memory is not None and teacher_memory is not None:
            memory_change = teacher_memory - self.prev_teacher_memory
            memory_divergence = torch.norm(memory_change).item()
        
        self.prev_teacher_memory = teacher_memory.detach() if teacher_memory is not None else None
        
        # WM uncertainty
        wm_logits = wm_output.get("wm_logits", torch.zeros(1, device=self.device))
        wm_probs = F.softmax(wm_logits, dim=-1)
        wm_entropy = -torch.sum(wm_probs * torch.log(wm_probs + 1e-8), dim=-1)
        wm_uncertainty = wm_entropy.mean().item()
        
        # Combined reward
        reward = (
            self.memory_divergence_weight * memory_divergence +
            self.wm_uncertainty_weight * wm_uncertainty
        )
        
        return reward, {
            "memory_divergence": memory_divergence,
            "wm_uncertainty": wm_uncertainty,
        }
    
    def _compute_advantages(self, transitions: List[Dict[str, Any]]):
        """Compute GAE advantages for PPO."""
        rewards = [t["reward"] for t in transitions]
        values = [t["value"] for t in transitions]
        dones = [t["done"] for t in transitions]
        
        # Bootstrap value
        next_value = 0.0 if dones[-1] else values[-1]
        
        # GAE
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(transitions))):
            next_val = next_value if i == len(transitions) - 1 else values[i + 1]
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        # Store in transitions
        for i, t in enumerate(transitions):
            t["advantage"] = advantages[i]
            t["return"] = advantages[i] + values[i]
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one PPO training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        actions, log_probs, values = self._forward_student(batch)
        
        # PPO loss
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        # Policy loss (clipped)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        std = torch.exp(self.log_std)
        entropy = 0.5 * (1 + torch.log(2 * np.pi * std**2)).sum()
        
        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        # Backward
        loss.backward()
        clip_gradients(self.student_policy, max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }
    
    def train(self, start_episode: int = 0):
        """Main training loop."""
        num_episodes = self.config.num_episodes
        
        logger.info(f"Starting student training for {num_episodes} episodes")
        logger.info(f"Starting from episode: {start_episode}")
        
        # Setup environment
        self.setup_environment()
        
        for episode in range(start_episode, num_episodes):
            self.metrics["episode"] = episode
            
            # Collect episode
            transitions = self.collect_episode()
            
            # Track episode reward
            episode_reward = sum(t["reward"] for t in transitions)
            self.metrics["episode_reward"].append(episode_reward)
            
            # Track reward components
            avg_memory_div = np.mean([t["info"]["memory_divergence"] for t in transitions])
            avg_wm_unc = np.mean([t["info"]["wm_uncertainty"] for t in transitions])
            self.metrics["memory_divergence"].append(avg_memory_div)
            self.metrics["wm_uncertainty"].append(avg_wm_unc)
            
            # Build PPO batch
            batch = self._build_ppo_batch(transitions)
            
            # Multiple PPO epochs
            for _ in range(4):  # PPO epochs
                loss_dict = self.train_step(batch)
                self.metrics["policy_loss"].append(loss_dict["policy_loss"])
                self.metrics["value_loss"].append(loss_dict["value_loss"])
                self.metrics["entropy"].append(loss_dict["entropy"])
            
            self.metrics["total_steps"] += len(transitions)
            self.scheduler.step()
            
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
                        "value_head": self.value_head.state_dict(),
                        "log_std": self.log_std.data,
                    }
                )
        
        # Final save
        self.save_checkpoint(num_episodes)
        logger.info("Training complete!")
    
    def _build_ppo_batch(self, transitions: List[Dict]) -> Dict[str, torch.Tensor]:
        """Build PPO training batch."""
        batch = {
            "observation.state": torch.stack([
                torch.from_numpy(t["obs"]["state"]).float() 
                for t in transitions
            ]).to(self.device),
            "old_log_probs": torch.tensor(
                [t["log_prob"] for t in transitions],
                device=self.device
            ),
            "advantages": torch.tensor(
                [t["advantage"] for t in transitions],
                device=self.device
            ),
            "returns": torch.tensor(
                [t["return"] for t in transitions],
                device=self.device
            ),
        }
        
        # Normalize advantages
        batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (
            batch["advantages"].std() + 1e-8
        )
        
        return batch
    
    def _log_episode_metrics(self, episode: int):
        """Log training metrics."""
        self.log_metrics(episode, {
            "policy_loss": np.mean(self.metrics["policy_loss"]) if self.metrics["policy_loss"] else 0,
            "value_loss": np.mean(self.metrics["value_loss"]) if self.metrics["value_loss"] else 0,
            "entropy": np.mean(self.metrics["entropy"]) if self.metrics["entropy"] else 0,
            "memory_divergence": np.mean(self.metrics["memory_divergence"]) if self.metrics["memory_divergence"] else 0,
            "wm_uncertainty": np.mean(self.metrics["wm_uncertainty"]) if self.metrics["wm_uncertainty"] else 0,
            "episode_reward": np.mean(self.metrics["episode_reward"]) if self.metrics["episode_reward"] else 0,
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
    if args.output_dir is not None:
        rl_config.student.output_dir = args.output_dir
    if args.memory_divergence_weight is not None:
        rl_config.student.rewards.memory_divergence_weight = args.memory_divergence_weight
    if args.wm_uncertainty_weight is not None:
        rl_config.student.rewards.wm_uncertainty_weight = args.wm_uncertainty_weight
    if args.sequential_training is not None:
        rl_config.training.sequential_training = args.sequential_training
    if args.device is not None:
        rl_config.device = args.device
    if args.debug is not None:
        rl_config.debug = args.debug
    
    device = rl_config.get("device", "cuda")
    debug = rl_config.get("debug", False)
    
    logger.info("=" * 70)
    logger.info("Student Policy Training (Phase 2)")
    logger.info("=" * 70)
    
    # Load model config file path
    model_config_file = rl_config.get("model", {}).get(
        "config_file",
        "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml"
    )
    
    # Get LoRA config
    lora_config = get_lora_config_from_dict(rl_config)
    
    # Load teacher policy (frozen)
    logger.info(f"Loading teacher policy from: {args.teacher_path}")
    teacher_policy, _, _ = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=lora_config,
        checkpoint_path=args.teacher_path,
    )
    
    # Load student policy (trainable)
    student_policy, policy_config, model_config = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=lora_config,
    )
    
    logger.info("Models loaded successfully")
    
    # Create trainer
    trainer = StudentTrainer(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
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
