#!/usr/bin/env python3
"""
Teacher Policy Training Script for F1-VLA (Phase 1)

Phase 1 Training (Supervised Learning):
- Train World Model to predict next frame observation
- Input: history images, actions, states
- Output: predicted next frame image tokens
- Loss: cross-entropy on VQ-VAE image tokens

Note: This is supervised learning, not reinforcement learning.
Random actions are used only for data collection/exploration.
"""

# ============== MUST BE FIRST: Suppress warnings before ANY imports ==============
import os
import sys
import warnings

# Set matplotlib backend BEFORE importing matplotlib (avoids IPython issues)
os.environ["MPLBACKEND"] = "Agg"

# Set environment variables BEFORE any torch/cuda imports
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")
os.environ.setdefault("CUROBO_LOG_LEVEL", "ERROR")

# Suppress warnings BEFORE importing packages that trigger them
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============== Setup paths BEFORE importing other modules ==============
script_dir = os.path.dirname(os.path.abspath(__file__))  # rl/training
rl_dir = os.path.dirname(script_dir)                      # rl
robotwin_dir = os.path.dirname(rl_dir)                    # RoboTwin
f1_vla_dir = os.path.dirname(robotwin_dir)                # F1-VLA
sys.path.insert(0, f1_vla_dir)
sys.path.insert(0, robotwin_dir)

# Import log suppression module (must be before any CuRobo imports)
from rl.suppress_logs import suppress_curobo_logs

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image

from PIL import Image

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
    setup_optimizer,
    setup_scheduler,
    clip_gradients,
    count_trainable_params,
    setup_logging_from_config,
    set_policy_requires_grad,
)

# Default logging (will be overridden by config)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train World Model (Phase 1) - Supervised Learning")
    
    # Config file
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to training config YAML file")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Override model config file path")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    
    # Device & debug
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint directory to resume from")
    
    return parser.parse_args()


class WorldModelTrainer:
    """
    Trainer for World Model (supervised learning).
    
    Training loop:
    1. Collect trajectory with random actions (for exploration)
    2. Feed observation + action history to world model
    3. Predict next observation tokens
    4. Compute cross-entropy loss on predicted vs actual tokens
    5. Update model parameters via gradient descent
    """
    
    def __init__(
        self,
        policy: nn.Module,
        policy_config,
        rl_config: OmegaConf,
        model_config_file: str,
        device: str = "cuda",
    ):
        self.policy = policy
        self.policy_config = policy_config
        self.rl_config = rl_config
        self.device = device
        
        # Load model config (debug_test.yaml) to get n_obs_img_steps and obs_img_stride
        import yaml
        with open(model_config_file, 'r') as f:
            model_cfg = yaml.safe_load(f)
        
        # Extract n_obs_img_steps and obs_img_stride from first dataset
        train_datasets = model_cfg.get('dataset', {}).get('train_dir', {})
        if not train_datasets:
            raise ValueError("No train datasets found in model config")
        
        first_dataset = next(iter(train_datasets.values()))
        self.n_obs_img_steps = first_dataset.get('n_obs_img_steps', 4)
        self.obs_img_stride = first_dataset.get('obs_img_stride', 1)
        
        # Get training config
        self.config = get_training_config(rl_config)
        self.n_pred_img_steps = self.config.n_pred_img_steps
        
        # history_length is the observation buffer size (n_obs_img_steps)
        # The prediction target (next_obs) will be appended separately in batch building
        self.history_length = self.n_obs_img_steps
        
        print(f"World model config from {model_config_file}:")
        print(f"  n_obs_img_steps: {self.n_obs_img_steps} (input frames)")
        print(f"  n_pred_img_steps: {self.n_pred_img_steps} (prediction frames)")
        print(f"  history_length: {self.history_length} (observation buffer)")
        print(f"  obs_img_stride: {self.obs_img_stride}")
        
        teacher_config = rl_config.get("teacher", {})
        self.output_dir = Path(teacher_config.get("output_dir", "./outputs/teacher_rl"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup policy for training - configure to train world model only
        self.policy.train()
        
        # Set gradient flags: train world model (gen expert) only
        # freeze_vision_encoder=True: saves ~30-40% VRAM, 2-3x faster, PaliGemma already strong
        print("\nConfiguring training mode: World Model only")
        set_policy_requires_grad(
            self.policy,
            freeze_vision_encoder=True,   # Freeze vision encoder (recommended for all phases)
            freeze_gen_expert=False,
            train_act_expert_only=False,
            train_gen_expert_only=True,   # Only train world model
        )
        
        # Setup optimizer and scheduler
        trainable, total = count_trainable_params(self.policy)
        print(f"Trainable parameters: {trainable:,} / {total:,}")
        
        self.optimizer = setup_optimizer(
            self.policy,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = setup_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            T_max=self.config.num_episodes * self.config.steps_per_episode,
            eta_min=1e-6,
        )
        
        # Environment config
        self.env_config = get_environment_config(rl_config)
        
        # Batch size for training (larger batch = better GPU utilization)
        self.batch_size = self.config.batch_size
        
        # Replay buffer for accumulating transitions
        self.replay_buffer = deque(maxlen=10000)  # Store up to 10K transitions
        
        # Batch builder - teacher uses head + wrist cameras
        self.batch_builder = BatchBuilder(
            device=device,
            image_keys=["head_rgb", "wrist_rgb"],
            use_head_camera=True,  # Teacher: head_rgb (image0) + wrist_rgb (image1)
        )
        
        # Memory state manager (for sequential processing)
        self.memory_manager = MemoryStateManager()
        
        # Training metrics
        self.metrics = {
            "wm_loss": deque(maxlen=100),
            "wm_accuracy": deque(maxlen=100),
            "total_steps": 0,
        }
        
        # Environment (lazy init)
        self.env = None
        # Ensure output directories for logging and samples
        self.metrics_log_path = self.output_dir / "episode_metrics.jsonl"
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
    
    def setup_environment(self):
        """Setup the simulation environment."""
        print("Setting up environment...")
        from rl.f1_rl_env import TeacherEnv
        
        # Get single_arm and scene_reset_interval from config
        single_arm = self.env_config.get("single_arm", False)
        scene_reset_interval = self.env_config.get("scene_reset_interval", 1)
        
        self.env = TeacherEnv(
            task_config=self.env_config,
            history_length=self.history_length,  # Use calculated history_length
            max_steps=self.config.steps_per_episode,
            device=self.device,
            action_scale=self.config.action_scale,
            single_arm=single_arm,
            scene_reset_interval=scene_reset_interval,
        )
        print(f"Environment ready! single_arm={single_arm}, scene_reset_interval={scene_reset_interval}")
        logger.debug(f"Environment ready (action_scale={self.config.action_scale}, single_arm={single_arm})")
    
    def collect_trajectory(self) -> List[Dict[str, Any]]:
        """
        Collect one trajectory with random actions.
        Returns list of (obs, action, next_obs) tuples for training.
        """
        obs, _ = self.env.reset()
        trajectory = []
        done = False
        
        while not done:
            # Random action for exploration
            action = np.random.uniform(-1, 1, self.config.action_dim).astype(np.float32)
            next_obs, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            trajectory.append({
                "obs": obs,
                "action": info.get("action_executed", action),
                "next_obs": next_obs,
            })
            obs = next_obs
        
        return trajectory
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        # Ensure model is in training mode (required for CuDNN RNN backward)
        self.policy.train()
        
        # Forward pass - predict next frame tokens
        loss_dict = self.policy.forward_with_world_model(
            batch,
            cur_n_obs_img_steps=self.n_obs_img_steps,
            cur_n_pred_img_steps=self.n_pred_img_steps,
            train_gen_expert_only=True,  # Only train world model
        )
        
        # Backward pass
        loss = loss_dict["loss"]
        loss.backward()
        
        # Gradient clipping
        clip_gradients(self.policy, max_norm=self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss": loss.item(),
            "accuracy": loss_dict.get("wm_acc_mean", torch.tensor(0.0)).item(),
        }
    
    def train(self, start_episode: int = 0):
        """Main training loop with tqdm progress bar."""
        from tqdm import tqdm
        import time
        
        num_episodes = self.config.num_episodes
        
        # Setup environment (suppress the log, tqdm will show progress)
        self.setup_environment()
        
        # Create progress bar
        pbar = tqdm(
            range(start_episode, num_episodes),
            desc="Training",
            unit="ep",
            ncols=120,
            initial=start_episode,
            total=num_episodes,
        )
        
        start_time = time.time()
        
        for episode in pbar:
            # Collect trajectory and add to replay buffer
            trajectory = self.collect_trajectory()
            self.replay_buffer.extend(trajectory)
            
            # Train on mini-batches from replay buffer
            ep_loss = []
            ep_acc = []
            
            # Number of training steps per episode
            num_train_steps = max(1, len(trajectory) // self.batch_size)
            
            for _ in range(num_train_steps):
                # Sample mini-batch from replay buffer
                if len(self.replay_buffer) >= self.batch_size:
                    # Random sample
                    indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
                    batch_transitions = [self.replay_buffer[i] for i in indices]
                else:
                    # Use all available if buffer smaller than batch_size
                    batch_transitions = list(self.replay_buffer)
                
                batch = self.batch_builder.build_batch(
                    batch_transitions, include_memory_states=False
                )
                
                result = self.train_step(batch)
                
                ep_loss.append(result["loss"])
                ep_acc.append(result["accuracy"])
                self.metrics["wm_loss"].append(result["loss"])
                self.metrics["wm_accuracy"].append(result["accuracy"])
                self.metrics["total_steps"] += 1
            
            # Update progress bar with current metrics
            avg_loss = np.mean(ep_loss) if ep_loss else 0
            avg_acc = np.mean(ep_acc) if ep_acc else 0
            lr = self.scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            fps = self.metrics["total_steps"] / elapsed if elapsed > 0 else 0
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.3f}",
                "acc": f"{avg_acc:.3f}",
                "lr": f"{lr:.1e}",
                "steps": self.metrics["total_steps"],
                "fps": f"{fps:.1f}",
            })

            # Log metrics and save image samples periodically
            log_every = getattr(self.config, "log_every", None) or self.rl_config.get("training", {}).get("log_every", None)
            # Default to 10 if not set
            if log_every is None:
                log_every = 10
            if (episode + 1) % int(log_every) == 0:
                # Episode reward may be tracked by env
                episode_reward = getattr(self.env, "episode_reward", None)
                metrics_entry = {
                    "episode": int(episode),
                    "avg_loss": float(avg_loss),
                    "avg_acc": float(avg_acc),
                    "episode_reward": float(episode_reward) if episode_reward is not None else None,
                    "total_steps": int(self.metrics["total_steps"]),
                    "lr": float(lr),
                    "timestamp": time.time(),
                }
                try:
                    # Append JSON line
                    with open(self.metrics_log_path, "a") as fh:
                        fh.write(json.dumps(metrics_entry) + "\n")
                except Exception:
                    logger.exception("Failed to write metrics log")

                # Save one sample comparison image (pred vs gt) using the last transition if available
                try:
                    if len(trajectory) > 0:
                        last = trajectory[-1]
                        obs_before = last["obs"]
                        action = last.get("action")
                        obs_after = last.get("next_obs")
                        if obs_before is not None and obs_after is not None:
                            self._save_prediction_sample(episode + 1, obs_before, action, obs_after)
                except Exception:
                    logger.exception("Failed to save sample image")

            # Save checkpoint periodically
            if (episode + 1) % self.config.save_every == 0:
                pbar.write(f"[Checkpoint] Saving episode {episode+1}...")
                self.save_checkpoint(episode + 1)
        
        pbar.close()
        
        # Final save
        self.save_checkpoint(num_episodes)
        print("\nTraining complete!")
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{episode}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.policy.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer & scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "episode": episode,
            "total_steps": self.metrics["total_steps"],
        }, checkpoint_dir / "training_state.pt")
        
        logger.debug(f"Checkpoint saved: {checkpoint_dir}")

    def _save_prediction_sample(self, episode: int, obs_before: Dict[str, Any], action: Any, obs_after: Dict[str, Any]):
        """Run the world model to predict next image and save a side-by-side comparison with ground truth.

        Saves to `self.output_dir / 'samples'` with filename containing episode number.
        """
        try:
            # Build batch using environment helper (use head camera)
            # action should be numpy array of raw/normalized action; env helper expects raw action
            batch = self.env._build_policy_batch(obs_before, np.array(action, dtype=np.float32), use_head_camera=True)

            # Fix batch observation shape if needed: [1, history, C, H, W] -> [history, C, H, W]
            if "observation.images.head_camera" in batch:
                obs_imgs = batch["observation.images.head_camera"]
                if obs_imgs.ndim == 5 and obs_imgs.shape[0] == 1:
                    batch["observation.images.head_camera"] = obs_imgs.squeeze(0)

            # Use policy to predict images only
            self.policy.eval()  # Set to eval mode for inference
            with torch.no_grad():
                pred_out = self.policy.predict_images_only(batch)
            pred_imgs = pred_out.get("pred_imgs")  # Tensor [B, C, H, W] or [B, T, C, H, W]
            if pred_imgs is None:
                return

            # Use first sample in batch
            pred = pred_imgs.detach().cpu()
            if pred.ndim == 5:
                # [B, T, C, H, W] -> take last predicted frame
                pred = pred[:, -1]
            pred = pred[0]  # [C, H, W]

            # Ground truth image (head camera from obs_after)
            gt = obs_after.get("head_rgb")
            if gt is None:
                gt = obs_after.get("wrist_rgb")
            if gt is None:
                return
            # gt is CHW (as in env._get_raw_observation produced CHW); convert to HWC uint8
            if isinstance(gt, np.ndarray):
                if gt.shape[0] == 3:
                    gt_img = np.transpose(gt, (1, 2, 0)).astype(np.uint8)
                else:
                    gt_img = gt.astype(np.uint8)
            else:
                # Fallback
                gt_img = np.array(gt).astype(np.uint8)

            # Convert pred from [-1,1] to uint8 HWC
            pred_img = ((pred + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
            pred_img = np.transpose(pred_img, (1, 2, 0)) * 255.0
            pred_img = pred_img.astype(np.uint8)

            # Compose side-by-side image using PIL (avoid matplotlib IPython issues)
            # Resize both images to same height for side-by-side comparison
            gt_pil = Image.fromarray(gt_img)
            pred_pil = Image.fromarray(pred_img)
            
            # Create combined image
            h = max(gt_pil.height, pred_pil.height)
            w = gt_pil.width + pred_pil.width + 20  # 20px gap
            combined = Image.new('RGB', (w, h), color=(255, 255, 255))
            combined.paste(gt_pil, (0, 0))
            combined.paste(pred_pil, (gt_pil.width + 20, 0))
            
            out_path = self.output_dir / "samples" / f"episode_{episode:06d}.png"
            combined.save(out_path)

        except Exception:
            logger.exception("Error while saving prediction sample")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return starting episode."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.metrics["total_steps"] = state.get("total_steps", 0)
            return state.get("episode", 0)
        
        return 0


def main():
    args = parse_args()
    
    # Load config
    rl_config = load_rl_config(args.rl_config)
    
    # Setup logging
    setup_logging_from_config(rl_config)
    
    # Apply command-line overrides
    if args.model_config:
        rl_config.model.config_file = args.model_config
    if args.num_episodes is not None:
        rl_config.training.num_episodes = args.num_episodes
    if args.steps_per_episode is not None:
        rl_config.training.steps_per_episode = args.steps_per_episode
    if args.learning_rate is not None:
        rl_config.training.learning_rate = args.learning_rate
    if args.output_dir is not None:
        rl_config.teacher.output_dir = args.output_dir
    if args.save_every is not None:
        rl_config.training.save_every = args.save_every
    if args.log_every is not None:
        rl_config.training.log_every = args.log_every
    if args.device is not None:
        rl_config.device = args.device
    if args.debug is not None:
        rl_config.debug = args.debug
    
    device = rl_config.get("device", "cuda")
    debug = rl_config.get("debug", False)
    
    # Print startup info (use print for banner, it's cleaner)
    print("\n" + "=" * 60)
    print("World Model Training (Phase 1 - Supervised Learning)")
    print("=" * 60)
    
    # Load model config
    model_config_file = rl_config.get("model", {}).get(
        "config_file", 
        "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml"
    )
    print(f"Loading config from: {model_config_file}")
    
    # Get LoRA config
    lora_config = get_lora_config_from_dict(rl_config)
    print("Loading policy...")
    
    # Load policy
    policy, policy_config, model_config = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=debug,
        lora_config=lora_config,
    )
    
    print("Model loaded successfully")
    
    # Create trainer
    trainer = WorldModelTrainer(
        policy=policy,
        policy_config=policy_config,
        rl_config=rl_config,
        model_config_file=model_config_file,
        device=device,
    )
    
    # Resume if specified
    start_episode = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        start_episode = trainer.load_checkpoint(args.resume)
    
    print("")  # Empty line before training
    
    # Train
    trainer.train(start_episode=start_episode)


if __name__ == "__main__":
    main()
