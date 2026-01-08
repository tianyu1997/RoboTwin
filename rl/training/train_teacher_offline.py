#!/usr/bin/env python3
"""
Offline Teacher Training Script for F1-VLA (Phase 1)

This script trains the World Model using offline data collected by collect_data_teacher.py.

Usage:
    accelerate launch train_teacher_offline.py --data_dir ./data/teacher_offline --output_dir ./outputs/teacher_offline
"""

import os
import sys
import argparse
import logging
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
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
    get_training_config,
    adjust_config_for_ddp,
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
    resolve_device_and_process,
    print_startup_header,
    create_summary_writer,
    add_video_to_writer,
)
from rl.training.parallel_utils import (
    AcceleratorWrapper,
    create_accelerator,
    SequentialEpisodeBuffer,
    WeightedEpisodeBuffer,
    print_rank0,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train World Model (Offline)")
    
    parser.add_argument("--rl_config", type=str,
                       default="/mnt/data2/ty/F1-VLA/RoboTwin/rl/rl_config.yaml",
                       help="Path to RL training config YAML file")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Override model config file path")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing collected .pt episode files (optional if datasets defined in config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save checkpoints")
    
    # Training params
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of training epochs over the dataset")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--save_every", type=int, default=None, help="Save every N epochs")
    
    # Distributed
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=1, help="Ignored in offline mode (compatibility)")
    
    # Device
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    
    # Visualization
    parser.add_argument("--vis_every_episodes", type=int, default=None,
                       help="Visualize every N episodes (default: use config or 20)")
    
    return parser.parse_args()

class OfflineTrainer:
    def __init__(
        self,
        policy: torch.nn.Module,
        policy_config,
        rl_config: OmegaConf,
        model_config_file: str,
        data_dir: Optional[str] = None,
        device: str = "cuda",
        accelerator: Optional[AcceleratorWrapper] = None,
    ):
        self.policy = policy
        self.policy_config = policy_config
        self.rl_config = rl_config
        self.data_dir = Path(data_dir) if data_dir else None
        self.device = device
        self.accelerator = accelerator
        
        # Output dir
        teacher_config = rl_config.get("teacher", {})
        self.output_dir = Path(teacher_config.get("output_dir", "./outputs/teacher_offline"))
        if self._is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Load model config for image steps
        import yaml
        with open(model_config_file, 'r') as f:
            model_cfg = yaml.safe_load(f)
            
        train_datasets = model_cfg.get('dataset', {}).get('train_dir', {})
        first_dataset = next(iter(train_datasets.values()))
        self.n_obs_img_steps = first_dataset.get('n_obs_img_steps', 4)
        
        self.config = get_training_config(rl_config)
        adjust_config_for_ddp(self.config, self.accelerator)
        self.n_pred_img_steps = self.config.n_pred_img_steps
        
        # Memory config
        self.memory_enabled = policy_config.memory_enabled if hasattr(policy_config, 'memory_enabled') else True
        self.memory_hidden = policy_config.memory_hidden if hasattr(policy_config, 'memory_hidden') else 2048
        self.memory_num_layers = policy_config.memory_num_layers if hasattr(policy_config, 'memory_num_layers') else 4
        
        # Setup policy
        self.policy.train()
        set_policy_requires_grad(
            self.policy,
            freeze_vision_encoder=True,
            freeze_gen_expert=False,
            train_act_expert_only=False,
            train_gen_expert_only=True,
        )
        
        # Print trainable params
        trainable, total = count_trainable_params(self.policy)
        self._print(f"Trainable parameters: {trainable:,} / {total:,}")
        
        # Optimizer
        self.optimizer = setup_optimizer(
            self.policy,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # Scheduler (will be set in train() once we know total steps)
        self.scheduler = None
        
        if self.accelerator:
            self.policy, self.optimizer = self.accelerator.prepare(self.policy, self.optimizer)
            
        # Buffer setup
        self.datasets_config = teacher_config.get("datasets", None)
        if self.datasets_config:
            self._print(f"Using mixed datasets: {self.datasets_config}")
            buffers = [SequentialEpisodeBuffer(max_episodes=10000, max_transitions=1000000) for _ in self.datasets_config]
            weights = [d.get("weight", 1.0) for d in self.datasets_config]
            self.replay_buffer = WeightedEpisodeBuffer(buffers, weights)
        else:
            self.replay_buffer = SequentialEpisodeBuffer(max_episodes=10000, max_transitions=1000000)
        
        # Batch builder
        self.batch_builder = BatchBuilder(
            device=self.device,
            image_keys=["head_rgb", "wrist_rgb"],
            use_head_camera=True,
        )
        
        self.memory_manager = MemoryStateManager()
        self.metrics = {
            "wm_loss": deque(maxlen=100), 
            "wm_accuracy": deque(maxlen=100), 
            "total_steps": 0,
            "total_transitions": 0,
            "total_episodes": 0,
        }
        
        # Tensorboard
        self.writer = create_summary_writer(self.output_dir, self._is_main_process())
        
        # Sequential training config
        train_cfg = rl_config.get("training", {})
        self.sequential_training = train_cfg.get("sequential_training", False)
        self.bptt_length = train_cfg.get("bptt_length", 8)
        self.memory_backprop = train_cfg.get("memory_backprop", False)
        
        if self.memory_backprop:
            policy_unwrapped = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
            if hasattr(policy_unwrapped, 'model') and hasattr(policy_unwrapped.model, 'memory_backprop'):
                policy_unwrapped.model.memory_backprop = True

    def _is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    def _print(self, msg: str):
        print_rank0(msg, self.accelerator)

    def load_data(self):
        """
        Load .pt files from data_dir(s) into replay buffer.
        In DDP mode, only loads the subset of files assigned to this process.
        """
        if self.datasets_config:
            total_count = 0
            for i, ds_cfg in enumerate(self.datasets_config):
                path = Path(ds_cfg["path"])
                self._print(f"Loading dataset {i} from {path} (weight={ds_cfg.get('weight', 1.0)})...")
                count = self._load_data_from_dir(path, buffer_idx=i)
                total_count += count
            self._print(f"Total loaded {total_count} episodes across all datasets.")
        else:
            if self.data_dir is None:
                raise ValueError("No data_dir provided and no datasets configured in rl_config.")
            self._print(f"Loading data from {self.data_dir}...")
            count = self._load_data_from_dir(self.data_dir)
            self._print(f"Loaded {count} episodes (per process). Buffer size: {len(self.replay_buffer)} transitions.")

    def _load_data_from_dir(self, data_dir: Path, buffer_idx: Optional[int] = None) -> int:
        files = sorted(list(data_dir.glob("episode_*.pt")))
        if not files:
            logger.warning(f"No episode_*.pt files found in {data_dir}")
            return 0
        
        # DDP Sharding: Only load files assigned to this process
        if self.accelerator and self.accelerator.is_distributed:
            total_files = len(files)
            files = files[self.accelerator.process_index::self.accelerator.num_processes]
            logger.info(f"Rank {self.accelerator.process_index}: Loading {len(files)}/{total_files} episodes from {data_dir}")
        
        count = 0
        for f in tqdm(files, desc=f"Loading {data_dir.name}", disable=not self._is_main_process()):
            try:
                episode = torch.load(f, weights_only=False)
                
                # Drop the last transition to avoid "teleporting" bug in old data
                # and because predicting beyond the final step is ambiguous.
                if len(episode) > 0:
                    episode = episode[:-1]
                
                if buffer_idx is not None:
                    self.replay_buffer.add_episode(episode, buffer_idx=buffer_idx)
                else:
                    self.replay_buffer.add_episode(episode)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        if self.accelerator and self.accelerator.is_distributed:
            self.accelerator.wait_for_everyone()
            
        return count

    def _init_memory_state(self, batch_size: int) -> Any:
        memory_type = getattr(self.policy_config, "memory_type", "gru")
        if memory_type == "kv":
            return None
            
        return torch.zeros(
            self.memory_num_layers, batch_size, self.memory_hidden,
            device=self.device, dtype=torch.float32
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        policy.train()
        
        batch_size = batch["observation.state"].shape[0]
        if "initial_memory_state" not in batch or batch["initial_memory_state"] is None:
            batch["initial_memory_state"] = self._init_memory_state(batch_size)
            
        loss_dict = policy.forward_with_world_model(
            batch,
            cur_n_obs_img_steps=self.n_obs_img_steps,
            cur_n_pred_img_steps=self.n_pred_img_steps,
            train_gen_expert_only=True,
        )
        
        output_memory_state = loss_dict.get("memory_state")
        if output_memory_state is not None:
            self.memory_manager.update(output_memory_state.detach())
            
        loss = loss_dict["loss"]
        if self.accelerator:
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        else:
            loss.backward()
            clip_gradients(self.policy, max_norm=self.config.max_grad_norm)
            
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            
        return {"loss": loss.item(), "accuracy": loss_dict.get("wm_acc_mean", torch.tensor(0.0)).item()}


    def train_step_sequential(self, sequences: List[List[Dict[str, Any]]], initial_memory_state: Optional[torch.Tensor] = None, episode_step_start: int = 0) -> Dict[str, Any]:
        # Simplified sequential step (similar to original but condensed)
        self.optimizer.zero_grad()
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        policy.train()
        
        batch_size = len(sequences)
        seq_length = len(sequences[0])
        
        # Init memory
        memory_state = None
        if initial_memory_state is not None:
            memory_state = initial_memory_state
        else:
            first_transitions = [seq[0] for seq in sequences]
            for t in first_transitions:
                if t.get("initial_memory_state") is not None:
                    ms = t["initial_memory_state"]
                    if isinstance(ms, torch.Tensor):
                        memory_state = ms.unsqueeze(0) if ms.dim() == 2 else ms
                        memory_state = memory_state.detach()
                        break
        
        if memory_state is None:
            memory_state = self._init_memory_state(batch_size)
        else:
            memory_state = memory_state.detach()
            
        total_loss = 0.0
        total_raw_loss = 0.0
        total_acc = 0.0
        valid_steps = 0
        
        output_memory_state = None
        
        for step_idx in range(seq_length):
            current_episode_step = episode_step_start + step_idx
            
            # Loss Schedule: Down-weight early steps in the episode
            warmup_steps = self.config.loss_warmup_steps
            start_weight = self.config.loss_warmup_start_weight
            
            if current_episode_step < warmup_steps:
                # Linear interpolation from start_weight to 1.0
                progress = current_episode_step / warmup_steps
                loss_weight = start_weight + (1.0 - start_weight) * progress
            else:
                loss_weight = 1.0
            
            step_transitions = [seq[step_idx] for seq in sequences]
            batch = self.batch_builder.build_batch(step_transitions, include_memory_states=True)
            batch["initial_memory_state"] = memory_state.to(self.device)
            
            loss_dict = policy.forward_with_world_model(
                batch,
                cur_n_obs_img_steps=self.n_obs_img_steps,
                cur_n_pred_img_steps=self.n_pred_img_steps,
                train_gen_expert_only=True,
            )
            
            output_memory_state = loss_dict.get("memory_state")
            if output_memory_state is not None:
                if self.memory_backprop and step_idx < seq_length - 1:
                    memory_state = output_memory_state
                else:
                    memory_state = output_memory_state.detach()
            
            # Apply weight
            step_loss = loss_dict["loss"] * loss_weight
            
            total_loss += step_loss
            total_raw_loss += loss_dict["loss"].item()
            total_acc += loss_dict.get("wm_acc_mean", torch.tensor(0.0)).item()
            valid_steps += 1
            
        avg_loss = total_loss / valid_steps if valid_steps > 0 else total_loss
        avg_raw_loss = total_raw_loss / valid_steps if valid_steps > 0 else 0.0
        
        if self.accelerator:
            self.accelerator.backward(avg_loss)
            self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        else:
            avg_loss.backward()
            clip_gradients(self.policy, max_norm=self.config.max_grad_norm)
            
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            
        if output_memory_state is not None:
            self.memory_manager.update(output_memory_state.detach())
            
        return {
            "loss": avg_raw_loss, 
            "weighted_loss": avg_loss.item(),
            "accuracy": total_acc / valid_steps if valid_steps > 0 else 0.0,
            "final_memory_state": output_memory_state.detach() if output_memory_state is not None else None
        }

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_path}")
            return 0
        
        self._print(f"Loading checkpoint from {checkpoint_path}")
        
        # Unwrap DDP model if needed
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        
        # Load model weights
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                policy.load_state_dict(state_dict, strict=False)
                self._print("Loaded model weights from model.pt")
            except Exception as e:
                logger.warning(f"Could not load model.pt: {e}")
        
        # Load trainer state
        trainer_state_path = checkpoint_path / "training_state.pt"
        start_epoch = 0
        if trainer_state_path.exists():
            try:
                state = torch.load(trainer_state_path, map_location=self.device, weights_only=False)
                
                if self.optimizer is not None and "optimizer" in state:
                    self.optimizer.load_state_dict(state["optimizer"])
                
                if self.scheduler is not None and "scheduler" in state:
                    self.scheduler.load_state_dict(state["scheduler"])
                
                self.metrics["total_steps"] = state.get("total_steps", 0)
                self.metrics["total_transitions"] = state.get("total_transitions", 0)
                self.metrics["total_episodes"] = state.get("total_episodes", 0)
                start_epoch = state.get("epoch", 0)
                
                self._print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Could not load trainer state: {e}")
        
        return start_epoch

    def find_latest_checkpoint(self) -> Optional[Path]:
        if not self.output_dir.exists():
            return None
        
        checkpoints = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-epoch-"):
                try:
                    epoch = int(item.name.split("-")[-1])
                    checkpoints.append((epoch, item))
                except (ValueError, IndexError):
                    continue
        
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]

    def visualize_episode(self, episode: List[Dict[str, Any]], step: int):
        """
        Visualize a full episode by predicting the next frame at each step.
        Generates a side-by-side video of Ground Truth vs Prediction.
        """
        if not self._is_main_process() or self.writer is None:
            return
            
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        policy.eval()
        
        pred_frames = []
        gt_frames = []
        
        # Initialize memory for this episode
        memory_state = self._init_memory_state(1) # Batch size 1
        
        try:
            with torch.no_grad():
                # Iterate through episode step-by-step to update memory correctly
                for i, transition in enumerate(episode):
                    # Build batch of size 1
                    batch = self.batch_builder.build_batch([transition], include_memory_states=True)
                    batch["initial_memory_state"] = memory_state
                    
                    # Get Ground Truth (Next Frame)
                    # BatchBuilder puts history + next_frame in "observation.images.image0_history"
                    # Shape: [1, T_hist+1, C, H, W]
                    # The last frame is the target (next frame)
                    gt_hist = batch.get("observation.images.image0_history")
                    gt_img = None
                    if gt_hist is not None:
                        gt_img = gt_hist[0, -1] # [C, H, W]

                    # Predict
                    outputs = policy.predict_images_only(batch)
                    # outputs["pred_imgs"] shape: [1, n_pred, C, H, W]
                    
                    # We take the first predicted frame (immediate next frame)
                    if "pred_imgs" in outputs and gt_img is not None:
                        pred_imgs_tensor = outputs["pred_imgs"]
                        # Handle [B, C, H, W] case (T=1 squeezed) or [B*T, C, H, W]
                        if pred_imgs_tensor.ndim == 4:
                            pred_imgs_tensor = pred_imgs_tensor.unsqueeze(1)
                            
                        pred_img = pred_imgs_tensor[0, 0] # [C, H, W]
                        
                        # Resize to match GT if needed
                        if pred_img.shape[-2:] != gt_img.shape[-2:]:
                             pred_img = F.interpolate(
                                 pred_img.unsqueeze(0), 
                                 size=gt_img.shape[-2:], 
                                 mode='bilinear', 
                                 align_corners=False
                             ).squeeze(0)

                        pred_frames.append(pred_img.cpu())
                        gt_frames.append(gt_img.cpu())
                    
                    # Update memory for next step
                    if outputs.get("memory_state") is not None:
                        memory_state = outputs["memory_state"]
            
            # Stitch video
            if pred_frames and gt_frames:
                # Stack: [T, C, H, W]
                pred_vid = torch.stack(pred_frames)
                gt_vid = torch.stack(gt_frames)
                
                # Denormalize [-1, 1] -> [0, 1]
                pred_vid = (pred_vid + 1.0) / 2.0
                gt_vid = (gt_vid + 1.0) / 2.0
                
                pred_vid = torch.clamp(pred_vid, 0, 1)
                gt_vid = torch.clamp(gt_vid, 0, 1)
                
                # Concatenate side-by-side: [T, C, H, 2W]
                combined = torch.cat([gt_vid, pred_vid], dim=-1)
                
                # Save to disk
                try:
                    # [T, C, H, W] -> [T, H, W, C]
                    video_np = combined.permute(0, 2, 3, 1).cpu().numpy()
                    video_np = (video_np * 255).astype(np.uint8)
                    
                    video_dir = self.output_dir / "videos"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_dir / f"step_{step}.mp4"
                    
                    import imageio
                    imageio.mimsave(str(video_path), video_np, fps=10)
                except Exception as e:
                    logger.warning(f"Failed to save video to disk: {e}")

                # Add to writer: [1, T, C, H, W]
                combined = combined.unsqueeze(0)
                add_video_to_writer(self.writer, "val/episode_prediction", combined.numpy(), step, fps=10)
                
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
        
        policy.train()

    def get_episode_iterator(self, batch_size: int):
        """
        Get an iterator over batches of full episodes.
        If using WeightedEpisodeBuffer, constructs an epoch by:
        1. Identifying the bottleneck dataset (min N_i / w_i).
        2. Taking all episodes from the bottleneck dataset.
        3. Sampling from other datasets to match the weight ratios relative to the bottleneck.
        
        Otherwise, shuffles all episodes and iterates.
        """
        if self.datasets_config:
            buffers = self.replay_buffer.buffers
            weights = self.replay_buffer.weights
            counts = [len(b.episodes) for b in buffers]
            
            if any(c == 0 for c in counts):
                logger.warning("One of the datasets is empty. Epoch will be empty.")
                return
            
            # 1. Identify bottleneck
            # We want to find i such that counts[i] / weights[i] is minimized
            ratios = [c / w for c, w in zip(counts, weights)]
            bottleneck_idx = np.argmin(ratios)
            
            base_count = counts[bottleneck_idx]
            base_weight = weights[bottleneck_idx]
            
            # 2. Collect episodes for this epoch
            epoch_episodes = []
            for i, buffer in enumerate(buffers):
                # Calculate target count for this dataset
                # count_i / count_base = weight_i / weight_base
                target_count = int(base_count * (weights[i] / base_weight))
                target_count = min(target_count, len(buffer.episodes))
                
                if i == bottleneck_idx:
                    # Take all
                    epoch_episodes.extend(buffer.episodes)
                else:
                    # Sample without replacement
                    indices = np.random.choice(len(buffer.episodes), size=target_count, replace=False)
                    for idx in indices:
                        epoch_episodes.append(buffer.episodes[idx])
            
            # 3. Shuffle and yield
            np.random.shuffle(epoch_episodes)
            
            for i in range(0, len(epoch_episodes), batch_size):
                yield epoch_episodes[i : i + batch_size]
        else:
            # 1. Get all episodes (already sharded)
            all_episodes = list(self.replay_buffer.episodes)
            
            # 2. Shuffle
            np.random.shuffle(all_episodes)
            
            # 3. Yield batches
            for i in range(0, len(all_episodes), batch_size):
                yield all_episodes[i : i + batch_size]

    def train(self, num_epochs: int, start_epoch: int = 0, vis_every_episodes: int = 20):
        self.load_data()
        
        # Calculate total steps (approximate for scheduler)
        # We iterate through all episodes, chunked by bptt_length
        total_episodes_per_epoch = 0
        if self.datasets_config:
            buffers = self.replay_buffer.buffers
            weights = self.replay_buffer.weights
            counts = [len(b.episodes) for b in buffers]
            if any(c == 0 for c in counts):
                total_transitions = 0
            else:
                ratios = [c / w for c, w in zip(counts, weights)]
                bottleneck_idx = np.argmin(ratios)
                base_count = counts[bottleneck_idx]
                base_weight = weights[bottleneck_idx]
                
                total_transitions = 0
                for i, buffer in enumerate(buffers):
                    target_count = int(base_count * (weights[i] / base_weight))
                    target_count = min(target_count, len(buffer.episodes))
                    total_episodes_per_epoch += target_count
                    
                    # Estimate transitions: average length * target_count
                    avg_len = np.mean([len(ep) for ep in buffer.episodes]) if buffer.episodes else 0
                    total_transitions += int(avg_len * target_count)
        else:
            total_episodes_per_epoch = len(self.replay_buffer.episodes)
            total_transitions = sum(len(ep) for ep in self.replay_buffer.episodes)
        # Note: total_transitions is already per-process since we only loaded local data
            
        # If sequential, we process in chunks of bptt_length
        if self.sequential_training:
            steps_per_epoch = total_transitions // (self.config.batch_size * self.bptt_length)
        else:
            steps_per_epoch = total_transitions // self.config.batch_size
            
        total_steps = steps_per_epoch * num_epochs
        
        self._print(f"Training for {num_epochs} epochs (approx {steps_per_epoch} steps/epoch, total {total_steps} steps)")
        
        # Setup scheduler now that we know total steps
        self.scheduler = setup_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            T_max=total_steps,
            eta_min=1e-6,
        )
        if self.accelerator:
            self.scheduler = self.accelerator.prepare(self.scheduler)
            
        pbar = tqdm(range(start_epoch, num_epochs), desc="Epochs", disable=not self._is_main_process())
        
        for epoch in pbar:
            epoch_loss = []
            epoch_acc = []
            
            if self.sequential_training:
                # Sequential training: Iterate through episodes
                episode_iterator = self.get_episode_iterator(self.config.batch_size)
                
                # Use tqdm for inner loop if main process
                if self._is_main_process():
                    # Estimate number of batches
                    num_batches = total_episodes_per_epoch // self.config.batch_size
                    step_pbar = tqdm(episode_iterator, total=num_batches, desc=f"Ep {epoch+1}", leave=False)
                else:
                    step_pbar = episode_iterator
                
                for batch_episodes in step_pbar:
                    if not batch_episodes: continue
                    
                    # Update episode count
                    self.metrics["total_episodes"] += len(batch_episodes)
                    
                    # Visualization (based on episodes)
                    # Check if we crossed a multiple of vis_every_episodes
                    # We use a simple check: if (total - len) // interval < total // interval
                    prev_total = self.metrics["total_episodes"] - len(batch_episodes)
                    if self._is_main_process() and (prev_total // vis_every_episodes < self.metrics["total_episodes"] // vis_every_episodes):
                        self.visualize_episode(batch_episodes[0], self.metrics["total_steps"])

                    # Process this batch of episodes sequentially
                    # 1. Initialize memory state
                    memory_state = self._init_memory_state(len(batch_episodes))
                    
                    # 2. Find max length
                    max_len = max(len(ep) for ep in batch_episodes)
                    
                    # 3. Iterate in chunks
                    for t in range(0, max_len, self.bptt_length):
                        # Construct batch of sequences
                        sequences = []
                        active_indices = [] # Indices in batch_episodes that are still active
                        
                        for idx, ep in enumerate(batch_episodes):
                            if t < len(ep):
                                # Get sequence
                                end_t = min(t + self.bptt_length, len(ep))
                                seq = ep[t : end_t]
                                sequences.append(seq)
                                active_indices.append(idx)
                            else:
                                # Episode finished, append dummy empty sequence to maintain list size?
                                # train_step_sequential expects list of sequences.
                                # If we filter, batch size changes, and memory_state needs slicing.
                                pass
                        
                        if not sequences:
                            break
                            
                        # If some episodes finished, we need to slice memory_state
                        if len(sequences) < len(batch_episodes):
                            # This is complex because memory_state corresponds to original batch indices
                            # We need to gather memory states for active indices
                            # memory_state: [layers, batch, hidden]
                            memory_state = memory_state[:, active_indices, :]
                            
                            # Update batch_episodes to only include active ones for next iteration?
                            # No, indices in active_indices refer to original batch_episodes list
                            # But next iteration we iterate batch_episodes again.
                            # We should filter batch_episodes
                            batch_episodes = [batch_episodes[i] for i in active_indices]
                        
                        # Train step
                        result = self.train_step_sequential(sequences, initial_memory_state=memory_state, episode_step_start=t)
                        
                        # Visualization (every 1000 steps)
                        # Removed old visualization logic
                        
                        # Update memory state for next chunk
                        memory_state = result["final_memory_state"]
                        
                        epoch_loss.append(result["loss"])
                        epoch_acc.append(result["accuracy"])
                        self.metrics["total_steps"] += 1
                        
                        # Log to TensorBoard
                        if self._is_main_process() and self.writer is not None:
                            self.writer.add_scalar("train/loss", result["loss"], self.metrics["total_steps"])
                            self.writer.add_scalar("train/accuracy", result["accuracy"], self.metrics["total_steps"])
                        
                        # Count transitions processed in this step
                        # sequences is list of chunks. Each chunk has length <= bptt_length
                        num_transitions = sum(len(s) for s in sequences)
                        self.metrics["total_transitions"] += num_transitions
                        
                        if self._is_main_process() and isinstance(step_pbar, tqdm):
                            step_pbar.set_postfix({"loss": f"{result['loss']:.3f}", "acc": f"{result['accuracy']:.3f}"})

            else:
                # Random sampling (original behavior)
                step_pbar = tqdm(range(steps_per_epoch), desc=f"Ep {epoch+1}", leave=False, disable=not self._is_main_process())
                
                for _ in step_pbar:
                    transitions = self.replay_buffer.sample_batch(self.config.batch_size)
                    if not transitions: continue
                    batch = self.batch_builder.build_batch(transitions, include_memory_states=True)
                    result = self.train_step(batch)
                    
                    epoch_loss.append(result["loss"])
                    epoch_acc.append(result["accuracy"])
                    self.metrics["total_steps"] += 1
                    self.metrics["total_transitions"] += len(transitions)
                    
                    # Log to TensorBoard
                    if self._is_main_process() and self.writer is not None:
                        self.writer.add_scalar("train/loss", result["loss"], self.metrics["total_steps"])
                        self.writer.add_scalar("train/accuracy", result["accuracy"], self.metrics["total_steps"])
                    
                    step_pbar.set_postfix({"loss": f"{result['loss']:.3f}", "acc": f"{result['accuracy']:.3f}"})
            
            avg_loss = np.mean(epoch_loss) if epoch_loss else 0
            avg_acc = np.mean(epoch_acc) if epoch_acc else 0
            
            if self._is_main_process():
                pbar.set_postfix({"avg_loss": f"{avg_loss:.3f}", "avg_acc": f"{avg_acc:.3f}"})
                
            # Save checkpoint
            save_every = self.rl_config.get("training", {}).get("save_every", 1)
            save_every = min(save_every, num_epochs)
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
                
    def save_checkpoint(self, epoch: int):
        if not self._is_main_process(): return
        
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        policy = self.accelerator.unwrap_model(self.policy) if self.accelerator else self.policy
        torch.save(policy.state_dict(), checkpoint_dir / "model.pt")
        
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "total_steps": self.metrics["total_steps"],
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

def main():
    args = parse_args()
    accelerator = create_accelerator(mixed_precision=args.mixed_precision)
    rl_config = load_rl_config(args.rl_config)
    setup_logging_from_config(rl_config, is_main_process=accelerator.is_main_process)
    
    # Overrides
    if args.output_dir: rl_config.teacher.output_dir = args.output_dir
    if args.batch_size: rl_config.training.batch_size = args.batch_size
    if args.learning_rate: rl_config.training.learning_rate = args.learning_rate
    if args.save_every: rl_config.training.save_every = args.save_every
    if args.num_epochs: rl_config.training.num_epochs = args.num_epochs
    
    
    device, is_main_process = resolve_device_and_process(accelerator, rl_config, args)
    print_startup_header("Offline Teacher Training", device, is_main_process, use_ddp=args.use_ddp)
    
    model_config_file = rl_config.get("model", {}).get("config_file", "/mnt/data2/ty/F1-VLA/f1_vla/config/debug_test.yaml")
    lora_config = get_lora_config_from_dict(rl_config)
    
    policy, policy_config, _ = load_f1_policy(
        config_file=model_config_file,
        device=device,
        debug=rl_config.get("debug", False),
        lora_config=lora_config,
        is_main_process=is_main_process,
    )
    
    trainer = OfflineTrainer(
        policy=policy,
        policy_config=policy_config,
        rl_config=rl_config,
        model_config_file=model_config_file,
        data_dir=args.data_dir,
        device=device,
        accelerator=accelerator,
    )
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    elif args.auto_resume:
        latest = trainer.find_latest_checkpoint()
        if latest:
            start_epoch = trainer.load_checkpoint(str(latest))
            
    if args.num_epochs is None:
        args.num_epochs = rl_config.get("training", {}).get("num_epochs", 100)
    
    # Determine visualization frequency
    vis_every = args.vis_every_episodes
    if vis_every is None:
        vis_every = rl_config.get("training", {}).get("video_save_every", 20)
    
    trainer.train(
        num_epochs= args.num_epochs, 
        start_epoch=start_epoch,
        vis_every_episodes=vis_every
    )

if __name__ == "__main__":
    main()
