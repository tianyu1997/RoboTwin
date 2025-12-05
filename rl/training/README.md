# F1-VLA RL Training

This directory contains the reinforcement learning training pipeline for F1-VLA.

## Overview

The training consists of three phases:

1. **Phase 1 (Teacher/World Model)**: Train the World Model to predict next frame observations
2. **Phase 2 (Student/Explorer)**: Train a student policy to explore and learn from teacher
3. **Phase 3 (Adversarial)**: Adversarial training between World Model and Explorer

## Quick Start

### Single GPU Training

```bash
# Train World Model (Phase 1)
cd /mnt/data2/ty/F1-VLA/RoboTwin
bash rl/training/train.sh teacher 1 1

# With more episodes
bash rl/training/train.sh teacher 1 1 --num_episodes 500
```

### Multi-GPU Distributed Training

```bash
# 4 GPUs with 2 environments per GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash rl/training/train.sh teacher 4 2

# 4 GPUs with custom config
CUDA_VISIBLE_DEVICES=0,1,2,3 bash rl/training/train.sh teacher 4 2 --num_episodes 500
```

## Usage

```bash
./train.sh <phase> [num_gpus] [num_envs] [extra_args...]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `phase` | Training phase: `teacher`, `student`, `adversarial` | `teacher` |
| `num_gpus` | Number of GPUs for distributed training | `1` |
| `num_envs` | Number of parallel environments per GPU | `1` |
| `extra_args` | Additional arguments passed to training script | |

### Phase Aliases

- **Phase 1**: `teacher`, `phase1`, `wm`
- **Phase 2**: `student`, `phase2`, `explorer`
- **Phase 3**: `adversarial`, `phase3`, `adv`

## Training Scripts

| Script | Phase | Description |
|--------|-------|-------------|
| `train_teacher_rl.py` | 1 | World Model training (supervised) |
| `train_student_rl.py` | 2 | Student policy training (exploration) |
| `train_adversarial_rl.py` | 3 | Adversarial WM vs Explorer |

## Features

### Distributed Training (DDP)
- Uses HuggingFace Accelerate for multi-GPU training
- Automatic gradient synchronization across processes
- Only main process logs INFO level (reduces log noise)
- Checkpoints saved only by main process

### Multi-Environment Collection
- Uses gymnasium's `SyncVectorEnv` for parallel environments
- Multiple envs per GPU for faster data collection
- Sequential episode buffer maintains temporal ordering

### Video Recording
- Environment frames saved as MP4 videos
- Useful for debugging and visualization
- Saved to `outputs/teacher_rl/videos/`

### Sample Comparison Images
- Side-by-side GT vs Predicted images
- Saved periodically to `outputs/teacher_rl/samples/`

## Output Directory Structure

```
outputs/teacher_rl/
├── samples/              # GT vs Predicted comparison images
│   ├── episode_000010.png
│   └── ...
├── videos/               # Episode videos from environment
│   ├── episode_000010.mp4
│   └── ...
├── checkpoint-100/       # Model checkpoints
│   ├── model.pt
│   └── training_state.pt
└── episode_metrics.jsonl # Training metrics log
```

## Configuration

Main configuration file: `rl/rl_config.yaml`

Key settings:
```yaml
training:
  num_episodes: 100
  steps_per_episode: 50
  batch_size: 8
  learning_rate: 1e-4
  save_every: 50
  log_every: 10

teacher:
  output_dir: "./outputs/teacher_rl"

logging:
  console_level: INFO
  file_level: DEBUG
```

## Requirements

- PyTorch with CUDA
- HuggingFace Accelerate (`pip install accelerate`)
- gymnasium (`pip install gymnasium`)
- OpenCV (`pip install opencv-python`)

## Common Commands

```bash
# Phase 1: World Model training with 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash rl/training/train.sh teacher 4 2 --num_episodes 500

# Phase 2: Student training (requires teacher checkpoint)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash rl/training/train.sh student 4 1 \
    --teacher_path ./outputs/teacher_rl/checkpoint-500

# Phase 3: Adversarial training
CUDA_VISIBLE_DEVICES=0,1,2,3 bash rl/training/train.sh adversarial 4 1 \
    --teacher_checkpoint ./outputs/teacher_rl/checkpoint-500
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config
- Reduce `num_envs`
- Use fewer GPUs

### Slow Training
- Increase `num_envs` for faster data collection
- Use more GPUs

### DDP Hang
- Check all GPUs are visible: `nvidia-smi`
- Ensure `CUDA_VISIBLE_DEVICES` is set correctly
- Check for firewall issues on multi-node setup

## File Descriptions

| File | Description |
|------|-------------|
| `train.sh` | Main launcher script |
| `train_teacher_rl.py` | Phase 1 training script |
| `train_student_rl.py` | Phase 2 training script |
| `train_adversarial_rl.py` | Phase 3 training script |
| `parallel_utils.py` | DDP and multi-env utilities |
| `rl_training_common.py` | Shared training utilities |
