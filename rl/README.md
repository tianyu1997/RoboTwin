# F1-VLA RL Training Module

This directory contains the Reinforcement Learning training framework for F1-VLA.

## Directory Structure

```
rl/
├── __init__.py              # Module exports
├── f1_rl_env.py             # F1-VLA RL environment (Gym compatible)
├── gym_wrapper.py           # Core Gym environment wrapper
├── vec_env.py               # Vectorized environment for parallel training
├── normalizers.py           # Action/State normalization
├── suppress_logs.py         # Log suppression utilities
├── rl_config.yaml           # Training configuration
├── GYM_README.md            # Gym wrapper documentation
├── README.md                # This file
│
└── training/                # Training scripts
    ├── __init__.py          # Training module exports
    ├── rl_training_common.py    # Shared training utilities
    ├── train_teacher_rl.py      # Phase 1: World Model training
    ├── train_student_rl.py      # Phase 2: Explorer training  
    ├── train_adversarial_rl.py  # Phase 3: Adversarial training
    ├── train_rl.py              # Unified Python entry point
    ├── train_rl.sh              # Shell script wrapper
    ├── train_ppo.py             # Basic PPO example
    ├── train_rl_example.py      # Simple RL example
    └── evaluate_rl.py           # Policy evaluation
```

## Three-Phase Training

### Phase 1: Teacher Training (World Model)
Train the LLM + World Model using random exploration.
- Input: history actions, states, head observation
- Action: randomly generated (for exploration)
- Reward: prediction accuracy of next frame observation

```bash
cd training/
python train_rl.py --phase teacher
# or
./train_rl.sh teacher
```

### Phase 2: Student Training (Explorer)
Train the Explorer policy using PPO with frozen World Model.
- Initialize new LLM but REUSE frozen World Model from teacher
- Use only wrist camera observations (no head camera)
- Reward based on memory divergence and WM prediction error

```bash
python train_rl.py --phase student --teacher_checkpoint outputs/teacher/checkpoint-10000
# or
./train_rl.sh student
```

### Phase 3: Adversarial Training
Alternating training between World Model and Explorer.
- WM: Tries to accurately predict the next frame
- Explorer: Tries to find actions that make WM's predictions fail

```bash
python train_rl.py --phase adversarial --teacher_checkpoint outputs/teacher/checkpoint-10000
# or
./train_rl.sh adversarial
```

### Run All Phases
```bash
./train_rl.sh all
```

## Configuration

All training parameters are configured via `rl_config.yaml`:

- `model`: Model paths and LoRA configuration
- `environment`: Task and camera settings
- `training`: Shared training parameters (lr, batch_size, sequential_training)
- `teacher`: Phase 1 specific settings
- `student`: Phase 2 specific settings (PPO params, reward weights)
- `adversarial`: Phase 3 specific settings

## Key Components

### rl_training_common.py
Shared utilities across all training phases:
- `load_rl_config()`: Load and merge YAML configurations
- `load_f1_policy()`: Load F1-VLA model with LoRA support
- `MemoryStateManager`: Track memory state for sequential training
- `BaseRLTrainer`: Base class with optimizer, scheduler, checkpointing
- `BatchBuilder`: Build training batches with memory support

### f1_rl_env.py
F1-VLA specific RL environment:
- `F1RLEnv`: Main environment class (Gym compatible)
- `TeacherEnv`: Phase 1 environment wrapper
- `StudentEnv`: Phase 2/3 environment wrapper
- Handles image history, state normalization, reward computation

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU selection (default: 0)
- `CONDA_ENV`: Conda environment name (default: f1)
- `OUTPUT_BASE`: Output directory base (default: ./outputs)

## Usage Examples

```python
# Using the RL environment directly
from rl import F1RLEnv

env = F1RLEnv(
    task_config=config,
    phase="teacher",
    device="cuda",
)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

```python
# Using training utilities
from rl.training import (
    load_rl_config,
    load_f1_policy,
    MemoryStateManager,
)

config = load_rl_config("rl_config.yaml")
policy = load_f1_policy(config.model.config_file, device="cuda")
memory = MemoryStateManager()
```
