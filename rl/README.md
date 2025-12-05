# F1-VLA RL Training Module

This directory contains the Reinforcement Learning training framework for F1-VLA.

## 📁 Directory Structure

```
rl/
├── __init__.py                  # Module exports
├── f1_rl_env.py                 # F1-VLA RL environment (Gymnasium compatible)
├── gym_wrapper.py               # Core Gymnasium environment wrapper
├── vec_env.py                   # Vectorized environment utilities
├── normalizers.py               # Action/State normalization
├── suppress_logs.py             # Log suppression utilities
├── rl_config.yaml               # Training configuration
├── GYM_README.md                # Gymnasium wrapper documentation
├── README.md                    # This file
│
└── training/                    # Training scripts
    ├── __init__.py              # Training module exports
    ├── rl_training_common.py    # Shared training utilities (config, model loading, etc.)
    ├── parallel_utils.py        # Multi-GPU/multi-env utilities (Accelerate + SyncVectorEnv)
    ├── train_teacher_rl.py      # Phase 1: World Model training
    ├── train_student_rl.py      # Phase 2: Explorer training
    ├── train_adversarial_rl.py  # Phase 3: Adversarial training
    ├── train.sh                 # Shell script entry point (recommended)
    └── evaluate_rl.py           # Policy evaluation
```

## 🎯 Three-Phase Training Architecture

F1-VLA uses a three-phase training approach:

### Phase 1: Teacher Training (World Model)
Train the World Model to predict next frame observations.

- **Input**: History images (head + wrist cameras), action history, states, **memory state**
- **Output**: Predicted next wrist camera frame (VQ-VAE tokens)
- **Loss**: Cross-entropy on predicted vs actual image tokens
- **Actions**: Random (for exploration/data collection)
- **Memory**: GRU memory state propagated through sequence (initialized to zeros for first frame)

### Phase 2: Student Training (Explorer)
Train the action policy using the frozen World Model from Phase 1.

- **Input**: Wrist camera only (no head camera), **memory state**
- **Output**: Actions from F1-VLA actor
- **Reward**: Memory divergence + WM uncertainty
- **World Model**: Frozen (from Phase 1)

### Phase 3: Adversarial Training
Jointly train World Model and Explorer in adversarial manner.

- **World Model**: Tries to accurately predict next frame
- **Explorer**: Tries to find actions that make WM's predictions fail
- **Result**: Both become more robust

## ⚙️ Configuration

### Main Config: `rl_config.yaml`

```yaml
# Model configuration
model:
  config_file: "/path/to/f1_vla/config/debug_test.yaml"
  lora:
    r: 8
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]

# Training parameters (shared across phases)
training:
  num_episodes: 10000
  steps_per_episode: 50
  batch_size: 8
  learning_rate: 1e-4
  action_dim: 32
  state_dim: 32
  n_pred_img_steps: 1
  save_every: 1000
  log_every: 10
  video_save_every: 100
  sample_save_every: 100

# Environment configuration
environment:
  task_name: "random_exploration"
  control_mode: "delta_qpos"
  single_arm: true
  scene_reset_interval: 10
  embodiment: ["franka-panda"]
  domain_randomization:
    random_appearance: false
    random_background: true
    random_light: true
    cluttered_table: true

# Logging configuration
logging:
  console_level: "WARNING"  # Only show warnings+ in console
  file_level: "DEBUG"       # Full debug in log file
  enable_file_logging: true

# Phase-specific configs
teacher:
  output_dir: "./outputs/teacher_rl"

student:
  output_dir: "./outputs/student_rl"
  rewards:
    memory_divergence_weight: 0.5
    wm_uncertainty_weight: 0.5

adversarial:
  output_dir: "./outputs/adversarial_rl"
  total_iterations: 100000
```

### Model Config: `f1_vla/config/debug_test.yaml`

This config specifies:
- `n_obs_img_steps`: Number of history frames (default: 4)
- `n_pred_img_steps`: Number of prediction frames (default: 1)
- `obs_img_stride`: Stride for observation sampling

### Memory Configuration: `f1_vla/config/f1_config.json`

```json
{
  "memory_enabled": true,
  "memory_hidden": 2048,
  "memory_num_layers": 4,
  "memory_project_to_vae_dim": true
}
```

## 🚀 Training Entry Point

### Recommended: Shell Script

```bash
cd /path/to/F1-VLA

# Single GPU
bash ./RoboTwin/rl/training/train.sh teacher 1 1 --num_episodes 1000

# Multi-GPU (4 GPUs, 2 envs per GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./RoboTwin/rl/training/train.sh teacher 4 2 --num_episodes 10000

# Arguments:
#   $1: phase (teacher|student|adversarial)
#   $2: num_gpus (default: 1)
#   $3: num_envs_per_gpu (default: 1)
#   $4+: additional arguments passed to training script
```

### Direct Python

```bash
cd /path/to/F1-VLA/RoboTwin

# Single GPU
python -m rl.training.train_teacher_rl --num_episodes 1000

# Multi-GPU with Accelerate
accelerate launch --num_processes=4 \
    -m rl.training.train_teacher_rl \
    --num_envs 2 \
    --num_episodes 10000
```

## 🔧 Key Components

### 1. Memory State Management (⚠️ CRITICAL)

F1-VLA has a GRU-based memory that requires proper state propagation:

```python
# Memory state shape: (num_layers, batch_size, hidden_dim)
# Default: (4, batch_size, 2048)

# First frame: MUST initialize to zeros
memory_state = torch.zeros(4, batch_size, 2048, device=device)

# Subsequent frames: use output from previous step
loss_dict = policy.forward_with_world_model(batch, ...)
memory_state = loss_dict["memory_state"]  # Use for next step

# NEVER let memory_state be None!
if memory_state is None:
    raise ValueError("memory_state cannot be None!")
```

**⚠️ IMPORTANT**: 
- Memory state **MUST** be initialized to zeros for the first frame
- Memory state **MUST NOT** be None during training
- The training script will raise an error if memory_state becomes None

### 2. Multi-GPU Distribution

SAPIEN uses Vulkan rendering which requires explicit GPU assignment:

```python
# At script start (BEFORE importing SAPIEN):
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
visible_gpus = [int(x) for x in cuda_visible.split(",") if x]
physical_gpu_id = visible_gpus[local_rank] if local_rank < len(visible_gpus) else local_rank

os.environ["VK_DEVICE_INDEX"] = str(physical_gpu_id)
os.environ["SAPIEN_DEVICE_INDEX"] = str(physical_gpu_id)
os.environ["EGL_DEVICE_ID"] = str(physical_gpu_id)
```

### 3. Camera Configuration

- **Teacher Phase**: Uses both head and wrist cameras
  - `head_rgb` → `image0` (VLM input)
  - `wrist_rgb` → `image1` (VLM input) + `image0_history` (WM input)
  
- **Student Phase**: Uses only wrist camera
  - `wrist_rgb` → `image0` (VLM input) + `image0_history` (WM input)

### 4. Image History 与 Action History 的对应关系

**⚠️ 关键设计：时序对齐**

```
时间步:     t-3    t-2    t-1    t     t+1 (预测目标)
           ─────────────────────────────────────────
图像历史:  [img₀, img₁, img₂, img₃] → 预测 img₄
动作历史:  [a₋₃,  a₋₂,  a₋₁,  a₀  ]
状态:                           s_t
```

**对应关系**:
- `image_history[i]` 对应 `action_history[i]` **执行前**的观测
- `image_history[i+1]` 是执行 `action_history[i]` **后**的观测
- World Model 的目标是：给定 `[img₀...img₃]` + `[a₀...a₃]` → 预测 `img₄`

**配置参数**:
- `n_obs_img_steps`: 图像历史长度 (默认: 4)
- `obs_img_stride`: 采样步长 (默认: 1)
- 实际历史帧数 = `n_obs_img_steps * obs_img_stride`

**代码实现**:
```python
# 在 BatchBuilder.build_batch() 中:
# image_history 包含 [img_t-3, img_t-2, img_t-1, img_t, img_t+1]
# 其中 img_t+1 是 next_obs（预测目标），在训练时拼接

# 在 f1_rl_env.py 中:
# action_history 维护最近的动作序列
# 每次 step() 后更新: action_history.append(action); action_history.pop(0)

# ⚠️ 重要: 两个历史必须同步维护！
# 如果 image_history 有 4 帧，action_history 也必须有对应的 4 个动作
```

**验证检查**:
```python
assert len(image_history) == n_obs_img_steps, "Image history length mismatch"
assert len(action_history) == n_obs_img_steps, "Action history length mismatch"
```

### 5. Action Space

Default action dimension: 32 (padded)
- For single arm (Franka Panda): 8 actual DOFs (7 joints + 1 gripper)
- Control mode: `delta_qpos` (joint position deltas)

## 📊 Outputs

Each training phase produces:

```
outputs/teacher_rl/
├── checkpoint-{episode}/
│   ├── model.pt           # Model weights
│   └── training_state.pt  # Optimizer, scheduler, metrics
├── samples/
│   └── episode_{n}.png    # Prediction comparison images
├── videos/
│   └── episode_{n}.mp4    # Episode recordings (Head | GT Wrist | Predicted)
├── episode_metrics.jsonl  # Training metrics log
└── tensorboard/           # Tensorboard logs
```

## ⚠️ Important Notes

### 1. Memory State is Critical
- Memory state must be initialized to zeros for the first frame
- Must be propagated through the sequence (not None)
- The training script will raise an error if memory_state becomes None
- Logger records memory state info at DEBUG level for debugging

### 2. SAPIEN GPU Assignment
- SAPIEN uses Vulkan which doesn't respect `CUDA_VISIBLE_DEVICES`
- Must set `VK_DEVICE_INDEX` **before** importing SAPIEN
- This is done automatically in training scripts

### 3. collect_steps vs steps_per_episode
- `collect_steps(num_steps)` collects `num_steps` total across all envs
- For `num_envs=2` and `steps_per_episode=50`, use `num_steps = 50 * 2 = 100`

### 4. Logging Levels
- Console shows only WARNING+ by default (for clean output)
- Full DEBUG logs are written to `logs/rl_training_*.log`
- Set `console_level: "INFO"` in config for more verbose output
- Memory state logging is at DEBUG level

### 5. Domain Randomization
- `cluttered_table: true` adds random objects to the scene
- `random_background: true` randomizes wall/table textures
- `random_light: true` randomizes lighting conditions

## 🐛 Troubleshooting

### GPU Distribution Issues
```bash
# Check GPU assignment
nvidia-smi  # Should show processes on different GPUs

# If all processes on GPU 0:
# Ensure VK_DEVICE_INDEX is set BEFORE importing SAPIEN
```

### Memory State Errors
```
ValueError: CRITICAL: initial_memory_state is None
```
This means memory state wasn't properly initialized. Check:
1. `memory_enabled: true` in model config
2. `_init_memory_state()` is being called
3. Memory state is not being set to None anywhere in the pipeline

### SAPIEN Rendering Issues
```
SIGBUS or Vulkan errors
```
Try:
1. Use `PhysxCpuSystem` instead of `PhysxGpuSystem`
2. Check GPU memory availability
3. Verify Vulkan drivers are installed

### Video Only Has 21 Frames
This was a bug where `collect_steps(steps_per_episode)` was used instead of `collect_steps(steps_per_episode * num_envs)`. This has been fixed.

## 📚 API Reference

### rl_training_common.py

```python
# GPU assignment utilities
get_physical_gpu_id(accelerator=None) -> int
setup_sapien_gpu(gpu_id: Optional[int] = None)

# Config loading
load_rl_config(config_path: str) -> DictConfig
get_training_config(config: DictConfig) -> TrainingConfig
get_environment_config(config: DictConfig) -> Dict[str, Any]

# Model loading
load_f1_policy(config_file, device, debug, lora_config, checkpoint_path) -> Tuple[policy, config, full_config]

# Training utilities
set_policy_requires_grad(policy, freeze_vision_encoder, freeze_gen_expert, train_act_expert_only, train_gen_expert_only)
setup_optimizer(policy, lr, weight_decay) -> Optimizer
setup_scheduler(optimizer, scheduler_type, T_max, eta_min) -> Scheduler

# Batch building
class BatchBuilder:
    def build_batch(transitions, include_memory_states=True) -> Dict[str, Tensor]

# Memory management
class MemoryStateManager:
    def reset()
    def update(memory_state)
    def get_current() -> Optional[Tensor]
```

### f1_rl_env.py

```python
class F1RLEnv(gymnasium.Env):
    def __init__(task_config, phase, teacher_policy, history_length, max_steps, device, action_scale, single_arm)
    def reset(seed=None, options=None) -> Tuple[obs, info]
    def step(action) -> Tuple[obs, reward, terminated, truncated, info]
    def close()

class TeacherEnv(F1RLEnv):  # Convenience wrapper for phase="teacher"
class StudentEnv(F1RLEnv):  # Convenience wrapper for phase="student"
```

## 📚 References

- [F1-VLA Paper](https://arxiv.org/abs/...)
- [SAPIEN Documentation](https://sapien.ucsd.edu/docs/)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/)
- [Gymnasium](https://gymnasium.farama.org/)
