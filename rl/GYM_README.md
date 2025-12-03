# RoboTwin Gym Environment Wrapper

This module provides standard Gymnasium-compatible wrappers for RoboTwin manipulation tasks, enabling reinforcement learning training.

## Directory Structure

```
envs/
├── __init__.py              # Main package entry
├── _base_task.py            # Base task class
├── _GLOBAL_CONFIGS.py       # Global configurations
├── tasks/                   # All task definitions
│   ├── __init__.py          # Task registry
│   ├── beat_block_hammer.py
│   ├── place_bread_basket.py
│   └── ...                  # 50+ manipulation tasks
├── rl/                      # Reinforcement learning module
│   ├── __init__.py
│   ├── gym_wrapper.py       # Core Gym environment wrapper
│   ├── vec_env.py           # Vectorized environment
│   ├── train_ppo.py         # PPO training script
│   ├── train_rl_example.py  # Usage examples
│   ├── evaluate_rl.py       # Model evaluation
│   └── GYM_README.md        # This documentation
├── robot/                   # Robot control and planning
├── camera/                  # Camera and observation utilities
└── utils/                   # Common utilities
```

## Features

- **Standard Gym API**: Compatible with Gymnasium (gym) interface
- **Multiple Action Types**: Support for joint position (`qpos`), end-effector (`ee`), and delta end-effector (`delta_ee`) control
- **Configurable Observations**: Choose from RGB images, depth, pointcloud, joint positions, and end-effector poses
- **Vectorized Environments**: Support for parallel training with multiple environments
- **RL Library Integration**: Works with Stable-Baselines3, RLlib, and other RL frameworks

## Installation

The Gym wrapper is included in the RoboTwin package. Make sure you have the base RoboTwin dependencies installed:

```bash
cd RoboTwin
pip install -e .

# For RL training with Stable-Baselines3
pip install stable-baselines3[extra]
```

## Quick Start

### Basic Usage

```python
from rl import RoboTwinGymEnv, make_robotwin_env

# Create environment
env = RoboTwinGymEnv(
    task_name="beat_block_hammer",      # Task to run
    task_config="demo_randomized",       # Configuration file
    action_type="qpos",                  # 'qpos', 'ee', or 'delta_ee'
    obs_keys=['rgb', 'qpos', 'endpose'], # Observation modalities
    max_episode_steps=300,               # Maximum steps per episode
    reward_type="sparse",                # 'sparse' or 'dense'
    render_mode=None,                    # 'human' for visualization
    seed=42
)

# Standard Gym loop
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Vectorized Environment

```python
from rl import RoboTwinVecEnv, make_vec_env

# Create vectorized environment
vec_env = RoboTwinVecEnv(
    task_name="beat_block_hammer",
    num_envs=4,                          # Number of parallel environments
    task_config="demo_randomized",
    action_type="qpos",
    max_episode_steps=300,
    seed=42
)

# Batch operations
obs, infos = vec_env.reset()
actions = np.array([vec_env.action_space.sample() for _ in range(4)])
obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

vec_env.close()
```

### Using Gymnasium Registry

```python
import gymnasium as gym
from rl import register_robotwin_envs

# Register environments
register_robotwin_envs()

# Create using gym.make
env = gym.make("RoboTwin-beat_block_hammer-v0")
```

## Training with Stable-Baselines3

### PPO Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl import RoboTwinGymEnv

# Create environment
def make_env(seed):
    def _init():
        return RoboTwinGymEnv(
            task_name="beat_block_hammer",
            action_type="qpos",
            obs_keys=['qpos', 'endpose'],
            max_episode_steps=300,
            seed=seed
        )
    return _init

env = DummyVecEnv([make_env(i) for i in range(4)])

# Create and train PPO
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_robotwin")
```

### Command Line Training

```bash
# Train with PPO
python script/train_ppo.py \
    --task beat_block_hammer \
    --num_envs 4 \
    --total_timesteps 1000000 \
    --learning_rate 3e-4

# Evaluate trained model
python script/evaluate_rl.py \
    --model_path logs/ppo/final_model.zip \
    --task beat_block_hammer \
    --num_episodes 100
```

## Environment Configuration

### Action Types

| Type | Description | Action Dimension |
|------|-------------|-----------------|
| `qpos` | Joint positions for both arms + grippers | 16 (7+1+7+1) |
| `ee` | End-effector pose (position + quaternion) + grippers | 16 (8+8) |
| `delta_ee` | Relative end-effector movement + grippers | 16 |

### Observation Keys

| Key | Description | Shape |
|-----|-------------|-------|
| `rgb` | RGB images from cameras | (H, W, 3) per camera |
| `depth` | Depth images | (H, W) |
| `pointcloud` | Point cloud data | (N, 3) |
| `qpos` | Joint positions | (16,) |
| `endpose` | End-effector poses | (7,) per arm |

### Available Tasks

- `beat_block_hammer`
- `place_bread_basket`
- `stack_blocks_two`
- `stack_blocks_three`
- `pick_dual_bottles`
- `place_can_basket`
- `handover_block`
- `lift_pot`
- `open_laptop`
- `press_stapler`
- ... and many more (see `envs/` directory)

## Custom Reward Functions

You can create custom reward functions by subclassing `RoboTwinGymEnv`:

```python
class CustomRewardEnv(RoboTwinGymEnv):
    def _compute_reward(self) -> float:
        success = self._task_env.check_success()
        
        if success:
            return 100.0
        
        # Add custom dense rewards
        reward = -0.01  # Step penalty
        
        # Distance-based reward, etc.
        # ...
        
        return reward
```

## API Reference

### RoboTwinGymEnv

```python
class RoboTwinGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for RoboTwin tasks.
    
    Args:
        task_name: Name of the task
        task_config: Configuration file name
        action_type: 'qpos', 'ee', or 'delta_ee'
        obs_keys: List of observation modalities
        max_episode_steps: Maximum steps per episode
        reward_type: 'sparse' or 'dense'
        render_mode: 'human', 'rgb_array', or None
        seed: Random seed
    """
    
    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation."""
        
    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        
    def render(self):
        """Render the environment."""
        
    def close(self):
        """Clean up resources."""
```

### RoboTwinVecEnv

```python
class RoboTwinVecEnv:
    """
    Vectorized environment for parallel training.
    
    Args:
        task_name: Name of the task
        num_envs: Number of parallel environments
        task_config: Configuration file name
        seed: Base random seed
        **env_kwargs: Additional environment arguments
    """
    
    def reset(self, seed=None, options=None):
        """Reset all environments."""
        
    def step(self, actions):
        """Step all environments with batch of actions."""
```

## Troubleshooting

### Common Issues

1. **Environment hangs on reset**: Make sure to set `render_freq: 0` in your task config for headless training.

2. **CUDA out of memory**: Reduce `num_envs` or use CPU for simulation.

3. **Slow training**: Use `obs_keys=['qpos', 'endpose']` instead of image observations for faster training.

### Performance Tips

- Use vectorized environments for faster data collection
- Start with sparse rewards before adding dense rewards
- Use `action_type='qpos'` for more stable learning
- Normalize observations and rewards for better training stability

## Citation

If you use this Gym wrapper in your research, please cite:

```bibtex
@misc{robotwin2024,
    title={RoboTwin: Dual-Arm Robot Benchmark},
    year={2024},
}
```
