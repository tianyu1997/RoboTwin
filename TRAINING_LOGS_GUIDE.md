# F1-VLA Training Logs Guide

## 概述 (Overview)

本文档说明F1-VLA训练过程中的日志系统改进，包括：
1. 避免CuRobo JIT编译（在qpos控制模式下）
2. 结构化的per-episode指标记录
3. 预测图像vs真实图像的sample对比

This document explains the F1-VLA training log system improvements, including:
1. Avoiding CuRobo JIT compilation (in qpos control mode)
2. Structured per-episode metrics logging
3. Prediction vs ground-truth image samples

---

## 1. 避免CuRobo JIT编译 (Avoiding CuRobo JIT Compilation)

### 问题 (Problem)
- CuRobo的CUDA JIT编译非常耗时（数分钟）
- 使用`delta_qpos`控制时不需要planner，但仍会触发编译
- 训练日志被编译信息淹没

CuRobo CUDA JIT compilation is very time-consuming (several minutes)
- Not needed when using `delta_qpos` control mode
- Training logs flooded with compilation messages

### 解决方案 (Solution)
修改了`RoboTwin/envs/_base_task.py`，使planner只在需要时加载：

Modified `RoboTwin/envs/_base_task.py` to load planner only when needed:

```python
def load_robot(self, **kwags):
    if not hasattr(self, "robot"):
        self.robot = Robot(self.scene, self.need_topp, **kwags)
        # Only set planner when planning is required
        if self.need_plan:
            self.robot.set_planner(self.scene)
        self.robot.init_joints()
```

### 配置 (Configuration)
在`rl_config.yaml`中，环境配置自动设置`need_plan=False`：

In `rl_config.yaml`, environment config automatically sets `need_plan=False`:

```yaml
environment:
  control_mode: "delta_qpos"  # qpos control不需要planner
  # ...其他配置
```

### 验证 (Verification)
启动训练时，不应看到以下信息：
- "kinematics_fused_cu not found, JIT compiling..."
- PyTorch CUDA extension编译信息

When starting training, you should NOT see:
- "kinematics_fused_cu not found, JIT compiling..."
- PyTorch CUDA extension compilation messages

---

## 2. 结构化指标记录 (Structured Metrics Logging)

### 输出位置 (Output Location)
训练指标记录在：`outputs/{phase}_rl/episode_metrics.jsonl`

Training metrics are logged to: `outputs/{phase}_rl/episode_metrics.jsonl`

其中`{phase}`可以是：
- `teacher` - 教师策略训练（Phase 1）
- `student` - 学生策略训练（Phase 2）
- `adversarial` - 对抗训练（Phase 3）

Where `{phase}` can be:
- `teacher` - Teacher policy training (Phase 1)
- `student` - Student policy training (Phase 2)
- `adversarial` - Adversarial training (Phase 3)

### 记录内容 (Content)
每行是一个JSON对象，包含：

Each line is a JSON object containing:

```json
{
  "episode": 10,
  "avg_loss": 0.523,
  "avg_acc": 0.78,
  "episode_reward": 15.6,
  "total_steps": 500,
  "lr": 0.0001,
  "timestamp": 1701648000.123
}
```

### 记录频率 (Logging Frequency)
默认每10个episode记录一次，可在配置中调整：

Logged every 10 episodes by default, configurable in config:

```yaml
training:
  log_every: 10  # 每N个episode记录一次指标
```

### 使用示例 (Usage Examples)

#### 实时监控 (Real-time Monitoring)
```bash
# 实时查看最新指标
tail -f outputs/teacher_rl/episode_metrics.jsonl | jq '.'

# 只看loss和reward
tail -f outputs/teacher_rl/episode_metrics.jsonl | jq '{episode, avg_loss, episode_reward}'
```

#### Python分析 (Python Analysis)
```python
import json
import pandas as pd

# 加载所有指标
metrics = []
with open('outputs/teacher_rl/episode_metrics.jsonl') as f:
    for line in f:
        metrics.append(json.loads(line))

df = pd.DataFrame(metrics)

# 绘制loss曲线
import matplotlib.pyplot as plt
plt.plot(df['episode'], df['avg_loss'])
plt.xlabel('Episode')
plt.ylabel('Average Loss')
plt.title('Training Loss over Episodes')
plt.savefig('training_loss.png')

# 计算统计信息
print(f"Mean loss: {df['avg_loss'].mean():.4f}")
print(f"Mean accuracy: {df['avg_acc'].mean():.4f}")
print(f"Mean reward: {df['episode_reward'].mean():.4f}")
```

---

## 3. 预测图像样本 (Prediction Image Samples)

### 输出位置 (Output Location)
样本图像保存在：`outputs/{phase}_rl/samples/`

Sample images are saved to: `outputs/{phase}_rl/samples/`

### 文件命名 (File Naming)
```
episode_0010_pred_vs_gt.png
episode_0020_pred_vs_gt.png
episode_0030_pred_vs_gt.png
...
```

### 图像内容 (Image Content)
每张图像包含左右两部分：
- **左侧**：世界模型预测的下一帧图像
- **右侧**：实际观测的下一帧图像（Ground Truth）

Each image contains two side-by-side parts:
- **Left**: World model predicted next frame
- **Right**: Actual observed next frame (Ground Truth)

### 生成频率 (Generation Frequency)
与指标记录同步，默认每10个episode生成一次：

Generated at the same frequency as metrics logging, default every 10 episodes:

```yaml
training:
  log_every: 10  # 同时控制指标记录和样本图像生成
```

### 查看样本 (Viewing Samples)
```bash
# 列出所有样本
ls -lh outputs/teacher_rl/samples/

# 使用图像查看器
eog outputs/teacher_rl/samples/episode_0010_pred_vs_gt.png

# 或用Python
from PIL import Image
img = Image.open('outputs/teacher_rl/samples/episode_0010_pred_vs_gt.png')
img.show()
```

### 自动化分析 (Automated Analysis)
```python
import os
import glob
from PIL import Image
import numpy as np

# 计算所有样本的预测误差
sample_dir = 'outputs/teacher_rl/samples/'
errors = []

for img_path in sorted(glob.glob(os.path.join(sample_dir, '*.png'))):
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    
    # 分离预测和GT（左右两半）
    pred = img[:, :w//2]
    gt = img[:, w//2:]
    
    # 计算MSE
    mse = np.mean((pred - gt) ** 2)
    errors.append(mse)
    
    episode = int(img_path.split('_')[1])
    print(f"Episode {episode}: MSE = {mse:.2f}")

# 绘制误差趋势
import matplotlib.pyplot as plt
plt.plot(errors)
plt.xlabel('Sample Index')
plt.ylabel('Prediction MSE')
plt.title('World Model Prediction Error over Training')
plt.savefig('prediction_error_trend.png')
```

---

## 4. 日志级别配置 (Log Level Configuration)

### 当前配置 (Current Configuration)
在`rl_config.yaml`中：

In `rl_config.yaml`:

```yaml
logging:
  console_level: "WARNING"  # 控制台只显示警告和错误
  file_level: "DEBUG"       # 文件记录所有详细信息
  enable_file_logging: true
```

### 可用级别 (Available Levels)
- `DEBUG`: 最详细，包含所有调试信息
- `INFO`: 一般信息，如episode开始、模型加载等
- `WARNING`: 警告信息（默认控制台级别）
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

Available levels:
- `DEBUG`: Most verbose, includes all debugging info
- `INFO`: General information like episode start, model loading
- `WARNING`: Warning messages (default console level)
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### 调整建议 (Recommendations)

#### 正常训练 (Normal Training)
```yaml
logging:
  console_level: "INFO"     # 查看训练进度
  file_level: "DEBUG"       # 保留完整日志用于调试
```

#### 调试模式 (Debug Mode)
```yaml
logging:
  console_level: "DEBUG"    # 查看所有详细信息
  file_level: "DEBUG"
```

#### 生产模式 (Production Mode)
```yaml
logging:
  console_level: "WARNING"  # 最小输出
  file_level: "INFO"        # 只记录重要事件
```

---

## 5. 日志文件位置 (Log File Locations)

### 训练日志 (Training Logs)
```
logs/
├── rl_training_YYYYMMDD_HHMMSS.log  # 主训练日志
├── env/
│   └── f1_rl_env_YYYYMMDD_HHMMSS.log  # 环境日志
├── normalizer/
│   └── normalizer_YYYYMMDD_HHMMSS.log  # 归一化器日志
└── training/
    └── (空目录，未来可用)
```

### 输出文件 (Output Files)
```
outputs/
├── teacher_rl/
│   ├── episode_metrics.jsonl      # 训练指标
│   ├── samples/                   # 预测图像样本
│   │   ├── episode_0010_pred_vs_gt.png
│   │   ├── episode_0020_pred_vs_gt.png
│   │   └── ...
│   ├── checkpoint-1000/           # 模型检查点
│   │   ├── model.pt
│   │   └── training_state.pt
│   └── ...
├── student_rl/
│   └── (类似结构)
└── adversarial_rl/
    └── (类似结构)
```

---

## 6. 常见问题 (FAQ)

### Q: 如何知道JIT编译被成功跳过？
**A:** 启动训练时，不应看到"JIT compiling"相关信息。如果使用`delta_qpos`控制模式，planner不会被加载。

### Q: 为什么没有生成episode_metrics.jsonl？
**A:** 确保：
1. 至少运行了`log_every`个episode（默认10个）
2. 输出目录权限正确
3. 检查训练脚本日志是否有错误

### Q: 样本图像显示全黑或异常？
**A:** 可能原因：
1. 世界模型未正确加载
2. 图像归一化问题
3. 检查训练日志中的`_save_prediction_sample`相关错误

### Q: 如何减少日志文件大小？
**A:** 调整`file_level`为`INFO`或`WARNING`：
```yaml
logging:
  file_level: "INFO"  # 减少详细调试信息
```

### Q: 如何实时监控训练？
**A:** 使用以下命令：
```bash
# 监控指标
watch -n 5 "tail -1 outputs/teacher_rl/episode_metrics.jsonl | jq '.'"

# 监控日志
tail -f logs/rl_training_*.log

# 监控GPU使用
watch -n 1 nvidia-smi
```

---

## 7. 快速参考 (Quick Reference)

### 启动训练 (Start Training)
```bash
cd RoboTwin
bash rl/training/train_rl.sh teacher
```

### 实时监控 (Monitor Training)
```bash
# 终端1: 查看训练日志
tail -f logs/rl_training_*.log | grep -v "DEBUG"

# 终端2: 查看指标
watch -n 5 "tail -1 outputs/teacher_rl/episode_metrics.jsonl | jq '.'"

# 终端3: 查看GPU
watch -n 1 nvidia-smi
```

### 训练后分析 (Post-training Analysis)
```bash
# 统计指标
cat outputs/teacher_rl/episode_metrics.jsonl | jq -s '
  "Episodes: \(length)\n" +
  "Avg Loss: \(map(.avg_loss) | add / length)\n" +
  "Avg Acc: \(map(.avg_acc) | add / length)\n" +
  "Avg Reward: \(map(.episode_reward) | add / length)"
'

# 查看最新样本
eog outputs/teacher_rl/samples/episode_*_pred_vs_gt.png
```

---

## 更新日志 (Changelog)

### 2024-12-04
- ✅ 实现planner懒加载，避免qpos模式下的JIT编译
- ✅ 减少normalizer和环境的DEBUG日志噪音
- ✅ 确认per-episode JSONL指标记录已实现
- ✅ 确认预测vs GT图像sample已实现
- ✅ 创建本指南文档

### Changes
- ✅ Implemented lazy planner loading to avoid JIT in qpos mode
- ✅ Reduced DEBUG log noise from normalizer and environment
- ✅ Confirmed per-episode JSONL metrics logging is implemented
- ✅ Confirmed pred vs GT image samples are implemented
- ✅ Created this guide document

---

## 参考资料 (References)

- Training configuration: `rl/rl_config.yaml`
- Teacher training script: `rl/training/train_teacher_rl.py`
- Environment wrapper: `rl/f1_rl_env.py`
- Normalizers: `rl/normalizers.py`
- Base task: `envs/_base_task.py`
