#!/bin/bash
# =============================================================================
# F1-VLA RL Training Launcher
# Supports single-GPU and multi-GPU distributed training
# =============================================================================

set -e

# Default CUDA devices if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}

# =============================================================================
# Suppress verbose warnings for cleaner output
# =============================================================================
# Suppress HuggingFace tokenizers parallelism warning (safe to disable in subprocess)
export TOKENIZERS_PARALLELISM=false
# Suppress imageio ffmpeg warnings
export IMAGEIO_FFMPEG_EXE_LOGGING=quiet
# Filter Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"

# Get script directory and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(dirname "$SCRIPT_DIR")"
ROBOTWIN_DIR="$(dirname "$RL_DIR")"
F1_VLA_DIR="$(dirname "$ROBOTWIN_DIR")"

# Activate conda environment
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate f1

# Change to RoboTwin directory (required for relative paths in environment)
cd "$ROBOTWIN_DIR"

# Default values
PHASE="${1:-teacher}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_ENVS="${NUM_ENVS:-1}"  # Environments per GPU

# Shift processed arguments
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

# Print banner
echo "=============================================="
echo "F1-VLA RL Training"
echo "=============================================="
echo "Phase:          $PHASE"
echo "Num GPUs:       $NUM_GPUS"
echo "Envs per GPU:   $NUM_ENVS"
echo "Working Dir:    $ROBOTWIN_DIR"
echo "Extra args:     $EXTRA_ARGS"
echo ""

# Select training script based on phase
case "$PHASE" in
    teacher|phase1|wm)
        SCRIPT="$SCRIPT_DIR/train_teacher_rl.py"
        PHASE_NAME="Teacher/World Model (Phase 1)"
        ;;
    student|phase2|explorer)
        SCRIPT="$SCRIPT_DIR/train_student_rl.py"
        PHASE_NAME="Student/Explorer (Phase 2)"
        
        # Check if teacher_path is provided, if not use default
        if [[ "$EXTRA_ARGS" != *"--teacher_path"* ]]; then
            DEFAULT_TEACHER="/mnt/data2/ty/F1-VLA/RoboTwin/outputs/teacher_rl/checkpoint-memory"
            echo "[INFO] No teacher_path provided. Using default: $DEFAULT_TEACHER"
            EXTRA_ARGS="$EXTRA_ARGS --teacher_path $DEFAULT_TEACHER"
        fi
        ;;
    adversarial|phase3|adv)
        SCRIPT="$SCRIPT_DIR/train_adversarial_rl.py"
        PHASE_NAME="Adversarial (Phase 3)"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo ""
        echo "Usage: $0 <phase> [extra_args...]"
        echo ""
        echo "Phases:"
        echo "  teacher     Phase 1: Train World Model (supervised learning)"
        echo "  student     Phase 2: Train Student Policy (exploration)"
        echo "  adversarial Phase 3: Adversarial WM vs Explorer training"
        echo ""
        echo "Examples:"
        echo "  $0 teacher"
        echo "  $0 teacher --num_episodes 500"
        echo "  $0 student --teacher_path ./outputs/teacher_rl/checkpoint-100"
        exit 1
        ;;
esac

echo "Phase:          $PHASE_NAME"
echo "Training script: $SCRIPT"
echo "=============================================="
echo ""

# Setup logging
LOG_DIR="$ROBOTWIN_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LAUNCHER_LOG="$LOG_DIR/launcher_${TIMESTAMP}.log"

# Create symlink to latest log
ln -sf "$LAUNCHER_LOG" "$F1_VLA_DIR/latest_launcher.log"
echo "Created symlink: $F1_VLA_DIR/latest_launcher.log -> $LAUNCHER_LOG"

echo "Redirecting output to: $LAUNCHER_LOG"
echo "Running in background (detached)..."

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with HuggingFace Accelerate
    echo "[INFO] Launching distributed training with $NUM_GPUS GPUs..."
    
    # Create default accelerate config if not exists
    ACCELERATE_CONFIG_DIR="$HOME/.cache/huggingface/accelerate"
    if [ ! -f "$ACCELERATE_CONFIG_DIR/default_config.yaml" ]; then
        echo "[INFO] Creating default accelerate config..."
        mkdir -p "$ACCELERATE_CONFIG_DIR"
        cat > "$ACCELERATE_CONFIG_DIR/default_config.yaml" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: 'no'
num_machines: 1
num_processes: $NUM_GPUS
use_cpu: false
EOF
    fi
    
    # Launch with accelerate
    nohup accelerate launch \
        --num_processes=$NUM_GPUS \
        --multi_gpu \
        "$SCRIPT" \
        --use_ddp \
        --num_envs=$NUM_ENVS \
        $EXTRA_ARGS > "$LAUNCHER_LOG" 2>&1 &
else
    # Single GPU training
    echo "[INFO] Launching single GPU training..."
    nohup python "$SCRIPT" \
        --num_envs=$NUM_ENVS \
        $EXTRA_ARGS > "$LAUNCHER_LOG" 2>&1 &
fi

PID=$!
echo "Training started with PID: $PID"
echo "You can monitor progress with: tail -f $LAUNCHER_LOG"
