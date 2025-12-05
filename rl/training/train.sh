#!/bin/bash
# =============================================================================
# F1-VLA RL Training Launcher
# Supports single-GPU and multi-GPU distributed training
# =============================================================================

set -e

# Get script directory and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(dirname "$SCRIPT_DIR")"
ROBOTWIN_DIR="$(dirname "$RL_DIR")"
F1_VLA_DIR="$(dirname "$ROBOTWIN_DIR")"

# Change to RoboTwin directory (required for relative paths in environment)
cd "$ROBOTWIN_DIR"

# Default values
PHASE="${1:-teacher}"
NUM_GPUS="${2:-1}"
NUM_ENVS="${3:-1}"  # Environments per GPU

# Shift processed arguments
shift 3 2>/dev/null || true
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
        ;;
    adversarial|phase3|adv)
        SCRIPT="$SCRIPT_DIR/train_adversarial_rl.py"
        PHASE_NAME="Adversarial (Phase 3)"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo ""
        echo "Usage: $0 <phase> [num_gpus] [num_envs] [extra_args...]"
        echo ""
        echo "Phases:"
        echo "  teacher     Phase 1: Train World Model (supervised learning)"
        echo "  student     Phase 2: Train Student Policy (exploration)"
        echo "  adversarial Phase 3: Adversarial WM vs Explorer training"
        echo ""
        echo "Examples:"
        echo "  $0 teacher 1 1                    # Single GPU, 1 env"
        echo "  $0 teacher 4 2 --num_episodes 500 # 4 GPUs, 2 envs each"
        echo "  $0 student 4 1 --teacher_path ./outputs/teacher_rl/checkpoint-100"
        exit 1
        ;;
esac

echo "Phase:          $PHASE_NAME"
echo "Training script: $SCRIPT"
echo "=============================================="
echo ""

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
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --multi_gpu \
        "$SCRIPT" \
        --use_ddp \
        --num_envs=$NUM_ENVS \
        $EXTRA_ARGS
else
    # Single GPU training
    echo "[INFO] Launching single GPU training..."
    python "$SCRIPT" \
        --num_envs=$NUM_ENVS \
        $EXTRA_ARGS
fi
