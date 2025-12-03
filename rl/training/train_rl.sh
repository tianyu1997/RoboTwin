#!/bin/bash
# F1-VLA RL Training Script
# Usage:
#   ./train_rl.sh teacher              # Phase 1
#   ./train_rl.sh student              # Phase 2 (需要先完成 Phase 1)
#   ./train_rl.sh adversarial          # Phase 3 (需要先完成 Phase 1)
#   ./train_rl.sh all                  # 串联运行所有阶段

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(dirname "$SCRIPT_DIR")"
ROBOTWIN_DIR="$(dirname "$RL_DIR")"
cd "$ROBOTWIN_DIR"

# 配置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONDA_ENV=${CONDA_ENV:-f1}
OUTPUT_BASE=${OUTPUT_BASE:-"./outputs"}

# 激活 conda 环境
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || echo "Warning: conda env '$CONDA_ENV' not found"
fi

# 训练参数
TEACHER_DIR="$OUTPUT_BASE/teacher_rl"
STUDENT_DIR="$OUTPUT_BASE/student_rl"
ADVERSARIAL_DIR="$OUTPUT_BASE/adversarial_rl"

run_teacher() {
    echo "========== Phase 1: Teacher Training =========="
    python "$SCRIPT_DIR/train_rl.py" --phase teacher --output_dir "$TEACHER_DIR" "$@"
}

run_student() {
    local checkpoint="${1:-$TEACHER_DIR/checkpoint-latest}"
    shift 2>/dev/null || true
    echo "========== Phase 2: Student Training =========="
    python "$SCRIPT_DIR/train_rl.py" --phase student \
        --teacher_checkpoint "$checkpoint" \
        --output_dir "$STUDENT_DIR" "$@"
}

run_adversarial() {
    local teacher_ckpt="${1:-$TEACHER_DIR/checkpoint-latest}"
    local student_ckpt="${2:-}"
    shift 2 2>/dev/null || true
    echo "========== Phase 3: Adversarial Training =========="
    cmd="python $SCRIPT_DIR/train_rl.py --phase adversarial --teacher_checkpoint $teacher_ckpt --output_dir $ADVERSARIAL_DIR"
    [[ -n "$student_ckpt" ]] && cmd="$cmd --student_checkpoint $student_ckpt"
    $cmd "$@"
}

run_all() {
    run_teacher
    run_student "$TEACHER_DIR/checkpoint-latest"
    run_adversarial "$TEACHER_DIR/checkpoint-latest" "$STUDENT_DIR/checkpoint-latest"
}

# 主入口
case "${1:-}" in
    teacher|1)
        shift; run_teacher "$@" ;;
    student|2)
        shift; run_student "$@" ;;
    adversarial|3)
        shift; run_adversarial "$@" ;;
    all)
        shift; run_all "$@" ;;
    *)
        echo "Usage: $0 {teacher|student|adversarial|all} [options]"
        echo ""
        echo "Phases:"
        echo "  teacher (1)      - Train world model with random exploration"
        echo "  student (2)      - Train policy with PPO"
        echo "  adversarial (3)  - Adversarial WM vs Explorer training"
        echo "  all              - Run all phases sequentially"
        echo ""
        echo "Environment variables:"
        echo "  CUDA_VISIBLE_DEVICES  - GPU to use (default: 0)"
        echo "  CONDA_ENV             - Conda environment (default: f1)"
        echo "  OUTPUT_BASE           - Output directory base (default: ./outputs)"
        exit 1 ;;
esac
