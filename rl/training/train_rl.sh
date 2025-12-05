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
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0;7.5;8.0;8.6;8.9;9.0"}
export CUROBO_LOG_LEVEL=${CUROBO_LOG_LEVEL:-"ERROR"}
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"
export MPLBACKEND=${MPLBACKEND:-"Agg"}
CONDA_ENV=${CONDA_ENV:-f1}
OUTPUT_BASE=${OUTPUT_BASE:-"./outputs"}

# 过滤函数：过滤掉JIT编译等无关消息 (使用行缓冲避免输出阻塞)
filter_output() {
    stdbuf -oL -eL grep -v -E "(kinematics_fused_cu not found|geom_cu binary not found|tensor_step_cu not found|lbfgs_step_cu not found|line_search_cu not found|JIT compiling|jit compiling|TORCH_CUDA_ARCH_LIST|pkg_resources is deprecated|UserWarning)" || true
}

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
    echo "Note: First run may take 2-3 minutes for CUDA JIT compilation..."
    # Run with unbuffered Python output and filter noisy JIT/compiler messages
    python -u "$SCRIPT_DIR/train_teacher_rl.py" "$@" 2>&1 | filter_output
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        return $rc
    fi
}

run_student() {
    local checkpoint="${1:-$TEACHER_DIR/checkpoint-latest}"
    shift 2>/dev/null || true
    echo "========== Phase 2: Student Training =========="
    python -u "$SCRIPT_DIR/train_rl.py" --phase student \
        --teacher_checkpoint "$checkpoint" \
        --output_dir "$STUDENT_DIR" "$@" 2>&1 | filter_output
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        return $rc
    fi
}

run_adversarial() {
    local teacher_ckpt="${1:-$TEACHER_DIR/checkpoint-latest}"
    local student_ckpt="${2:-}"
    shift 2 2>/dev/null || true
    echo "========== Phase 3: Adversarial Training =========="
    cmd="python $SCRIPT_DIR/train_rl.py --phase adversarial --teacher_checkpoint $teacher_ckpt --output_dir $ADVERSARIAL_DIR"
    [[ -n "$student_ckpt" ]] && cmd="$cmd --student_checkpoint $student_ckpt"
    # Run and filter output
    eval "$cmd" 2>&1 | filter_output
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        return $rc
    fi
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
