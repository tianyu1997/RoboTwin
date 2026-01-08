#!/bin/bash

# Configuration
TOTAL_EPISODES=2048
OUTPUT_BASE_DIR="../data/clean_teacher_offline"
CONFIG_PATH="rl/rl_config_clean.yaml"
NUM_ENVS_PER_GPU=8

# Auto-detect free GPUs (memory usage < 500MB)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f1

echo "Detecting free GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found, defaulting to GPU 0"
    AVAILABLE_GPUS=(0)
else
    # Query nvidia-smi for memory used, filter those < 500MB
    # This creates an array of GPU indices, e.g., "0 1 2 3"
    AVAILABLE_GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 500 {print $1}'))
fi

if [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; then
    echo "Error: No free GPUs found (memory usage < 500MB)."
    exit 1
fi

# Limit to max 4 GPUs
MAX_GPUS=4
if [ ${#AVAILABLE_GPUS[@]} -gt $MAX_GPUS ]; then
    GPUS=(${AVAILABLE_GPUS[@]:0:$MAX_GPUS})
else
    GPUS=("${AVAILABLE_GPUS[@]}")
fi

NUM_GPUS=${#GPUS[@]}

# Calculate episodes per GPU (distribute remainder)
BASE_EPISODES_PER_GPU=$((TOTAL_EPISODES / NUM_GPUS))
REMAINDER=$((TOTAL_EPISODES % NUM_GPUS))

echo "Starting data collection with:"
echo "  Total Episodes: $TOTAL_EPISODES"
echo "  Available GPUs: ${#AVAILABLE_GPUS[@]} detected"
echo "  Using $NUM_GPUS GPUs: ${GPUS[@]}"
echo "  Base Episodes per GPU: $BASE_EPISODES_PER_GPU"
echo "  Envs per GPU: $NUM_ENVS_PER_GPU"
echo "  Output Dir: $OUTPUT_BASE_DIR"

# Create base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Launch parallel processes
pids=()
for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    
    # Calculate episodes for this GPU (add 1 if part of remainder)
    if [ $i -lt $REMAINDER ]; then
        episodes_for_this_gpu=$((BASE_EPISODES_PER_GPU + 1))
    else
        episodes_for_this_gpu=$BASE_EPISODES_PER_GPU
    fi

    # Create specific output directory for this GPU to avoid filename conflicts
    # We use GPU ID in the folder name to be explicit and unique
    gpu_output_dir="${OUTPUT_BASE_DIR}/part_gpu${gpu_id}"
    mkdir -p "$gpu_output_dir"
    
    echo "Launching on GPU $gpu_id -> Collecting $episodes_for_this_gpu eps -> Saving to $gpu_output_dir"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python rl/training/collect_data_teacher.py \
        --rl_config "$CONFIG_PATH" \
        --num_episodes "$episodes_for_this_gpu" \
        --output_dir "$gpu_output_dir" \
        --num_envs "$NUM_ENVS_PER_GPU" > "${OUTPUT_BASE_DIR}/log_gpu_${gpu_id}.txt" 2>&1 &
        
    pids+=($!)
done

# Wait for all processes
echo "Waiting for ${#pids[@]} processes to finish..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All data collection processes finished."
echo "Data stored in subdirectories of $OUTPUT_BASE_DIR"
