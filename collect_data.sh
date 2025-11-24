#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

# If DISPLAY is not set, default to :0 so SAPIEN can initialize an X/OpenGL context
if [ -z "$DISPLAY" ]; then
	export DISPLAY=:0
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
echo "Data collection for ${task_name} with config ${task_config} completed."