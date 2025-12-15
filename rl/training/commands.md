conda activate f1

bash '/mnt/data2/ty/F1-VLA/RoboTwin/rl/training/train_rl.sh' offline
pkill -f train_teacher_offline.py
tail -f /mnt/data2/ty/F1-VLA/RoboTwin/rl/logs/offline.log

CUDA_VISIBLE_DEVICES = 0  python 