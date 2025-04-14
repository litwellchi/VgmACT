#!/bin/bash
# 自动化杀掉所有相关进程
TARGET="replay_test_rlbench_ch.py"

# 找到相关进程并杀死
for pid in $(ps aux | grep $TARGET | grep -v grep | awk '{print $2}'); do
    echo "Killing process $pid"
    kill -9 $pid
done

# 确认 GPU 是否清理干净
nvidia-smi