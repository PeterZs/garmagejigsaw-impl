#!/bin/bash

# 如果没传参数，就提示用法并退出
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_cfg_yaml>"
    exit 1
fi

CFG_PATH=$1  # 第一个参数是 cfg 路径

# 固定设置
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/code/Jigsaw_matching

# 启动训练脚本
python garmage_jigsaw/train_garmage_jigsaw.py --cfg "$CFG_PATH"