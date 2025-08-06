# 使用的是1：0.2：1的Q124采样，并且使用了新的data augmentation，以及改用InstanceNorm
# sh scripts/train_stylexd_onDocker.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/code/Jigsaw_matching
python _my/train_matching_stylexd.py --cfg experiments/train_matching_stylexd_Q124_2_onDocker.yaml