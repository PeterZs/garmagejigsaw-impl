# sh scripts/train_stylexd_onDocker.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/code/Jigsaw_matching
python garmage_jigsaw/train_garmage_jigsaw.py --cfg experiments/finetune_garmage_jigsaw_Q124_onDocker.yaml