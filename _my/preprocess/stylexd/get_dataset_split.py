import os
from glob import glob
import json

import numpy as np

from torch.utils.data import random_split

def filter_toomuchpanel(garment_list, max_panel_num):
    filtered_list = []
    for garment_dir in garment_list:
        panel_num = len(glob(os.path.join(garment_dir, "piece_*")))
        if panel_num<=max_panel_num:
            filtered_list.append(garment_dir)
    return filtered_list

if __name__ == "__main__":
    max_panel_num = 64

    dataset_dir = "data/stylexd_jigsaw/train"
    output_dir = "_my/preprocess/stylexd/results/dataset_split"

    all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))
    filtered_garments_dir = filter_toomuchpanel(all_garment_dir, max_panel_num)

    filtered_garments_dir = [os.path.basename(dir_) for dir_ in filtered_garments_dir]

    split = [8, 1, 1]
    garment_num = len(filtered_garments_dir)
    train_size = int(garment_num * split[0]/sum(split))
    val_size = int(garment_num * split[1]/sum(split))
    test_size = garment_num-train_size-val_size

    # 按比例随机划分
    train_dataset, val_dataset, test_dataset = random_split(filtered_garments_dir, [train_size, val_size, test_size])
    train_split = sorted(list(train_dataset))
    val_split = sorted(list(val_dataset))
    test_split = sorted(list(test_dataset))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_split, f, ensure_ascii=False, indent=2)

    another_info = {"total_num": garment_num, "max_panel_num":max_panel_num}
    with open(os.path.join(output_dir, "another_info.json"), "w", encoding="utf-8") as f:
        json.dump(another_info, f, ensure_ascii=False, indent=2)