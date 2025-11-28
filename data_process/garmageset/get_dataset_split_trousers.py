"""
制作仅包含裤子的 dataset split
修改自 get_dataset_split.py
"""


import os
import json
from glob import glob
from tqdm import tqdm

from torch.utils.data import random_split
from get_dataset_split import filter_toomuchpanel, filter_toosmallpanel, keep_percentage

if __name__ == '__main__':
    max_panel_num = 64
    min_panel_boundary_len = 16

    dataset_dir = "data/stylexd_jigsaw/train"
    output_dir = "_my/preprocess/stylexd/results/dataset_split"

    filtered_garments_dirs_path = "/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/Jigsaw_matching/_my/preprocess/stylexd/filtered_garments_dirs.json"
    if os.path.exists(filtered_garments_dirs_path):
        with open(filtered_garments_dirs_path, "r", encoding="utf-8") as f:
            filtered_garments_dir = json.load(f)
    else:
        all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))

        filtered_garments_dir = filter_toomuchpanel(all_garment_dir, max_panel_num)

        filtered_garments_dir = filter_toosmallpanel(filtered_garments_dir, min_panel_boundary_len)

        filtered_garments_dir = [os.path.basename(dir_) for dir_ in filtered_garments_dir]

        with open("_my/preprocess/stylexd/filtered_garments_dirs.json", "w", encoding="utf-8") as f:
            json.dump(filtered_garments_dir, f)

    Q1_num, Q2_num, Q4_num = 0, 0, 0
    for garments_dir in filtered_garments_dir:
        garment_idx = int(garments_dir.split("_")[-1])
        if 0 <= garment_idx < 890: Q1_num += 1
        if 890 <= garment_idx < 11077: Q2_num += 1
        if 11077 <= garment_idx: Q4_num += 1

    Q_type = ["Q1", "Q2", "Q4"]
    Q_range = [0, 0, 1]
    Q_list = {k: [] for k in Q_type}

    for garments_dir in tqdm(filtered_garments_dir):
        garment_idx = int(garments_dir.split("_")[-1])
        if 0 <= garment_idx < 890:
            # Q_list["Q1"].append(garments_dir)
            continue
        elif 890 <= garment_idx < 11077:
            # Q_list["Q2"].append(garments_dir)
            continue
        else:
            orig_info_fp = os.path.join(dataset_dir, garments_dir, "annotations", "additional_info.json")
            with open(orig_info_fp, "r", encoding="utf-8") as f:
                obj_fp = json.load(f)['mesh_file_path']
                obj_fp = obj_fp.split("objs_with_stitch")[1]
            if "裤" in obj_fp:
                Q_list["Q4"].append(garments_dir)

    # 每个批次仅取一定百分比数量
    for i, Q in enumerate(Q_type):
        if len(Q_list[Q]) > 0:
            Q_list[Q] = keep_percentage(Q_list[Q], Q_range[i])

    garment_list = []
    for Q in Q_type:
        garment_list.extend(Q_list[Q])

    split = [8, 1, 1]
    garment_num = len(garment_list)
    train_size = int(garment_num * split[0] / sum(split))
    val_size = int(garment_num * split[1] / sum(split))
    test_size = garment_num - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(garment_list, [train_size, val_size, test_size])
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

    another_info = {
        "total_num": garment_num,
        "max_panel_num": max_panel_num,
        "size_train": train_size,
        "size_val": val_size,
        "size_test": test_size,
        "Q1_orig_num": 890,
        "Q2_orig_num": 10187,
        "Q4_orig_num": 1198,
        "Q1_filtered_num": Q1_num,
        "Q2_filtered_num": Q2_num,
        "Q4_filtered_num": Q4_num,
    }

    with open(os.path.join(output_dir, "another_info.json"), "w", encoding="utf-8") as f:
        json.dump(another_info, f, ensure_ascii=False, indent=2)
