"""
python _my/SigAsia2025_rebuttle/train_on_GCDMM/preprocess/get_dataset_split.py \
    --dataset_dir /home/lsr/data/obj_with_stitch_gcdmm2k/ \
    --output_dir /home/lsr/data/obj_with_stitch_gcdmm2k_datasplit/
"""


import os
import json
import pickle
import argparse
import random
from glob import glob
from tqdm import tqdm

import igl
import trimesh
from torch.utils.data import random_split


# 一个衣服上的Panel如果过多
def filter_toomuchpanel(garment_list, max_panel_num):
    filtered_list = []
    for garment_dir in tqdm(garment_list):
        panel_num = len(glob(os.path.join(garment_dir, "piece_*")))
        if panel_num<=max_panel_num:
            filtered_list.append(garment_dir)
    return filtered_list


def filter_toosmallpanel(garment_list, min_panel_boundary_len):
    filtered_list = []
    for garment_dir in tqdm(garment_list):
        mesh_files = sorted(glob(os.path.join(garment_dir, "piece_*.obj")))
        valid = True
        for idx, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(mesh_file, force = "mesh", process = False)
            sum_boundary_point=0
            for loop in igl.all_boundary_loop(mesh.faces):
                sum_boundary_point+=len(loop)
            if sum_boundary_point<min_panel_boundary_len:
                valid=False
                break
        if valid:
            filtered_list.append(garment_dir)
    return filtered_list


def filter_by_list(garment_list, list_file_path):
    if list_file_path is None or not os.path.exists(list_file_path):
        print("list_file not exist")
        return garment_list
    filtered_list = []
    with open(list_file_path, "r", encoding="utf-8") as f:
        list_f = json.load(f)
    for garment_dir in garment_list:
        if garment_dir in list_f:
            continue
        filtered_list.append(garment_dir)
    return filtered_list


def keep_percentage(lst, r):
    n = max(1, int(len(lst) * r))  # 至少保留1个元素，防止为空
    return random.sample(lst, n)


if __name__ == "__main__":
    max_panel_num = 64
    min_panel_boundary_len = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))

    # 过滤掉Panel数量过多的Garment
    filtered_garments_dir = filter_toomuchpanel(all_garment_dir, max_panel_num)

    # 过滤掉存在特别小的Panel的Garment
    filtered_garments_dir = filter_toosmallpanel(filtered_garments_dir, min_panel_boundary_len)

    # 仅保留文件名
    filtered_garments_dir = [os.path.basename(dir_) for dir_ in filtered_garments_dir]

    garment_list = filtered_garments_dir

    split = [9, 1, 0]
    garment_num = len(garment_list)
    train_size = int(garment_num * split[0]/sum(split))
    val_size = int(garment_num * split[1]/sum(split))
    test_size = garment_num-train_size-val_size

    # 按比例随机划分
    train_dataset, val_dataset, test_dataset = random_split(garment_list, [train_size, val_size, test_size])
    train_split = sorted(list(train_dataset))
    val_split = sorted(list(val_dataset))
    test_split = sorted(list(test_dataset))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)

    another_info = {
        "total_num": garment_num,
        "max_panel_num":max_panel_num,
        # 训练集、验证集、测试集 的长度
        "size_train":train_size,
        "size_val":val_size,
        "size_test":test_size,
    }

    with open(os.path.join(output_dir, "another_info.json"), "w", encoding="utf-8") as f:
        json.dump(another_info, f, ensure_ascii=False, indent=2)
