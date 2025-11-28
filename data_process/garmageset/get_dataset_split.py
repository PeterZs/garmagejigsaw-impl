"""
Generated dataset split.

Processing:
  - Filter out garments with too many panels
  - Filter out garments containing panel too small
  - Filter out garment where a single panel contains too many contours.
  - Generate train/validation/test splits for the filtered dataset
"""


import os
import json
import random
import argparse
from glob import glob
from tqdm import tqdm

import igl
import trimesh
from torch.utils.data import random_split


def filter_toomuchpanel(garment_list, max_panel_num):
    """
    Filter out garments with too many panels.
    """
    filtered_list = []
    for garment_dir in tqdm(garment_list):
        panel_num = len(glob(os.path.join(garment_dir, "piece_*")))
        if panel_num<=max_panel_num:
            filtered_list.append(garment_dir)
    return filtered_list


def filter_toosmallpanel(garment_list, min_panel_boundary_len):
    """
    Filter out garments containing panel too small.
    """
    filtered_list = []
    for garment_dir in tqdm(garment_list):
        mesh_files = sorted(glob(os.path.join(garment_dir, "piece_*.obj")))
        valid = True
        for idx, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(mesh_file, force = "mesh", process = False)
            sum_boundary_point=0
            loops = igl.all_boundary_loop(mesh.faces)
            if len(loops)>max_contour_num_in1panel:
                valid=False
                break
            for loop in loops:
                sum_boundary_point+=len(loop)
            if sum_boundary_point<min_panel_boundary_len:
                valid=False
                break
        if valid:
            filtered_list.append(garment_dir)
    return filtered_list


def filter_toomuch_contours(garment_list, max_contour_num_in1panel=7):
    """
    Filter out garment where a single panel contains too many contours.
    """
    filtered_list = []
    for garment_dir in tqdm(garment_list):
        mesh_files = sorted(glob(os.path.join(garment_dir, "piece_*.obj")))
        valid = True
        for idx, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(mesh_file, force = "mesh", process = False)
            loops = igl.all_boundary_loop(mesh.faces)
            if len(loops)>max_contour_num_in1panel:
                valid=False
                break
        if valid:
            filtered_list.append(garment_dir)
        print(f"{mesh_file} filtered: {len(loops)} contours.")
    return filtered_list


def keep_percentage(lst, r):
    n = max(1, int(len(lst) * r))
    return random.sample(lst, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/data/AIGP/objs_with_stitch")
    parser.add_argument("--output_dir", type=str, default="data_process/garmageset/dataset_split")
    parser.add_argument("--filtered_garments_fp", type=str, default="data_process/garmageset/cache/filtered_garments_dirs.json")
    args = parser.parse_args()

    max_panel_num = 32
    min_panel_boundary_len = 16
    max_contour_num_in1panel = 7

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    filtered_garments_fp = args.filtered_garments_fp

    # filter garment
    if os.path.exists(filtered_garments_fp):
        with open(filtered_garments_fp, "r", encoding="utf-8") as f:
            filtered_garments_dir = json.load(f)
    else:
        all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))

        filtered_garments_dir = filter_toomuchpanel(all_garment_dir, max_panel_num)

        filtered_garments_dir = filter_toosmallpanel(filtered_garments_dir, min_panel_boundary_len)

        filtered_garments_dir = filter_toomuch_contours(filtered_garments_dir, max_contour_num_in1panel=max_contour_num_in1panel)

        filtered_garments_dir = [os.path.basename(dir_) for dir_ in filtered_garments_dir]

        with open(filtered_garments_fp, "w", encoding="utf-8") as f:
            json.dump(filtered_garments_dir, f)

    Q1_num, Q2_num, Q4_num = 0, 0, 0
    for garments_dir in filtered_garments_dir:
        garment_idx = int(garments_dir.split("_")[-1])
        if 0 <= garment_idx < 890: Q1_num += 1
        if 890 <= garment_idx < 11077: Q2_num += 1
        if 11077 <= garment_idx: Q4_num += 1

    Q_type = ["Q1", "Q2", "Q4"]
    Q_range = [1, 0.2, 1]
    Q_list = {k:[] for k in Q_type}

    for garments_dir in tqdm(filtered_garments_dir):
        garment_idx = int(garments_dir.split("_")[-1])
        if 0 <= garment_idx < 890:
            Q_list["Q1"].append(garments_dir)
        elif 890 <= garment_idx < 11077:
            Q_list["Q2"].append(garments_dir)
        else:
            Q_list["Q4"].append(garments_dir)

    # 每个批次仅取一定百分比数量
    for i, Q in enumerate(Q_type):
        if len(Q_list[Q])>0:
            Q_list[Q] = keep_percentage(Q_list[Q], Q_range[i])

    garment_list = []
    for Q in Q_type:
        garment_list.extend(Q_list[Q])

    # # 数据集划分
    # split = [9., 1.]
    # data_list = {"train": [], "val": []}
    # split[0] = int(len(garment_list) * split[0]/sum(split))
    # split[1] = len(garment_list) - split[0]
    #
    # idx_list = range(len(garment_list))
    # train_dataset, val_dataset = random_split(idx_list, split)
    # train_list, val_list = list(train_dataset), list(val_dataset)
    # train_list = [garment_list[idx] for idx in train_list]
    # val_list = [garment_list[idx] for idx in val_list]
    #
    # data_list["train"] = train_list
    # data_list["val"] = val_list

    # with open(os.path.join("_LSR/gen_data_list/output", "stylexd_data_split_reso_256_Q1Q2Q4.pkl"), "wb") as f:
    #     pickle.dump(data_list, f)
    # ===END

    split = [8, 1, 1]
    garment_num = len(garment_list)
    train_size = int(garment_num * split[0]/sum(split))
    val_size = int(garment_num * split[1]/sum(split))
    test_size = garment_num-train_size-val_size

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
        "max_panel_num":max_panel_num,
        "size_train":train_size,
        "size_val":val_size,
        "size_test":test_size,
        "Q1_orig_num":890,
        "Q2_orig_num":10187,
        "Q4_orig_num":1198,
        "Q1_filtered_num":Q1_num,
        "Q2_filtered_num":Q2_num,
        "Q4_filtered_num":Q4_num,
    }

    with open(os.path.join(output_dir, "another_info.json"), "w", encoding="utf-8") as f:
        json.dump(another_info, f, ensure_ascii=False, indent=2)
