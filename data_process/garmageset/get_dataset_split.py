"""
Generated dataset split.

Processing:
  - Filter out garments with too many panels
  - Filter out garments containing panel too small
  - Filter out garment where a single panel contains too many contours.
  - Generate train/validation splits for the filtered dataset
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
    return filtered_list


def keep_percentage(lst, r):
    n = max(1, int(len(lst) * r))
    return random.sample(lst, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    args = parser.parse_args()

    max_panel_num = 32
    min_panel_boundary_len = 16
    max_contour_num_in1panel = 7

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # filter garment ===
    all_garment_dir = sorted(glob(os.path.join(dataset_dir, "*")))
    all_garment_dir = [dir for dir in all_garment_dir if os.path.isdir(dir) and len(os.path.join(dir,"*.obj"))>0]
    # filter out garment with too much panel.
    filtered_garments_dir = filter_toomuchpanel(all_garment_dir, max_panel_num)
    # filter out garment with panel`s boundary point too low.
    filtered_garments_dir = filter_toosmallpanel(filtered_garments_dir, min_panel_boundary_len)
    # filter out garment with too much seperated contours.
    filtered_garments_dir = filter_toomuch_contours(filtered_garments_dir, max_contour_num_in1panel=max_contour_num_in1panel)
    filtered_garments_dir = [os.path.basename(dir_) for dir_ in filtered_garments_dir]

    # get dataset split ===
    split = [9, 1, 0]
    garment_num = len(filtered_garments_dir)
    train_size = int(garment_num * split[0]/sum(split))
    val_size = garment_num-train_size

    train_dataset, val_dataset = random_split(filtered_garments_dir, [train_size, val_size])
    train_split = sorted(list(train_dataset))
    val_split = sorted(list(val_dataset))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)

    another_info = {
        "total_num": garment_num,
        "max_panel_num":max_panel_num,
        "size_train":train_size,
        "size_val":val_size,
    }

    with open(os.path.join(output_dir, "another_info.json"), "w", encoding="utf-8") as f:
        json.dump(another_info, f, ensure_ascii=False, indent=2)
