"""
cd
python _my/SigAsia2025_rebuttle/train_on_GCDMM/preprocess/preprocess_gcdmm.py
    --dataset_dir /data/AIGP/GCD_holdout/objs \
    --out_dir /home/lsr/data/obj_with_stitch_gcdmm2k
"""


import argparse
import os
from glob import glob
from tqdm import tqdm

from _my.preprocess.stylexd.utils_data_process import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str, default="XXXXXX")
    parser.add_argument("--out_dir", required=True, type=str, default="XXXXXX")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir

    file_list = sorted(glob(os.path.join(dataset_dir, "**", "*.obj"), recursive=True))
    for idx, file_path in tqdm(list(enumerate(file_list))):
        garment_idx = idx

        base_name = os.path.basename(file_path)

        garment_save_dir = os.path.join(out_dir,"garment_"+f"{garment_idx}".zfill(5))

        obj_dict = parse_obj_file(file_path)
        # stitch_visualize(np.array(obj_dict["vertices"]),np.array(obj_dict["stitch"]))

        meshes = split_mesh_into_parts(obj_dict)

        save_results(obj_dict, meshes, garment_save_dir, file_path)
