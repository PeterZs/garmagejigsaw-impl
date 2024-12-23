"""
获取数据集指定划分（比如Q1+Q2 或是 Q4）的统计数据
"""


import os
from glob import glob
import json

import igl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import trimesh

from tqdm import tqdm


def get_panels_num_info(choised_garments_dir, output_dir, results):
    panel_num_list = []
    max_panel_num, min_panel_num = 0, 9999
    min_panel_num_dir, max_panel_num_dir = None, None
    print("\nget_panels_num_info")
    for garment_dir in tqdm(choised_garments_dir):
        panel_num = len(glob(os.path.join(garment_dir, "piece_*")))
        panel_num_list.append(panel_num)
        if panel_num>max_panel_num:
            max_panel_num = panel_num
            max_panel_num_dir = garment_dir
        elif panel_num<min_panel_num:
            min_panel_num=panel_num
            min_panel_num_dir = garment_dir

    df =  pd.DataFrame(panel_num_list, columns=["Panel_Num"])
    sns.set_theme(style="darkgrid")
    sns.histplot(data=df, x="Panel_Num", kde=True)
    plt.xlim(0, 2000)
    plt.savefig(os.path.join(output_dir, "panel_num.jpg"))

    results["min_panel_num"] = min_panel_num
    results["min_panel_num_dir"] = min_panel_num_dir
    results["max_panel_num"] = max_panel_num
    results["max_panel_num_dir"] = max_panel_num_dir



def get_boundary_point_num_info(choised_garments_dir, results):
    boundary_num_list = []
    max_boundary_point_num, min_boundary_point_num = 0, 9999999
    min_boundary_point_num_file, max_boundary_point_num_file = None, None
    print("\nget_panels_num_info")
    for garment_dir in tqdm(choised_garments_dir):
        mesh_files = sorted(glob(os.path.join(garment_dir, "piece_*.obj")))

        for idx, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(mesh_file, force = "mesh", process = False)
            sum_boundary_point=0
            for loop in igl.all_boundary_loop(mesh.faces):
                sum_boundary_point+=len(loop)
            if sum_boundary_point>max_boundary_point_num:
                max_boundary_point_num=sum_boundary_point
                max_boundary_point_num_file=mesh_file
            elif sum_boundary_point<min_boundary_point_num:
                min_boundary_point_num=sum_boundary_point
                min_boundary_point_num_file=mesh_file
            boundary_num_list.append(sum_boundary_point)

    df = pd.DataFrame(boundary_num_list, columns=["Boundary_Num"])
    sns.set_theme(style="darkgrid")
    sns.histplot(data=df, x="Boundary_Num", kde=True)
    plt.savefig(os.path.join(output_dir, "boundary_num.jpg"))

    results["min_boundary_point_num"] = min_boundary_point_num
    results["min_boundary_point_num_file"] = min_boundary_point_num_file
    results["max_boundary_point_num"] = max_boundary_point_num
    results["max_boundary_point_num_file"] = max_boundary_point_num_file



def save_results(results, output_dir):
    with open(os.path.join(output_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    data_batchs = ["Q1", "Q2"]
    data_batchs_name = ""
    for b in data_batchs: data_batchs_name+=b

    dataset_dir = "data/stylexd_jigsaw/train"
    output_dir = os.path.join("_my/another_script/results", "dataset_statistic", data_batchs_name)
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    all_garment_dir = sorted(glob(os.path.join(dataset_dir, "garment_*")))
    choised_garments_dir = []
    if "Q1" in data_batchs:
        choised_garments_dir.extend(all_garment_dir[0:890])
    if "Q2" in data_batchs:
        choised_garments_dir.extend(all_garment_dir[890:11077])
    if "Q4" in data_batchs:
        choised_garments_dir.extend(all_garment_dir[11077:12275])

    get_panels_num_info(choised_garments_dir, output_dir, results)

    get_boundary_point_num_info(choised_garments_dir, results)

    save_results(results, output_dir)