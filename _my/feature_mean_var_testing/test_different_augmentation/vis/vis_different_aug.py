
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def vis_grid(x, y, value, labelx=None, labely=None, title=None):
    # 合并为 DataFrame
    df = pd.DataFrame({'x': x, 'y': y, 'value': value})

    # 以 (x, y) 分组，取均值
    df_grouped = df.groupby(['x', 'y'], as_index=False).mean()

    # 构建规则网格
    xi = np.linspace(df_grouped['x'].min(), df_grouped['x'].max(), 200)
    yi = np.linspace(df_grouped['y'].min(), df_grouped['y'].max(), 200)
    xi, yi = np.meshgrid(xi, yi)

    # 插值
    zi = griddata((x, y), value, (xi, yi), method='linear')

    # 绘图
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(xi, yi, zi, cmap='viridis', shading='auto')
    plt.colorbar(label='Value')
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.axis('auto')
    plt.savefig(os.path.join(output_dir, f"grid_{labelx}_{labely}.png"))
    plt.show()


output_dir = "_my/feature_mean_var_testing/test_different_augmentation/vis"


if __name__ == "__main__":
    data_dir_list = sorted(glob(os.path.join("_my/feature_mean_var_testing/test_different_augmentation/output", "*")))

    diff_list = []
    diff_mean_list = []
    diff_var_list = []
    num_points_list = []
    bbox_noise_strength_list = []
    stitch_noise_strength_list = []

    for data_dir in tqdm(data_dir_list):
        try:
            configc_fp = os.path.join(data_dir, "config.json")
            mean_fp = os.path.join(data_dir, "mean.npy")
            var_fp = os.path.join(data_dir, "var.npy")

            with open(configc_fp, "r", encoding="utf-8") as f:
                data_config = json.load(f)

            diff_mean = data_config["diff_mean"]
            diff_var = data_config["diff_var"]
            num_points = data_config["num_points"]
            bbox_noise_strength = data_config["bbox_noise_strength"]
            stitch_noise_strength = data_config["stitch_noise_strength"]

            diff_list.append(diff_mean + diff_var)
            diff_mean_list.append(diff_mean)
            diff_var_list.append(diff_var)
            num_points_list.append(num_points)
            bbox_noise_strength_list.append(bbox_noise_strength)
            stitch_noise_strength_list.append(stitch_noise_strength)
        except Exception as e:
            print(e)
            continue

    vis_grid(
        x=num_points_list,
        y=bbox_noise_strength_list,
        value=diff_list,
        labelx="num_points",
        labely="bbox_noise_strength"
    )

    vis_grid(
        x=num_points_list,
        y=stitch_noise_strength_list,
        value=diff_list,
        labelx="num_points",
        labely="stitch_noise_strength"
    )
    vis_grid(
        x=bbox_noise_strength_list,
        y=stitch_noise_strength_list,
        value=diff_list,
        labelx="bbox_noise_strength",
        labely="titch_noise_strength"
    )