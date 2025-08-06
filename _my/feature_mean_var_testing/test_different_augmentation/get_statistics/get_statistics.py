
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    statistics_dict_fp = "_my/feature_mean_var_testing/test_different_augmentation/get_statistics/statistics.json"
    statistics_dict = {
        "min_diff": 1e10,
        "min_diff_dir": "",
        "min_diff_mean": 1e10,
        "min_diff_mean_dir": "",
        "min_diff_var": 1e10,
        "min_diff_var_dir": "",
    }


    # distribution of generation data
    mean_inf = np.load("_my/feature_mean_var_testing/cal_infset_mean_var/result/mean.npy")
    var_inf = np.load("_my/feature_mean_var_testing/cal_infset_mean_var/result/var.npy")

    data_dir_list = sorted(glob(os.path.join("_my/feature_mean_var_testing/test_different_augmentation/output", "*")))

    for data_dir in tqdm(data_dir_list):
        configc_fp = os.path.join(data_dir, "config.json")
        mean_fp = os.path.join(data_dir, "mean.npy")
        var_fp = os.path.join(data_dir, "var.npy")

        with open(configc_fp, "r", encoding="utf-8") as f:
            data_config = json.load(f)

        mean_val = np.load(mean_fp)
        var_val = np.load(var_fp)

        diff_mean = np.abs(mean_val - mean_inf).sum().item()
        diff_var = np.abs(var_val - var_inf).sum().item()


        # 记录哪个配置的特征分布和生成数据最相似
        if diff_mean+diff_var < statistics_dict["min_diff"]:
            statistics_dict["min_diff"] = diff_mean+diff_var
            statistics_dict["min_diff_dir"] = data_dir
        if diff_mean < statistics_dict["min_diff_mean"]:
            statistics_dict["min_diff_mean"] = diff_mean
            statistics_dict["min_diff_mean_dir"] = data_dir
        if diff_var < statistics_dict["min_diff_var"]:
            statistics_dict["min_diff_var"] = diff_var
            statistics_dict["min_diff_var_dir"] = data_dir
        with open(statistics_dict_fp, "w", encoding="utf-8") as f:
            json.dump(statistics_dict, f, indent=2, ensure_ascii=False)

