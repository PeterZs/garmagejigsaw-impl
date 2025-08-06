import os
import json
import torch
import numpy as np
from tqdm import tqdm
from model import build_model
from dataset import build_stylexd_dataloader_train_val
from utils import  to_device

from itertools import product

"""
    val_loader.dataset.num_points   # [1000, 2500]
    val_loader.dataset.bbox_noise_strength  # [0,12]
    val_loader.dataset.stitch_noise_strength    # [0, 12]
    val_loader.dataset.stitch_noise_random_min  # [-1, 0]
    val_loader.dataset.stitch_noise_random_max  # [0, 1]
"""






if __name__ == "__main__":
    data_type = "Garmage256"
    if not data_type in [
        "Garmage256",
        ]: raise ValueError(f"data_type{data_type} is not valid")
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    output_root = "_my/feature_mean_var_testing/test_different_augmentation/output"
    statistics_dict_fp = os.path.join(output_root, "statistics.json")
    if os.path.exists(statistics_dict_fp):
        with open(statistics_dict_fp, "r") as f:
            statistics_dict = json.load(f)
    else:
        statistics_dict = {
            "min_diff": 1e10,
            "min_diff_dir": "",
            "min_diff_mean": 1e10,
            "min_diff_mean_dir": "",
            "min_diff_var": 1e10,
            "min_diff_var_dir": "",
        }

    # 每一轮测多少数据
    sample_num_per_config = 30
    cfg.BATCH_SIZE = 6
    cfg.NUM_WORKERS = 15

    # init model
    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    # model.to("cpu")
    model.eval()
    model.pc_cls_threshold = 0.5

    # distribution of generation data
    mean_inf = np.load("_my/feature_mean_var_testing/cal_infset_mean_var/result/mean.npy")
    var_inf = np.load("_my/feature_mean_var_testing/cal_infset_mean_var/result/var.npy")

    trigger = False

    # init dataloader
    _, val_loader = build_stylexd_dataloader_train_val(cfg, shuffle_val_loader=True)
    for num_points in range(1000, 2501, 300):
        for bbox_noise_strength in range(0, 11, 1):
            for stitch_noise_strength in [0, 1, 2, 3, 4, 6, 8, 10, 12]:
                if num_points==2200 and bbox_noise_strength==10 and stitch_noise_strength==4:
                    trigger = True

                if not trigger:
                    continue

                print("num_points: ", num_points, "\n",
                      "bbox_noise_strength: ", bbox_noise_strength, "\n",
                      "stitch_noise_strength: ", stitch_noise_strength, "\n",
                      )
                output_dir = os.path.join(output_root,
                    "num_points_" + f"{num_points}".zfill(4) +
                    "_bbox_noise_strength_" + f"{bbox_noise_strength}".zfill(2) +
                    "_stitch_noise_strength_" + f"{stitch_noise_strength}".zfill(2)
                )
                os.makedirs(output_dir, exist_ok=True)

                # 如果config.json已经存在，则跳过
                config_fp = os.path.join(output_dir, "config.json")
                if os.path.exists(config_fp):
                    continue

                mean_list = []
                var_list = []
                for idx, batch in tqdm(enumerate(val_loader)):
                    try:
                        # 每轮随机挑一批数据
                        if idx >= sample_num_per_config:
                            break

                        batch = to_device(batch, model.device)

                        # 检查是否存在太大的
                        if batch["pcs"].shape[-2] > 3000:
                            print("num point too mach, continue...")
                            continue

                        with torch.no_grad():
                            inf_rst = model(batch)

                        features = inf_rst["features"]

                        mean = torch.mean(features, dim=-2)
                        var = torch.var(features, dim=-2)
                        mean_list.append(mean)
                        var_list.append(var)
                        # todo 分别保存点云和特征，存到pkl文件中
                    except Exception as e:
                        torch.cuda.empty_cache()
                        continue

                mean_val = torch.mean(torch.cat(mean_list), dim=-2)
                var_val = torch.mean(torch.cat(var_list), dim=-2)
                torch.cuda.empty_cache()

                mean_val = mean_val.detach().cpu().numpy()
                var_val = var_val.detach().cpu().numpy()

                # 误差
                diff_mean = np.abs(mean_val - mean_inf).sum().item()
                diff_var = np.abs(var_val - var_inf).sum().item()

                np.save(os.path.join(output_dir, "mean.npy"), mean_val)
                np.save(os.path.join(output_dir, "var.npy"), var_val)

                config = {
                    "diff_mean": diff_mean,
                    "diff_var": diff_var,
                    "num_points": num_points,
                    "bbox_noise_strength": bbox_noise_strength,
                    "stitch_noise_strength": stitch_noise_strength,
                }
                with open(config_fp, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                # 记录哪个配置的特征分布和生成数据最相似
                if diff_mean+diff_var < statistics_dict["min_diff"]:
                    statistics_dict["min_diff"] = diff_mean+diff_var
                    statistics_dict["min_diff_dir"] = output_dir
                if diff_mean < statistics_dict["min_diff_mean"]:
                    statistics_dict["min_diff_mean"] = diff_mean
                    statistics_dict["min_diff_mean_dir"] = output_dir
                if diff_var < statistics_dict["min_diff_var"]:
                    statistics_dict["min_diff_var"] = diff_var
                    statistics_dict["min_diff_var_dir"] = output_dir
                with open(statistics_dict_fp, "w", encoding="utf-8") as f:
                    json.dump(statistics_dict, f, indent=2, ensure_ascii=False)

