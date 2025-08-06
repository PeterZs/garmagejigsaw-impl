import os
import torch
import numpy as np
from tqdm import tqdm
from model import build_model
from dataset import build_stylexd_dataloader_inference
from utils import  to_device

if __name__ == "__main__":
    data_type = "Garmage256"
    if not data_type in [
        "Garmage256",
        ]: raise ValueError(f"data_type{data_type} is not valid")
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    model.to("cpu")
    model.pc_cls_threshold = 0.5

    # model.train()
    model.eval()

    mean_list = []
    var_list = []

    inference_loader = build_stylexd_dataloader_inference(cfg)
    all_features = []

    for idx, batch in tqdm(enumerate(inference_loader)):

        batch = to_device(batch, model.device)

        if batch["pcs"].shape[-2] > 3000:
            print("num point too much, continue...")
            continue

        with torch.no_grad():
            inf_rst = model(batch)

        features = inf_rst["features"]  # shape: (N, D)
        all_features.append(features.cpu()[0])

        torch.cuda.empty_cache()

    # 合并所有 features
    all_features = torch.cat(all_features, dim=0)  # shape: (Total_N, D)

    # 直接计算 mean/var
    mean_st = all_features.mean(dim=0)  # (D,)
    var_st = all_features.var(dim=0, unbiased=False)  # (D,)

    print(f"mean: {mean_st}\n"
          f"var: {var_st}\n")
    print(f"mean after norm: {torch.mean((all_features - mean_st) / (var_st + 1e-6).sqrt())}\n"
          f"var after norm: {torch.var((all_features - mean_st) / (var_st + 1e-6).sqrt())}\n")

    np.save(os.path.join("_my/feature_mean_var_testing/cal_infset_mean_var/result","mean.npy"), mean_st)
    np.save(os.path.join("_my/feature_mean_var_testing/cal_infset_mean_var/result","var.npy"), var_st)