import os
import torch
import numpy as np
from tqdm import tqdm
from model import build_model
from dataset import build_stylexd_dataloader_train_val
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
    mean_st = model.pc_classifier_layer[0].running_mean.detach().cpu().numpy()
    var_st = model.pc_classifier_layer[0].running_var.detach().cpu().numpy()

    np.save(os.path.join("_my/feature_mean_var_testing/get_running_mean_var/results", "mean.npy"), mean_st)
    np.save(os.path.join("_my/feature_mean_var_testing/get_running_mean_var/results", "var.npy"), var_st)