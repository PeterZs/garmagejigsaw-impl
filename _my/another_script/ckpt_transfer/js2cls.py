# this code is used to clip the checkpoint of all model to pointclassifier model

import torch
import os
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    ckpt_dir = "pretrained/PCglobal+PClocal+UV_tfB3_sn1"
    out_dir = "_my/another_script/ckpt_transfer/results"
    ckpt_list = sorted(glob(os.path.join(ckpt_dir,"*.ckpt")))
    filter_keyword = ["affinity"]
    for ckpt_path in tqdm(ckpt_list[::-1]):
        save_path = os.path.join(out_dir, os.path.basename(ckpt_path))
        ckpt = torch.load(ckpt_path)
        if "state_dict" in ckpt: state_dict = ckpt["state_dict"]
        else: state_dict = ckpt

        filtered_state_dict = {}
        for k, v in state_dict.items():
            need2filter = False
            for keyword in filter_keyword:
                if keyword in k:
                    need2filter = True
                    break
            if need2filter: continue
            filtered_state_dict[k] = v

        if "state_dict" in ckpt: ckpt["state_dict"] = filtered_state_dict
        else: ckpt = filtered_state_dict
        torch.save(ckpt, save_path)
        print(f"Filtered checkpoint saved to {save_path}")