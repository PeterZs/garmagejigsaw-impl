# inference point stitch
import os.path
import argparse
from glob import glob

from model import build_model
from dataset import build_stylexd_dataloader_inference
from utils import (to_device, get_pointstitch, composite_visualize)


def inf_args(parser:argparse.ArgumentParser):
     parser.add_argument('--data_dir', type=str, default=None)


if __name__ == "__main__":
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("GarmageJigsaw", inf_args)

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    model.eval()

    # pointcloud classify threshold
    model.pc_cls_threshold = cfg.MODEL.PC_CLS_THRESHOLD

    # Init dataloader
    inference_data_list=None
    if args.data_dir is not None:   # If not use data dir in the config file.
        inference_data_list = glob(os.path.join(args.data_dir, "*"))
        inference_data_list = [dir for dir in inference_data_list if os.path.isdir(dir)]
    test_loader = build_stylexd_dataloader_inference(
        cfg, inference_data_list=inference_data_list
    )

    for g_idx, batch in enumerate(test_loader):

        batch = to_device(batch, model.device)

        inf_rst = model(batch)

        stitch_mat_full, stitch_indices_full, logits = get_pointstitch(batch, inf_rst,
                         sym_choice = "sym_max", mat_choice = "hun",
                         filter_too_long = True, filter_length = 0.2,
                         filter_too_small = True, filter_logits = 0.15,
                         show_pc_cls = False, show_stitch = False)

        # export visualize html file.
        fig_comp = composite_visualize(batch, inf_rst, choice=[[True,True,False],[True,True,True]])
        garment_dir = os.path.join("_tmp/inference_ps_output", "garment_"+f"{g_idx}".zfill(5))
        os.makedirs(garment_dir, exist_ok=True)
        fig_comp.write_html(os.path.join(garment_dir,"vis.html"))