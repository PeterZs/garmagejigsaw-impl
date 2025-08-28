"""
将模型设置为Train，先进行一轮推理，更新各个BN层的mean var
然后设为eval进行推理
"""


import torch
from tqdm import tqdm
import os.path
import numpy as np
from model import build_model
from dataset import build_stylexd_dataloader_inference

from utils import  to_device, get_pointstitch, pointstitch_2_edgestitch, pointstitch_2_edgestitch2, pointstitch_2_edgestitch3, pointstitch_2_edgestitch4, export_video_results
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize, composite_visualize
from utils.inference.save_result import save_result

from scipy.ndimage import convolve1d

UPDATE_DIS_ITER = 0
ADD_NOISE_INFERENCE = True
NBOISE_STRENGTH = 6
def add_noise_inference(batch, noise_strength=3):
    if ADD_NOISE_INFERENCE:
        # def smooth_using_convolution(noise, k=3):
        #     k = [1] * k
        #     kernel = np.array(k) / sum(k)  # 平滑卷积核
        #     smoothed_noise = convolve1d(noise, kernel, axis=0, mode='nearest')
        #     return smoothed_noise
        print("Warning: add noise on infeerence pcs")
        stitch_noise_strength3 = noise_strength
        noise3 = (np.random.rand(*(batch["pcs"].shape)) * 2. - 1.)
        # for _ in range(1): noise3 = smooth_using_convolution(noise3, k=3)
        noise3 = noise3 / (np.linalg.norm(noise3, axis=1, keepdims=True) + 1e-6)
        noise3 = noise3 * stitch_noise_strength3 * 0.0072
        noise3 = torch.from_numpy(noise3).float().to(batch["pcs"].device)
        batch["pcs_before_add_noise_inference"] = batch["pcs"].clone()
        batch["pcs"] += noise3

def remove_noise_inference(batch):
    if ADD_NOISE_INFERENCE:
        batch["pcs"] = batch["pcs_before_add_noise_inference"]
        del batch["pcs_before_add_noise_inference"]


if __name__ == "__main__":
    data_type = "Garmage256"
    if not data_type in [
        "Garmage64",
        "Garmage64_ML",   # multi-layer of Garmage data
        "Garmage256",
        "brep_reso_128",
        "brep_reso_256",
        "brep_reso_512",
        "brep_reso_1024"
        ]: raise ValueError(f"data_type{data_type} is not valid")
    lst_tmp = []
    from utils.config import cfg
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw")

    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE).cuda()
    # model.to("cpu")
    model.pc_cls_threshold = 0.5

    #是否导出为视频（分图片）
    export_vis_result = False
    export_vis_source = True

    # Warm UP ===
    inference_loader = build_stylexd_dataloader_inference(cfg)

    model.train()
    for i in range(UPDATE_DIS_ITER):
        for idx, batch in tqdm(enumerate(inference_loader)):
            batch = to_device(batch, model.device)

            # [test] add noise on infeerence pcs
            add_noise_inference(batch, noise_strength=NBOISE_STRENGTH)

            with torch.no_grad():
                inf_rst = model(batch)

            torch.cuda.empty_cache()

    # Inference ===
    inference_loader = build_stylexd_dataloader_inference(cfg)
    model.eval()
    for idx, batch in tqdm(enumerate(inference_loader)):
        batch = to_device(batch, model.device)

        # 检查是否存在太大的
        if batch["pcs"].shape[-2]>5000:
            print("num point too mach, continue...")
            continue


        # [test] add noise on infeerence pcs
        add_noise_inference(batch, noise_strength=NBOISE_STRENGTH)

        with torch.no_grad():
            inf_rst = model(batch)

        # [test] add noise on infeerence pcs
        remove_noise_inference(batch)

        if data_type == "Garmage256":
            try:
                data_id = int(os.path.basename(batch['mesh_file_path'][0]).split("_")[1])
            except Exception:
                data_id = idx
            g_basename = os.path.basename(batch['mesh_file_path'][0])
        else: data_id=int(batch['data_id'])


        # 获取点点缝合关系 ------------------------------------------------------------------------------------------------
        if data_type == "Garmage256":
            stitch_mat_full, stitch_pcs, unstitch_pcs, stitch_indices, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="", mat_choice="hun",
                                filter_neighbor_stitch=True, filter_neighbor = 2,
                                filter_too_long=True, filter_length=0.15,
                                filter_too_small=True, filter_logits=0.2,
                                only_triu=True, filter_uncontinue=True,
                                show_pc_cls=False, show_stitch=False))
        else:
            raise NotImplementedError
        if export_vis_result:
            export_video_results(batch, inf_rst, stitch_pcs, unstitch_pcs, stitch_indices, logits, data_id,
                                 vid_len=240, output_dir=os.path.join("_tmp","video_rotate"))
        if export_vis_source:
            batch_np = {}
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch_np[k] = batch[k].detach().cpu()
                else:
                    batch_np[k] = batch[k]
            inf_rst_np = {}
            for k in inf_rst:
                if isinstance(inf_rst[k], torch.Tensor):
                    inf_rst_np[k] = inf_rst[k].detach().cpu()
                else:
                    inf_rst_np[k] = inf_rst[k]
            vis_resource = {
                "batch":batch_np,
                "inf_rst":inf_rst_np,
                "stitch_pcs": stitch_pcs.detach().cpu(),
                "unstitch_pcs": unstitch_pcs.detach().cpu(),
                "stitch_indices": stitch_indices,
                "logits": logits,
            }
        else:
            vis_resource = None

        # 从点点缝合关系获取边边缝合关系 -------------------------------------------------------------------------------------
        batch = to_device(batch, "cpu")
        edgestitch_results = pointstitch_2_edgestitch4(batch, inf_rst,
                                                       stitch_mat_full, stitch_indices_full,
                                                       unstitch_thresh=12, fliter_len=2, division_thresh = 5,
                                                       optimize_thresh_neighbor_index_dis=6,
                                                       optimize_thresh_side_index_dis=12,
                                                       auto_adjust=False)

        garment_json = edgestitch_results["garment_json"]

        # 保存可视化结果 ---------------------------------------------------------------------------------------------------
        fig_comp = composite_visualize(batch, inf_rst,
                                       stitch_indices_full=stitch_indices_full, logits=logits)

        # 保存结果 -------------------------------------------------------------------------------------------------------
        save_dir = "_tmp/inference_ps2es_output"
        data_dir_list = cfg["DATA"]["DATA_TYPES"]["INFERENCE"]
        data_dir = "+".join(data_dir_list)
        save_dir  = os.path.join(save_dir, data_dir)
        save_result(save_dir, data_id=data_id, garment_json=garment_json, fig=fig_comp, g_basename=g_basename
                    , vis_resource=vis_resource, mesh_file_path=batch['mesh_file_path'][0])
        # input("Press ENTER to continue")
        torch.cuda.empty_cache()
