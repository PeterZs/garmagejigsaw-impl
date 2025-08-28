# first inference point stitch，then obtain edge stitch

import time
import random
import os.path
from tqdm import tqdm

import torch
from torch import nn
import numpy as np

from model import build_model
from dataset import build_stylexd_dataloader_inference
from utils import  (
    to_device,
    get_pointstitch,
    # pointstitch_2_edgestitch,
    # pointstitch_2_edgestitch2,
    # pointstitch_2_edgestitch3,
    pointstitch_2_edgestitch4,
    export_video_results)
from utils import pointcloud_visualize, pointcloud_and_stitch_visualize, pointcloud_and_stitch_logits_visualize, composite_visualize
from utils.inference.save_result import save_result

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

    # set random seed ===
    def set_seed(seed: int = 42):
        random.seed(seed)  # Python 内置 random
        np.random.seed(seed)  # NumPy
        torch.manual_seed(seed)  # CPU 上的 torch
        torch.cuda.manual_seed(seed)  # 当前 GPU
        torch.cuda.manual_seed_all(seed)  # 所有 GPU（如果使用多卡）

        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积结果一致
        torch.backends.cudnn.benchmark = False  # 避免 cuDNN 使用非确定性算法
    set_seed(seed = 42)

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

    # TEST 找内鬼 ===
    # model.train()
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()  # 这样 BN 使用 batch 统计量
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
    # model.pc_encoder.eval()
    # model.uv_encoder.eval()
    # try:    model.feature_conv.eval()
    # except: pass
    # model.tf_layers_ml.eval()
    # for m in model.tf_layers_ml:
    #     m.bn1.train()
    #     m.bn2.train()
    #     m.bn3.train()
    # model.pc_classifier_layer.eval()
    # model.pc_classifier_layer[0].train()
    # model.affinity_extractor.eval()
    # model.affinity_extractor[0].train()
    # END TEST 找内鬼 ===

    #是否导出为视频（分图片）
    export_vis_result = False
    export_vis_source = True

    inference_loader = build_stylexd_dataloader_inference(cfg)
    for idx, batch in tqdm(enumerate(inference_loader)):

        time1 = time.time()
        batch = to_device(batch, model.device)

        # 检查是否存在太大的
        if batch["pcs"].shape[-2]>3000:
            print("num point too mach, continue...")
            continue

        # [test] add noise on infeerence pcs
        add_noise_inference(batch, noise_strength=NBOISE_STRENGTH)

        if UPDATE_DIS_ITER>0:
            model.train()
            for i in range(UPDATE_DIS_ITER):
                inf_rst = model(batch)
            model.eval()

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
        # # [test]
        # if not data_id==1:
        #     continue
        # 获取并优化点点缝合关系 ------------------------------------------------------------------------------------------------
        # if data_type == "Garmage64":
        #     stitch_mat_full, stitch_indices_full, logits = (
        #         get_pointstitch(batch, inf_rst,
        #                         sym_choice="sym_max", mat_choice="col_max",
        #                         filter_neighbor_stitch=True, filter_neighbor = 7,
        #                         filter_too_long=True, filter_length=0.2,
        #                         filter_too_small=True, filter_logits=0.18,
        #                         only_triu=True, filter_uncontinue=False,
        #                         show_pc_cls=False, show_stitch=False, export_vis_result = False))
        # elif data_type == "Garmage64_ML":
        #     stitch_mat_full, stitch_indices_full, logits = (
        #         get_pointstitch(batch, inf_rst,
        #                         sym_choice="sym_max", mat_choice="col_max",
        #                         filter_neighbor_stitch=True, filter_neighbor = 5,
        #                         filter_too_long=True, filter_length=0.2,
        #                         filter_too_small=True, filter_logits=0.05,
        #                         only_triu=True, filter_uncontinue=False,
        #                         show_pc_cls=False, show_stitch=False, export_vis_result = False))
        pass
        # 一个还算不错的配置 ===
        # edgestitch_results = pointstitch_2_edgestitch4(batch, inf_rst,
        #                                                stitch_mat_full, stitch_indices_full,
        #                                                unstitch_thresh=12, fliter_len=3, division_thresh = 1000,
        #                                                optimize_thresh_neighbor_index_dis=10,
        #                                                optimize_thresh_side_index_dis=30,
        #                                                auto_adjust=False)
        # optimize point-point stitch
        if data_type == "Garmage256":
            stitch_mat_full, stitch_pcs, unstitch_pcs, stitch_indices, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="", mat_choice="col_max",
                                filter_neighbor_stitch=True, filter_neighbor = 1,
                                filter_too_long=True, filter_length=0.12,
                                filter_too_small=True, filter_logits=0.11,
                                only_triu=True, filter_uncontinue=True,
                                show_pc_cls=False, show_stitch=False))
            batch = to_device(batch, "cpu")

        # elif data_type == "brep_reso_128":
        #     stitch_mat_full, stitch_indices_full, logits = (
        #         get_pointstitch(batch, inf_rst,
        #                         sym_choice="sym_max", mat_choice="col_max",
        #                         filter_neighbor_stitch=True, filter_neighbor = 3,
        #                         filter_too_long=True, filter_length=0.1,
        #                         filter_too_small=True, filter_logits=0.2,
        #                         only_triu=True, filter_uncontinue=False,
        #                         show_pc_cls=False, show_stitch=False, export_vis_result = False))
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

        # # 优化缝合关系【OPTIONAL】[有问题，而且不是很必要。。。]
        # optimized_stitch_indices = optimize_pointstitch(batch, inf_rst, stitch_mat_full, stitch_indices_full, show_stitch = True)

        # 从点点缝合关系获取边边缝合关系 -------------------------------------------------------------------------------------

        # edgestitch_results = pointstitch_2_edgestitch2(batch, inf_rst,
        #                                                stitch_mat_full, stitch_indices_full,
        #                                                unstitch_thresh=3, fliter_len=2,
        #                                                optimize_thresh_neighbor_index_dis=12,
        #                                                optimize_thresh_side_index_dis=6)

        # edgestitch_results = pointstitch_2_edgestitch2(batch, inf_rst,
        #                                                stitch_mat_full, stitch_indices_full,
        #                                                unstitch_thresh=12, fliter_len=2,
        #                                                optimize_thresh_neighbor_index_dis=12,
        #                                                optimize_thresh_side_index_dis=6,
        #                                                auto_adjust=False)

        # edgestitch_results = pointstitch_2_edgestitch2(batch, inf_rst,
        #                                                stitch_mat_full, stitch_indices_full,
        #                                                unstitch_thresh=12, fliter_len=2,
        #                                                optimize_thresh_neighbor_index_dis=12,
        #                                                optimize_thresh_side_index_dis=6,
        #                                                auto_adjust=False)

        # # todo stitch_mat_full 丢信息了，只能用stitch_indices_full
        # edgestitch_results = pointstitch_2_edgestitch3(batch, inf_rst,
        #                                                stitch_mat_full, stitch_indices_full,
        #                                                unstitch_thresh=12, fliter_len=3, division_thresh = 9,
        #                                                optimize_thresh_neighbor_index_dis=12,
        #                                                optimize_thresh_side_index_dis=6,
        #                                                auto_adjust=False)

        # edgestitch_results = pointstitch_2_edgestitch4(batch, inf_rst,
        #                                                stitch_mat_full, stitch_indices_full,
        #                                                unstitch_thresh=12, fliter_len=2, division_thresh = 3,
        #                                                optimize_thresh_neighbor_index_dis=12,
        #                                                optimize_thresh_side_index_dis=15,
        #                                                auto_adjust=False)

        edgestitch_results = pointstitch_2_edgestitch4(batch, inf_rst,
                                                       stitch_mat_full, stitch_indices_full,
                                                       unstitch_thresh=12, fliter_len=3, division_thresh = 3,
                                                       optimize_thresh_neighbor_index_dis=6,
                                                       optimize_thresh_side_index_dis=3,
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

        time2 = time.time()
        lst_tmp.append([time2-time1])
        print(np.mean(np.array(lst_tmp)[:,0]))

        # if idx >10:
        #     print("break")
        #     break

