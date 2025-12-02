# first inference point stitch，then obtain edge stitch
import os
from tqdm import tqdm

import torch

from model import build_model
from dataset import build_stylexd_dataloader_inference
from utils import  (
    to_device,
    get_pointstitch,
    pointstitch_2_edgestitch4,
    export_video_results)
from utils import composite_visualize
from utils.inference.save_result import save_result


if __name__ == "__main__":
    data_type = "Garmage256"
    if not data_type in [
        "Garmage256",
        ]: raise ValueError(f"data_type{data_type} is not valid")

    from utils.config import cfg
    from utils.config_2 import cfg2
    from utils.parse_args import parse_args

    args = parse_args("Jigsaw", multimodel=True)

    cfg.DATA.DATA_TYPES.INFERENCE = ["Garmage_SigAisia2025/_补充材料_点云生成/常规1"]
    model = build_model(cfg).load_from_checkpoint(cfg.WEIGHT_FILE,strict=False).cuda()
    model.pc_cls_threshold = 0.5
    if True:
        # model.train()
        model.eval()
        # model.to("cpu")
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

    model2 = build_model(cfg2).load_from_checkpoint(cfg2.WEIGHT_FILE,strict=False).cuda()
    model2.pc_cls_threshold = 0.5
    if True:
        # model2.train()
        model2.eval()
        # model2.to("cpu")
        # model2.pc_encoder.eval()
        # model2.uv_encoder.eval()
        # try:    model2.feature_conv.eval()
        # except: pass
        # model2.tf_layers_ml.eval()
        # for m in model2.tf_layers_ml:
        #     m.bn1.train()
        #     m.bn2.train()
        #     m.bn3.train()
        # model2.pc_classifier_layer.eval()
        # model2.pc_classifier_layer[0].train()
        # model2.affinity_extractor.eval()
        # model2.affinity_extractor[0].train()

    export_vis_result = False
    export_vis_source = True

    inference_loader = build_stylexd_dataloader_inference(cfg)
    for idx, batch in tqdm(enumerate(inference_loader)):

        batch = to_device(batch, model.device)
        inf_rst_cls = model2(batch)
        pc_cls_mask = inf_rst_cls["pc_cls_mask"]

        inf_rst = model(batch, pc_cls_mask)

        if data_type == "Garmage256":
            try:
                data_id = int(os.path.basename(batch['mesh_file_path'][0]).split("_")[1])
            except Exception:
                data_id = idx
            g_basename = os.path.basename(batch['mesh_file_path'][0])
        else: data_id=int(batch['data_id'])

        # 获取点点缝合关系 -------------------------------------------------------------------------------------------------
        if data_type == "Garmage256":
            stitch_mat_full, stitch_pcs, unstitch_pcs, stitch_indices, stitch_indices_full, logits = (
                get_pointstitch(batch, inf_rst,
                                sym_choice="", mat_choice="hun",
                                filter_neighbor_stitch=True, filter_neighbor = 1,
                                filter_too_long=True, filter_length=0.12,
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
        # edgestitch_results = pointstitch_2_edgestitch3(batch, inf_rst,
        #                                                stitch_mat_full, stitch_indices_full,
        #                                                unstitch_thresh=12, fliter_len=3, division_thresh = 6,
        #                                                optimize_thresh_neighbor_index_dis=12,
        #                                                optimize_thresh_side_index_dis=3,
        #                                                auto_adjust=False)
        edgestitch_results = pointstitch_2_edgestitch4(batch, inf_rst,
                                                       stitch_mat_full, stitch_indices_full,
                                                       unstitch_thresh=12, fliter_len=3, division_thresh = 9,
                                                       optimize_thresh_neighbor_index_dis=12,
                                                       optimize_thresh_side_index_dis=6,
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

        torch.cuda.empty_cache()