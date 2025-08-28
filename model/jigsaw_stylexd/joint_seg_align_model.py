import warnings
from copy import deepcopy

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import BCELoss

from model import MatchingBaseModel, build_encoder
from .affinity_layer import build_affinity
from .pc_classifier_layer import build_pc_classifier
from .attention_layer import PointTransformerLayer, CrossAttentionLayer, PointTransformerBlock
from .feature_conv_layer import feature_conv_layer_contourwise

from utils import permutation_loss
from utils import get_batch_length_from_part_points, merge_c2p_byPanelIns  # , is_contour_OutLine
# from utils import pointcloud_visualize, pointcloud_and_stitch_visualize
from utils import Sinkhorn  # , hungarian, stitch_indices2mat, stitch_mat2indices, get_pointstitch
# from sklearn.metrics import f1_score
# from concurrent.futures import ThreadPoolExecutor


class JointSegmentationAlignmentModel(MatchingBaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.N_point = cfg.DATA.NUM_PC_POINTS               # 点数量

        self.mat_loss_type = cfg.MODEL.LOSS.get("MAT_LOSS_TYPE", "local")
        if self.mat_loss_type not in ["local", "global"]:
            raise ValueError(f"self.mat_loss_type = {self.mat_loss_type} is wrong")
        self.w_cls_loss = self.cfg.MODEL.LOSS.w_cls_loss    # 点分类损失
        self.w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss    # 缝合损失
        self.cal_mat_loss_sym = self.cfg.MODEL.LOSS.get("MAT_LOSS_SYM", True)   # 用斜对称的gt stitching mat来计算loss
        self.pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD  # 二分类结果的阈值

        self.use_point_feature = cfg.MODEL.get("USE_POINT_FEATURE", True)                   # 是否提取点的特征
        self.use_local_point_feature = cfg.MODEL.get("USE_LOCAL_POINT_FEATURE", True)       # 是否提取点的局部特征
        self.use_global_point_feature = cfg.MODEL.get("USE_GLOBAL_POINT_FEATURE", True)     # 是否提取点的局部特征

        self.use_uv_feature = cfg.MODEL.get("USE_UV_FEATURE", False)                                    # 是否使用UV特征
        self.use_local_uv_feature = cfg.MODEL.get("USE_LOCAL_UV_FEATURE", self.use_uv_feature)                     # 是否提取UV的局部特征（为了兼容）
        self.use_global_uv_feature = cfg.MODEL.get("USE_GLOBAL_UV_FEATURE", False)     # 是否提取UV的局部特征

        self.pc_feat_dim = self.cfg.MODEL.get("PC_FEAT_DIM", 128)
        self.uv_feat_dim = self.cfg.MODEL.get("UV_FEAT_DIM", 128)

        # 是否用同一encoder提取特征（假设输入有 点云、uv、normal，concat到一起用同一个encoder提取特征）
        self.encode_all_once = self.cfg.MODEL.get("ENCODE_ALL_ONCE", False)
        if self.encode_all_once:
            assert not self.use_global_point_feature
            assert not self.use_global_uv_feature

        # === 计算backbone提取的特征维度 ===
        self.backbone_feat_dim = 0
        self.in_dim_sum = 0
        if self.use_point_feature:
            if self.use_local_point_feature:
                self.backbone_feat_dim += self.pc_feat_dim
                self.in_dim_sum+=3
            if self.use_global_point_feature:
                self.backbone_feat_dim += self.pc_feat_dim
        if self.use_uv_feature:
            if self.use_local_uv_feature:
                self.backbone_feat_dim += self.uv_feat_dim
                self.in_dim_sum+=2
            if self.use_global_uv_feature:
                self.backbone_feat_dim += self.uv_feat_dim

        assert self.backbone_feat_dim!=0, "No feature will be extracted"

        if not self.encode_all_once:
            if self.use_point_feature:
                self.pc_encoder = self._init_pc_encoder()
            if self.use_uv_feature:
                self.uv_encoder = self._init_uv_encoder()
        else:
            self.all_encoder = self._init_all_encoder(in_feat_dim=self.in_dim_sum, feat_dim=self.backbone_feat_dim)


        # === feature conv ===
        """
        在PointTransformer之前，对属于同一个环的点的特征进行1d卷积操作
        """
        self.use_feature_conv = self.cfg.MODEL.get("FEATURE_CONV", {}).get("USE_FEATURE_CONV", False)
        if self.use_feature_conv:
            feature_conv_ks = self.cfg.MODEL.get("FEATURE_CONV", {}).get("KERNEL_SIZE", 3)
            feature_conv_dilation = self.cfg.MODEL.get("FEATURE_CONV", {}).get("DILATION", 1)
            feature_conv_type = self.cfg.MODEL.get("FEATURE_CONV", {}).get("TYPE", "default")
            self.feature_conv = feature_conv_layer_contourwise(
                in_channels=self.backbone_feat_dim,
                out_channels=self.backbone_feat_dim,
                type = feature_conv_type,
                kernel_size=feature_conv_ks,
                dilation=feature_conv_dilation,
            )
        """
        在PointTransformer之后，对属于同一个环的点的特征进行1d卷积操作
        """
        self.use_feature_conv_2 = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("USE_FEATURE_CONV", False)
        if self.use_feature_conv_2:
            feature_conv_ks = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("KERNEL_SIZE", 3)
            feature_conv_dilation = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("DILATION", 1)
            feature_conv_type = self.cfg.MODEL.get("FEATURE_CONV_2", {}).get("TYPE", "default")
            self.feature_conv_2 = feature_conv_layer_contourwise(
                in_channels=self.backbone_feat_dim,
                out_channels=self.backbone_feat_dim,
                type = feature_conv_type,
                kernel_size=feature_conv_ks,
                dilation=feature_conv_dilation,
            )

        self.aff_feat_dim = self.cfg.MODEL.AFF_FEAT_DIM
        assert self.aff_feat_dim % 2 == 0, "The affinity feature dimension must be even!"
        self.half_aff_feat_dim = self.aff_feat_dim // 2

        self.pccls_feat_dim = self.backbone_feat_dim
        self.pc_classifier_layer = self._init_pc_classifier_layer()
        self.affinity_extractor = self._init_affinity_extractor()
        self.affinity_layer = self._init_affinity_layer()
        self.sinkhorn = self._init_sinkhorn()

        self.tf_layer_num = cfg.MODEL.get("TF_LAYER_NUM", 1)
        assert self.tf_layer_num >= 0, "tf_layer_num too small"
        self.use_tf_block = cfg.MODEL.get("USE_TF_BLOCK", False)
        # === 如果不使用 PointTransformer Block === (这种方法仅能够支持 self.tf_layer_num <= 2，在层数过多的情况下会出现训练过程中的梯度骤增)
        if not self.use_tf_block:
            # [todo] 分成以下三种情况是为了兼容过去的checkpoints，将来可以考虑将这三个if整合到一起，重新训练一遍
            if self.tf_layer_num == 1:
                self.tf_self1 = PointTransformerLayer(
                    in_feat=self.backbone_feat_dim, out_feat=self.backbone_feat_dim,
                    n_heads=self.cfg.MODEL.TF_NUM_HEADS, nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE,
                )
                self.tf_cross1 = CrossAttentionLayer(d_in=self.backbone_feat_dim,
                                                     n_head=self.cfg.MODEL.TF_NUM_HEADS, )
                self.tf_layers = [("self", self.tf_self1), ("cross", self.tf_cross1)]
            elif self.tf_layer_num > 1:
                # 加入ModuleList是为了让这些模型在训练开始时自动装入GPU
                self.tf_layers_ml = nn.ModuleList()
                self.tf_layers = []
                for i in range(self.tf_layer_num):
                    self_tf_layer = PointTransformerLayer(
                        in_feat=self.backbone_feat_dim,
                        out_feat=self.backbone_feat_dim,
                        n_heads=self.cfg.MODEL.TF_NUM_HEADS,
                        nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE, )
                    cross_tf_layer = CrossAttentionLayer(
                        d_in=self.backbone_feat_dim,
                        n_head=self.cfg.MODEL.TF_NUM_HEADS, )
                    self.tf_layers_ml.append(self_tf_layer)
                    self.tf_layers_ml.append(cross_tf_layer)
                    self.tf_layers.append(("self", self_tf_layer))
                    self.tf_layers.append(("cross", cross_tf_layer))
            elif self.tf_layer_num == 0:
                self.tf_layers = []
        # === 如果使用 PointTransformer Block ===
        else:
            layers_name = ["self", "cross"] * self.tf_layer_num
            self.tf_layer_num = len(layers_name)
            self.tf_layers_ml = nn.ModuleList()
            self.tf_layers = []
            for i in range(self.tf_layer_num):
                tf_block = PointTransformerBlock(
                    name=layers_name[i],
                    backbone_feat_dim=self.backbone_feat_dim,
                    num_points=self.N_point,
                    n_heads=self.cfg.MODEL.TF_NUM_HEADS,
                    nsampmle=self.cfg.MODEL.TF_NUM_SAMPLE,
                )
                self.tf_layers_ml.append(tf_block)
                self.tf_layers.append(("block", tf_block))

        # 分阶段学习 -----------------------------------------------------------------------------------------------------
        self.is_train_in_stages = self.cfg.MODEL.get("IS_TRAIN_IN_STAGE", False)  # 是否分阶段学习
        self.init_dynamic_adjustment()  # 分阶段学习的初始化

    def _init_pc_encoder(self):
        # 提取点云特征的pointnet2
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.pc_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_uv_encoder(self):
        # 提取UV特征的pointnet2
        in_feat_dim = 3
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=self.uv_feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_all_encoder(self, in_feat_dim, feat_dim):
        # 提取点云特征的pointnet2
        encoder = build_encoder(
            self.cfg.MODEL.ENCODER,
            feat_dim=feat_dim,
            global_feat=False,
            in_feat_dim=in_feat_dim,
        )
        return encoder

    def _init_affinity_extractor(self):
        norm = self.cfg.MODEL.STITCHPREDICTOR.get("NORM", "batch")

        assert norm in ["batch", "instance"]

        def get_norm(norm_type, dim):
            if norm_type == "batch":
                return nn.BatchNorm1d(dim)
            elif norm_type == "instance":
                return nn.InstanceNorm1d(dim)
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")

        affinity_extractor = nn.Sequential(
            get_norm(norm, self.backbone_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.backbone_feat_dim, self.aff_feat_dim, 1),
        )
        return affinity_extractor

    def _init_affinity_layer(self):
        affinity_layer = build_affinity(
            self.cfg.MODEL.AFFINITY.lower(),
            self.aff_feat_dim,
            norm=self.cfg.MODEL.POINTCLASSIFIER.get("NORM", "batch"),
        )
        return affinity_layer

    def _init_pc_classifier_layer(self):
        pc_classifier_layer = build_pc_classifier(
            self.pccls_feat_dim,
            norm=self.cfg.MODEL.POINTCLASSIFIER.get("NORM", "batch"),
        )
        return pc_classifier_layer

    def _init_sinkhorn(self):
        return Sinkhorn(
            max_iter=self.cfg.MODEL.SINKHORN_MAXITER, tau=self.cfg.MODEL.SINKHORN_TAU
        )

    def _extract_pointcloud_feats(self, part_pcs, batch_length):
        B, N_sum, _ = part_pcs.shape  # [B, N_sum, 3]
        valid_pcs = part_pcs.reshape(B * N_sum, -1)
        valid_feats = self.pc_encoder(valid_pcs, batch_length)  # [B * N_sum, F]
        pc_feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return pc_feats

    def _extract_uv_feats(self, uv, batch_length):
        B, N_sum, _ = uv.shape  # [B, N_sum, 3]
        valid_uv = uv.reshape(B * N_sum, -1)
        valid_feats = self.uv_encoder(valid_uv, batch_length)  # [B * N_sum, F]
        uv_feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return uv_feats

    def _extract_all_feats(self, pcs_list:list, batch_length):
        pcs_all = torch.cat(pcs_list, dim=-1)
        B, N_sum, _ = pcs_all.shape  # [B, N_sum, 3]
        valid_pcs = pcs_all.reshape(B * N_sum, -1)
        valid_feats = self.all_encoder(valid_pcs.to(torch.float32), batch_length)  # [B * N_sum, F]
        feats = valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        return feats

    def _get_stitch_pcs_feats(self, feat, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point, F_dim):
        critical_feats = torch.zeros(B_size, N_point, F_dim, device=self.device, dtype=feat.dtype)
        for b in range(B_size):
            critical_feats[b, : n_stitch_pcs_sum[b]] = feat[b, pc_cls_mask[b] == 1]
        return critical_feats

    def _get_stitch_pcs_gt_mat(self, mat_gt, pc_cls_mask, B_size,N_point,n_stitch_pcs_sum):
        stitch_pcs_gt_mat = torch.zeros((B_size,N_point,N_point),device=mat_gt.device)
        for B in range(B_size):
            stitch_pcs_gt_mat[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]] = mat_gt[B][pc_cls_mask[B] == 1][:,pc_cls_mask[B] == 1]
        return stitch_pcs_gt_mat

    def forward(self, data_dict, pc_cls_mask_predefine=None):
        out_dict = dict()
        # 根据 panel_instance_seg，将contours合并为panels(为了更有效的提取panel内部的环的特征)
        data_dict = merge_c2p_byPanelIns(deepcopy(data_dict))

        pcs = data_dict.get("pcs", None)  # [B_size, N_point, 3]
        uv = data_dict.get("uv", None)

        B_size, N_point, _ = pcs.shape
        piece_id = data_dict["piece_id"]
        part_valids = data_dict["part_valids"]
        n_valid = torch.sum(part_valids, dim=1).to(torch.long)  # [B]
        # panel_instance_seg = data_dict["panel_instance_seg"]
        n_pcs = data_dict["n_pcs"]
        num_parts = data_dict["num_parts"]
        # contour_n_pcs = data_dict["contour_n_pcs"]
        # num_contours = data_dict["num_contours"]

        batch_length = get_batch_length_from_part_points(n_pcs, n_valids=n_valid).to(self.device)


        # # test
        # test_encoder = build_encoder(
        #     self.cfg.MODEL.ENCODER,
        #     feat_dim=self.backbone_feat_dim,
        #     global_feat=False,
        #     in_feat_dim=5,
        # )
        # test_encoder = test_encoder.to(self.device)
        # test_pcs_all = torch.cat([pcs, uv[..., :-1]], dim=-1)
        # B, N_sum, _ = test_pcs_all.shape  # [B, N_sum, 3]
        # test_valid_pcs = test_pcs_all.reshape(B * N_sum, -1)
        # test_valid_feats = test_encoder(test_valid_pcs.to(torch.float32), batch_length)  # [B * N_sum, F]
        # test_feats = test_valid_feats.reshape(B, N_sum, -1)  # [B, N_sum, F]
        # # test end

        # === 用PointNet从点云或UV中提取特征，并拼接 ===
        if not self.encode_all_once:
            features = []
            if self.use_point_feature:
                if self.use_local_point_feature:
                    local_pcs_feats = self._extract_pointcloud_feats(pcs, batch_length)
                    features.append(local_pcs_feats)
                if self.use_global_point_feature:
                    pcs_feats_global = self._extract_pointcloud_feats(pcs, torch.tensor([N_point] * B_size))
                    features.append(pcs_feats_global)
            if self.use_uv_feature:
                if self.use_local_uv_feature:
                    uv_feats = self._extract_uv_feats(uv.to(torch.float32), batch_length)
                    features.append(uv_feats)
                if self.use_global_uv_feature:
                    uv_feats = self._extract_uv_feats(uv.to(torch.float32), torch.tensor([N_point] * B_size))
                    features.append(uv_feats)
            assert len(features) > 0, "None feature extracted!"
            features = torch.concat(features, dim=-1)
        else:
            # 用同一个encoder提取所有特征
            pcs_list = []
            if self.use_point_feature:
                if self.use_local_point_feature:
                    pcs_list.append(pcs)
            if self.use_uv_feature:
                if self.use_local_uv_feature:
                    pcs_list.append(uv[..., :-1])
            features = self._extract_all_feats(pcs_list, batch_length)

        # === 按板片为单位进行一维卷积
        if self.use_feature_conv:
            features_list = []

            for b in range(B_size):
                n_parts = num_parts[b]
                n_pcs_part = n_pcs[b][:n_parts]
                # [test]
                # print(n_pcs_part)
                n_pcs_part_cumsum = torch.cumsum(n_pcs_part, dim=-1)

                features_b = []
                for i in range(len(n_pcs_part_cumsum)):
                    st = 0 if i == 0 else n_pcs_part_cumsum[i - 1]
                    ed = n_pcs_part_cumsum[i]
                    # 一个板片上的特征
                    part_feature = self.feature_conv(features[b][st:ed])  # shape: (N_i, D)
                    features_b.append(part_feature)
                features_list.append(torch.cat(features_b, dim=0))  # shape: (N_b, D)

            features = torch.stack(features_list, dim=0)

        # === 提取出的特征输入到PointTransformer Layers\Blocks ===
        pcs_flatten = pcs.reshape(-1, 3).contiguous()
        # 顶点特征输入到PointTransformer层中，获取点与点之间的关系
        for name, layer in self.tf_layers:
            # 如果是自注意力层
            if name == "self":
                features = (
                    layer(
                        pcs_flatten,
                        features.view(-1, self.backbone_feat_dim),
                        batch_length,
                    ).view(B_size, N_point, -1).contiguous()
                )
            # 如果是交叉注意力层
            elif name == "cross":
                features = layer(features)
            # 如果是被封装成块了
            elif name == "block" and self.use_tf_block:
                features = (
                    layer(
                        pcs_flatten,
                        features,
                        batch_length,
                        B_size,
                        N_point
                    )
                )

        # === 按板片为单位进行一维卷积
        if self.use_feature_conv_2:
            features_list = []

            for b in range(B_size):
                n_parts = num_parts[b]
                n_pcs_part = n_pcs[b][:n_parts]
                n_pcs_part_cumsum = torch.cumsum(n_pcs_part, dim=-1)

                features_b = []
                for i in range(len(n_pcs_part_cumsum)):
                    st = 0 if i == 0 else n_pcs_part_cumsum[i - 1]
                    ed = n_pcs_part_cumsum[i]
                    # 一个板片上的特征
                    part_feature = self.feature_conv_2(features[b][st:ed])  # shape: (N_i, D)
                    features_b.append(part_feature)
                features_list.append(torch.cat(features_b, dim=0))  # shape: (N_b, D)

            features = torch.stack(features_list, dim=0)

        out_dict.update({"features": features})

        # 预测点分类
        pc_cls = self.pc_classifier_layer(features.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        pc_cls = torch.sigmoid(pc_cls)
        pc_cls_mask = ((pc_cls>self.pc_cls_threshold) * 1)

        if pc_cls_mask_predefine is not None:
            pc_cls_mask = pc_cls_mask_predefine

        out_dict.update({"pc_cls": pc_cls,
                         "pc_cls_mask": pc_cls_mask,})

        # === 预测点点缝合关系 ===
        # pointcloud_visualize(pcs[0][pc_cls_mask[0]==1])
        n_stitch_pcs_sum = torch.sum(pc_cls_mask, dim=-1)

        stitch_pcs_feats = self._get_stitch_pcs_feats(features, n_stitch_pcs_sum, pc_cls_mask, B_size, N_point, self.backbone_feat_dim)
        out_dict.update({"n_stitch_pcs_sum": n_stitch_pcs_sum,})

        affinity_feat = self.affinity_extractor(stitch_pcs_feats.permute(0, 2, 1))
        affinity_feat = affinity_feat.permute(0, 2, 1)
        affinity_feat = torch.cat(
            [
                F.normalize(
                    affinity_feat[:, :, : self.half_aff_feat_dim], p=2, dim=-1
                ),
                F.normalize(
                    affinity_feat[:, :, self.half_aff_feat_dim:], p=2, dim=-1
                ),
            ],
            dim=-1,
        )
        # 预测 Affinity matrix: s
        affinity = self.affinity_layer(affinity_feat, affinity_feat)
        B_point_num = torch.tensor([N_point]*B_size)
        out_dict.update({"B_point_num": B_point_num})

        mat = self.sinkhorn(affinity, n_stitch_pcs_sum, n_stitch_pcs_sum)
        out_dict.update({"ds_mat": mat, })  # [B, N_, N_]

        return out_dict


    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        pcs = data_dict["pcs"]
        # pcs_gt = data_dict["pcs_gt"]
        B_size, N_point, _ = pcs.shape
        n_stitch_pcs_sum = out_dict["n_stitch_pcs_sum"]

        loss_dict = {
            "batch_size": B_size,
        }

        ds_mat = out_dict["ds_mat"]  # 预测的点点匹配概率
        # 【下三角为0】所有采样点的gt缝合关系
        gt_mat = data_dict.get("mat_gt", None)
        # calculate cls loss ---------------------------------------------------------------------------------------
        # 【概率】预测的点的二分类概率
        pc_cls = (out_dict.get("pc_cls", None)).squeeze(-1)
        # 【1为是缝合点】pc_cls>pc_cls_threshold得到的分类结果
        pc_cls_mask = (out_dict.get("pc_cls_mask", None)).squeeze(-1)
        pc_cls_gt = (torch.sum(gt_mat + gt_mat.transpose(-1, -2),
                               dim=-1) == 1) * 1.0
        cls_loss = BCELoss()(pc_cls, pc_cls_gt)
        loss_dict.update({"cls_loss": cls_loss,})

        # calculate matching loss ----------------------------------------------------------------------------------
        """
        【gt_mat是单向的】
        
        之所以让模型与预测一个尽量对称的ds_mat，而不是仅取ds_mat的上半部分与gt_mat计算损失，是因为我希望模型能真正学到通过几何关系来计算
        缝合关系，即：如果点A被预测和点B缝合，那么对B进行预测的结果也应当是A
        
        因此，我选择让模型去和双向的gt_mat（上三角被复制到下三角）来计算损失，从而让affinity layer学到从几何角度推出缝合关系，从而
        解决那些错误缝合。在使用模型的inference结果时，可以先将行中最大值小于阈值的行（以及对应的列）全部剔除，然后进行匈牙利算法
        """

        if self.mat_loss_type=="local":
            n_stitch_pcs_sum = n_stitch_pcs_sum.reshape(-1)

            # 获取 对称的 gt_mat
            stitch_pcs_gt_mat_half = self._get_stitch_pcs_gt_mat(gt_mat, pc_cls_mask, B_size, N_point, n_stitch_pcs_sum)
            if self.cal_mat_loss_sym:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half + stitch_pcs_gt_mat_half.transpose(-1, -2)
            else:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half
            mat_loss = permutation_loss(
                ds_mat, stitch_pcs_gt_mat.float(), n_stitch_pcs_sum, n_stitch_pcs_sum
            )
        elif self.mat_loss_type=="global":
            n_stitch_pcs_sum = n_stitch_pcs_sum.reshape(-1)

            # 获取 global的 ds_mat
            ds_mat_global = torch.zeros((B_size,N_point,N_point), device=ds_mat.device)
            for B in range(B_size):
                mask = pc_cls_mask[B] == 1
                indices = torch.where(mask)[0]
                ds_mat_global[B].index_put_((indices[:, None], indices), ds_mat[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]])

            # 获取 对称的 gt_mat
            stitch_pcs_gt_mat_half = gt_mat

            if self.cal_mat_loss_sym:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half + stitch_pcs_gt_mat_half.transpose(-1, -2)
            else:
                stitch_pcs_gt_mat = stitch_pcs_gt_mat_half
            n_pcs_sum = torch.ones((B_size), device=pcs.device, dtype=torch.int64) * N_point
            mat_loss = permutation_loss(
                ds_mat_global, stitch_pcs_gt_mat.float(), n_pcs_sum, n_pcs_sum
            )
        else:
            raise  NotImplementedError(f"self.mat_loss_type={self.mat_loss_type}")
        # stitch_pcs=pcs[0][pc_cls_mask[0] == 1]
        # pointcloud_visualize(stitch_pcs)
        # pointcloud_and_stitch_visualize(stitch_pcs, stitch_mat2indices(stitch_pcs_gt_mat[0].cpu().detach().numpy()))
        # 【上下三角sum相等】双向的的缝合关系

        loss_dict.update(
            {
                "mat_loss": mat_loss,
            }
        )


        loss = (cls_loss * self.w_cls_loss+
                mat_loss * self.w_mat_loss)
        loss_dict.update({"loss": loss,})


        # ------------------------- Following Only For Evaluation --------------------------------------------------

        # calculate stitch_dis_loss --------------------------------------------------------------------------------
        # mean distance between stitched points
        with torch.no_grad():
            # calculate mean dist between stitched points
            Dis = torch.sqrt(((pcs[:, :, None, :] - pcs[:, None, :, :]) ** 2).sum(dim=-1)) + (
                        torch.eye(pcs.shape[1])).to(pcs.device)

            for B in range(B_size):
                Dis[B][:n_stitch_pcs_sum[B], :n_stitch_pcs_sum[B]] = Dis[B][pc_cls_mask[B] == 1][:,pc_cls_mask[B] == 1]

            stitch_dis_loss = torch.sum(torch.mul(Dis, ds_mat)) / torch.sum(n_stitch_pcs_sum)
            loss_dict.update(
                {
                    "stitch_dis_loss": stitch_dis_loss,
                }
            )

        # calculate TP FP TN FN ACC TPR TNR ------------------------------------------------------------------------
        with torch.no_grad():
            B_size, N_point, _ = data_dict["pcs"].shape

            stitch_mat_ = out_dict["ds_mat"]
            mat_gt = data_dict["mat_gt"]
            pc_cls = out_dict["pc_cls"]
            threshold = self.pc_cls_threshold

            pc_cls_ = pc_cls.squeeze(-1)
            pc_cls_gt = (torch.sum(mat_gt[:, :N_point] + mat_gt[:, :N_point].transpose(-1, -2),
                                   dim=-1) == 1) * 1.0
            indices = pc_cls_ > threshold
            TP = torch.sum(torch.sum(indices[pc_cls_gt == 1] * 1))
            # print(f"TP={TP}")
            indices = pc_cls_ > threshold
            FP = torch.sum(torch.sum((indices[pc_cls_gt == 1] == False) * 1))
            # print(f"TN={FP}")
            indices = pc_cls_ < threshold
            TN = torch.sum(torch.sum(indices[pc_cls_gt == 0] * 1))
            # print(f"FP={TN}")
            indices = pc_cls_ < threshold
            FN = torch.sum(torch.sum((indices[pc_cls_gt == 0] == False) * 1))
            # print(f"FN={FN}")

            ACC = (TP + TN) / (TP + FP + TN + FN)
            # print(f"ACC={ACC:.4f}")
            TPR = TP / (TP + FN)
            # print(f"TPR={TPR:.4f}")
            TNR = TN / (FP + TN)
            # print(f"FPR={TNR:.4f}")
            PRECISION = TP / (TP + FP + 1e-5)
            loss_dict.update(
                {
                    "pcs_1_ACC": ACC,
                    "pcs_2_TPR": TPR,
                    "pcs_3_TNR": TNR,
                    "pcs_4_PRECISION": PRECISION,
                }
            )

        if self.is_train_in_stages and self.trainer.validating:
            with torch.no_grad():
                self.val_ACC_list.append(float(ACC))

        # # point-point stitching metrics ------------------------------------------------------------------------
        # with torch.no_grad(), warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #
        #     if torch.randint(0, 20, (1,)) == 0: # 避免过多执行，导致训练速度降低
        #         data_dict_clone = {k: v.cpu().clone().detach() if isinstance(v, torch.Tensor) else v for k, v in zip(data_dict.keys(), data_dict.values())}
        #         out_dict_clone = {k:v.cpu().clone().detach() if isinstance(v, torch.Tensor) else v for k,v in zip(out_dict.keys(),out_dict.values())}
        #
        #         item_num = data_dict_clone["pcs"].shape[0]
        #         for i in range(item_num):
        #             data_dict_c = {k:v[i][None,...] if isinstance(v[i],torch.Tensor) else v[i] for k, v in zip(data_dict_clone.keys(), data_dict_clone.values())}
        #             out_dict_c = {k:v[i][None,...] if isinstance(v[i],torch.Tensor) else v[i] for k, v in zip(out_dict_clone.keys(), out_dict_clone.values())}
        #
        #             # stitch_mat_full, _, _, _, stitch_indices_full, logits = (
        #             #     get_pointstitch(data_dict_c,
        #             #                     out_dict_c,
        #             #                     sym_choice="", mat_choice="hun",
        #             #                     filter_neighbor_stitch=False, filter_neighbor=3,
        #             #                     filter_too_long=False, filter_length=0.1,
        #             #                     filter_too_small=True, filter_logits=0.12,
        #             #                     only_triu=False, filter_uncontinue=False,
        #             #                     show_pc_cls=False, show_stitch=False))
        #
        #             # get_pointstitch ===
        #             def safe_get_pointstitch(data_dict_c, out_dict_c):
        #                 try:
        #                     return get_pointstitch(                               data_dict_c, out_dict_c,
        #                         sym_choice="", mat_choice="hun",
        #                         filter_neighbor_stitch=False, filter_neighbor=3,
        #                         filter_too_long=False, filter_length=0.1,
        #                         filter_too_small=True, filter_logits=0.12,
        #                         only_triu=False, filter_uncontinue=False,
        #                         show_pc_cls=False, show_stitch=False
        #                     )
        #                 except Exception as e:
        #                     print("[跳过 get_pointstitch] 异常:", e)
        #                     return None, None, None, None, None, None
        #
        #             # 定义任务函数
        #             def task(data_dict_c, out_dict_c):
        #                 stitch_mat_full, _, _, _, stitch_indices_full, logits = safe_get_pointstitch(data_dict_c, out_dict_c)
        #                 return stitch_mat_full, stitch_indices_full, logits
        #
        #             data_list = [(data_dict_c, out_dict_c)]
        #
        #             results = []
        #             with ThreadPoolExecutor(max_workers=1) as executor:
        #                 futures = [executor.submit(task, data_dict_c, out_dict_c) for data_dict_c, out_dict_c in data_list]
        #                 for i, future in enumerate(futures):
        #                     try:
        #                         result = future.result()
        #                         results.append(result)
        #                     except Exception as e:
        #                         print(f"任务{i} 异常:", e)
        #                         results.append((None, None, None))
        #
        #             stitch_mat_full=results[0][0]
        #             stitch_indices_full=results[0][1]
        #             logits=results[0][2]
        #             if stitch_mat_full is None and stitch_indices_full is None and logits is None:
        #                 continue
        #
        #             # END get_pointstitch ===
        #
        #
        #             device = stitch_mat_full.device
        #             mat_gt = data_dict_c["mat_gt"]
        #             mat_gt = mat_gt == 1
        #             pcs = data_dict_c["pcs"].squeeze(0)
        #             normalize_range = data_dict_c['normalize_range']
        #
        #             pc_cls = out_dict_c["pc_cls"]
        #             threshold = self.pc_cls_threshold
        #             pc_cls_ = pc_cls.squeeze(-1)
        #
        #             point_num = pcs.shape[-2]
        #             if self.cal_mat_loss_sym:
        #                 mat_gt = mat_gt + mat_gt.transpose(-1, -2)
        #             stitch_mat_full = stitch_mat_full >= 0.9
        #
        #             # === 缝合Recall ===
        #             RECALL_STITCH = torch.sum(torch.bitwise_and(stitch_mat_full == mat_gt, mat_gt)) / torch.sum(mat_gt)
        #             loss_dict.update({
        #                 "STITCH_RECALL": RECALL_STITCH,
        #             })
        #
        #             # === 缝合 AMD (average mean distance) ===
        #             stitch_indices_pred = stitch_mat2indices(stitch_mat_full)
        #             stitch_indices_gt = stitch_mat2indices(mat_gt)
        #             if stitch_indices_pred.shape[-2]>0:
        #                 mask_cls_pred = (pc_cls_ > threshold).squeeze(0)
        #                 idx_range = torch.arange(0, point_num, dtype=torch.int64, device=mat_gt.device)
        #                 stitch_map = torch.zeros((point_num), dtype=torch.int64).to(device) - 1
        #                 stitch_map[stitch_indices_pred[:, 0]] = stitch_indices_pred[:, 1]
        #                 stitch_map[stitch_indices_pred[:, 1]] = stitch_indices_pred[:, 0]
        #
        #                 stitch_map_gt = torch.zeros((point_num), dtype=torch.int64).to(device) - 1
        #                 stitch_map_gt[stitch_indices_gt[:, 0]] = stitch_indices_gt[:, 1]
        #                 stitch_map_gt[stitch_indices_gt[:, 1]] = stitch_indices_gt[:, 0]
        #
        #                 stitch_point_idx = idx_range[mask_cls_pred]
        #
        #                 stitch_cor_pred = stitch_map[stitch_point_idx]
        #                 stitch_cor_gt = stitch_map_gt[stitch_point_idx]
        #
        #                 stitch_pair_valid_mask = torch.bitwise_and(stitch_cor_pred != -1, stitch_cor_gt != -1)
        #                 stitch_point_idx_valid = stitch_point_idx[stitch_pair_valid_mask]
        #
        #                 stitch_cor_pred = stitch_map[stitch_point_idx_valid]
        #                 stitch_cor_gt = stitch_map_gt[stitch_point_idx_valid]
        #
        #                 stitch_cor_position_pred = pcs[stitch_cor_pred]
        #                 stitch_cor_position_gt = pcs[stitch_cor_gt]
        #
        #                 if len(stitch_point_idx_valid)>0:
        #                     STITCH_AMD = torch.sum(torch.norm(stitch_cor_position_pred - stitch_cor_position_gt, dim=1)) / len(stitch_point_idx_valid)
        #                     STITCH_AMD = STITCH_AMD.reshape(1)
        #                     if not torch.isnan(STITCH_AMD):
        #                         STITCH_AMD *= normalize_range
        #                         loss_dict.update({
        #                             "STITCH_AMD": STITCH_AMD,
        #                         })
        #
        #             # === 缝合 F1 Score ===
        #                 if len(stitch_point_idx_valid)>0:
        #                     # 计算 F1
        #                     STITCH_F1 = f1_score(
        #                         stitch_cor_gt.detach().cpu().numpy(),
        #                         stitch_cor_pred.detach().cpu().numpy(),
        #                         average='macro'
        #                     )
        #                     if not np.isnan(STITCH_F1):
        #                         loss_dict.update({
        #                             "STITCH_F1": STITCH_F1,
        #                         })
        # if self.is_train_in_stages and self.training:
        #     with torch.no_grad():
        #         self.cls_loss_list.append(float(cls_loss))
        #         self.mat_loss_list.append(float(mat_loss))

        return loss_dict

    # 在训练开始时执行的
    def init_dynamic_adjustment(self):
        if not self.is_train_in_stages:
            return

        # self.training_stage = 0
        self.cls_loss_list = []
        self.mat_loss_list = []
        self.val_ACC_list = []
        self.w_mat_loss = 0
        self.pc_cls_threshold = 0.5

    # # 在一个epoce结束时动态调整超参数，来实现分阶段学习
    # def training_epoch_end(self, outputs):
    #     super().training_epoch_end(outputs)
    #     self.dynamic_adjustment_epoch_end()

    def validation_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        self.dynamic_adjustment_epoch_end()

    def dynamic_adjustment_epoch_end(self):
        if not self.is_train_in_stages:
            return

        # 是否不可恢复：恢复指的是一个超参数可以回到上一阶段的状态
        is_unrecoverable = True

        with torch.no_grad():
            # cls_loss_mean = torch.mean(torch.tensor(self.cls_loss_list))
            ACC_mean = torch.mean(torch.tensor(self.val_ACC_list))
            print(ACC_mean)
            print(self.cfg.MODEL.TRAIN_IN_STAGE.VAL_ACC)
            # if cls_loss_mean < self.cfg.MODEL.TRAIN_IN_STAGE.W_CLS:
            if ACC_mean > self.cfg.MODEL.TRAIN_IN_STAGE.VAL_ACC:
                new_pc_cls_threshold = self.cfg.MODEL.PC_CLS_THRESHOLD * 1.0
                new_w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss * 1.0
                # 将之冻结
                for param in self.pc_classifier_layer.parameters():
                    param.requires_grad = False
            else:
                new_pc_cls_threshold = 0.5
                new_w_mat_loss = self.cfg.MODEL.LOSS.w_mat_loss * 0.0

            if is_unrecoverable:
                new_w_mat_loss = max(self.w_mat_loss, new_w_mat_loss)
                new_pc_cls_threshold = max(self.pc_cls_threshold, new_pc_cls_threshold)

            self.w_mat_loss = new_w_mat_loss
            self.pc_cls_threshold = new_pc_cls_threshold

            self.cls_loss_list = []
            self.mat_loss_list = []
            self.val_ACC_list = []

            self.log_dict({
                # "Charts/training_stage": self.training_stage,
                "Charts/w_cls_loss": torch.tensor(self.w_cls_loss, dtype=torch.float32),
                "Charts/w_mat_loss": torch.tensor(self.w_mat_loss, dtype=torch.float32),
                # "train/cls_loss_mean": cls_loss_mean,
                "VAL/ACC_mean": ACC_mean,
            },logger=True, sync_dist=False, rank_zero_only=True)
