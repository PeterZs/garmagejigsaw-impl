import torch
import numpy as np
from utils import (
    hungarian,
    stitch_mat2indices,
    stitch_indices2mat
)


def get_pointstitch(batch, inf_rst,
                    sym_choice = "sym_max", mat_choice = "hun",
                    filter_neighbor_stitch = True, filter_neighbor=7,
                    filter_too_long = True, filter_length = 0.2,
                    filter_too_small = True, filter_prob = 0.2,
                    only_triu = False):
    """
    :param batch:           # garment_jigsaw model input
    :param inf_rst:         # garment_jigsaw model output
    :param sym_choice:      # sym_max;  sym_avg;  sym_min;
    :param mat_choice:      # hun:hungarian algorithm;  col_max:choose max num of each column;
    :param filter_neighbor_stitch:  # filter stitches between filter_neighbor step neighbor
    :param filter_neighbor
    :param filter_too_long:         # filter stitches whose distance longer than filter_length
    :param filter_length:
    :param filter_too_small:        # filter stitches whose probabilities smaller than filter_logits
    :param filter_prob:
    :param only_triu:       # if true, the result map will only include triu part of stitches
    :return:
    """

    pcs = batch["pcs"].squeeze(0)
    pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)

    stitch_pcs = pcs[pc_cls_mask == 1]
    unstitch_pcs = pcs[pc_cls_mask == 0]

    n_stitch_pcs_sum = inf_rst['n_stitch_pcs_sum']
    stitch_mat_pred_ = inf_rst["ds_mat"][:, :n_stitch_pcs_sum, :n_stitch_pcs_sum]

    # Avoid neighbor point stitching
    stitch_mat_pred_[0][torch.eye(stitch_mat_pred_.shape[-1], stitch_mat_pred_.shape[-1]) == 1] = 0
    if filter_neighbor_stitch:
        assert filter_neighbor <= 2
        i_indices = torch.arange(stitch_mat_pred_.shape[-1]).view(-1, 1).repeat(1, stitch_mat_pred_.shape[-1])
        j_indices = torch.arange(stitch_mat_pred_.shape[-1]).view(1, -1).repeat(stitch_mat_pred_.shape[-1], 1)
        mask_neighbor = torch.abs(i_indices - j_indices) < filter_neighbor
        stitch_mat_pred_[0][mask_neighbor] = 0

    # Force the predicted stitching probility mat symmetry
    stitch_mat_pred = torch.zeros_like(stitch_mat_pred_)
    if sym_choice == "sym_max":
        stitch_mat_pred_mask = stitch_mat_pred_ > stitch_mat_pred_.transpose(1, 2)
        stitch_mat_pred[stitch_mat_pred_mask] = stitch_mat_pred_[stitch_mat_pred_mask]
        stitch_mat_pred[~stitch_mat_pred_mask] = stitch_mat_pred_.transpose(1, 2)[~stitch_mat_pred_mask]
    elif sym_choice == "sym_avg":
        stitch_mat_pred = (stitch_mat_pred_.transpose(1, 2) + stitch_mat_pred_) / 2
    elif sym_choice == "sym_min":
        stitch_mat_pred_mask = stitch_mat_pred_ < stitch_mat_pred_.transpose(1, 2)
        stitch_mat_pred[stitch_mat_pred_mask] = stitch_mat_pred_[stitch_mat_pred_mask]
        stitch_mat_pred[~stitch_mat_pred_mask] = stitch_mat_pred_.transpose(1, 2)[~stitch_mat_pred_mask]
    else:
        stitch_mat_pred[:] = stitch_mat_pred_[:]

    """
    Tansfer predict stitching probilities mat to stitching mat
    Two options: 
        1. col_max: take the row-wise maximum.
        2. hun: use the Hungarian algorithm.
    """
    if mat_choice == "col_max":
        stitch_mat = torch.zeros_like(stitch_mat_pred)
        max_values, max_indices = stitch_mat_pred.max(dim=-1)
        stitch_mat.scatter_(-1, max_indices.unsqueeze(-1), 1)
    elif mat_choice == "hun":
        stitch_mat = hungarian(stitch_mat_pred)
    else:
        raise NotImplementedError

    stitch_mat = stitch_mat.int()
    pc_cls_mask = inf_rst["pc_cls_mask"].squeeze(0)
    stitch_pcs = pcs[pc_cls_mask == 1]

    # Filter out stitching where spacing of point pair too large.
    if filter_too_long:
        stitch_indices = stitch_mat2indices(stitch_mat[:, :].detach().cpu().numpy().squeeze(0))
        stitch_dis = torch.sqrt(
            torch.sum((stitch_pcs[stitch_indices[:, 0]] - stitch_pcs[stitch_indices[:, 1]]) ** 2, dim=-1))
        stitch_dis = stitch_dis.detach().cpu().numpy()
        stitch_indices = stitch_indices[stitch_dis < filter_length]
        stitch_mat = torch.tensor(stitch_indices2mat(stitch_pcs.shape[-2], stitch_indices),
                                  device=pcs.device, dtype=torch.int64).unsqueeze(0)

    # filter out stitching with too low probabilities
    if filter_too_small:
        pc_stitch_threshold = filter_prob
    else:
        pc_stitch_threshold = 0

    # probabilities of stitchings
    logits_ = stitch_mat_pred[stitch_mat == 1]
    pc_stitch_mask = logits_ < pc_stitch_threshold
    logits_ = logits_[~pc_stitch_mask]
    stitch_mat = stitch_mat.to(torch.int64)
    stitch_mat[stitch_mat==1] = ~pc_stitch_mask*1
    stitch_indices = stitch_mat2indices(stitch_mat[:, :].detach().cpu().numpy().squeeze(0))

    # probabilities of stitchings
    logits = np.zeros(stitch_indices.shape[0])
    logits[:] = (logits_.detach().cpu().numpy())

    # Transfer stitching mat of stitching points to the
    # stitching mat of all points (stitching points and non-stitching points)
    stitch_mat_full = torch.zeros((1, pcs.shape[0], pcs.shape[0]), dtype=torch.int64, device=pcs.device)
    mask1 = torch.zeros((1, pcs.shape[0], pcs.shape[0]), dtype=torch.bool, device=pcs.device)
    mask1[:, pc_cls_mask == 1, :] = True
    mask2 = torch.zeros((1, pcs.shape[0], pcs.shape[0]), dtype=torch.bool, device=pcs.device)
    mask2[:, :, pc_cls_mask == 1] = True
    stitch_mat_full_mask = torch.bitwise_and(mask1, mask2)
    stitch_mat_full[stitch_mat_full_mask] = stitch_mat.reshape(-1)

    # transfer stitch mat to the stitched point index pairs
    stitch_indices_full = stitch_mat2indices(stitch_mat_full)

    # filter triu part of mat
    if only_triu:
        mask = stitch_indices_full[:, 0] > stitch_indices_full[:, 1]
        stitch_indices_full[mask] = stitch_indices_full[mask][:, [1, 0]]

    stitch_mat_full.to(pcs.device)
    stitch_indices_full = torch.tensor(stitch_indices_full, device=pcs.device, dtype=torch.int64)

    return stitch_mat_full, stitch_pcs, unstitch_pcs, stitch_indices, stitch_indices_full, logits