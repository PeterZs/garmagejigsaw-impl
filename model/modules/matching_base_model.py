import os
import pickle

import numpy as np
import pytorch_lightning
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from scipy.spatial.transform import Rotation as R
from torch import optim

from utils import filter_wd_parameters, CosineAnnealingWarmupRestarts

import matplotlib.pyplot as plt
import time


class MatchingBaseModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg):
        super(MatchingBaseModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self._setup()
        self.test_results = None
        self.cd_list = []
        if len(cfg.STATS):
            os.makedirs(cfg.STATS, exist_ok=True)
            self.stats = dict()
            self.stats['part_acc'] = []
            self.stats['chamfer_distance'] = []
            for metric in ['mse', 'rmse', 'mae']:
                self.stats[f'trans_{metric}'] = []
                self.stats[f'rot_{metric}'] = []
            self.stats['pred_trans'] = []
            self.stats['gt_trans'] = []
            self.stats['pred_rot'] = []
            self.stats['gt_rot'] = []
            self.stats['part_valids'] = []
        else:
            self.stats = None

    def _setup(self):
        self.max_num_part = self.cfg.DATA.MAX_NUM_PART

        self.pc_feat_dim = self.cfg.MODEL.PC_FEAT_DIM

    # The flow for this base model is:
    # training_step -> forward_pass -> loss_function ->
    # _loss_function -> forward

    def forward(self, data_dict):
        """Forward pass to predict matching."""
        raise NotImplementedError("forward function should be implemented per model")

    def training_step(self, data_dict, batch_idx, optimizer_idx=-1):
        loss_dict = self.forward_pass(
            data_dict, mode='train', optimizer_idx=optimizer_idx
        )
        return loss_dict['loss']  # 这个被拿去计算梯度

    def validation_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode='val', optimizer_idx=-1)
        return loss_dict

    def validation_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size

        func = torch.tensor if \
            isinstance(outputs[0]['batch_size'], int) else torch.stack
        batch_sizes = func([output.pop('batch_size') for output in outputs
                            ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'val/{k}': torch.stack([output[k] for output in outputs]).reshape(-1)
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        self.log_dict(avg_loss, sync_dist=True)

    def test_step(self, data_dict, batch_idx):
        torch.cuda.synchronize()
        start = time.time()
        loss_dict = self.forward_pass(data_dict, mode='test', optimizer_idx=-1)
        torch.cuda.synchronize()
        end = time.time()
        time_elapsed = end - start
        loss_dict['time'] = torch.tensor(time_elapsed, device=loss_dict['loss'].device, dtype=torch.float64)
        return loss_dict

    def test_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size
        if isinstance(outputs[0]['batch_size'], int):
            func_bs = torch.tensor
            func_loss = torch.stack
        else:
            func_bs = torch.cat
            func_loss = torch.cat
        batch_sizes = func_bs([output.pop('batch_size') for output in outputs
                               ]).type_as(outputs[0]['loss'])  # [num_batches]
        losses = {
            f'test/{k}': func_loss([output[k] for output in outputs])
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum()
            for k, v in losses.items()
        }
        print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))

        total_shape_cd = torch.mean(torch.cat(self.cd_list))
        print(f'total_shape_cd: {total_shape_cd.item():.6f}')

        # this is a hack to get results outside `Trainer.test()` function
        self.test_results = avg_loss
        self.log_dict(avg_loss, sync_dist=True)
        if self.cfg.STATS is not None:
            with open(os.path.join(self.cfg.STATS, 'saved_stats.pk'), 'wb') as f:
                pickle.dump(self.stats, f)

    @torch.no_grad()
    def _cus_vis(self, data_dict, pred_trans, pred_rot, gt_trans, gt_rot, part_acc):
        B = data_dict["num_parts"].shape[0]
        pred_trans_rots = torch.cat([pred_trans, pred_rot.to_quat()], dim=-1)
        gt_trans_tots = torch.cat([gt_trans, gt_rot.to_quat()], dim=-1)
        for i in range(B):
            save_dir = os.path.join("inference", str(data_dict["data_id"][i].item()))
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "mesh_file_path.txt"), "w") as f:
                f.write(data_dict["mesh_file_path"][i])
            mask = data_dict["part_valids"][i] == 1
            c_pred_trans_rots = pred_trans_rots[i, mask]
            c_gt_trans_rots = gt_trans_tots[i, mask]
            np.save(os.path.join(save_dir, f"predict_{part_acc[i]}.npy"), c_pred_trans_rots.cpu().numpy())
            np.save(os.path.join(save_dir, f"gt.npy"), c_gt_trans_rots.cpu().numpy())

            

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        raise NotImplementedError("loss_function should be implemented per model")

    def transformation_loss(self, data_dict, out_dict):
        perm_mat = out_dict['perm_mat'].cpu().numpy()  # [B, N_, N_]
        ds_mat = out_dict['ds_mat'].cpu().numpy()  # [B, N_, N_]
        gt_pcs = data_dict['gt_pcs'].cpu().numpy()
        part_pcs = data_dict['part_pcs'].cpu().numpy()
        part_quat = data_dict['part_quat'].cpu().numpy()
        part_trans = data_dict['part_trans'].cpu().numpy()
        n_pcs = data_dict.get('n_pcs', None)
        if n_pcs is not None:
            n_pcs = n_pcs.cpu().numpy()

        part_valids = data_dict.get('part_valids', None)
        if part_valids is not None:
            part_valids = part_valids.cpu().numpy()
            n_valid = np.sum(part_valids, axis=1, dtype=np.int32)  # [B]
        else:
            n_valid = None

        gt_pcs = gt_pcs[:, :, :3]
        part_pcs = part_pcs[:, :, :3]
        B, N_sum, _ = gt_pcs.shape
        assert n_pcs is not None
        assert part_valids is not None
        assert n_valid is not None
        P = part_valids.shape[-1]
        N_ = ds_mat.shape[-1]

        critical_pcs_idx = data_dict.get('critical_pcs_idx', None)  # [B, N_sum]
        # critical_pcs_pos = data_dict.get('critical_pcs_pos', None)  # [\sum_B \sum_P n_critical_pcs[B, P], 3]
        n_critical_pcs = data_dict.get('n_critical_pcs', None)  # [B, P]

        n_critical_pcs = n_critical_pcs.cpu().numpy()
        critical_pcs_idx = critical_pcs_idx.cpu().numpy()

        density_match_mat = ds_mat
        best_match = np.argmax(ds_mat.reshape(-1, N_), axis=-1)
        density_match_mat_mask = np.zeros((B * N_, N_))
        density_match_mat_mask[np.arange(B * N_), best_match] = 1
        density_match_mat_mask = density_match_mat_mask.reshape(ds_mat.shape)

        pred_dict = self.compute_global_transformation(n_critical_pcs,
                                                       perm_mat,
                                                       gt_pcs, critical_pcs_idx,
                                                       part_pcs, n_valid, n_pcs,
                                                       part_quat, part_trans,
                                                       data_dict["data_id"],
                                                       data_dict["mesh_file_path"],
                                                       )
        metric_dict = self.calc_metric(data_dict, pred_dict)
        return metric_dict



    def loss_function(self, data_dict, optimizer_idx, mode):
        # loss_dict = None
        out_dict = self.forward(data_dict)

        loss_dict = self._loss_function(data_dict, out_dict, optimizer_idx)

        if 'loss' not in loss_dict:
            # if loss is composed of different losses, should combine them together
            # each part should be of shape [B, ] or [int]
            total_loss = 0.
            for k, v in loss_dict.items():
                if k.endswith('_loss'):
                    total_loss += v * eval(f'self.cfg.LOSS.{k.upper()}_W')
            loss_dict['loss'] = total_loss

        total_loss = loss_dict['loss']
        if total_loss.numel() != 1:
            loss_dict['loss'] = total_loss.mean()

        # log the batch_size for avg_loss computation
        if not self.training:
            if 'batch_size' not in loss_dict:
                loss_dict['batch_size'] = out_dict['batch_size']
        if mode == 'test':
            loss_dict.update(self.transformation_loss(data_dict, out_dict))
        return loss_dict

    def forward_pass(self, data_dict, mode, optimizer_idx):
        loss_dict = self.loss_function(data_dict, optimizer_idx=optimizer_idx, mode=mode)
        # in training we log for every step
        if mode == 'train' and self.local_rank == 0:
            log_dict = {f'{mode}/{k}': v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in loss_dict.items()}
            data_name = [
                k for k in self.trainer.profiler.recorded_durations.keys()
                if 'prepare_data' in k
            ][0]
            log_dict[f'{mode}/data_time'] = \
                self.trainer.profiler.recorded_durations[data_name][-1]
            self.log_dict(
                log_dict, logger=True, sync_dist=False, rank_zero_only=True)
        return loss_dict

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.cfg.TRAIN.LR
        wd = self.cfg.TRAIN.WEIGHT_DECAY

        if wd > 0.:
            params_dict = filter_wd_parameters(self)
            params_list = [{
                'params': params_dict['no_decay'],
                'weight_decay': 0.,
            }, {
                'params': params_dict['decay'],
                'weight_decay': wd,
            }]
            optimizer = optim.AdamW(params_list, lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.)

        if self.cfg.TRAIN.LR_SCHEDULER:
            assert self.cfg.TRAIN.LR_SCHEDULER.lower() in ['cosine']
            total_epochs = self.cfg.TRAIN.NUM_EPOCHS
            warmup_epochs = int(total_epochs * self.cfg.TRAIN.WARMUP_RATIO)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                total_epochs,
                max_lr=lr,
                min_lr=lr / self.cfg.TRAIN.LR_DECAY,
                warmup_steps=warmup_epochs,
            )
            return (
                [optimizer],
                [{
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }],
            )
        return optimizer


    # def _move_points(self, pc1, pc2, critical_pc1, critical_pc2, distance=0.1):
    #     # Move the points away from each other
    #     centroid1 = np.mean(pc1, axis=0)
    #     centroid2 = np.mean(pc2, axis=0)
        
    #     # Calculate the direction vector from centroid1 to centroid2
    #     direction = centroid2 - centroid1
    #     # Normalize the direction vector
    #     direction_norm = direction / np.linalg.norm(direction)
        
    #     # Move each point in pc1 away from centroid2
    #     pc1_moved = pc1 - direction_norm * distance
    #     # Move each point in pc2 away from centroid1
    #     pc2_moved = pc2 + direction_norm * distance

    #     # Move the critical points away from each other
    #     critical_pc1_moved = critical_pc1 - direction_norm * distance
    #     critical_pc2_moved = critical_pc2 + direction_norm * distance
        
    #     return pc1_moved, pc2_moved, critical_pc1_moved, critical_pc2_moved


    # def _visualize_pair(self, corr, gt_pc1, gt_pc2, critical_pc1, critical_pc2, data_id, idx1, idx2):
    #     # Visualize pair information
    #     save_dir = "vis_correspondance/vis_by_data_id" 
    #     save_dir = os.path.join(save_dir, f"{data_id}")
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f"{idx1}_{idx2}.png")
        
    #     # TODO: 1. Visualize teh ground truth

    #     # TODO: 2. Visualize the correspondence for the points
    #     # 2.1: Move all points away from each other
    #     gt_pc1_moved, gt_pc2_moved, critical_pc1_moved, critical_pc2_moved = self._move_points(gt_pc1, gt_pc2, critical_pc1, critical_pc2)
        
    #     # 2.1: Get the correspondence points
    #     corr_points_1 = critical_pc1_moved[corr[:, 0]]
    #     corr_points_2 = critical_pc2_moved[corr[:, 1]]
        
    #     # 2.3: Visualize the correspondence points by connecting them with lines
    #     # 2.3.1: Create a new plot
    #     fig = plt.figure(figsize=(14, 7))
    #     ax1 = fig.add_subplot(121, projection='3d')
    #     ax1.set_xlim([- 0.5, 0.5])
    #     ax1.set_ylim([- 0.5, 0.5])
    #     ax1.set_zlim([- 0.5, 0.5])

    #     # 2.3.2: Plot the points
    #     ax1.scatter(gt_pc1_moved[:, 0], gt_pc1_moved[:, 1], gt_pc1_moved[:, 2], c='r', s=1)
    #     ax1.scatter(gt_pc2_moved[:, 0], gt_pc2_moved[:, 1], gt_pc2_moved[:, 2], c='b', s=1)
    #     ax1.scatter(corr_points_1[:, 0], corr_points_1[:, 1], corr_points_1[:, 2], c='g')
    #     ax1.scatter(corr_points_2[:, 0], corr_points_2[:, 1], corr_points_2[:, 2], c='g')

    #     # 2.3.3: Connect the correspondence points with lines
    #     for i in range(corr_points_1.shape[0]):
    #         ax1.plot(
    #             [corr_points_1[i, 0], corr_points_2[i, 0]], 
    #             [corr_points_1[i, 1], corr_points_2[i, 1]], 
    #             [corr_points_1[i, 2], corr_points_2[i, 2]], 
    #             c='g', linewidth=0.1
    #         )

    #     ax2 = fig.add_subplot(122, projection='3d')
    #     # Visualize ground truth points on the right side
    #     ax2.scatter(gt_pc1[:, 0], gt_pc1[:, 1], gt_pc1[:, 2], c='r', s=1)
    #     ax2.scatter(gt_pc2[:, 0], gt_pc2[:, 1], gt_pc2[:, 2], c='b', s=1)
        
    #     ax2.set_xlim([- 0.5, 0.5])
    #     ax2.set_ylim([- 0.5, 0.5])
    #     ax2.set_zlim([- 0.5, 0.5])
    #     # 2.3.5: Show the plot
    #     plt.savefig(save_path)
    #     plt.close()



    def _save_data(self, edges, corr_list, gt_pcs, critical_pcs_idx, n_pcs, n_critical_pcs, data_id, mesh_file_path):
        """
        save to a dictionary
        edges: current 2 index of the pair
        correspondance: list of correspondance of critical points
        gt_pc: list of ground truth points
        critical_pcs: list of critical points
        """
        save_dir = "matching_data/everyday"
        os.makedirs(save_dir, exist_ok=True)

        # save to npz
        save_path = os.path.join(save_dir, f"{data_id}.npz")
        # if save_path exists, skip
        if os.path.exists(save_path):
            return

        data = {
            "edges": edges,
            "correspondence": corr_list,
            "gt_pcs": gt_pcs,
            "critical_pcs_idx": critical_pcs_idx,
            "n_pcs": n_pcs,
            "n_critical_pcs": n_critical_pcs,
            "mesh_file_path": mesh_file_path
        }

        np.savez(save_path, **data)
        

