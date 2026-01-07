
import os
import json
import math
import random
from copy import copy
from glob import glob

import igl
import torch
import numpy as np
import trimesh
import trimesh.sample
from diffusers import DDPMScheduler
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import convolve1d

from utils import balancedSample
from utils import styleXD_normalize, get_pc_bbox
from utils import stitch_indices2mat, stitch_indices_order, cal_mean_edge_len


def smooth_using_convolution(noise, k=3):
    k = [1] * k
    kernel = np.array(k) / sum(k)
    smoothed_noise = convolve1d(noise, kernel, axis=0, mode='nearest')
    return smoothed_noise


class AllPieceMatchingDataset_stylexd(Dataset):
    def __init__(
            self,
            data_dir,
            # data_types = (),

            num_points=1000,    # point sample num per garment
            min_num_part=2,
            max_num_part=20,
            read_uv = False,

            shrink_mesh=False,
            shrink_mesh_param=1,
            shrink_bystitch=False,
            shrink_bystitch_strength=0,

            panel_noise_type="default",
            trans_range=1,
            rot_range=-1,
            scale_range=0,
            bbox_noise_strength=15,

            pcs_sample_type="area",
            pcs_sample_type_stitch_only_sample_boundary = False,
            use_stitch_noise = False,
            stitch_noise_strength = 1,
            stitch_noise_random_range = (0.8, 2.2),

            min_part_point=30,
            mode= "train",
            dataset_split_dir = "data/stylexd_jigsaw/dataset_split",
            inference_data_list=None,   # data list while running inference.
            **kwargs
    ):
        if mode not in ["train", "val", "test", "inference"]:
            raise ValueError(f"mode=\"{mode}\" is not valid.")

        self.mode = mode
        self.data_dir = data_dir
        # self.data_types = data_types

        self.dataset_split_dir = dataset_split_dir
        self.inference_data_list = inference_data_list
        self.data_list = self._read_data()

        self.num_points = num_points

        # Used to filter data by part num and panel`s point num
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part
        self.min_part_point = min_part_point

        self.read_uv = read_uv

        # === panel noise ===
        self.panel_noise_type = panel_noise_type
        if self.panel_noise_type == "default":
            self.scale_range = scale_range
            self.rot_range = rot_range
            self.trans_range = trans_range
        elif self.panel_noise_type == "bbox":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule='linear',
                prediction_type='epsilon',
                beta_start=0.0001,
                beta_end=0.02,
                clip_sample=False,
            )
            self.bbox_noise_strength = bbox_noise_strength

        # === point sample ===
        """
        stitch: sample points on obj for training.
        boundary_pcs: sample points on generated Garmage for inference.
        """
        assert pcs_sample_type in [ "stitch", "boundary_pcs"]
        self.pcs_sample_type = pcs_sample_type
        self.pcs_sample_type_stitch_only_sample_boundary = pcs_sample_type_stitch_only_sample_boundary
        self.shrink_mesh=shrink_mesh
        self.shrink_mesh_param=shrink_mesh_param
        self.shrink_bystitch=shrink_bystitch
        self.shrink_bystitch_strength=shrink_bystitch_strength

        # === point noise ===
        self.use_stitch_noise = use_stitch_noise
        self.stitch_noise_strength = stitch_noise_strength
        self.stitch_noise_random_min = min(stitch_noise_random_range)
        self.stitch_noise_random_max = max(stitch_noise_random_range)

        print("dataset length: ", len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def _read_data(self):
        if self.mode in ["train", "val"]:
            with open(os.path.join(self.dataset_split_dir, f"{self.mode}.json") ,"r", encoding="utf-8") as f:
                split = json.load(f)
            data_list = [os.path.join(self.data_dir, dir_) for dir_ in split]
            # Ensure Valid
            data_list_filter = []
            for dir in data_list:
                if os.path.isdir(dir) and len(glob(os.path.join(dir,"*.obj")))>0:
                    data_list_filter.append(dir)
            return data_list_filter
        elif self.mode == "inference":
            assert self.inference_data_list is not None
            # Ensure Valid
            data_list_filter = []
            for dir in self.inference_data_list:
                if os.path.isdir(dir) and len(glob(os.path.join(dir,"*.obj")))>0:
                    data_list_filter.append(dir)
            del self.inference_data_list
            return data_list_filter
        else:
            raise KeyError(f"mode {self.mode} not supported.")

    def _random_SRM_default(self, pc, pcs, pc_idx, mean_edge_len):
        """
        Panel Noise: random scale+rotate+move
        pc: [N, 3]
        """
        noise_strength_S = random.random()*0.5 + 0.2
        noise_strength_R = random.random()*0.5 + 0.5
        noise_strength_M = random.random()*0.6 + 0.8

        pc_centroid = get_pc_bbox(pc)[0]

        # SCALE
        pc = pc - pc_centroid[None]
        scale_gt = (np.random.rand(1) * self.scale_range) * noise_strength_S + 1
        pc = pc*scale_gt

        # ROTATE
        if self.rot_range >= 0.0:
            rot_euler = (np.random.rand(3) - 0.5) * 2.0 * self.rot_range
            rot_mat = R.from_euler("xyz", rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        quat_gt = quat_gt[[3, 0, 1, 2]] * noise_strength_R
        pc = pc + pc_centroid[None]

        # MOVE
        bkg_pcs = np.concatenate(pcs[:pc_idx]+pcs[pc_idx+1:])
        pc_centroid = get_pc_bbox(pc)[0]
        bkg_pcs_centroid = get_pc_bbox(bkg_pcs)[0]

        move_vec = pc_centroid-bkg_pcs_centroid
        move_vec = move_vec/np.linalg.norm(move_vec)

        move_vec_horizon = np.zeros_like(move_vec)
        move_vec_horizon[:] = move_vec[:]
        move_vec_horizon[1] = move_vec_horizon[1]/2
        move_vec_horizon = move_vec_horizon/np.linalg.norm(move_vec_horizon)

        move_vec = move_vec + move_vec_horizon
        move_vec = move_vec / np.linalg.norm(move_vec)

        trans_gt = (mean_edge_len * move_vec * (np.random.rand(3) + 1.) / 4. * self.trans_range).reshape(-1, 3)
        pc = pc + trans_gt

        trans_gt = (mean_edge_len * (np.random.rand(3) - 0.5) * 2.0 * self.trans_range).reshape(-1,3) * noise_strength_M
        pc = pc + trans_gt
        return pc, trans_gt, quat_gt

    def transform_pc(self, pc, bbox_old, bbox_new):
        """
        apply bbox changing to the pointcloud
        :param pc:
        :param bbox_old:
        :param bbox_new:
        :return:
        """
        xyz_min_old, xyz_max_old = np.array(bbox_old[:3]), np.array(bbox_old[3:])
        xyz_min_new, xyz_max_new = np.array(bbox_new[:3]), np.array(bbox_new[3:])

        center_old = (xyz_min_old + xyz_max_old) / 2
        size_old = xyz_max_old - xyz_min_old

        center_new = (xyz_min_new + xyz_max_new) / 2
        size_new = xyz_max_new - xyz_min_new

        scale = size_new / size_old

        pc_transformed = (pc - center_old) * scale + center_new

        return pc_transformed

    def _random_SM_byBbox(self, pc):
        """
        Panel Noise: random scale+move panel by add noise on their bbox
        """
        bbox_old = get_pc_bbox(pc, type="xyxy")
        bbox_old = torch.Tensor(np.concatenate(bbox_old), device="cpu").unsqueeze(0)
        aug_ts_float = torch.rand(1) * self.bbox_noise_strength
        if self.bbox_noise_strength>0:
            aug_ts = torch.ceil(aug_ts_float).long()
            aug_noise = torch.randn(bbox_old.shape, device="cpu")
            bbox_new = self.noise_scheduler.add_noise(bbox_old, aug_noise, aug_ts)
            bbox_inter = (aug_ts_float)/aug_ts * bbox_new + (aug_ts - aug_ts_float)/aug_ts * bbox_old
            pc_new = self.transform_pc(pc, bbox_old.squeeze(0), bbox_inter.squeeze(0))
        else:
            pc_new = pc
        return pc_new

    def _pad_data(self, data, pad_size=None, pad_num=0):
        if pad_size is None:
            pad_size = self.max_num_part
        data = np.array(data)
        if len(data.shape) > 1:
            pad_shape = (pad_size,) + tuple(data.shape[1:])
        else:
            pad_shape = (pad_size,)
        pad_data = np.ones(pad_shape, dtype=data.dtype) * pad_num
        pad_data[: data.shape[0]] = data
        return pad_data

    def sample_point_byBoundaryPcs(self, meshes):
        """
        sample point from generated Garmage.
        """
        pcs = [np.array(mesh.vertices) for mesh in meshes]
        nps = np.array([len(pc) for pc in pcs])
        piece_id = np.concatenate([[idx]*len(pcs[idx]) for idx in range(len(pcs))], axis=0)
        mean_edge_len = 0.0072
        sample_result =dict(
            pcs=pcs,
            nps=nps,
            piece_id=piece_id,
            mat_gt=None,
            mean_edge_len=mean_edge_len,
            normalize_range=1,
            panel_instance_seg=None
        )
        return sample_result

    def sample_point_byStitch(self, meshes, stitches, full_uv_info, n_rings = 2, max_check_times=4, min_samplenum_prepanel=4):
        """
        Sample point from garment mesh.

        :param meshes:          List of trimesh.Trimesh
        :param num_points:      Target point sample num
        :param stitches:        Stitch relation in original mesh
        :param full_uv_info:    UV info in original mesh
        :param n_rings:         Expand rings in getting sampled_points_CH
        :param max_check_times:         Max re-sample times
        :param min_samplenum_prepanel:  Min point sample number in each Panel
        :return:
        """

        pcs = []
        nps =  [len(mesh.vertices) for mesh in meshes]      # point num of each panel
        num_parts = len(nps)    # Panel num
        piece_id_global = np.concatenate([[idx]*i for idx,i in enumerate(nps)], axis=-1)
        num_points = self.num_points

        # === Get boundary ===
        # Get indices of mesh`s boundary vertices
        vertices = np.concatenate([np.array(mesh.vertices) for mesh in meshes], axis=0)
        boundary_point_list = []
        points_end_idxs = np.cumsum(nps)
        for idx, mesh in enumerate(meshes):
            if idx ==0 : point_start_idx = 0
            else: point_start_idx = points_end_idxs[idx-1]
            # A panel may contain multi contours
            new_list = []
            for loop in igl.all_boundary_loop(mesh.faces):
                new_list.extend(np.array(loop)+point_start_idx)
            boundary_point_list.append(new_list)
        boundary_points_idx = np.concatenate(boundary_point_list,axis=0)

        # === Allocate sample num ===
        # Sample points with and without stitching relationships according to a ratio

        # This option will filter out stitches where each side is not on the boundary of panel.
        if self.pcs_sample_type_stitch_only_sample_boundary:
            m0 = np.isin(stitches[:, 0], boundary_points_idx)
            m1 = np.isin(stitches[:, 1], boundary_points_idx)
            boundary_stitch_mask = np.logical_and(m0, m1)
            stitches = stitches[boundary_stitch_mask]

        stitch_map = np.zeros(shape=(len(vertices)), dtype=np.int32) - 1
        stitch_map[stitches[:, 0]] = stitches[:, 1]
        stitch_map[stitches[:, 1]] = stitches[:, 0]  # [todo]

        stitched_vertices_idx = np.concatenate([stitches[:, 0], stitches[:, 1]], axis=0)
        unstitched_vertices_idx = boundary_points_idx[
            np.array([s not in stitched_vertices_idx for s in boundary_points_idx])]

        # Allocate the number of sampled points for each panel
        boundary_len = sum([len(b) for b in boundary_point_list])
        expect_sample_nums = np.array([int(num_points*len(b)/boundary_len) for b in boundary_point_list])
        sample_num_arrangement = np.zeros(num_parts, dtype=np.int16)
        min_expect = min_samplenum_prepanel * 4
        pre_arranged = int(min_samplenum_prepanel * 3)
        assert pre_arranged * num_parts < num_points, "min_samplenum_prepanel too large may cause error"
        sample_num_arrangement[expect_sample_nums <= min_expect] = pre_arranged
        sample_num_arrangement[expect_sample_nums > min_expect] = int(pre_arranged/2)
        arranged_num = np.sum(sample_num_arrangement)
        sample_num_arrangement = np.array([int((num_points-arranged_num)*len(b)/boundary_len) for b in boundary_point_list]) + sample_num_arrangement
        # Allocate the remaining points.
        for i in range(num_points - np.sum(sample_num_arrangement)):
            sample_num_arrangement[random.randint(0, num_parts - 1)]+=1

        # Get the number of stitching and unstitching points on each panel
        panel_stitched_num = [sum(piece_id_global[stitched_vertices_idx]==i) for i in range(num_parts)]
        panel_unstitched_num = [sum(piece_id_global[unstitched_vertices_idx]==i) for i in range(num_parts)]
        # Get the allocated point sample number of stitching and unstitching points on each panel
        sample_num_arrangement_stitched = np.array([int((sample_num_arrangement[i] * panel_stitched_num[i] / (panel_stitched_num[i]+panel_unstitched_num[i]))/2) for i in range(num_parts)])
        sample_num_arrangement_unstitched = np.array([sample_num_arrangement[i]-sample_num_arrangement_stitched[i]*2 for i in range(num_parts)])

        # Get stitching and non-stitching points on the boundary of each panel
        stitched_list, unstitched_list = [], []
        for part_idx in range(num_parts):
            mask = piece_id_global[stitched_vertices_idx] == part_idx
            stitched_list.append(stitched_vertices_idx[mask])
            mask = piece_id_global[unstitched_vertices_idx] == part_idx
            unstitched_list.append(unstitched_vertices_idx[mask])

        # Handle the allocation of sampled points for panels without stitching or non-stitching points
        unarranged = 0
        for i in range(num_parts):
            if len(unstitched_list[i])==0:
                transfer_volumn = math.floor((sample_num_arrangement_unstitched[i] + unarranged)/2)
                unarranged = (sample_num_arrangement_unstitched[i] + unarranged)%2
                sample_num_arrangement_stitched[i] += transfer_volumn
                sample_num_arrangement_unstitched[i] = 0
        for i in range(num_parts):
            if len(stitched_list[i])==0:
                transfer_volumn = sample_num_arrangement_stitched[i]*2
                if unarranged>0:
                    transfer_volumn+=unarranged
                    unarranged=0
                sample_num_arrangement_unstitched[i] += transfer_volumn
                sample_num_arrangement_stitched[i] = 0
        if unarranged>0:
            min_idx, min_arranged = 0, 9999999
            for i in range(num_parts):
                if min_arranged > sample_num_arrangement_unstitched[i] > 0:
                    min_arranged = sample_num_arrangement_unstitched[i]
                    min_idx = i
            sample_num_arrangement_unstitched[min_idx]+=unarranged

        # === Sample points ===
        for check_time in range(max_check_times):
            # Sample stitching points
            sample_list = []
            for part_idx in range(num_parts):
                part_points = stitched_list[part_idx]
                if sample_num_arrangement_stitched[part_idx]>0:
                    sample_result = balancedSample(len(part_points), sample_num_arrangement_stitched[part_idx])
                    sample_list.append(part_points[sample_result])
            stitched_sample_idx = np.concatenate(sample_list)
            stitched_sample_idx = sorted(stitched_sample_idx)
            stitched_sample_idx_cor = stitch_map[stitched_sample_idx]

            # Sample non-stitching points
            sample_list = []
            for part_idx in range(num_parts):
                part_points = unstitched_list[part_idx]
                if sample_num_arrangement_unstitched[part_idx]>0:
                    sample_result = balancedSample(len(part_points), sample_num_arrangement_unstitched[part_idx])
                    sample_list.append(part_points[sample_result])

            if len(sample_list) > 0:
                unstitched_sample_idx = np.concatenate(sample_list)
            else:
                unstitched_sample_idx = None

            if unstitched_sample_idx is not None:
                all_sample_idx = np.concatenate([stitched_sample_idx, stitched_sample_idx_cor, unstitched_sample_idx], axis=0)
            else:
                all_sample_idx = np.concatenate([stitched_sample_idx, stitched_sample_idx_cor], axis=0)

            sampled_piece_id = piece_id_global[all_sample_idx]
            piece_sample_num = np.array([np.sum(sampled_piece_id==i) for i in range(num_parts)])
            if np.sum(piece_sample_num<min_samplenum_prepanel)==0:
                break
            else:
                if check_time!=max_check_times-1:
                    continue
                else:
                    raise ValueError(f"sample num:{np.sum(piece_sample_num<min_samplenum_prepanel)}, NUM_PC_POINTS too small in cfg")

        # Sample results
        stitched_sample_num = np.sum(sample_num_arrangement_stitched)
        mat_gt = np.array([np.array([i, i + stitched_sample_num]) for i in range(stitched_sample_num)])
        piece_id = np.concatenate([np.array([idx]*n) for idx, n in enumerate(nps)],axis=0)[all_sample_idx]
        sampled_points_CH =  vertices[all_sample_idx]

        # normalize sampled pointcloud
        points_normalized, normalize_range = styleXD_normalize(sampled_points_CH)
        mean_edge_len = cal_mean_edge_len(meshes)/normalize_range

        # sort by index
        sorted_indices = np.argsort(all_sample_idx)
        points_normalized = points_normalized[sorted_indices]
        piece_id = piece_id[sorted_indices]
        if full_uv_info is not None:
            uv_sampled = full_uv_info[all_sample_idx]
            uv_sampled = uv_sampled[sorted_indices]

        mat_gt = stitch_indices_order(mat_gt, sorted_indices)
        mask = mat_gt[:, 0] > mat_gt[:, 1]
        mat_gt[mask] = mat_gt[mask][:, ::-1]
        mat_gt = mat_gt[np.argsort(mat_gt[:, 0])]

        # === segment contour-wise ===
        for i in range(len(nps)):
            pcs.append(points_normalized[piece_id==i])
        nps = np.array([len(pc) for pc in pcs])

        if(sum([len(pc) for pc in pcs])!=num_points):
            raise AssertionError("Pointcloud Sample Num Wrong.")

        pcs_dict = dict(
            pcs=pcs,                    # Sampled 3D pointcloud
            pcs_idx=all_sample_idx,
            piece_id=piece_id,          # Contour idx of each point
            nps=np.array(nps),          # each panel`s point num
            mat_gt=mat_gt,              # stitches
            mean_edge_len=mean_edge_len,
            normalize_range=normalize_range,
            panel_instance_seg=np.arange(len(nps))
        )
        if full_uv_info is not None:
            pcs_dict["uv"] = uv_sampled
        normals = np.concatenate([np.array(mesh.vertex_normals) for mesh in meshes], axis=0)
        sampled_normals = normals[all_sample_idx]
        pcs_dict["normal"] = sampled_normals
        return pcs_dict

    def load_meshes(self, data_folder):
        mesh_files = sorted(glob(os.path.join(data_folder, "piece_*.obj")))

        if not self.min_num_part <= len(mesh_files) <= self.max_num_part:
            raise ValueError(f"Part num of {data_folder}({len(mesh_files)}) out of range [{self.min_num_part}:{self.max_num_part}]")

        meshes = [trimesh.load(mesh_file, force="mesh", process=False) for mesh_file in mesh_files]

        return meshes

    def load_full_uv_info(self, data_folder):
        if self.read_uv:
            full_uv_info = np.load(os.path.join(data_folder, "annotations", "uv.npy"))
            return full_uv_info
        else:
            return None

    def shrink_meshes(self, meshes, shrink_param=0.5):
        """
        Shrink each boundary point of the mesh inwardly.
        """
        if shrink_param > 1 or shrink_param < 0:
            raise ValueError(f"shrink_param={shrink_param} is invalid")

        for mesh in meshes:
            if len(mesh.vertices) < 400: continue
            # Identify non-boundary neighbors for each boundary point
            edge_points_loops = igl.all_boundary_loop(mesh.faces)
            all_boundary_points = np.concatenate(edge_points_loops)
            neighbor_points = {}
            neighbor_boundary_points = {}
            for loop in edge_points_loops:
                for p_idx, point in enumerate(loop):
                    neighbors = mesh.vertex_neighbors[point]
                    neighbor_points[point] = neighbors

                    if p_idx==0:
                        if len(loop) == 1: neighbors_boundaey = [loop[p_idx], loop[p_idx]]
                        else: neighbors_boundaey = [loop[-1], loop[p_idx + 1]]
                    elif p_idx==len(loop)-1:
                        neighbors_boundaey = [loop[p_idx-1], loop[0]]
                    else:
                        neighbors_boundaey = [loop[p_idx - 1], loop[p_idx+1]]
                    neighbor_boundary_points[point] = neighbors_boundaey

            # Compute the positions of boundary points after shrinking.
            new_vertices_positions = {}
            for b_point in all_boundary_points:

                neighbors_boundaey = neighbor_boundary_points[b_point]
                neighbors_boundaey_position = mesh.vertices[neighbors_boundaey]
                neighbors_boundaey_vector = neighbors_boundaey_position[1] - neighbors_boundaey_position[0]
                if np.linalg.norm(neighbors_boundaey_vector)==0:
                    new_vertices_positions[b_point] =  np.array(mesh.vertices[b_point])
                else:
                    neighbors_boundaey_vector = neighbors_boundaey_vector / np.linalg.norm(neighbors_boundaey_vector)

                    b_point_pos = np.array(mesh.vertices[b_point])
                    center = get_pc_bbox(mesh.vertices[neighbor_points[b_point]], type="ccwh")[0]

                    AB = center - b_point_pos
                    proj_v_AB = (np.dot(AB, neighbors_boundaey_vector) / np.dot(neighbors_boundaey_vector, neighbors_boundaey_vector)) * neighbors_boundaey_vector
                    perpendicular = AB - proj_v_AB

                    target_position = b_point_pos + perpendicular

                    new_position = (shrink_param * b_point_pos +
                                    (1 - shrink_param) * target_position)
                    new_vertices_positions[b_point] = new_position

            for b_point in all_boundary_points:
                mesh.vertices[b_point] = new_vertices_positions[b_point]

    def _shrink_bystitch(self, pcs, stitch_mat, piece_id, mean_edge_len):
        """
        Shrink pcs by stitches info
        """
        nps = [len(pc) for pc in pcs]

        if isinstance(pcs, list):
            pcs_all = np.concatenate(pcs)

        self_stitch_mask = piece_id[stitch_mat[:, 1]] == piece_id[stitch_mat[:, 0]]

        shrink_strength = mean_edge_len * self.shrink_bystitch_strength * np.random.rand(1)

        vec = copy(pcs_all[stitch_mat[:, 1]] - pcs_all[stitch_mat[:, 0]])
        vec2 = copy(pcs_all[stitch_mat[:, 0]] - pcs_all[stitch_mat[:, 1]])

        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        vec2 = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-12)

        shrink1 = vec * shrink_strength
        shrink2 = vec2 * shrink_strength

        shrink1[~self_stitch_mask] = 0
        shrink2[~self_stitch_mask] = 0

        pcs_all[stitch_mat[:, 1]] += shrink1
        pcs_all[stitch_mat[:, 0]] += shrink2

        pcs = []
        nps_cumsum = np.cumsum(nps)
        for i in range(len(nps_cumsum)):
            if i==0:
                st = 0
            else:
                st = nps_cumsum[i-1]
            ed = nps_cumsum[i]
            pcs.append(pcs_all[st:ed])
        return pcs


    def _get_pcs(self, data_folder, annotations_json_path):
        meshes = self.load_meshes(data_folder)

        # Shrink each boundary point of the mesh inwardly to create gaps between panels.
        if self.shrink_mesh:
            self.shrink_meshes(meshes, self.shrink_mesh_param)

        full_uv_info = self.load_full_uv_info(data_folder)

        if self.pcs_sample_type == "boundary_pcs":  # inference Only
            sample_result = self.sample_point_byBoundaryPcs(meshes)
            if full_uv_info is not None:
                sample_result["uv"] = full_uv_info
            with open(annotations_json_path, "r", encoding="utf-8") as f:
                ann_json = json.load(f)
            sample_result["panel_instance_seg"] = ann_json["panel_instance_seg"]
            return sample_result
        elif self.pcs_sample_type == "stitch":      # train val Only
            if self.num_points%2!=0:
                raise ValueError("self.num_points should be an even number when self.pcs_sample_type==\"stitch\"")
            stitches = np.load(os.path.join(data_folder, "annotations", "stitch.npy"))
            max_check_times = 10        # Maximum number of resampling attempts
            min_samplenum_prepanel = 4  # Minimum number of sampled points on a single panel
            sample_result = self.sample_point_byStitch(
                meshes, stitches, full_uv_info, n_rings=2,
                max_check_times=max_check_times, min_samplenum_prepanel=min_samplenum_prepanel
            )

            return sample_result
        else:
            raise NotImplementedError(f"pcs_sample_type: {self.pcs_sample_type} hasen't been implemented")


    def __getitem__(self, index):
        # === get paths ===
        mesh_file_path = self.data_list[index]
        try:
            garment_json_path = glob(os.path.join(mesh_file_path, "annotations", "garment*.json"))[0]
            garment_json_path = garment_json_path if os.path.exists(garment_json_path) else ""
        except Exception:
            garment_json_path = ""
        annotations_json_path = os.path.join(mesh_file_path, "annotations", "annotations.json")
        annotations_json_path = annotations_json_path if os.path.exists(annotations_json_path) else ""

        # === sample pointcloud ===
        sample_result= self._get_pcs(mesh_file_path, annotations_json_path)
        pcs = sample_result["pcs"]
        num_parts = len(pcs)
        nps = sample_result["nps"]
        mat_gt = sample_result["mat_gt"]
        piece_id = sample_result["piece_id"]
        mean_edge_len = sample_result["mean_edge_len"]
        normalize_range = sample_result["normalize_range"]
        panel_instance_seg = sample_result["panel_instance_seg"]

        # Shrink pcs by stitches info
        if self.shrink_bystitch:
            pcs = self._shrink_bystitch(pcs, mat_gt, piece_id, mean_edge_len)

        # normalize uv
        if "uv" in sample_result.keys():
            uv = sample_result["uv"]
            uv = styleXD_normalize(uv)[0]
        else: uv=None
        if "normal" in sample_result.keys():
            # [todo] Normal processing (Model input like xyz, uv).
            normal = sample_result["normal"]

        # === panel noise ===
        cur_pcs = []
        for idx, (pc, n_p) in enumerate(zip(pcs, nps)):
            # Add noise on panel`s position
            if self.panel_noise_type == "default":
                pc, gt_trans, gt_quat = self._random_SRM_default(pc, pcs, idx, mean_edge_len)
            elif self.panel_noise_type == "bbox":
                pc = self._random_SM_byBbox(pc)
            else:
                raise NotImplementedError("")
            cur_pcs.append(pc)
        cur_pcs = np.concatenate(cur_pcs).astype(np.float32)  # [N_sum, 3]

        n_pcs = self._pad_data(np.array(nps), self.max_num_part).astype(np.int64)  # [P]
        valids = np.zeros(self.max_num_part, dtype=np.float32)
        valids[:num_parts] = 1.0
        panel_instance_seg = self._pad_data(np.array(panel_instance_seg), self.max_num_part, -1).astype(np.int64)
        data_dict = {
            "pcs": cur_pcs,             # pointclouds after random transformation
            "n_pcs": n_pcs,             # point num of each part
            "num_parts": num_parts,
            "part_valids": valids,
            "data_id": index,
            "piece_id": piece_id,
            "mesh_file_path": mesh_file_path,               # path of this garment
            "garment_json_path": garment_json_path,         # path of garment.json if exist
            "annotations_json_path": annotations_json_path, # path of annotations.json if exist
            "normalize_range": normalize_range,
            "panel_instance_seg": panel_instance_seg
        }
        if uv is not None:
            uv = np.hstack((uv, np.zeros((uv.shape[0], 1))))
            data_dict["uv"] = uv

        # === Add stitch noise ===
        if self.mode == "train" or self.mode == "val":
            if self.use_stitch_noise:
                # Random noise strength
                rand_param = random.random()*0.8+0.2
                stitch_noise_strength_base = (self.stitch_noise_random_min * rand_param +
                                              self.stitch_noise_random_max * (1-rand_param))
                stitch_noise_strength_base *= self.stitch_noise_strength

                # Stitching direction
                vec =  copy(cur_pcs[mat_gt[:, 1]] - cur_pcs[mat_gt[:,0]])
                vec2 = copy(cur_pcs[mat_gt[:, 0]] - cur_pcs[mat_gt[:, 1]])
                vec = vec/(np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
                vec2 = vec2/(np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-12)

                # 1. Add noise along the point-to-point stitching direction to generate non-smooth panel`s boundary
                stitch_noise_strength1 = stitch_noise_strength_base
                stitch_noise1_strength_per_point1 = np.random.rand(len(mat_gt), 1) * 0.9 + 0.1
                stitch_noise2_strength_per_point2 = np.random.rand(len(mat_gt), 1) * 0.9 + 0.1
                noise1 = vec * stitch_noise1_strength_per_point1 * stitch_noise_strength1 * mean_edge_len
                noise2 = vec2 * stitch_noise2_strength_per_point2 * stitch_noise_strength1 * mean_edge_len
                cur_pcs[mat_gt[:, 1]] += noise1
                cur_pcs[mat_gt[:, 0]] += noise2

                # 2. Add noise on unstitching points
                stitch_noise_strength2 = stitch_noise_strength_base * (random.random() * 1 + 0.5)
                unstitch_mask = np.zeros(cur_pcs.shape[0])
                unstitch_mask[mat_gt[:, 0]] = 1
                unstitch_mask[mat_gt[:, 1]] = 1
                unstitch_mask = unstitch_mask == 0
                noise2 = (np.random.rand(np.sum(unstitch_mask), 3)*2.-1.)
                for _ in range(1): noise2 = smooth_using_convolution(noise2, k=3)
                noise2 = noise2 / (np.linalg.norm(noise2, axis=1, keepdims=True) + 1e-6)
                noise2 = noise2 * stitch_noise_strength2 * mean_edge_len
                cur_pcs[unstitch_mask] += noise2

                # 3. Add noise (very small) on stitching points
                stitch_noise_strength3 = stitch_noise_strength_base * (random.random()*0.2+0.1)
                noise3 = (np.random.rand(len(mat_gt), 3)*2.-1.)
                noise3 = noise3 / (np.linalg.norm(noise3, axis=1, keepdims=True) + 1e-6)
                noise3 = noise3 * stitch_noise_strength3 * mean_edge_len
                cur_pcs[mat_gt[:, 0]] += noise3
                cur_pcs[mat_gt[:, 1]] += noise3

            # GT point 2 point stitching
            mat_gt = stitch_indices2mat(self.num_points, mat_gt)
            data_dict["mat_gt"] = mat_gt

            # mean stitching distance
            Dis = np.sqrt(np.sum(((cur_pcs[:,None,:] - cur_pcs[None,:,:])**2), axis=-1))
            data_dict["mean_stitch_dis_gt"] = np.mean(Dis[mat_gt == 1])
        else:
            pass
        return data_dict


def build_stylexd_dataloader_train_val(cfg, shuffle_val_loader=False):
    if cfg.NUM_WORKERS > 4:
        print("Too much workers may cause fault.")

    # === TRAIN DATASET ===
    data_dict = dict(
        mode="train",
        data_dir=cfg.DATA.DATA_DIR,

        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,

        shrink_mesh=cfg.DATA.SHRINK_MESH.TRAIN,
        shrink_mesh_param=cfg.DATA.SHRINK_MESH_PARAM.TRAIN,
        shrink_bystitch=cfg.DATA.SHRINK_BYSTITCH.TRAIN,
        shrink_bystitch_strength=cfg.DATA.SHRINK_BYSTITCH_STRENGTH.TRAIN,

        # stitch_noise only used for train data
        use_stitch_noise=cfg.DATA.USE_STITCH_NOISE,
        stitch_noise_strength=cfg.DATA.STITCH_NOISE_STRENGTH,
        stitch_noise_random_range=cfg.DATA.STITCH_NOISE_RANDOM_RANGE,

        pcs_sample_type=cfg.DATA.PCS_SAMPLE_TYPE.TRAIN,
        pcs_sample_type_stitch_only_sample_boundary=cfg.DATA.PCS_SAMPLE_TYPE.STITCH.ONLY_SAMPLE_BOUNDARY,

        panel_noise_type=cfg.DATA.PANEL_NOISE_TYPE.TRAIN,
        scale_range=cfg.DATA.SCALE_RANGE,
        rot_range=cfg.DATA.ROT_RANGE,
        trans_range=cfg.DATA.TRANS_RANGE,
        bbox_noise_strength=cfg.DATA.BBOX_NOISE_STRENGTH,
        min_part_point=cfg.DATA.MIN_PART_POINT,

        read_uv=cfg.MODEL.USE_UV_FEATURE,

        dataset_split_dir=cfg.DATA.DATASET_SPLIT_DIR,
    )
    train_set = AllPieceMatchingDataset_stylexd(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.DATA.SHUFFLE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    # === VAL DATASET ===
    data_dict["mode"] = "val"
    data_dict["pcs_sample_type"] = cfg.DATA.PCS_SAMPLE_TYPE.VAL
    data_dict["panel_noise_type"] = cfg.DATA.PANEL_NOISE_TYPE.VAL
    data_dict["shrink_mesh"] = cfg.DATA.SHRINK_MESH.VAL
    data_dict["shrink_mesh_param"] = cfg.DATA.SHRINK_MESH_PARAM.VAL
    data_dict["shrink_bystitch"] =  cfg.DATA.SHRINK_BYSTITCH.VAL
    data_dict["shrink_bystitch_strength"] = cfg.DATA.SHRINK_BYSTITCH_STRENGTH.VAL

    val_set = AllPieceMatchingDataset_stylexd(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle_val_loader,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader


def build_stylexd_dataloader_inference(cfg, inference_data_list=None):
    """
    used to load generated Garmage
    """
    data_dict = dict(
        mode="inference",
        data_dir=cfg.DATA.DATA_DIR,
        # data_types=cfg.DATA.DATA_TYPES.INFERENCE,

        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,

        shrink_mesh=cfg.DATA.SHRINK_MESH.INFERENCE,
        shrink_mesh_param=cfg.DATA.SHRINK_MESH_PARAM.INFERENCE,
        shrink_bystitch=cfg.DATA.SHRINK_BYSTITCH.INFERENCE,
        shrink_bystitch_strength=cfg.DATA.SHRINK_BYSTITCH_STRENGTH.INFERENCE,

        pcs_sample_type=cfg.DATA.PCS_SAMPLE_TYPE.INFERENCE,
        pcs_sample_type_stitch_only_sample_boundary=False,

        panel_noise_type=cfg.DATA.PANEL_NOISE_TYPE.INFERENCE,
        scale_range=cfg.DATA.SCALE_RANGE,
        rot_range=cfg.DATA.ROT_RANGE,
        trans_range=cfg.DATA.TRANS_RANGE,
        bbox_noise_strength=cfg.DATA.BBOX_NOISE_STRENGTH,

        min_part_point=cfg.DATA.MIN_PART_POINT,

        read_uv = True,

        dataset_split_dir=cfg.DATA.DATASET_SPLIT_DIR,
        inference_data_list=inference_data_list,
    )
    inference_mode_set = AllPieceMatchingDataset_stylexd(**data_dict)
    inference_loader = DataLoader(
        dataset=inference_mode_set,
        batch_size=1,
        shuffle=cfg.DATA.SHUFFLE,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return inference_loader