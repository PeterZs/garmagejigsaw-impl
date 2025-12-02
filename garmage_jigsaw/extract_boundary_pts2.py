
import os
import json
import pickle
import shutil
from glob import glob
from tqdm import tqdm

import cv2
import uuid
import trimesh
from matplotlib.colors import to_hex
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils import (pointcloud_visualize, draw_per_panel_geo_imgs, draw_bbox_geometry)

import plotly.graph_objects as go


def pointcloud_condition_visualize(vertices: np.ndarray, output_fp=None):
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices 应为 (N, 3) 的 numpy 数组"

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    color = "#717388"  # 用 z 来着色
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    zrange = z.max() - z.min()
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                colorscale='Viridis',
                opacity=1,
                showscale=False  # 不显示 colorbar
            ),
            showlegend=False  # 不显示图例
        )
    ])

    # 隐藏坐标轴、网格、背景等
    axis_style = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        visible=False  # 最直接隐藏整个轴
    )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2)
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode='manual',
            aspectratio=dict(
                x=xrange,
                y=yrange,
                z=zrange
            )
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    RESO = 512
    if output_fp:
        # fig.write_html(output_fp.replace(".pkl", "") + "_pcCond_vis.html")
        fig.write_image(output_fp.replace(".pkl", "") + "_pcCond.png", width=RESO, height=RESO)



# 判断一个contour是不是净边
def is_contour_OutLine(contour_idx, panel_instance_seg):
    return contour_idx == 0 or panel_instance_seg[contour_idx] != panel_instance_seg[contour_idx - 1]


# 获取一段随机的 uuid
def get_random_uuid():
    id = str(uuid.uuid4().hex)
    result = id[0:8] + "-" + id[8:12] + "-" + id[12:16] + "-" + id[16:20] + "-" + id[20:]
    return result


def angle_between_vectors(vectors1, vectors2):
    # 点积
    dot_product = np.sum(vectors1 * vectors2, axis=-1)
    # 向量的模长
    norm_v1 = np.linalg.norm(vectors1, axis=-1)
    norm_v2 = np.linalg.norm(vectors2, axis=-1)
    # 计算cos(θ)
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)  # [todo] 会报错 invalid value encountered in divide
    # 反余弦，转换为度
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)


def _denormalize_pts(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset


def _pad_arr(arr, pad_size=10):
    return np.pad(
        arr,
        ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        # pad size to each dimension, require tensor to have size (H,W, C)
        mode='constant',
        constant_values=0)


def load_data(data_type="Garmage256", fp={}, save_vis=False):
    """
    Load a garmage file.
    :param data_type:
    :param fp:
    :return:
    """
    if data_type == "Garmage256":

        with open(fp["garment_fp"], "rb") as f:
            data = pickle.load(f)
        geo_orig = data["surf_ncs"]
        mask = data["surf_mask"]
        n_surfs = len(geo_orig)

        # save some visualize results (Only for watch)
        if save_vis:
            # panel wise geometry image
            mask_draw_per_panel = mask
            if np.max(mask_draw_per_panel)>0.8 and np.min(mask_draw_per_panel)<-0.8:
                mask_draw_per_panel = (mask_draw_per_panel+1)/2
                mask_draw_per_panel[mask_draw_per_panel<0.1] = 0
                mask_draw_per_panel[mask_draw_per_panel>1] = 1
            colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
            draw_per_panel_geo_imgs(geo_orig.reshape(n_surfs,-1,3), mask_draw_per_panel.reshape(n_surfs,-1), colors, pad_size=5, out_dir=fp["garment_fp"].replace(".pkl","")+"_per_panel_images")

            # 3D pointcloud+bbox
            try:
                _surf_bbox_ = data["surf_bbox"]
            except Exception:
                _surf_bbox_ = data["surf_bbox_wcs"]

            _surf_ncs_ = data["surf_ncs"].reshape(n_surfs,-1,3)
            _surf_wcs_ = _denormalize_pts(_surf_ncs_, _surf_bbox_)
            _mask_ = data["surf_mask"].reshape(n_surfs,-1)

            if _mask_.dtype == np.float32 or _mask_.dtype == np.float64:
                _mask_ = _mask_>0

            draw_bbox_geometry(
                bboxes=_surf_bbox_,
                bbox_colors=colors,
                points=_surf_wcs_,
                point_masks=_mask_,
                point_colors=colors,
                num_point_samples=2000,
                all_bboxes=_surf_bbox_,
                output_fp=fp["garment_fp"].replace(".pkl","")+"_3d_PC_BBOX.png",
                visatt_dict={
                    "bboxmesh_opacity": 0.12,
                    "point_size": 12,
                    "point_opacity": 0.8,
                    "bboxline_width": 8,
                }
            )

            try:
                _surf_uv_bbox = data["surf_uv_bbox"]
            except Exception:
                _surf_uv_bbox = data["surf_uv_bbox_wcs"]
            _surf_uv_bbox_wcs_ = np.zeros((n_surfs, 6))
            _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = _surf_uv_bbox

        if "surf_uv_bbox" in data: uv_bbox = data["surf_uv_bbox"]
        else: uv_bbox = data["surf_uv_bbox_wcs"]
        if "surf_bbox" in data: _surf_bbox = data["surf_bbox"]
        else: _surf_bbox = data["surf_bbox_wcs"]

        if geo_orig.ndim == 4:
            geo_orig = geo_orig.reshape(geo_orig.shape[0], -1, geo_orig.shape[3])

        geo_orig = _denormalize_pts(geo_orig, _surf_bbox)
        geo_orig = geo_orig.reshape(-1, 256, 256, 3)
        mask = mask.reshape(-1, 256, 256, 1)

        # 董远的有些数据没有过sigmoid，需要在这里重新过一遍
        if np.min(mask) < -0.1 or np.max(mask) > 1.1:
            mask = torch.sigmoid(torch.tensor(mask, dtype=torch.float64))
            mask = mask.detach().numpy()
            mask = mask>0.5

        # pad geometry image
        geo_orig = _pad_arr(geo_orig, pad_size=5)
        mask = _pad_arr(mask, pad_size=5)

    else:
        raise NotImplementedError

    return geo_orig, mask, uv_bbox


def resample_boundary(points, contour, corner_index, delta, c_idx, outlier_thresh=0.05):
    """

    :param points:          Nx3
    :param contour:         Nx2
    :param corner_index:    哪些点是角点
    :param delta:           resample间距
    :param c_idx:
    :param outlier_thresh:  用于过滤间距异常的点
    :return:
    """

    points = np.asarray(points)
    TEST = False

    if TEST:
        # 假设 ndarrayA 和 ndarrayB 是已定义的 Mx2 和 Nx2 的 ndarray
        A = contour[corner_index]  # 示例数据，Mx2
        B = contour  # 示例数据，Nx2

        # 创建一个图形
        plt.figure(figsize=(8, 6))

        # 绘制 B 点集，使用蓝色
        plt.scatter(B[:, 0], B[:, 1], color='blue', label='Points B')

        # 绘制 A 点集，使用红色
        plt.scatter(A[:, 0], A[:, 1], color='red', label='Points A')

        # 添加图例
        plt.legend()

        # 设置标题和标签
        plt.title('Scatter Plot of Points A and B')
        plt.xlabel('X')
        plt.ylabel('Y')

        # 显示图形
        plt.show()
        input("Press Enter to continue...")

    # remove outliers
    deltas_prev = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    deltas_next = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    # 筛选出前后距离都小于阈值的有效点
    valid_pts = np.logical_and(deltas_prev < outlier_thresh, deltas_next < outlier_thresh)
    valid_pts[corner_index] = True
    points = points[valid_pts, :]
    contour = contour[valid_pts, :]
    # 如果存在valid_pts，需要同步修改corner_index的内容
    _old_indices = np.arange(len(valid_pts))
    _new_indices = np.cumsum(valid_pts) - 1
    corner_index = _new_indices[corner_index]

    # Compute distances between consecutive points
    # 将边界点连接成一个闭合的环
    points = np.vstack([points, points[0]])  # Close the boundary
    contour = np.vstack([contour, contour[0]])

    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)  # 计算连续点之间的距离

    # Compute cumulative arc length  # 计算累积弧长
    arc_lengths = np.insert(np.cumsum(deltas), 0, 0)

    # Total length of the boundary
    total_length = arc_lengths[-1]

    # Number of new points
    num_points = int(np.ceil(total_length / delta)) + 1

    # New equally spaced arc lengths
    new_arc_lengths = np.linspace(0, total_length, num=num_points)

    # resample后的顶点在原始顶点中的位置应该在哪两个顶点之间
    # 长度归一化 (1e-x为了防止new_points_insert_idx[-1]越界)
    new_arc_lengths = new_arc_lengths * (arc_lengths[-1] / (new_arc_lengths[-1] + 1e-12))

    # 将resample后的点中，与原本corner最接近那些点会被移动到corner上 ===
    corner_arc = arc_lengths[corner_index]    # 角点对应的 param
    new_corner_index = np.abs(new_arc_lengths[:, None] - corner_arc).argmin(axis=0)  # 新的采样点中最接近角点的几个点
    dis_2_old_corner = new_arc_lengths[new_corner_index] - corner_arc
    # resample 后的点中，找到离 resample 前的角点中最近的点。
    min_dis_mask = np.abs(dis_2_old_corner) < delta/3
    # 离这些角点最近的边缘点被移到角点上
    new_arc_lengths[new_corner_index[min_dis_mask]] = corner_arc[min_dis_mask]
    # 为其它的角点在resample后的点中插入新的点
    new_arc_lengths = np.sort(np.concatenate((new_arc_lengths, corner_arc[~min_dis_mask])))
    # 重新计算每个角点的index
    new_corner_index = np.abs(new_arc_lengths[:, None] - corner_arc).argmin(axis=0)

    num_points = len(new_arc_lengths)

    # 删除空的拟合边
    diff = np.abs(new_corner_index - np.roll(new_corner_index, -1)) > 0
    if np.sum(~diff)>0:
        print("Empty approx edge current, deleted.")
    new_corner_index = new_corner_index[diff]

    # Interpolate 3D-points
    resampled_points_3D = np.zeros((num_points, 3))
    for i in range(3):
        resampled_points_3D[:, i] = np.interp(new_arc_lengths, arc_lengths, points[:, i])
    # Interpolate 2D-points
    resampled_points_2D = np.zeros((num_points, 2))
    for i in range(2):
        resampled_points_2D[:, i] = np.interp(new_arc_lengths, arc_lengths, contour[:, i])

    if TEST:
        # 假设 ndarrayA 和 ndarrayB 是已定义的 Mx2 和 Nx2 的 ndarray
        A = resampled_points_2D[new_corner_index]  # 示例数据，Mx2
        B = resampled_points_2D  # 示例数据，Nx2
        # 创建一个图形
        plt.figure(figsize=(8, 6))
        # 绘制 B 点集，使用蓝色
        plt.scatter(B[:, 0], B[:, 1], color='blue', label='Points B')
        # 绘制 A 点集，使用红色
        plt.scatter(A[:, 0], A[:, 1], color='red', label='Points A')
        # 添加图例
        plt.legend()
        # 设置标题和标签
        plt.title('Scatter Plot of Points A and B')
        plt.xlabel('X')
        plt.ylabel('Y')
        # 显示图形
        plt.show()
        input("Press Enter to continue...")

    return resampled_points_3D, resampled_points_2D, new_corner_index, valid_pts#, old_point_param, new_point_param


def compute_cross_angles(arr1, arr2):
    N, k1, _ = arr1.shape
    _, k2, _ = arr2.shape

    vec1 = arr1[:, :, np.newaxis, :]  # (N, k1, 1, 2)
    vec2 = arr2[:, np.newaxis, :, :]  # (N, 1, k2, 2)

    # 计算点积
    dot_product = np.sum(vec1 * vec2, axis=-1)  # (N, k1, k2)

    # 计算模长
    norm1 = np.linalg.norm(vec1, axis=-1)  # (N, k1, 1)
    norm2 = np.linalg.norm(vec2, axis=-1)  # (N, 1, k2)
    norms = norm1 * norm2  # (N, k1, k2)

    # 计算夹角，避免数值误差超出 [-1,1]
    cos_theta = np.clip(dot_product / norms, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta))  # (N, k1, k2)

    # 变形为 Nx(k1*k2)x1
    return angles.reshape(N, k1 * k2, 1)


def get_boundary_corner(contour, resized_uv, corner_min_thresh=25, kernal_size_shrink=0, show_2d_approx=False):
    """
    BEST IS
    corner_min_thresh = 25
    kernal = np.array([-9, -7, -5, -4, -3, 0, 3, 4, 5, 7, 9])
    """

    # 检测角点的阈值
    # 检测角点的卷积核
    kernal = np.array([-9, -7, -5, -4, -3, 0, 3, 4, 5, 7, 9])

    if len(kernal)-kernal_size_shrink*2<3:
        kernal_size_shrink=(len(kernal)-3)/2
    kernal_mid = np.where(kernal==0)[0][0]  # 卷积核的中心位置

    # === 获取每一个边缘点的 【平均本地角度】 ===
    # 每个点以及其卷积到的其它点的index [N x K]
    index_matrix = np.repeat(np.arange(0, len(contour)).reshape(-1, 1), len(kernal), axis=-1)
    index_matrix += kernal
    np.where(index_matrix < 0)
    index_matrix[index_matrix < 0] += len(contour)
    index_matrix[index_matrix >= len(contour)] -= len(contour)
    # index_matrix对应的点的2D位置
    pos2d_matrix = contour[index_matrix]
    # [TODO] 上面这行改成下面这行，重新调参
    # pos2d_matrix = resized_uv[index_matrix]

    # Contour上的点
    contour_pos2d = pos2d_matrix[:, kernal_mid, :].reshape(len(contour), 1, 2)
    # Contour上的点到左右卷积到的点的向量
    vec_matrix_left = pos2d_matrix[:, :kernal_mid, :].reshape(len(contour), -1, 2)
    vec_matrix_right = pos2d_matrix[:, kernal_mid + 1:, :].reshape(len(contour), -1, 2)
    vec_matrix_left = contour_pos2d - vec_matrix_left
    vec_matrix_right = vec_matrix_right - contour_pos2d
    # 左右向量间的pairwise angle
    # 两种计算角度矩阵的方法
    # 1. 左右任一一对向量的角度
    angles_matrix = compute_cross_angles(vec_matrix_left, vec_matrix_right)
    # 2. 左右同一相对位置的向量的角度
    # angles_matrix = angle_between_vectors(vec_matrix_left.reshape(-1,2), vec_matrix_right[:,::-1,:].reshape(-1,2))
    # angles_matrix = angles_matrix.reshape(vec_matrix_left.shape[0], vec_matrix_left.shape[1], 1)

    # 每个contour点的平均夹角
    angles_matrix_mean = np.mean(angles_matrix, axis=-2).reshape(len(contour))

    # === 提取 平均本地角度 较大的一些区间 ===
    too_high_mask = angles_matrix_mean > corner_min_thresh

    # 找到index最小的False的位置（后面的遍历会从这个开始）
    start_index = np.where(~too_high_mask)[0][0]

    # 获取存在极大值的区间
    corner_interval_list = []  # 所有存在极大值的区间
    previous_valid = False
    corner_interval = []
    for point_idx in range(len(contour)):
        point_idx += start_index
        if point_idx >= len(contour): point_idx -= len(contour)
        next_idx = point_idx + 1
        if next_idx >= len(contour): next_idx -= len(contour)
        # 是否是区间的起点
        if too_high_mask[point_idx] and not previous_valid:
            previous_valid = True
            corner_interval.append(point_idx)
        # 是否是区间的终点
        if too_high_mask[point_idx] and not too_high_mask[next_idx]:
            previous_valid = False
            corner_interval.append(point_idx)
            corner_interval_list.append(corner_interval)
            corner_interval = []

    # === 将一些间距过近的(且太小的)的区间合并 ===
    interval_dis_thresh = 10  # 两个被合并区间的距离的阈值
    def merge_intervals_on_ring(A, N, T):
        class Node:
            def __init__(self, start, end):
                self.start = start
                self.end = end
                self.prev = None
                self.next = None
        if len(A) == 0:
            return np.array([], dtype=int)

        # 创建所有节点
        nodes = [Node(a[0], a[1]) for a in A]
        num_nodes = len(nodes)

        # 双向环形链表
        for i in range(num_nodes):
            prev_index = (i - 1) % num_nodes
            next_index = (i + 1) % num_nodes
            nodes[i].prev = nodes[prev_index]
            nodes[i].next = nodes[next_index]

        has_merged = True
        while has_merged:
            has_merged = False
            current = nodes[0]  # 起始点
            start_node = current
            visited = set()
            while True:
                if current in visited:
                    break
                visited.add(current)
                next_node = current.next
                # 计算间距
                spacing = (next_node.start - (current.end + 1) + N) % N
                if spacing < T:
                    # 合并当前节点和下一个节点
                    merged_start = current.start
                    merged_end = next_node.end
                    current.end = merged_end
                    current.next = next_node.next
                    next_node.next.prev = current
                    # 如果下一个节点是起始点，更新起始点
                    if next_node == start_node:
                        start_node = current
                    # 在nodes列表中移除
                    if next_node in nodes:
                        nodes.remove(next_node)
                    has_merged = True
                else:
                    current = next_node
            # 如果所有节点都被合并成一个，则退出循环
            if len(nodes) == 1:
                break

        result = []
        if len(nodes) == 0:
            return np.array(result, dtype=int)
        current = nodes[0]
        visited = set()
        while current not in visited:
            visited.add(current)
            result.append([current.start, current.end])
            current = current.next
            if current is None:
                break

        return np.array(result, dtype=int)

    corner_interval_list = merge_intervals_on_ring(A=corner_interval_list, N=len(contour), T=interval_dis_thresh)

    # === 将每个区间转化为索引 ===
    corner_interval_index_list = []
    for c in corner_interval_list:
        if c[0] < c[1]:
            corner_interval_index_list.append(np.arange(c[0], c[1] + 1))
        elif c[0] > c[1]:
            corner_interval_index_list.append(np.concatenate((
                np.arange(c[0], len(contour)),
                np.arange(0, c[1] + 1))
            ))
        else:
            corner_interval_index_list.append(c[0])

    # === 对每一个区间的索引，找到极大值点的位置 ===
    corner_index = []
    for c in corner_interval_index_list:
        # 获取极大值点（max_angle_index）[todo]这里获取极大值的方法并不严谨，仅仅是取最大值，但一个区间中可能存在多个极大值
        if c.ndim == 1:
            max_angle_index = []
            if len(c) == 1:
                max_angle_index = max_angle_index.append(c[0])
            else:
                # 取最大值的方法
                max_angle_i = np.argmax(angles_matrix_mean[c])
                max_angle_index = [c[max_angle_i]]
                # # 取极大值的方法
                # for idx in range(len(c)):
                #     if idx == 0 and angles_matrix_mean[c][idx] >= angles_matrix_mean[c][idx + 1]:
                #         max_angle_index.append(c[idx])
                #     elif idx == len(angles_matrix_mean[c])-1 and angles_matrix_mean[c][idx] > angles_matrix_mean[c][idx - 1]:
                #         max_angle_index.append(c[idx])
                #     elif angles_matrix_mean[c][idx] > angles_matrix_mean[c][idx - 1] and angles_matrix_mean[c][idx] > angles_matrix_mean[c][idx + 1]:
                #         max_angle_index.append(c[idx])
        elif c.ndim == 0:
            max_angle_index = [c]
        else:
            raise NotImplementedError()  # 先留着，应该不会报错
        corner_index.extend(max_angle_index)


    # === 可视化 ===
    if show_2d_approx:
        corner_pos2d = contour[corner_index]
        image = np.ones((266, 266), dtype=np.uint8) * 255  # 白色背景
        for x, y in contour:
            if 0 <= x < 266 and 0 <= y < 266:  # 确保坐标在图像范围内
                image[y, x] = 0  # 将点绘制为黑色
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', origin='upper')
        ax.axis('off')
        circle_radius = 5
        for x, y in corner_pos2d:
            circle = Circle((x, y), circle_radius, color='blue', fill=False, linewidth=2)
            ax.add_patch(circle)

        plt.show()

    return corner_index


def get_boundary_corner_v2(contour, resized_uv, corner_min_thresh=25, kernal_size_shrink=0, show_2d_approx=False):
    """
    BEST IS
    corner_min_thresh = 25
    kernal = np.array([-9, -7, -5, -4, -3, 0, 3, 4, 5, 7, 9])
    """

    # 检测角点的阈值
    # 检测角点的卷积核
    kernal = np.array([-5, -4, -3, 0, 3, 4, 5])

    if len(kernal)-kernal_size_shrink*2<3:
        kernal_size_shrink=(len(kernal)-3)/2
    kernal_mid = np.where(kernal==0)[0][0]  # 卷积核的中心位置

    # === 获取每一个边缘点的 【平均本地角度】 ===
    # 每个点以及其卷积到的其它点的index [N x K]
    index_matrix = np.repeat(np.arange(0, len(contour)).reshape(-1, 1), len(kernal), axis=-1)
    index_matrix += kernal
    np.where(index_matrix < 0)
    index_matrix[index_matrix < 0] += len(contour)
    index_matrix[index_matrix >= len(contour)] -= len(contour)
    # index_matrix对应的点的2D位置
    pos2d_matrix = contour[index_matrix]
    # [TODO] 上面这行改成下面这行，重新调参
    # pos2d_matrix = resized_uv[index_matrix]

    # Contour上的点
    contour_pos2d = pos2d_matrix[:, kernal_mid, :].reshape(len(contour), 1, 2)
    # Contour上的点到左右卷积到的点的向量
    vec_matrix_left = pos2d_matrix[:, :kernal_mid, :].reshape(len(contour), -1, 2)
    vec_matrix_right = pos2d_matrix[:, kernal_mid + 1:, :].reshape(len(contour), -1, 2)
    vec_matrix_left = contour_pos2d - vec_matrix_left
    vec_matrix_right = vec_matrix_right - contour_pos2d
    # 左右向量间的pairwise angle
    # 两种计算角度矩阵的方法
    # 1. 左右任一一对向量的角度
    angles_matrix = compute_cross_angles(vec_matrix_left, vec_matrix_right)
    # 2. 左右同一相对位置的向量的角度
    # angles_matrix = angle_between_vectors(vec_matrix_left.reshape(-1,2), vec_matrix_right[:,::-1,:].reshape(-1,2))
    # angles_matrix = angles_matrix.reshape(vec_matrix_left.shape[0], vec_matrix_left.shape[1], 1)

    # 每个contour点的平均夹角
    angles_matrix_mean = np.mean(angles_matrix, axis=-2).reshape(len(contour))

    # === 提取 平均本地角度 较大的一些区间 ===
    too_high_mask = angles_matrix_mean > corner_min_thresh

    # 找到index最小的False的位置（后面的遍历会从这个开始）
    start_index = np.where(~too_high_mask)[0][0]

    # 获取存在极大值的区间
    corner_interval_list = []  # 所有存在极大值的区间
    previous_valid = False
    corner_interval = []
    for point_idx in range(len(contour)):
        point_idx += start_index
        if point_idx >= len(contour): point_idx -= len(contour)
        next_idx = point_idx + 1
        if next_idx >= len(contour): next_idx -= len(contour)
        # 是否是区间的起点
        if too_high_mask[point_idx] and not previous_valid:
            previous_valid = True
            corner_interval.append(point_idx)
        # 是否是区间的终点
        if too_high_mask[point_idx] and not too_high_mask[next_idx]:
            previous_valid = False
            corner_interval.append(point_idx)
            corner_interval_list.append(corner_interval)
            corner_interval = []

    # === 将一些间距过近的(且太小的)的区间合并 ===
    interval_dis_thresh = 3  # 两个被合并区间的距离的阈值
    def merge_intervals_on_ring(A, N, T):
        class Node:
            def __init__(self, start, end):
                self.start = start
                self.end = end
                self.prev = None
                self.next = None
        if len(A) == 0:
            return np.array([], dtype=int)

        # 创建所有节点
        nodes = [Node(a[0], a[1]) for a in A]
        num_nodes = len(nodes)

        # 双向环形链表
        for i in range(num_nodes):
            prev_index = (i - 1) % num_nodes
            next_index = (i + 1) % num_nodes
            nodes[i].prev = nodes[prev_index]
            nodes[i].next = nodes[next_index]

        has_merged = True
        while has_merged:
            has_merged = False
            current = nodes[0]  # 起始点
            start_node = current
            visited = set()
            while True:
                if current in visited:
                    break
                visited.add(current)
                next_node = current.next
                # 计算间距
                spacing = (next_node.start - (current.end + 1) + N) % N
                if spacing < T:
                    # 合并当前节点和下一个节点
                    merged_start = current.start
                    merged_end = next_node.end
                    current.end = merged_end
                    current.next = next_node.next
                    next_node.next.prev = current
                    # 如果下一个节点是起始点，更新起始点
                    if next_node == start_node:
                        start_node = current
                    # 在nodes列表中移除
                    if next_node in nodes:
                        nodes.remove(next_node)
                    has_merged = True
                else:
                    current = next_node
            # 如果所有节点都被合并成一个，则退出循环
            if len(nodes) == 1:
                break

        result = []
        if len(nodes) == 0:
            return np.array(result, dtype=int)
        current = nodes[0]
        visited = set()
        while current not in visited:
            visited.add(current)
            result.append([current.start, current.end])
            current = current.next
            if current is None:
                break

        return np.array(result, dtype=int)

    corner_interval_list = merge_intervals_on_ring(A=corner_interval_list, N=len(contour), T=interval_dis_thresh)

    # === 将每个区间转化为索引 ===
    corner_interval_index_list = []
    for c in corner_interval_list:
        if c[0] < c[1]:
            corner_interval_index_list.append(np.arange(c[0], c[1] + 1))
        elif c[0] > c[1]:
            corner_interval_index_list.append(np.concatenate((
                np.arange(c[0], len(contour)),
                np.arange(0, c[1] + 1))
            ))
        else:
            corner_interval_index_list.append(c[0])

    # === 对每一个区间的索引，找到极大值点的位置 ===
    corner_index = []
    for c in corner_interval_index_list:
        # 获取极大值点（max_angle_index）[todo]这里获取极大值的方法并不严谨，仅仅是取最大值，但一个区间中可能存在多个极大值
        if c.ndim == 1:
            max_angle_index = []
            if len(c) == 1:
                max_angle_index = max_angle_index.append(c[0])
            else:
                # 取最大值的方法
                max_angle_i = np.argmax(angles_matrix_mean[c])
                max_angle_index = [c[max_angle_i]]
                # # 取极大值的方法
                # for idx in range(len(c)):
                #     if idx == 0 and angles_matrix_mean[c][idx] >= angles_matrix_mean[c][idx + 1]:
                #         max_angle_index.append(c[idx])
                #     elif idx == len(angles_matrix_mean[c])-1 and angles_matrix_mean[c][idx] > angles_matrix_mean[c][idx - 1]:
                #         max_angle_index.append(c[idx])
                #     elif angles_matrix_mean[c][idx] > angles_matrix_mean[c][idx - 1] and angles_matrix_mean[c][idx] > angles_matrix_mean[c][idx + 1]:
                #         max_angle_index.append(c[idx])
        elif c.ndim == 0:
            max_angle_index = [c]
        else:
            raise NotImplementedError()  # 先留着，应该不会报错
        corner_index.extend(max_angle_index)

    if len(corner_index) == 0:
        corner_index = np.floor(np.linspace(0, len(contour_pos2d), 4, endpoint=False)).astype(int).tolist()

    # === 可视化 ===
    # show_2d_approx = True
    if show_2d_approx:
        corner_pos2d = contour[corner_index]
        image = np.ones((266, 266), dtype=np.uint8) * 255  # 白色背景
        for x, y in contour:
            if 0 <= x < 266 and 0 <= y < 266:  # 确保坐标在图像范围内
                image[y, x] = 0  # 将点绘制为黑色
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', origin='upper')
        ax.axis('off')
        circle_radius = 5
        for x, y in corner_pos2d:
            circle = Circle((x, y), circle_radius, color='blue', fill=False, linewidth=2)
            ax.add_patch(circle)

        plt.show()

    return corner_index


def resize_uv(points_2D, panel_instance_seg, uv_bbox):
    """
    :param points_2D:     2D (UV) resampled points
    :param panel_instance_seg:      Determine the panel to which each contour belongs.
    :param uv_bbox:                 2D (UV) bbox of each panel
    :return:
    """
    points_2D = [ri.astype(np.float64) for ri in points_2D]
    nps = [len(ri) for ri in points_2D]
    uv_local = np.concatenate(points_2D)
    nps_cumsum = np.cumsum(nps)

    # 将每个panel的点放入uv_bbox中对应的bbox里 ------------------------------------------------------------------------------
    uv_global = np.zeros_like(uv_local, dtype=np.float32)
    bbox_uv_local_prev = None
    for contour_idx in range(len(nps_cumsum)):
        instance_idx = panel_instance_seg[contour_idx]
        if contour_idx == 0: start_point_idx = 0
        else: start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        bbox = uv_bbox[instance_idx]

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            bbox_uv_local = np.array([np.min(all_coords[:, 0]), np.min(all_coords[:, 1]),
                                      np.max(all_coords[:, 0]), np.max(all_coords[:, 1])])
            bbox_uv_local_prev = bbox_uv_local
        else:
            bbox_uv_local = bbox_uv_local_prev  # 如果不是 OutLine，则采用对应 OutLine 的 BBOX 进行调整

        uv_bbox_scale = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        uv_bbox_offset = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

        all_coords[:, 0] -= (bbox_uv_local[0] + bbox_uv_local[2]) / 2
        all_coords[:, 1] -= (bbox_uv_local[1] + bbox_uv_local[3]) / 2
        all_coords[:, 0] /= bbox_uv_local[2] - bbox_uv_local[0]
        all_coords[:, 1] /= bbox_uv_local[3] - bbox_uv_local[1]

        all_coords[:, 0] *= uv_bbox_scale[0]
        all_coords[:, 1] *= uv_bbox_scale[1]
        all_coords[:, 0] += uv_bbox_offset[0]
        all_coords[:, 1] += uv_bbox_offset[1]

        uv_global[start_point_idx:end_point_idx] = all_coords

    # 将整个 garment 放到一个更大的box里 -------------------------------------------------------------------------------------
    uv_bbox_scale = 1000
    uv_bbox_offset = [0, 1000]
    for contour_idx in range(len(nps_cumsum)):
        if contour_idx == 0:
            start_point_idx = 0
        else:
            start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        all_coords *= uv_bbox_scale
        all_coords += uv_bbox_offset

        uv_global[start_point_idx:end_point_idx] = all_coords

    result = []
    for contour_idx in range(len(nps_cumsum)):
        if contour_idx == 0: start_point_idx = 0
        else: start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]
        result.append(uv_global[start_point_idx:end_point_idx])
    return result


def extract_boundary_pts(geo_orig, uv_bbox, mask, delta=0.023, RESO=64, erode_size=1, show_2d_approx=False):
    """
    提取边缘点
    :param geo_orig:        几何图 的几何信息
    :param mask:            几何图 的mask
    :param delta:           Resample间距
    :param RESO:            几何图分辨率
    :param show_3D_garment:     调试用
    :param show_2d_approx:      调试用
    :param export_vis_result:   调试用
    :return:
    """
    panel_nes = []              # 每个panel上有几个边
    contour_nes = []            # 每个contouur上有几个边
    resampled_points_3D = []    # resample后的3D的点
    resampled_points_2D = []    # resample后的2D的点
    panel_instance_seg = []     # 一个contour所属的panel实例
                                # 一个Panel上可能出现多个contours，可以根据每个contour所属的Panel实例来进行合并
    edge_approx = []            # 拟合边

    # 根据RESO，获取contour的最小长度
    thresh_dict={64:16, 128:32, 256:64, 512:128, 1024:256}  # 随分辨率变化（理论上是应该是线性变化的，but it just work//)
    contour_min_thresh = thresh_dict.get(RESO, 16)

    contour_list = []
    empty_GeoImg_num = 0
    for panel_idx in range(mask.shape[0]):
        # filter empty GeoImg
        geo_dist = np.linalg.norm(geo_orig[panel_idx], axis=-1)
        if geo_dist.min() < 1e-6 and geo_dist.max() < 1e-6:
            empty_GeoImg_num+=1
            continue

        # erode img by mask
        mask_img = (mask[panel_idx] * 255.0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_img = cv2.erode(mask_img, kernel, iterations=erode_size)
        """
        [todo]
        参考 https://steam.oxxostudio.tw/category/python/ai/opencv-erosion-dilation.html 
        提到的处理图像杂讯的办法，合理的做法应该是：
            img = cv2.erode(img, kernel)
            img = cv2.dilate(img, kernel)
        先erode消除杂讯（板片边缘像素），再dilate回来
        
        虽然erode后会影响板片的形状，但是dilate可以恢复大部分，且原本板片边缘不可靠的像素也被消除了
        """
        mask_img[mask_img >= 150] = 255
        mask_img[mask_img < 150] = 0

        # extract contours by mask
        _, thresh = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contours_pts = [np.squeeze(contour, axis=1) for contour in contours if contour.shape[0] > 16]

        for contour in contours:
            # filter contour too small
            if contour.shape[0] < contour_min_thresh:
                continue

            contour_list.append(np.squeeze(contour, axis=-2))
            panel_instance_seg.append(panel_idx-empty_GeoImg_num)  # 所属的panel实例

    resized_uv = resize_uv(contour_list, panel_instance_seg, uv_bbox)

    for contour_idx, contour in enumerate(contour_list):
        p_idx = panel_instance_seg[contour_idx]  # contour 属于的 panel 的 idx

        # 获取角点的index
        # corner_index = get_boundary_corner(contour, resized_uv[contour_idx], corner_min_thresh=25, kernal_size_shrink=0, show_2d_approx=show_2d_approx)
        corner_index = get_boundary_corner_v2(contour, resized_uv[contour_idx], corner_min_thresh=30, kernal_size_shrink=0, show_2d_approx=show_2d_approx)

        # 端点太少的情况，直接平均挑选4个点作为端点
        if len(corner_index) <= 2:
            corner_index = np.arange(0, len(contour), len(contour)/4 + 1, dtype=np.int32)
            corner_index = np.unique(corner_index)

        # extract boundary points
        geo_arr = geo_orig[p_idx]
        geo_sample_pts = geo_arr[contour[:, 1], contour[:, 0], :]

        # resample boundary pts（delta 是 resample 的间距）
        new_points_3D, new_points_2D, new_corner_index, valid_dis_pts = (
                    resample_boundary(geo_sample_pts, contour, corner_index, delta, contour_idx, outlier_thresh=0.05))

        contour = contour[valid_dis_pts]
        contour_list[contour_idx] = contour

        contour_nes.append(len(new_corner_index))  # 每个Panel（contour）上几个edge
        resampled_points_3D.append(new_points_3D)  # 3D上的采样点
        resampled_points_2D.append(new_points_2D)  # 2D上的采样点
        edge_approx.append(np.array([[new_corner_index[i], new_corner_index[(i+1)%len(new_corner_index)]] for i in range(len(new_corner_index))]))

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            panel_nes.append(len(new_corner_index))
        else:
            panel_nes[-1] += len(new_corner_index)

    for e_i, e_ap in enumerate(edge_approx):
        edge_approx[e_i] = edge_approx[e_i][np.argsort(edge_approx[e_i][:, 0])]

    # Check whether invalid results occur.
    test = np.concatenate([np.array(e) for e in edge_approx])
    if np.sum(test[:,1]==test[:,0]):
        raise ValueError("Wrong edge approx result...")

    contour_nes = np.array(contour_nes)

    contour_nps = np.array([len(ri) for ri in resampled_points_2D])

    """
    resampled_points_3D:    提取出的3D边缘点 (Contour-wise)
    resampled_points_2D：   提取出的2D边缘点 (Contour-wise)
    contour_nes:            每个 Contour 上有几个 Edge
    panel_nes:              每个 Panel 上有几个 Edge
    edge_approx:            每个 contour 的拟合边
    panel_instance_seg:     每个 contour 分别属于哪个 Panel 
    contour_nps:            每个 contour 有几个采样点
    """
    return resampled_points_3D, resampled_points_2D, contour_nes, np.array(panel_nes), edge_approx, np.array(panel_instance_seg), contour_nps


def approx_curve(resampled_points_2D, edge_approx, contour_nps):
    """
    对每一对相邻的角点（端点）之间，采样一定数量的控制点，用于曲线拟合

    :param resampled_points_2D:     resample 后的 2D 边缘点
    :param edge_approx:             拟合边
    :return:
    """

    garment_approx_curve = []

    for contour_idx, contour_edge_approx in enumerate(edge_approx):

        panel_approx_curve = []

        contour_point_2d = resampled_points_2D[contour_idx]

        for e_idx, e_approx in enumerate(contour_edge_approx):

            # 获取一个拟合边上按顺时针排序的所有的点
            if e_approx[0] > e_approx[1]:
                edge_point_indices = np.vstack((contour_point_2d[e_approx[0]:], contour_point_2d[:e_approx[1] + 1]))
            else:
                edge_point_indices = contour_point_2d[e_approx[0]:e_approx[1] + 1]


            side_gap = 5
            middle_point_indices = edge_point_indices[side_gap:-side_gap]  # 不使用离端点太近的点作为控制点
            if len(middle_point_indices)>0:
                # 限制采样点数量范围
                adjust_range = [1, 36]  # 控制点数量根据实际采样点数量动态调整，但是设定范围
                min_sample_num = 1
                max_sample_num = 6  # Maximum sample size when N ≥ 36
                assert 0 < min_sample_num <= max_sample_num

                # 计算控制点数量 sample_size
                # sample_size = 1 when N = 1; sample_size = 6 when N = 36
                N = len(middle_point_indices)
                sample_size = int(min_sample_num + (max_sample_num - min_sample_num) * (N - adjust_range[0]) / (adjust_range[1] - adjust_range[0]))
                sample_size = np.clip(sample_size, min_sample_num, min(max_sample_num, N - adjust_range[0]))  # Avoid sample_size ≥ N

                # 均匀采样控制点
                sample_indices = np.linspace(0, N - 1, num=sample_size, dtype=int)
                middle_sample = middle_point_indices[sample_indices]
                edge_point_indices = np.vstack((edge_point_indices[0], middle_sample, edge_point_indices[-1]))
            else:
                edge_point_indices = np.vstack((edge_point_indices[0], edge_point_indices[-1]))
            panel_approx_curve.append(edge_point_indices)

        garment_approx_curve.append(panel_approx_curve)

    return garment_approx_curve


def panel_Layout(garment_approx_curve, uv_bbox, panel_instance_seg, show_layout_panels=False):
    """
    将每个Panel放置到对应的bbox中

    :param garment_approx_curve:    端点以及拟合曲线
    :param uv_bbox:                 每个Panel的UV的BBOX
    :param panel_instance_seg:      每个Contour对应的Panel
    :param show_layout_panels:
    :return:
    """

    garment_approx_curve = [[u.astype(np.float64) for u in k] for k in garment_approx_curve]

    # 将每个panel的点放入uv_bbox中对应的bbox里 ------------------------------------------------------------------------------
    bbox_approx_panel_prev = None
    for contour_idx in range(len(garment_approx_curve)):
        # 获得曲线 p_approx_curve 和其对应的 Panel 的UV的bbox（Panel和Contour存在一对多关系）
        p_idx = panel_instance_seg[contour_idx]
        approx_curve = garment_approx_curve[contour_idx]
        bbox = uv_bbox[p_idx]

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            all_coords = np.concatenate(approx_curve, axis=-2)
            bbox_approx_panel = np.array([np.min(all_coords[:, 0]), np.min(all_coords[:, 1]),
                                          np.max(all_coords[:, 0]), np.max(all_coords[:, 1])])
            bbox_approx_panel_prev = bbox_approx_panel
        else:
            bbox_approx_panel = bbox_approx_panel_prev

        uv_bbox_scale = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        uv_bbox_offset = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

        for curve in approx_curve:
            curve[:, 0] -= (bbox_approx_panel[0] + bbox_approx_panel[2]) / 2
            curve[:, 1] -= (bbox_approx_panel[1] + bbox_approx_panel[3]) / 2
            curve[:, 0] /= bbox_approx_panel[2] - bbox_approx_panel[0]
            curve[:, 1] /= bbox_approx_panel[3] - bbox_approx_panel[1]

            curve[:, 0] *= uv_bbox_scale[0]
            curve[:, 1] *= uv_bbox_scale[1]
            curve[:, 0] += uv_bbox_offset[0]
            curve[:, 1] += uv_bbox_offset[1]

    # 将整个garment 放到一个更大的box里 -------------------------------------------------------------------------------------
    all = np.concatenate([u for u in [np.concatenate(k, axis=-2) for k in garment_approx_curve]], axis=-2)
    uv_bbox_scale = 1500
    uv_bbox_offset = [0, 1500]
    for contour_idx in range(len(garment_approx_curve)):
        for curve in garment_approx_curve[contour_idx]:
            curve *= uv_bbox_scale
            curve += uv_bbox_offset

    return garment_approx_curve



def get_garment_json(garment_approx_curve, panel_instance_seg, uv_bbox):
    """
    将曲线拟合转换成不包含缝合的AIGP-JSON文件

    :param garment_approx_curve:    端点+拟合边
    :param panel_instance_seg:      每个 contour 属于哪一个 Panel
    :param uv_bbox:
    :return:
    """

    garment_json = {"panels": [], "stitches": []}

    for contour_idx in range(len(garment_approx_curve)):
        panel_instance_idx = panel_instance_seg[contour_idx]

        # 如果这个是 OutLine
        if is_contour_OutLine(contour_idx, panel_instance_seg):
            p_approx_curve = garment_approx_curve[contour_idx]
            bbox = uv_bbox[panel_instance_idx]
            panel_json = {
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "seqEdges": [],
                "id": get_random_uuid(),
                "label": f"{panel_instance_idx}".zfill(2),  # [todo] 填入板片的编号
                # "translation": [0.0, 0.0, 0.0],
                # "rotation": [0.0, 0.0, 0.0],
            }
            garment_json["panels"].append(panel_json)
            seqEdge_json = {
                "type": 3,  # 基本线
                "circleType": 0,
                "edges": [],
                "vertices": [],
            }
            for e_idx in range(len(p_approx_curve)):
                edge_json = {
                    "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                    "controlPoints": [[ct_p[0] - panel_json["center"][0], ct_p[1] - panel_json["center"][1], 0] for ct_p in
                                      p_approx_curve[e_idx]],
                    "id": get_random_uuid(),
                }
                seqEdge_json["edges"].append(edge_json)
                seqEdge_json["vertices"].append(edge_json["controlPoints"][0])

            panel_json["seqEdges"].append(seqEdge_json)
        else:
            panel_json = garment_json["panels"][panel_instance_idx]
            p_approx_curve = garment_approx_curve[contour_idx]
            seqEdge_json = {
                "type": 4,  # 内部线
                "circleType": 1,
                "edges": [],
                "vertices": [],
            }
            for e_idx in range(len(p_approx_curve)):
                edge_json = {
                    "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                    "controlPoints": [[ct_p[0] - panel_json["center"][0], ct_p[1] - panel_json["center"][1], 0] for ct_p in
                                      p_approx_curve[e_idx]],
                    "id": get_random_uuid(),
                }
                seqEdge_json["edges"].append(edge_json)
                seqEdge_json["vertices"].append(edge_json["controlPoints"][0])

            panel_json["seqEdges"].append(seqEdge_json)
            # raise NotImplementedError("调试下上面的代码，然后删掉这个")
    return garment_json


def get_full_uv_info(resampled_points_2D, panel_instance_seg, uv_bbox, contour_nps, show_full_uv=False):
    """
    获取 resampled 的 2D 点的UV坐标
    :param resampled_points_2D:     2D (UV) resampled points
    :param panel_instance_seg:      Determine the panel to which each contour belongs.
    :param uv_bbox:                 2D (UV) bbox of each panel
    :param show_full_uv:            调试用
    :return:
    """

    uv_local = np.concatenate(resampled_points_2D)
    nps_cumsum = np.cumsum(contour_nps)

    # 将每个panel的点放入uv_bbox中对应的bbox里 ------------------------------------------------------------------------------
    uv_global = np.zeros_like(uv_local, dtype=np.float32)
    bbox_uv_local_prev = None
    for contour_idx in range(len(nps_cumsum)):
        instance_idx = panel_instance_seg[contour_idx]
        if contour_idx == 0: start_point_idx = 0
        else: start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        bbox = uv_bbox[instance_idx]

        if is_contour_OutLine(contour_idx, panel_instance_seg):
            bbox_uv_local = np.array([np.min(all_coords[:, 0]), np.min(all_coords[:, 1]),
                                      np.max(all_coords[:, 0]), np.max(all_coords[:, 1])])
            bbox_uv_local_prev = bbox_uv_local
        else:
            bbox_uv_local = bbox_uv_local_prev  # 如果不是 OutLine，则采用对应 OutLine 的 BBOX 进行调整

        uv_bbox_scale = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        uv_bbox_offset = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]

        all_coords[:, 0] -= (bbox_uv_local[0] + bbox_uv_local[2]) / 2
        all_coords[:, 1] -= (bbox_uv_local[1] + bbox_uv_local[3]) / 2
        all_coords[:, 0] /= bbox_uv_local[2] - bbox_uv_local[0]
        all_coords[:, 1] /= bbox_uv_local[3] - bbox_uv_local[1]

        all_coords[:, 0] *= uv_bbox_scale[0]
        all_coords[:, 1] *= uv_bbox_scale[1]
        all_coords[:, 0] += uv_bbox_offset[0]
        all_coords[:, 1] += uv_bbox_offset[1]

        uv_global[start_point_idx:end_point_idx] = all_coords

    # 将整个 garment 放到一个更大的box里 -------------------------------------------------------------------------------------
    uv_bbox_scale = 1000
    uv_bbox_offset = [0, 1000]
    for contour_idx in range(len(nps_cumsum)):
        if contour_idx == 0:
            start_point_idx = 0
        else:
            start_point_idx = nps_cumsum[contour_idx - 1]
        end_point_idx = nps_cumsum[contour_idx]

        all_coords = uv_local[start_point_idx:end_point_idx]
        all_coords *= uv_bbox_scale
        all_coords += uv_bbox_offset

        uv_global[start_point_idx:end_point_idx] = all_coords

    if show_full_uv:
        plt.scatter(x=uv_global[:, 0], y=uv_global[:, 1], s=2)
        plt.show()

    return uv_global



def save_resaults(output_dir, g_idx,
                  resampled_points_3D, edge_approx, contour_nes, panel_nes, panel_instance_seg,
                  garment_json, full_uv_info, cfg, fp, g_basename=None):
    """
    保存结果

    :param output_dir:              这一批输出的根目录
    :param g_idx:                   garment index
    :param resampled_points_3D:     3D边缘点
    :param edge_approx:             拟合边
    :param contour_nes:             每个contour上有几个点
    :param panel_nes:               每个panel上有几个点
    :param panel_instance_seg:      每个contour属于哪一个Panel
    :param garment_json:            服装的AIGP文件（仅包含Panels）
    :param full_uv_info:            每个3D边缘点对应的 2D-UV位置
    :param cfg:                     进行resample时采用的配置
    :return:
    """
    if g_basename is None:
        garment_name = "garment_" + f"{g_idx}".zfill(5)
    else:
        garment_name = "garment_" + g_basename
    garment_dir = os.path.join(output_dir,garment_name)
    os.makedirs(garment_dir, exist_ok=True)

    # save data_info ===
    datainfo_json_save_path = os.path.join(output_dir, "data_info.json")
    with open(datainfo_json_save_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)

    # save piece_XX.obj files ===
    piece_list = [trimesh.Trimesh(vertices=b_pts, process=False) for b_pts in resampled_points_3D]
    for p_idx, piece in enumerate(piece_list):
        piece_save_path = os.path.join(garment_dir, "piece_" + f"{p_idx}".zfill(2) + ".obj")
        piece.export(piece_save_path)

    # save garment.json file ===
    annotation_dir = os.path.join(garment_dir, "annotations")
    os.makedirs(annotation_dir, exist_ok=True)
    garment_json_save_path = os.path.join(annotation_dir, garment_name + ".json")
    with open(garment_json_save_path, 'w', encoding='utf-8') as f:
        json.dump(garment_json, f, indent=4)

    # save annotations.json file ===
    annotations_json_save_path = os.path.join(annotation_dir, "annotations.json")
    annotations_json = {
        "panel_nes":panel_nes.tolist(),
        "contour_nes": contour_nes.tolist(),
        "edge_approx": np.concatenate(edge_approx).tolist(),
        "panel_instance_seg": panel_instance_seg.tolist(),
        "data_path": fp["garment_fp"]
    }
    with open(annotations_json_save_path, 'w', encoding='utf-8') as f:
        json.dump(annotations_json, f, indent=4, ensure_ascii=False)

    # save uv info in the full image ===
    if full_uv_info is not None:
        full_uv_info_save_path = os.path.join(annotation_dir, "uv.npy")
        np.save(full_uv_info_save_path, full_uv_info)

    # save original data ===
    # 这些数据仅用于 Jigsaw项目中的 composite_visualize 方法中，用于可视化
    original_data_dir = os.path.join(garment_dir, "original_data")
    os.makedirs(original_data_dir, exist_ok=True)
    # shutil.copy(fp["garment_fp"], original_data_dir)  # 不能直接复制，jigsaw 的环境无法读取
    with open(fp["garment_fp"], 'rb') as f:
        _data = pickle.load(f)
    with open(os.path.join(original_data_dir, os.path.basename(fp["garment_fp"])), "wb") as f:
        pickle.dump(_data, f)



if __name__ == "__main__":
    data_type = "Garmage256"

    data_dir = "/data/lsr/resources/Anta/251111_训练集_对比原图和增强/一些生成结果导入软件/generated"

    # 根据数据类型, 可以 采用不同的配置、读取不同的路径
    """
    cfg 中的几个key:
        RESO：GeoImg的分辨率
        delta：resample的间距
        gr：进行边拟合时的粒度粗细
    """
    if data_type=="Garmage256":  # 生成的 256x256 的 Garmage
        """
        feature convK55D11:
            output_dir = data_dir + "_output"
            data_path_list = sorted(glob(os.path.join(data_dir, "*.pkl")))
            garment_num = len(data_path_list)
            cfg = {"RESO":256, "delta": 0.008, "erode_size":3}
            + 4 INF NOISE
        """
        output_dir = data_dir + "_output"
        data_path_list = sorted(glob(os.path.join(data_dir, "*.pkl")))
        garment_num = len(data_path_list)
        cfg = {"RESO":256, "delta": 0.012, "erode_size":3}
    else:
        raise NotImplementedError
    os.makedirs(output_dir, exist_ok=True)

    # 对每个数据进行处理
    for g_idx in tqdm(range(0, garment_num), desc=f"Processing {data_type} data: "):
        if data_type == "Garmage256":
            garment_fp = data_path_list[g_idx]
            g_basename = os.path.basename(garment_fp).replace(".pkl","")
            fp = {"garment_fp": garment_fp}

            try:   g_idx = int(garment_fp.split("/")[-1].split(".")[-2])
            except Exception:   g_idx = g_idx

        else:
            raise NotImplementedError

        # read data
        geo_orig, mask, uv_bbox = load_data(data_type, fp, save_vis=True)

        # 提取边缘点，并获得边拟合的初始状态
        resampled_points_3D, resampled_points_2D, contour_nes, panel_nes, edge_approx, panel_instance_seg, contour_nps = (
            extract_boundary_pts(geo_orig, uv_bbox, mask, delta=cfg["delta"], RESO=cfg["RESO"], erode_size=cfg.get("erode_size", 3), show_2d_approx=False))

        # 根据 uv_bbox 将 resampled_points_2D denormalize 到 UV坐标
        full_uv_info = get_full_uv_info(resampled_points_2D, panel_instance_seg, uv_bbox, contour_nps, show_full_uv=False)

        # 获取用于拟合曲线的采样点
        approx_curve_samples = approx_curve(resampled_points_2D, edge_approx, contour_nps)

        # # 根据 uv_bbox 将每个Panel的矢量图 denormalize 到 UV坐标
        garment_approx_curve = panel_Layout(approx_curve_samples, uv_bbox, panel_instance_seg)

        # 获取 AIGP 的json文件（仅包含panels部分）
        garment_json = get_garment_json(garment_approx_curve, panel_instance_seg, uv_bbox)

        # 保存结果
        save_resaults(output_dir, g_idx,
                      resampled_points_3D, edge_approx,
                      contour_nes, panel_nes, panel_instance_seg,
                      garment_json, full_uv_info, cfg, fp,
                      g_basename=g_basename)
