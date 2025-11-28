
from copy import deepcopy

import torch
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go


def pointcloud_and_stitch_visualize(vertices:np.array, stitches:np.array, title=""):
    vertices = deepcopy(vertices)

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.clone()
        vertices = vertices.detach().cpu().numpy()

    if isinstance(vertices, list):
        vertices = np.array(vertices)
    if vertices.ndim==2:
        vertices = vertices.reshape(1,-1,3)

    fig = go.Figure()

    all_coords = np.concatenate(vertices, axis=0)

    min_val = np.min(all_coords)
    max_val = np.max(all_coords)
    colors = cm.get_cmap('tab20', 32)
    color_norm = mcolors.Normalize(vmin=0, vmax=10)

    for i, vertex in enumerate(vertices):
        color = mcolors.to_hex(colors(color_norm(i)))

        surf_pnts = vertex

        x = surf_pnts[:, 0]
        y = surf_pnts[:, 1]
        z = surf_pnts[:, 2]

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                opacity=0.8
            )
        ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X Axis',
                range=[min_val, max_val]
            ),
            yaxis=dict(
                title='Y Axis',
                range=[min_val, max_val]
            ),
            zaxis=dict(
                title='Z Axis',
                range=[min_val, max_val]
            ),
            aspectmode='cube'
        ),
        title='3D Global Coordinates of Faces'
    )

    colors_2 = cm.get_cmap('tab20', 50)
    color_norm_2 = mcolors.Normalize(vmin=0, vmax=50)
    for i, pair in enumerate(stitches):
        color = mcolors.to_hex(colors_2(color_norm_2(i)))

        surf_pnts = all_coords[np.array(pair)]

        x = surf_pnts[:, 0]
        y = surf_pnts[:, 1]
        z = surf_pnts[:, 2]

        x_line = [x[0], x[1]]
        y_line = [y[0], y[1]]
        z_line = [z[0], z[1]]

        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='lines',
            line=dict(
                color=color,
                width=4
            )
        ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X Axis',
                range=[min_val, max_val]
            ),
            yaxis=dict(
                title='Y Axis',
                range=[min_val, max_val]
            ),
            zaxis=dict(
                title='Z Axis',
                range=[min_val, max_val]
            ),
            aspectmode='cube'
        ),
        title=title
    )

    fig.show("browser")