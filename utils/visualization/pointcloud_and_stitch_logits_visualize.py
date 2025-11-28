import os
import math
import os.path
from copy import deepcopy

import torch
import numpy as np
import matplotlib.cm as cm
import plotly.graph_objects as go
import matplotlib.colors as mcolors


def pointcloud_and_stitch_logits_visualize(vertices:np.array, stitches:np.array, logistic, colormap="tab20", colornum=32, color_norm = [0,10], title="", export_data_config=None):
    vertices = deepcopy(vertices)

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.clone()
        vertices = vertices.detach().cpu().numpy()
    if isinstance(vertices, list):
        if isinstance(vertices[0], torch.Tensor):
            for idx, _ in enumerate(vertices):
                vertices[idx] = vertices[idx].detach().cpu().numpy()
        elif isinstance(vertices[0], np.ndarray):
            pass
        else:
            vertices = np.array(vertices)

    if isinstance(vertices, np.ndarray) and vertices.ndim==2:
        vertices = vertices.reshape(1,-1,3)

    if isinstance(logistic,torch.Tensor):
        logistic = logistic.detach().cpu().numpy()
    if logistic.ndim==2:
        logistic = logistic.reshape(-1)


    if not export_data_config:
        point_size = 6
        line_width = 5
    else:
        point_size = 5
        line_width = 13

    fig = go.Figure()
    all_coords = np.concatenate(vertices, axis=0)
    min_val = np.min(all_coords)
    max_val = np.max(all_coords)
    colors = cm.get_cmap(colormap, colornum)
    color_norm = mcolors.Normalize(vmin=color_norm[0], vmax=color_norm[1])

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
                size=point_size,
                color=color,
                opacity=0.65
            ),
            showlegend=False
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

    colors_2 = cm.get_cmap('coolwarm_r', 10)
    color_norm_2 = mcolors.Normalize(vmin=0, vmax=1)
    for i, pair in enumerate(stitches):
        color = mcolors.to_hex(colors_2(color_norm_2(logistic[i])))
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
                width=line_width
            ),
            showlegend=False
        ))
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(
            x=(np.max(all_coords[:, 0]) + np.min(all_coords[:, 0])) / 2,
            y=0,
            z=(np.max(all_coords[:, 2]) + np.min(all_coords[:, 2])) / 2),
        eye=dict(x=0, y=0, z=1.5)
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X Axis',
                range=[min_val, max_val],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                title='Y Axis',
                range=[min_val, max_val],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            ),
            zaxis=dict(
                title='Z Axis',
                range=[min_val, max_val],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            ),
            aspectmode='cube',
            camera=camera
        ))

    # === exporet rotate video ===
    if export_data_config:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='X Axis',
                    range=[min_val, max_val],
                    showgrid = False,
                    showticklabels = False,
                    zeroline = False,
                    visible=False
                ),
                yaxis=dict(
                    title='Y Axis',
                    range=[min_val, max_val],
                    showgrid = False,
                    showticklabels = False,
                    zeroline = False,
                    visible=False
                ),
                zaxis=dict(
                    title='Z Axis',
                    range=[min_val, max_val],
                    showgrid = False,
                    showticklabels = False,
                    zeroline=False,
                    visible=False
                ),
                aspectmode='cube'
            ),
            title=""
        )
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(
                x=0,
                y=0,
                z=0,),
            eye=dict(x=0, y=0, z=1.5)
        )
        fig.update_layout(scene_camera=camera)

        pic_num = export_data_config["pic_num"]
        os.makedirs(export_data_config["export_path"], exist_ok=True)
        for i in range(export_data_config["pic_num"]+1):
            img_path = os.path.join(export_data_config["export_path"],f"{i}".zfill(3)+".png")

            math.sin(pic_num)
            eye_z = math.cos(2 * math.pi * i / pic_num) * 1.5
            eye_x = math.sin(2 * math.pi * i / pic_num) * 1.5
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(
                    x=0,
                    y=0,
                    z=0),
                eye=dict(x=eye_x, y=0, z=eye_z)
            )
            # Update layout
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    yaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    zaxis=dict(
                        visible=False,
                        showbackground=False,
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        title=''
                    ),
                    aspectmode='cube'  # Keep the aspect ratio of data
                ),
                width=800,
                height=800,
                margin=dict(r=0, l=0, b=0, t=0),
                showlegend=False,
                title=dict(text=title, automargin=True),
                scene_camera=camera,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
                paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
            )
            fig.write_image(img_path, width=1920, height=1920, scale=2)

    if export_data_config is None:
        fig.show("browser")