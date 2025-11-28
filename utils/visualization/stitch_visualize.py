
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go


def stitch_visualize(vertices:np.array, stitches:np.array):
    fig = go.Figure()

    all_coords = vertices

    min_val = np.min(all_coords)
    max_val = np.max(all_coords)
    colors = cm.get_cmap('tab20', 1000)
    color_norm = mcolors.Normalize(vmin=0, vmax=1000)

    for i, pair in enumerate(stitches):
        color = mcolors.to_hex(colors(color_norm(i)))
        surf_pnts = all_coords[np.array(pair)]

        x = surf_pnts[:, 0]
        y = surf_pnts[:, 1]
        z = surf_pnts[:, 2]

        # add point
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

        x_line = [x[0], x[1]]
        y_line = [y[0], y[1]]
        z_line = [z[0], z[1]]

        # add line
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='lines',
            line=dict(
                color=color,
                width=1
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

    fig.show()