import numpy as np


def cal_mean_edge_len(meshes:list):
    """
    calculate mean edge length of a list of mesh.
    :param meshes:
    :return:
    """
    v_el, v_er = [], []
    try:
        for mesh in meshes:
            edges = np.array(mesh.edges)
            # Set e_sample, the sample frequency for a mesh
            if len(edges) < 400:
                e_sample = 2
            elif len(edges) < 1000:
                e_sample = 4
            elif len(edges) < 10000:
                e_sample = 6
            elif len(edges) < 50000:
                e_sample = 20
            else:
                e_sample = 50
            v_el.append(np.array(mesh.vertices)[np.concatenate(edges, axis=0)[0::e_sample]])
            v_er.append(np.array(mesh.vertices)[np.concatenate(edges, axis=0)[1::e_sample]])
        v_el = np.concatenate(v_el, axis=0)
        v_er = np.concatenate(v_er, axis=0)
        mean_edge_len = np.mean(np.sqrt(np.sum((v_el - v_er) ** 2, axis=1)))
    except:
        # get by statistics
        mean_edge_len = 5.937708012501726

    return  mean_edge_len

