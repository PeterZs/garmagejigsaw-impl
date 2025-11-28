import numpy as np
from trimesh import Trimesh


def split_mesh_into_parts(obj_dict):
    """
    Split garmageset format obj into part objs
    """
    nps = np.array(obj_dict["nps"])
    nfs = np.array(obj_dict["nfs"])
    vertices = np.array(obj_dict["vertices"])
    faces = np.array(obj_dict["faces"])

    meshes = []
    num_parts = len(nps)
    end_point_idx = np.cumsum(nps)
    end_face_idx = np.cumsum(nfs)

    for idx in range(num_parts):
        if idx == 0:
            point_start = 0
            face_start = 0
        else:
            point_start = end_point_idx[idx - 1]
            face_start = end_face_idx[idx - 1]
        point_end = end_point_idx[idx]
        face_end = end_face_idx[idx]

        Sub_vertices_idx = list(range(point_start, point_end))
        Sub_faces_idx = list(range(face_start, face_end))
        Sub_vertices, Sub_faces = get_sub_mesh(vertices, faces, Sub_vertices_idx, Sub_faces_idx)

        mesh = Trimesh(vertices=Sub_vertices, faces=Sub_faces, process=False)
        meshes.append(mesh)

    return meshes


def get_sub_mesh(Vertices:np.array, Faces:np.array, Sub_vertices_idx:np.array, Sub_faces_idx:np.array):
    Sub_faces = Faces[Sub_faces_idx]

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(Sub_vertices_idx)}

    Sub_vertices = Vertices[Sub_vertices_idx]

    Sub_faces = np.array([[index_map[vert] for vert in face] for face in Sub_faces])

    return Sub_vertices, Sub_faces
