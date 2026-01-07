
from torch import nn


def build_pc_classifier(dim, norm="batch"):
    assert norm in ["batch", "instance"]

    def get_norm(norm_type, dim):
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif norm_type == "instance":
            return nn.InstanceNorm1d(dim, affine=True)
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")

    affinity_layer = nn.Sequential(
        get_norm(norm, dim),
        nn.ReLU(inplace=True),
        nn.Conv1d(dim, 1, 1),
    )
    return affinity_layer
