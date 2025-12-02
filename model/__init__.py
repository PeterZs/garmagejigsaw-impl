import os
from .modules import *

def build_model(cfg):
    from .garmage_jigsaw import GarmageJigsawModel
    return GarmageJigsawModel(cfg)