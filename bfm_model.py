# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import numpy as np
from scipy.io import loadmat
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from utils import normalize_mesh

def load_bfm_model(device = torch.device('cpu')):
    bfm_meta_data = loadmat('BFM/BFM09_model_info.mat')
    vertex = bfm_meta_data['meanshape'].reshape(-1, 3)
    faces = bfm_meta_data['tri']
    lm_index = bfm_meta_data['keypoints']
    color = bfm_meta_data['meantex'].reshape(-1, 3)
    color = torch.from_numpy(color).to(device).unsqueeze(0)
    vertex = torch.from_numpy(vertex).to(device)
    faces = torch.from_numpy(faces).long().to(device) - 1
    lm_index = torch.from_numpy(lm_index).long().to(device)
    textures = TexturesVertex(color)
    bfm_mesh = Meshes([vertex], [faces], textures)
    norm_mesh, _ = normalize_mesh(bfm_mesh)
    return norm_mesh, lm_index

if __name__ == "__main__":
    import render
    import torchvision
    bfm_mesh, lm_index = load_bfm_model(torch.device('cuda:0'))
    dummy_render = render.create_dummy_render([1, 0, 0], device = torch.device('cuda:0'))
    images = dummy_render(bfm_mesh).squeeze()
    torchvision.utils.save_image(images.permute(2, 0, 1) / 255, 'test_data/test_bfm.png')