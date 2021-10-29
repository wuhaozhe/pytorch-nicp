# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import io3d
import render
import numpy as np
import json
from utils import normalize_mesh, normalize_pcl
from landmark import get_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2pcl, non_rigid_icp_mesh2mesh
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes, Pointclouds

device = torch.device('cuda:0')
# get the first point cloud
verts, _ = load_ply('./test_data/test2.ply')
verts = verts.to(device)
# get the second point cloud
perm = torch.randperm(verts.size(0))
idx = perm[:50000]
verts2 = verts[idx]
pcls = Pointclouds(points = [verts, verts2])
norm_pcls, norm_param = normalize_pcl(pcls)
# pcl_lm_file = open('./test_data/test2_lm.txt')
# lm_list = []
# for line in pcl_lm_file:
#     line = int(line.strip())
#     lm_list.append(line)

# target_lm_index = torch.from_numpy(np.array(lm_list)).to(device)
# lm_mask = (target_lm_index >= 0)
# target_lm_index = target_lm_index.unsqueeze(0)
# bfm_meshes, bfm_lm_index = load_bfm_model(torch.device('cuda:0'))
# bfm_lm_index_m = bfm_lm_index[:, lm_mask]
# target_lm_index_m = target_lm_index[:, lm_mask]
# coarse_config = json.load(open('config/coarse_grain.json'))
# registered_mesh = non_rigid_icp_mesh2pcl(bfm_meshes, norm_pcls, bfm_lm_index_m, target_lm_index_m, coarse_config)
# io3d.save_meshes_as_objs(['test_data/final2.obj'], registered_mesh, save_textures = False)