# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import face_alignment
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRenderer, TexturesVertex
from pytorch3d.ops import knn_points
# from utils import visualize_points

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
eps = 1e-5

def get_mesh_landmark(meshes: Meshes, dummy_renderer: MeshRenderer):
    '''
        The meshes should be textured
    '''
    global fa

    device = meshes.device
    mesh_verts = meshes.verts_padded()
    mesh_faces = meshes.faces_padded()
    textures = TexturesVertex(mesh_verts)
    shape_meshes = Meshes(mesh_verts, mesh_faces, textures)
    rgb_img = dummy_renderer(meshes)[:, :, :, 0, :]
    shape_img = dummy_renderer(shape_meshes)[:, :, :, 0, :]
    rgb_img_uint8 = (rgb_img * 255).permute(0, 3, 1, 2)
    landmarks = fa.get_landmarks_from_batch(rgb_img_uint8)
    landmarks = torch.from_numpy(np.array(landmarks)).to(device).long()

    row_index = landmarks[:, :, 1].view(landmarks.shape[0], -1)
    column_index = landmarks[:, :, 0].view(landmarks.shape[0], -1)
    row_index = row_index.unsqueeze(2).unsqueeze(3).expand(landmarks.shape[0], landmarks.shape[1], shape_img.shape[2], shape_img.shape[3])
    column_index = column_index.unsqueeze(1).unsqueeze(3).expand(landmarks.shape[0], landmarks.shape[1], landmarks.shape[1], shape_img.shape[3])

    lm_vertex = torch.gather(shape_img, 1, row_index)
    lm_vertex = torch.gather(lm_vertex, 2, column_index)
    lm_vertex = torch.diagonal(lm_vertex, dim1 = 1, dim2 = 2)
    lm_vertex = lm_vertex.transpose(1, 2)

    lm_norm = torch.norm(lm_vertex, p = 1, dim = 2)
    on_surface_mask = lm_norm > eps

    # measure whether the lip points locate on the surfaces
    # outer lip is supposed to locate on the surface
    # inner lip is possible to locate on the mouth interior, we shall remove these points during registration
    outer_lip = lm_vertex[:, 48: 61]
    inner_lip = lm_vertex[:, 61:]
    lip_threshold = outer_lip[:, 6, 0] - outer_lip[:, 0, 0]
    outer_lip_z = torch.mean(outer_lip[:, :, 2], dim = 1, keepdim = True)
    inner_lip_z = torch.abs(inner_lip[:, :, 2] - outer_lip_z)
    inner_lip_mask = inner_lip_z < lip_threshold.unsqueeze(1)
    
    on_surface_mask[:, 61:] = torch.logical_and(on_surface_mask[:, 61:], inner_lip_mask)
    lm_index = knn_points(lm_vertex, mesh_verts).idx[:, :, 0]
    
    # visualize_points('./test_data/test_point.png', lm_vertex.squeeze().transpose(0, 1))
    
    return lm_index, on_surface_mask