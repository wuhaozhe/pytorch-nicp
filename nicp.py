# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import numpy as np
from pytorch3d.ops.laplacian_matrices import laplacian
import render
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import io3d
from pytorch3d.structures import Meshes, Pointclouds
from utils import batch_vertex_sample
from pytorch3d.ops import (
    corresponding_points_alignment,
    knn_points,
    knn_gather
)
from pytorch3d.loss import mesh_laplacian_smoothing
from local_affine import LocalAffine
from tqdm import tqdm
from utils import convert_mesh_to_pcl, pointcloud_normal, mesh_boundary

def non_rigid_icp_mesh2mesh(
    template_mesh: Meshes, 
    target_mesh: Meshes, 
    template_lm_index: torch.LongTensor,
    target_lm_index: torch.LongTensor,
    config: dict,
    device = torch.device('cuda:0')
):
    target_pcl = convert_mesh_to_pcl(target_mesh)
    pcl_normal = target_mesh.verts_normals_padded()
    return non_rigid_icp_mesh2pcl(template_mesh, target_pcl, template_lm_index, target_lm_index, config, pcl_normal, device)

def non_rigid_icp_mesh2pcl(
    template_mesh: Meshes, 
    target_pcl: Pointclouds, 
    template_lm_index: torch.LongTensor,
    target_lm_index: torch.LongTensor,
    config: dict,
    pcl_normal: torch.FloatTensor = None,
    device = torch.device('cuda:0'),
    out_affine = False,
    in_affine = None
):
    '''
        deform template mesh to target pointclouds

        The template mesh and target pcl should be normalized with utils.normalize_mesh api. 
        The mesh should look at +z axis, the x define the width of mesh, and the y define the height of mesh
    '''
    
    template_mesh = template_mesh.to(device)
    target_pcl = target_pcl.to(device)
    template_lm_index = template_lm_index.to(device)
    target_lm_index = target_lm_index.to(device)

    template_vertex = template_mesh.verts_padded()
    target_vertex = target_pcl.points_padded()

    #TODO: currently, batch NICP is not supported
    assert target_vertex.shape[0] == 1

    boundary_mask = mesh_boundary(template_mesh.faces_padded()[0], template_vertex.shape[1])
    boundary_mask = boundary_mask.unsqueeze(0).unsqueeze(2)
    inner_mask = torch.logical_not(boundary_mask)

    # masking abnormal points according to the normal seems to be useless, we use distance mask in our framework
    # if pcl_normal is None:
    #     # estimate normal for point cloud
    #     with torch.no_grad():
    #         pcl_normal = pointcloud_normal(target_pcl).unsqueeze(0).repeat(target_vertex.shape[0], 1, 1)

    # rigid align
    target_lm = batch_vertex_sample(target_lm_index, target_vertex)
    template_lm = batch_vertex_sample(template_lm_index, template_vertex)
    R, T, s = corresponding_points_alignment(template_lm, target_lm, estimate_scale = True)
    transformed_vertex = s[:, None, None] * torch.bmm(template_vertex, R) + T[:, None, :]

    # define the transformation model
    template_edges = template_mesh.edges_packed()
    if in_affine is None:
        local_affine_model = LocalAffine(template_vertex.shape[1], template_vertex.shape[0], template_edges).to(device)
    else:
        local_affine_model = in_affine
    optimizer = torch.optim.AdamW([{'params': local_affine_model.parameters()}], lr=1e-4, amsgrad=True)

    # train param config
    inner_iter = config['inner_iter']
    outer_iter = config['outer_iter']
    loop = tqdm(range(outer_iter))
    log_iter = config['log_iter']

    milestones = set(config['milestones'])
    stiffness_weights = np.array(config['stiffness_weights'])
    landmark_weights = np.array(config['landmark_weights'])
    laplacian_weight = config['laplacian_weight']
    w_idx = 0

    # original 3d model
    # dummy_render = render.create_dummy_render([1, 0, 0], device = device)
    # transformed_mesh = template_mesh.update_padded(transformed_vertex)
    # images = dummy_render(transformed_mesh).squeeze()
    # torchvision.utils.save_image(images.permute(2, 0, 1) / 255, 'test_data/nicp.png')

    for i in loop:
        new_deformed_verts, stiffness = local_affine_model(transformed_vertex, pool_num = 0, return_stiff = True)
        new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
        old_verts = new_deformed_verts
        new_deform_mesh = template_mesh.update_padded(new_deformed_verts)

        # we can randomly sample the target point cloud for speed up
        target_sample_verts = target_vertex

        knn = knn_points(new_deformed_verts, target_sample_verts)
        close_points = knn_gather(target_sample_verts, knn.idx)[:, :, 0]
        # close_normals = knn_gather(pcl_normal, knn.idx)[:, :, 0]

        if (i == 0) and (in_affine is None):
            inner_loop = range(100)
        else:
            inner_loop = range(inner_iter)

        for _ in inner_loop:
            optimizer.zero_grad()

            # masking abnormal points according to the normal seems to be useless, we use distance mask in our framework
            # new_deformed_normal = new_deform_mesh.verts_normals_padded()
            # normal_cos_sim = torch.abs(F.cosine_similarity(close_normals, new_deformed_normal, dim = 2)).unsqueeze(2)
            # weight_mask = torch.logical_and(inner_mask, normal_cos_sim > 0.5)

            vert_distance = (new_deformed_verts - close_points) ** 2
            vert_distance_mask = torch.sum(vert_distance, dim = 2) < 0.04**2
            weight_mask = torch.logical_and(inner_mask, vert_distance_mask.unsqueeze(2))

            vert_distance = weight_mask * vert_distance
            landmark_distance = (new_deformed_lm - target_lm) ** 2

            bsize = vert_distance.shape[0]
            vert_distance = vert_distance.view(bsize, -1)
            vert_sum = torch.sum(vert_distance) / bsize
            landmark_distance = landmark_distance.view(bsize, -1)
            landmark_sum = torch.sum(landmark_distance) * landmark_weights[w_idx] / bsize
            stiffness = stiffness.view(bsize, -1)
            stiffness_sum = torch.sum(stiffness) * stiffness_weights[w_idx] / bsize
            laplacian_loss = mesh_laplacian_smoothing(new_deform_mesh) * laplacian_weight
            loss = torch.sqrt(vert_sum + landmark_sum + stiffness_sum) + laplacian_loss
            loss.backward()
            optimizer.step()
            new_deformed_verts, stiffness = local_affine_model(transformed_vertex, pool_num = 0, return_stiff = True)
            new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
            new_deform_mesh = template_mesh.update_padded(new_deformed_verts)

        distance = torch.mean(torch.sqrt(torch.sum((old_verts - new_deformed_verts) ** 2, dim = 2)))
        
        if i % log_iter == 0:
            print(distance, stiffness_sum.item(), landmark_sum.item(), vert_sum.item(), laplacian_loss.item())
            # new_deformed_verts, _ = local_affine_model(transformed_vertex, return_stiff = True)
            # new_deform_mesh = template_mesh.update_padded(new_deformed_verts)
            # images = dummy_render(new_deform_mesh).squeeze()
            # torchvision.utils.save_image(images.permute(2, 0, 1) / 255, 'test_data/nicp{}.png'.format(i))

        if i in milestones:
            w_idx += 1

    new_deform_mesh = template_mesh.update_padded(new_deformed_verts)
    if out_affine:
        return new_deform_mesh, local_affine_model
    else:
        return new_deform_mesh