# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torch
import open3d
from pytorch3d.structures import Meshes, Pointclouds


def normalize_mesh(in_mesh: Meshes):
    '''
        Detect the scale of the mesh, centralize and normalize the mesh, 
        record the inverse transformation of mesh

        input: Meshes object
        return: Meshes object, inverse transformation param
    '''
    assert in_mesh.verts_padded().shape[0] == 1
    aabb = in_mesh.get_bounding_boxes()
    distance = aabb[:, :, 1] - aabb[:, :, 0]
    max_distance, _ = torch.max(distance, dim = 1)
    mesh_verts = in_mesh.verts_padded()
    center = torch.mean(mesh_verts, dim = 1, keepdim = True).repeat(1, mesh_verts.shape[1], 1)
    offset = -1 * center.reshape(-1, 3)
    scale = 1 / max_distance
    out_mesh = in_mesh.offset_verts(offset)
    out_mesh.scale_verts_(scale)
    return out_mesh, (offset, scale)

def normalize_pcl(in_pcl: Pointclouds):
    assert in_pcl.points_padded().shape[0] == 1
    aabb = in_pcl.get_bounding_boxes()
    distance = aabb[:, :, 1] - aabb[:, :, 0]
    max_distance, _ = torch.max(distance, dim = 1)
    pcl_points = in_pcl.points_padded()
    center = torch.mean(pcl_points, dim = 1, keepdim = True).repeat(1, pcl_points.shape[1], 1)
    offset = -1 * center.reshape(-1, 3)
    scale = 1 / max_distance
    out_pcl = in_pcl.offset(offset)
    out_pcl.scale_(scale)
    return out_pcl, (offset, scale)

def pointcloud_normal(in_pcl: Pointclouds):
    '''
        pytorch3d normal estimation is so slow
        so we use the open3d normal estimation
    '''
    points = in_pcl.points_padded()[0]
    points_numpy = points.squeeze().cpu().data.numpy()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points_numpy)
    pcd.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return torch.from_numpy(np.asarray(pcd.normals)).to(points.device)

def mesh_boundary(in_faces: torch.LongTensor, num_verts: int):
    '''
    input:
        in edges: N * 3, is the vertex index of each face, where N is number of faces
        num_verts: the number of vertexs mesh
    return:
        boundary_mask: bool tensor of num_verts, if true, point is on the boundary, else not
    '''
    in_x = in_faces[:, 0]
    in_y = in_faces[:, 1]
    in_z = in_faces[:, 2]
    in_xy = in_x * (num_verts) + in_y
    in_yx = in_y * (num_verts) + in_x
    in_xz = in_x * (num_verts) + in_z
    in_zx = in_z * (num_verts) + in_x
    in_yz = in_y * (num_verts) + in_z
    in_zy = in_z * (num_verts) + in_y
    in_xy_hash = torch.minimum(in_xy, in_yx)
    in_xz_hash = torch.minimum(in_xz, in_zx)
    in_yz_hash = torch.minimum(in_yz, in_zy)
    in_hash = torch.cat((in_xy_hash, in_xz_hash, in_yz_hash), dim = 0)
    output, count = torch.unique(in_hash, return_counts = True, dim = 0)
    boundary_edge = output[count == 1]
    boundary_vert1 = boundary_edge // num_verts
    boundary_vert2 = boundary_edge % num_verts
    boundary_mask = torch.zeros(num_verts).bool().to(in_faces.device)
    boundary_mask[boundary_vert1] = True
    boundary_mask[boundary_vert2] = True
    return boundary_mask

def convert_mesh_to_pcl(in_mesh: Meshes):
    '''
        Convert Meshes object to Pointclouds object(only converting vertexes)
    '''
    points = in_mesh.verts_padded()
    return Pointclouds(points)

def visualize_points(fp, in_points: torch.Tensor):
    '''
        inpoints has shape of 'N*3',
        where N is the points number
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    in_points_np = in_points.cpu().detach().numpy()
    for point in in_points_np:
        x = point[0]
        y = point[1]
        z = point[2]
        ax.scatter(x, y, z)
    ax.view_init(elev=90., azim=0)
    plt.savefig(fp)
    plt.close()

def batch_vertex_sample(batch_idx: torch.LongTensor, vertex: torch.Tensor):
    '''
    input:
        batch_idx: shape of (B * L), B is the batch size, L is the select point length
        vertex: shape of (B * N * 3), N is the vertex size
    output:
        vertex: (B * L * 3)
    '''
    batch_idx_expand = batch_idx.unsqueeze(2).expand(batch_idx.shape[0], batch_idx.shape[1], vertex.shape[2])
    sampled_vertex = torch.gather(vertex, 1, batch_idx_expand)
    return sampled_vertex