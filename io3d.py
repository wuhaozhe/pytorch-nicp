# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

'''
Read and write mesh with textures
Read and write pointclouds
'''

import torch
from pytorch3d.io import (
    load_obj,
    load_ply,
    load_objs_as_meshes,
    save_obj,
    save_ply
)
from pytorch3d.structures import Meshes, Pointclouds

def load_obj_as_mesh(fp, device = None, load_textures = True):
    '''
        if load mesh with texture, only a single texture image with its mtl is permitted
        return mesh structure
    '''
    mesh = load_objs_as_meshes([fp], device, load_textures)
    return mesh

def load_ply_as_pointcloud(fp, device = None):
    '''
        no texture is permitted for pointcloud
    '''
    verts, _ = load_ply(fp)
    if not device is None:
        verts = verts.to(device)
    pointcloud = Pointclouds(points = [verts])
    return pointcloud

def save_pointclouds_as_plys(fp_list, pointcloud):
    '''
        no texture is permitted for pointcloud
        only save points
    '''
    for idx, fp in enumerate(fp_list):
        verts = pointcloud.points_padded()[idx]
        save_ply(fp, verts)

def save_meshes_as_objs(fp_list, mesh, save_textures = True):
    '''
        input Meshes object
        save obj
    '''
    for idx, fp in enumerate(fp_list):
        verts = mesh.verts_padded()[idx]
        faces = mesh.faces_padded()[idx]
        if save_textures:
            if mesh.textures.isempty():
                raise Exception('Save untextured mesh with param save_textures=True')
            texture_map = mesh.textures.maps_padded()[idx]
            verts_uvs = mesh.textures.verts_uvs_padded()[idx]
            faces_uvs = mesh.textures.faces_uvs_padded()[idx]
            save_obj(fp, verts, faces, verts_uvs = verts_uvs, faces_uvs = faces_uvs, texture_map = texture_map)
        else:
            save_obj(fp, verts, faces)

# testing code
if __name__ == "__main__":
    # test_mesh = load_obj_as_mesh('pjanic.obj')
    # save_meshes_as_objs(['test2.obj'], test_mesh)

    test_pl = load_ply_as_pointcloud('./test_data/test.ply')
    save_pointclouds_as_plys(['./test_data/test2.ply'], test_pl)