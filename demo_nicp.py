import torch
import io3d
import render
import numpy as np
from utils import normalize_mesh, normalize_pcl
from landmark import get_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2pcl, non_rigid_icp_mesh2mesh

# demo for registering mesh
# estimate landmark for target meshes
# the face must face toward z axis
# the mesh or point cloud must be normalized with normalize_mesh/normalize_pcl function before feed into the nicp process
device = torch.device('cuda:0')
meshes = io3d.load_obj_as_mesh('./test_data/pjanic.obj', device = device)

with torch.no_grad():
    norm_meshes, norm_param = normalize_mesh(meshes)
    dummy_render = render.create_dummy_render([1, 0, 0], device = device)
    target_lm_index, lm_mask = get_mesh_landmark(norm_meshes, dummy_render)
    bfm_meshes, bfm_lm_index = load_bfm_model(torch.device('cuda:0'))
    lm_mask = torch.all(lm_mask, dim = 0)
    bfm_lm_index_m = bfm_lm_index[:, lm_mask]
    target_lm_index_m = target_lm_index[:, lm_mask]
    
registered_mesh = non_rigid_icp_mesh2mesh(bfm_meshes, norm_meshes, bfm_lm_index_m, target_lm_index_m)
io3d.save_meshes_as_objs(['final.obj'], registered_mesh, save_textures = False)


# demo for registering point cloud
device = torch.device('cuda:0')
pcls = io3d.load_ply_as_pointcloud('./test_data/test2.ply', device = device)
norm_pcls, norm_param = normalize_pcl(pcls)
pcl_lm_file = open('./test_data/test2_lm.txt')
lm_list = []
for line in pcl_lm_file:
    line = int(line.strip())
    lm_list.append(line)

target_lm_index = torch.from_numpy(np.array(lm_list)).to(device)
lm_mask = (target_lm_index >= 0)
target_lm_index = target_lm_index.unsqueeze(0)
bfm_meshes, bfm_lm_index = load_bfm_model(torch.device('cuda:0'))
bfm_lm_index_m = bfm_lm_index[:, lm_mask]
target_lm_index_m = target_lm_index[:, lm_mask]
registered_mesh = non_rigid_icp_mesh2pcl(bfm_meshes, norm_pcls, bfm_lm_index_m, target_lm_index_m)
io3d.save_meshes_as_objs(['test_data/final2.obj'], registered_mesh, save_textures = False)