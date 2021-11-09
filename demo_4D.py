# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

'''
    Demo for registering on 4D depth data
'''

import torch
import io3d
import numpy as np
import json
import cv2
import face_alignment
import time
from utils import normalize_mesh, normalize_pcl
from landmark import get_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2pcl
from pytorch3d.structures import Meshes, Pointclouds

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def depth_to_point(depth_image, color_image, intrinsics, min_depth, max_depth):
    '''
        intrinsics format
        {
            fx
            fy
            ppx
            ppy
            width
            height
            scale
        }
    '''
    fx, fy, ppx, ppy, width, height, scale = intrinsics
    width, height = int(width), int(height)
    red = depth_image[:, :, 0].astype(np.int32)
    green = depth_image[:, :, 1].astype(np.int32)
    blue = depth_image[:, :, 2].astype(np.int32)
    zero_mask = (blue + green + red) < 128
    rbigg = (red >= green)
    rbigb = (red >= blue)
    gbigb = (green >= blue)
    rbigg_and_rbigb = np.logical_and(rbigg, rbigb)
    cond1 = np.logical_and(rbigg_and_rbigb, gbigb)
    cond2 = np.logical_and(rbigg_and_rbigb, np.logical_not(gbigb))
    cond3 = np.logical_and(np.logical_not(rbigg), gbigb)
    cond4 = np.logical_and(np.logical_not(gbigb), np.logical_not(rbigb))
    cond1 = np.logical_and(cond1, np.logical_not(zero_mask))
    cond2 = np.logical_and(cond2, np.logical_not(zero_mask))
    cond3 = np.logical_and(cond3, np.logical_not(zero_mask))
    cond4 = np.logical_and(cond4, np.logical_not(zero_mask))
    out_img = cond1 * (green - blue) + cond2 * (green - blue + 1535) + cond3 * (blue - red + 512) + cond4 * (red - green + 1024)
    out_img = (min_depth + (max_depth - min_depth) * out_img / 1535.0) / scale + 0.5
    out_img_flat = out_img.reshape(-1)

    x = np.tile(np.arange(width), height)
    y = np.repeat(np.arange(height), width)
    point_x = out_img_flat * (x - ppx) / fx * scale
    point_y = out_img_flat * (y - ppy) / fy * scale
    point_z = out_img_flat * scale

    point_xyz = np.transpose(np.array([point_x, point_y, point_z]))
    lm = fa.get_landmarks(color_image[:, :, ::-1])[0].astype(np.longlong)
    lm_index = lm[:, 1] * 1280 + lm[:, 0]
    lm_xyz = point_xyz[lm_index]
    lm_mask = np.zeros(68).astype(bool)
    lm_mask[27:] = True

    return point_xyz, lm_index, lm_mask

video_path = './test_data/depth_wuhz.mp4'
video_cap = cv2.VideoCapture(video_path)
min_depth = 0.3
max_depth = 1.5

point_list = []
lm_list = []
lm_mask_list = []

idx = 0
while video_cap.isOpened():
    ret, frame = video_cap.read()
    if ret == False:
        break

    color_frame, depth_frame = frame[:720], frame[720:]
    cv2.imwrite('test.png', color_frame)
    intrinsics = [908.882, 909.781, 638.245, 354.673, 1280, 720, 0.00025]
    point, lm, mask = depth_to_point(depth_frame, color_frame, intrinsics, min_depth, max_depth)
    point_list.append(point)
    lm_list.append(lm)
    lm_mask_list.append(mask)
    idx += 1
    if idx == 30:
        break

bfm_meshes, bfm_lm_index = load_bfm_model(torch.device('cuda:0'))
device = torch.device('cuda:0')
coarse_config = json.load(open('config/coarse_grain.json'))
fourd_config = json.load(open('config/4d.json'))
for i in range(len(point_list)):
    point = point_list[i]
    lm = lm_list[i]
    lm_mask = lm_mask_list[i]
    point = torch.from_numpy(point).to(device).float()
    lm = torch.from_numpy(lm).to(device).unsqueeze(0)
    pcls = Pointclouds(points = [point])
    norm_pcls, norm_param = normalize_pcl(pcls)

    bfm_lm_index_m = bfm_lm_index[:, lm_mask]
    target_lm_index_m = lm[:, lm_mask]
    if i == 0:
        registered_mesh, returned_affine = non_rigid_icp_mesh2pcl(bfm_meshes, norm_pcls, bfm_lm_index_m, target_lm_index_m, coarse_config, None, out_affine = True)
    else:
        t1 = time.time()
        registered_mesh, returned_affine = non_rigid_icp_mesh2pcl(bfm_meshes, norm_pcls, bfm_lm_index_m, target_lm_index_m, fourd_config, returned_affine, out_affine = True)
        t2 = time.time()
        print(t2 - t1)
    
    if i == 12:
        io3d.save_meshes_as_objs(['test_data/final3.obj'], registered_mesh, save_textures = False)