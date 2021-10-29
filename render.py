# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer
)
from shader import DummyShader

def create_dummy_render(camera_direction, device = torch.device('cpu'), image_size = 512):
    '''
        the dummy render directly use texture as final color without lighting model
    '''
    R, T = look_at_view_transform(*camera_direction)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    render = MeshRenderer(
        rasterizer = MeshRasterizer(
            cameras = cameras, 
            raster_settings = raster_settings
        ),
        shader = DummyShader(
            device = device
        )
    )
    return render

if __name__ == "__main__":
    render = create_dummy_render([3, 0, 0])