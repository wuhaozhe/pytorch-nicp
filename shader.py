# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import torch.nn as nn
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes

class DummyShader(nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        return texels