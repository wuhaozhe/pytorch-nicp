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