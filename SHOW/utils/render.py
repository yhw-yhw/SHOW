import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras
from enum import Enum

class View(Enum):
    GROUND_TRUTH = 1
    COLOR_OVERLAY = 2
    SHAPE_OVERLAY = 4
    SHAPE = 8
    LANDMARKS = 16
    HEATMAP = 32
    DEPTH = 64


def render_shape(vertices, flame_faces, mesh_rasterizer, debug_renderer, cameras, white=True):
    # mesh_file=('./../data/head_template_mesh.obj')
    # flame_faces = load_obj(mesh_file)[1]
    # mesh_rasterizer: MeshRasterizer
    # debug_renderer: MeshRenderer
    # cameras: PerspectiveCameras

    B = vertices.shape[0]
    V = vertices.shape[1]

    faces = flame_faces.verts_idx.cuda()[None].repeat(B, 1, 1)

    if not white:
        verts_rgb = torch.from_numpy(
            np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
    else:
        verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[
            None, None, :].repeat(B, V, 1)
    textures = TexturesVertex(verts_features=verts_rgb.cuda())
    meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[
                          faces[i] for i in range(B)], textures=textures)

    fragments = mesh_rasterizer(meshes_world, cameras=cameras)
    rendering = debug_renderer.shader(fragments, meshes_world, cameras=cameras)
    rendering = rendering.permute(0, 3, 1, 2).detach()
    mesh = rendering[:, 0:3, :, :]
    alpha = 1.0 - (mesh == 1.0).int()
    mesh *= alpha
    return mesh
