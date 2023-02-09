from pytorch3d.transforms import so3_exp_map
from torchvision.transforms.functional import gaussian_blur
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, Transform3d, axis_angle_to_matrix
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.io import load_obj

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn
import torch.nn.functional as F
import SHOW
from SHOW.utils import View


def save_tracker(
    img,valid_bool,valid_bs,
    ops,vertices,cameras,image_lmks,proj_lmks,
    flame_faces,mesh_rasterizer,debug_renderer,
    save_callback,
):
    with torch.no_grad():
        images=img[valid_bool.bool(),...]
        mask = SHOW.utils.parse_mask(ops)
        gt_images = images
        
        predicted_images_s = (ops['images'] * mask + (images * (1.0 - mask)))
        shape_mask_s = ((ops['alpha_images'] * ops['mask_images_mesh']) > 0.).int()
        
        for frame in range(valid_bs):
            gt_image=gt_images[frame:frame+1]
            predicted_images=predicted_images_s[frame]
            shape_mask=shape_mask_s[frame:frame+1]
            vertice=vertices[frame:frame+1]
            camera=cameras[frame]
            image=images[frame]
            image_lmk=image_lmks[frame]
            proj_lmk=proj_lmks[frame]
            
            
            visualizations=[
                [View.GROUND_TRUTH, View.LANDMARKS, View.HEATMAP], 
                [View.COLOR_OVERLAY, View.SHAPE_OVERLAY, View.SHAPE]
            ]
            
            final_views = []
            for views in visualizations:
                # views=visualizations[0]
                row = []
                for view in views:
                    # view=views[0]
                    if view == View.COLOR_OVERLAY:
                        row.append(predicted_images.detach().cpu().numpy())
                    if view == View.GROUND_TRUTH:
                        row.append(image.detach().cpu().numpy())
                    if view == View.SHAPE:#imp
                        shape = SHOW.render_shape(vertice, flame_faces,mesh_rasterizer,debug_renderer, camera,white=True).cpu().numpy()
                        row.append(shape[0])
                    if view == View.LANDMARKS:#imp
                        gt_lmk = SHOW.utils.tensor_vis_landmarks(image, image_lmk, isScale=False)
                        lmks = SHOW.utils.tensor_vis_landmarks(gt_lmk[0], proj_lmk, color='r', isScale=False).cpu().numpy()
                        row.append(lmks[0])
                    if view == View.SHAPE_OVERLAY:#imp
                        shape = SHOW.render_shape(vertice,flame_faces,mesh_rasterizer,debug_renderer, camera, white=False) * shape_mask
                        blend = gt_image * (1 - shape_mask) + gt_image * shape_mask * 0.3 + shape * 0.7 * shape_mask
                        row.append(blend[0].detach().cpu().numpy())
                    if view == View.HEATMAP:
                        t = gt_image.detach().cpu()
                        f = predicted_images.detach().cpu()
                        l2 = torch.pow(torch.abs(f - t), 2)
                        heatmap = SHOW.image.get_heatmap(l2)
                        row.append(heatmap)
                final_views.append(row)
            final_views = SHOW.utils.merge_views(final_views)
            
            save_callback(frame,final_views)

