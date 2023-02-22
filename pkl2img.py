import os
import sys
import os.path as osp
from pathlib import Path
import matplotlib
import platform
import warnings

if platform.system() == 'Windows':
    matplotlib.use('TkAgg')
    
import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
else:
    if 'PYOPENGL_PLATFORM' in os.environ:
        os.environ.__delitem__('PYOPENGL_PLATFORM')
    

sys.path.extend([
    osp.join(osp.dirname(__file__)),
    osp.join(osp.dirname(__file__), 'configs'),
])
warnings.filterwarnings("ignore")

from SHOW.utils.metric import MeterBuffer
from SHOW.load_models import load_save_pkl
from SHOW.save_results import save_one_results
from SHOW.load_models import load_smplx_model, load_vposer_model
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
import torch
from pathlib import Path
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from easydict import EasyDict
import SHOW.utils as utils
import mmcv
import SHOW
from SHOW.load_models import load_save_pkl
from SHOW.utils import default_timers
from SHOW.datasets import op_base
import os
import mmcv
import argparse
from loguru import logger
from SHOW.utils.video import images_to_video

def render_pkl_api(
        img_folder,
        ours_pkl_file_path,
        ours_images_path,
        output_video_path=None,
        output_obj_path=None,
        mica_all_dir=None,
        replace_from_mica=False,
        # use_npy_betas_ver='betas_2019_male',
        model_path='../models/smplx/SMPLX_NEUTRAL_2020_org.npz',
        body_model=None,
        use_input_renderer=True,
        **kwargs):
    utils.platform_init()

    op = op_base()
    dtype = torch.float32

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), model_path))

    all_var = mmcv.load(ours_pkl_file_path)
    # locals().update(all_var[0])
    if isinstance(all_var, list):
        all_var = all_var[0]
    all_var = EasyDict(all_var)


    if body_model is None:
        body_model = load_smplx_model(dtype=dtype,
                                      model_path=model_path,
                                      use_face_contour=True,
                                      use_pca=True,
                                      flat_hand_mean=False,
                                      use_hands=True,
                                      use_face=True,
                                      num_pca_comps=12,
                                      num_betas=300,
                                      num_expression_coeffs=100,
                                      batch_size=all_var.batch_size)

    camera_pose = op.get_smplx_to_pyrender_K(
        torch.from_numpy(all_var.camera_transl))
    smplx_params = dict(
        body_pose=all_var.body_pose_axis,
        betas=all_var.betas,
        global_orient=all_var.global_orient,
        transl=all_var.transl,
        left_hand_pose=all_var.left_hand_pose,
        right_hand_pose=all_var.right_hand_pose,
        jaw_pose=all_var.jaw_pose,
        leye_pose=all_var.leye_pose,
        reye_pose=all_var.reye_pose,
        expression=all_var.expression,
    )
    for key, val in smplx_params.items():
        if isinstance(val, torch.Tensor):
            smplx_params[key] = smplx_params[key].detach().to('cuda')
        if isinstance(val, np.ndarray):
            smplx_params[key] = torch.from_numpy(smplx_params[key]).to('cuda')
    model_output = body_model(return_verts=True, **smplx_params)

    if use_input_renderer:
        import pyrender
        input_renderer = pyrender.OffscreenRenderer(
            viewport_width=all_var.width,
            viewport_height=all_var.height,
            point_size=1.0)
    else:
        input_renderer = None

    vertices_ = model_output.vertices.detach().cpu().numpy()
    Path(ours_images_path).mkdir(parents=True, exist_ok=True)
    logger.info(f'saving final images to: {ours_images_path}')

    if output_obj_path is not None:
        Path(output_obj_path).mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(vertices_.shape[0])):  # idx=0
        vertices = vertices_[idx]

        input_img = None
        for ext in ['.png', '.jpg']:
            input_img = os.path.join(img_folder, f'{idx+1:06d}{ext}')
            if Path(input_img).exists():
                break

        if not Path(input_img).exists():
            logger.error(f'input_img not exists: {input_img}')
            return False

        meta_data = dict(
            input_img=input_img,
            output_name=os.path.join(ours_images_path, f'{idx+1:06d}.png'),
            obj_path=os.path.join(output_obj_path, f'{idx+1:06d}.obj')
            if output_obj_path is not None else None,
        )

        save_one_results(
            vertices,
            body_model.faces,
            img_size=[all_var.height, all_var.width],
            center=all_var.center,
            focal_length=[all_var.focal_length, all_var.focal_length],
            camera_pose=camera_pose,
            meta_data=meta_data,
            input_renderer=input_renderer,
        )

    if use_input_renderer:
        input_renderer.delete()

    if output_video_path is not None:
        logger.info(f'saving final video to: {output_video_path}')
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        if not SHOW.is_empty_dir(ours_images_path):
            images_to_video(
                input_folder=ours_images_path,
                output_path=output_video_path,
                img_format=None,
                fps=30,
            )
