import os
import sys
import os.path as osp
from pathlib import Path
import matplotlib
import platform
import warnings

if platform.system() == 'Windows':
    matplotlib.use('TkAgg')

if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

sys.path.extend([
    osp.join(osp.dirname(__file__)),
])
warnings.filterwarnings("ignore")

import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from loguru import logger
from easydict import EasyDict
import mmcv
import os
import mmcv
from loguru import logger
from pathlib import Path
import trimesh
import numpy as np
import pyrender
import os
import PIL.Image as pil_img
import cv2
import smplx
from loguru import logger

# from SHOW.utils.video import images_to_video
from mmhuman3d.utils.ffmpeg_utils import images_to_video


DEFAULT_SMPLX_CONFIG = dict(
    create_global_orient=True,
    create_body_pose=True,
    create_betas=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_expression=True,
    create_jaw_pose=True,
    create_leye_pose=True,
    create_reye_pose=True,
    create_transl=True,
)

colors_dict = {
    'red': np.array([0.5, 0.2, 0.2]),
    'pink': np.array([0.7, 0.5, 0.5]),
    'neutral': np.array([0.7, 0.7, 0.6]),
    # 'purple': np.array([0.5, 0.5, 0.7]),
    'purple': np.array([0.55, 0.4, 0.9]),
    'green': np.array([0.5, 0.55, 0.3]),
    'sky': np.array([0.3, 0.5, 0.55]),
    'white': np.array([1.0, 0.98, 0.94]),
}


def save_one_results(
    vertices,
    faces,
    img_size,  #(height,width)
    center,  #(cx,cy)
    focal_length,  #(focalx,focaly)
    camera_pose,  #K:(4,4)
    meta_data={},
    color_type='sky',
    input_renderer=None,
):

    input_img = meta_data.get('input_img', None)
    output_name = meta_data.get('output_name', None)

    if 1:
        color = colors_dict[color_type]

        out_mesh = trimesh.Trimesh(vertices, faces, process=False)

        out_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180),
                                                    [1, 0, 0]))

        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                roughnessFactor=0.6,
                alphaMode='OPAQUE',
                baseColorFactor=(color[0], color[1], color[2], 1.0)))
        # 黑色背景
        bg_color = [0.0, 0.0, 0.0, 0.0]

    # 渲染器参数设置
    if input_renderer is None:
        renderer = pyrender.OffscreenRenderer(viewport_width=img_size[1],
                                              viewport_height=img_size[0],
                                              point_size=1.0)
    else:
        renderer = input_renderer

    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera = pyrender.camera.IntrinsicsCamera(fx=focal_length[0],
                                              fy=focal_length[1],
                                              cx=center[0],
                                              cy=center[1])
    scene.add(camera, pose=camera_pose)

    if 0:
        light_node = pyrender.DirectionalLight()
        scene.add(light_node, pose=camera_pose)
    else:
        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.2,
                                    intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose)

        spot_l = pyrender.SpotLight(color=np.ones(3),
                                    intensity=15.0,
                                    innerConeAngle=np.pi / 3,
                                    outerConeAngle=np.pi / 2)

        light_pose[:3, 3] = [1, 2, 2]
        scene.add(spot_l, pose=light_pose)

        light_pose[:3, 3] = [-1, 2, 2]
        scene.add(spot_l, pose=light_pose)

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    if input_img != None:
        if type(input_img) == str:
            input_img = cv2.imread(input_img)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        assert (type(input_img) == np.ndarray)

        if input_img.max() > 1:
            input_img = input_img.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = (color[:, :, :-1] * valid_mask +
                      (1 - valid_mask) * input_img)
        output_img = pil_img.fromarray((output_img * 255.).astype(np.uint8))

    if output_name != None:
        output_img = (color[:, :, :-1])
        output_img = pil_img.fromarray((output_img * 255.).astype(np.uint8))

        Path(output_name).parent.mkdir(exist_ok=True, parents=True)
        output_img.save(output_name)

    if input_renderer is None:
        renderer.delete()

    return output_img


def render_pkl_api(
        pkl_file_path,  # pkl path
        out_images_path,  # image path folder
        output_video_path=None,  # endwith .mp4
        smplx_model_path='../models/smplx/SMPLX_NEUTRAL_2020_org.npz',  #smplx neutral 2020 npz file
        **kwargs):
    dtype = torch.float32

    smplx_model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), smplx_model_path))

    all_var = mmcv.load(pkl_file_path)
    if isinstance(all_var, list):
        all_var = all_var[0]
    all_var = EasyDict(all_var)

    body_model = smplx.create(**DEFAULT_SMPLX_CONFIG,
                              dtype=dtype,
                              model_path=smplx_model_path,
                              use_face_contour=True,
                              use_pca=True,
                              flat_hand_mean=False,
                              use_hands=True,
                              use_face=True,
                              num_pca_comps=12,
                              num_betas=300,
                              num_expression_coeffs=100,
                              batch_size=all_var.batch_size).to(device='cuda')

    def get_smplx_to_pyrender_K(cam_transl) -> np.ndarray:
        if isinstance(cam_transl, torch.Tensor):
            T = cam_transl.detach().cpu().numpy()
        T[1] *= -1
        K = np.eye(4)
        K[:3, 3] = T
        return K

    camera_pose = get_smplx_to_pyrender_K(
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

    import pyrender
    input_renderer = pyrender.OffscreenRenderer(viewport_width=all_var.width,
                                                viewport_height=all_var.height,
                                                point_size=1.0)

    vertices_ = model_output.vertices.detach().cpu().numpy()
    Path(out_images_path).mkdir(parents=True, exist_ok=True)
    logger.info(f'saving final images to: {out_images_path}')

    for idx in tqdm(range(vertices_.shape[0])):  # idx=0
        vertices = vertices_[idx]
        meta_data = dict(
            # input_img=os.path.join(img_folder, f'{idx+1:06d}.png'),
            output_name=os.path.join(out_images_path, f'{idx+1:06d}.png'), )

        tmp = save_one_results(
            vertices,
            body_model.faces,
            img_size=[all_var.height, all_var.width],
            center=all_var.center,
            focal_length=[all_var.focal_length, all_var.focal_length],
            camera_pose=camera_pose,
            meta_data=meta_data,
            input_renderer=input_renderer,
        )

    input_renderer.delete()

    def files_num_in_dir(dir_name):
        if not Path(dir_name).exists():
            return -1
        return len(os.listdir(dir_name))

    def is_empty_dir(dir_name):
        if not os.path.exists(dir_name):
            return 1
        return int(files_num_in_dir(dir_name) == 0)

    if output_video_path is not None:
        logger.info(f'saving final video to: {output_video_path}')
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        if not is_empty_dir(out_images_path):
            images_to_video(
                input_folder=out_images_path,
                output_path=output_video_path,
                img_format=None,
                fps=30,
            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file_path', type=str, default='all.pkl')
    parser.add_argument('--out_images_path',
                        type=str,
                        default='../results/ours_images')
    parser.add_argument('--output_video_path',
                        type=str,
                        default='../results/ours.mp4')
    parser.add_argument(
        '--smplx_model_path',
        type=str,
        default=
        'path_to_models/smplx/SMPLX_NEUTRAL_2020_org.npz')
    args = parser.parse_args()

    render_pkl_api(pkl_file_path=args.pkl_file_path,
                   out_images_path=args.out_images_path,
                   output_video_path=args.output_video_path,
                   smplx_model_path=args.smplx_model_path)
