#  -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Authors: paper author. 
# Special Acknowlegement:  Wojciech Zielonka and Justus Thies
# Contact: ps-license@tuebingen.mpg.de

from pathlib import Path
import numpy as np
from tqdm import tqdm
import mmcv
import numpy as np
import os
from tqdm import tqdm
import cv2
import os.path
from functools import reduce
from pathlib import Path
from loguru import logger
import face_alignment
import mmcv
from pathlib import Path
import numpy as np
from tqdm import tqdm
import mmcv
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import cv2
import glob
import os.path
from functools import reduce
from pathlib import Path
from loguru import logger
import face_alignment
import mmcv

from SHOW.utils.video import images_to_video
from torchvision.transforms.functional import gaussian_blur
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.io import load_obj

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn
import torch.nn.functional as F

import SHOW
from SHOW.utils import default_timers
from SHOW.datasets import op_base
from SHOW.detector.face_detector import FaceDetector
from SHOW.load_models import load_smplx_model, load_vposer_model
from SHOW.save_results import save_one_results
from SHOW.load_models import load_save_pkl
from SHOW.flame.FLAME import FLAMETex
from SHOW.smplx_dataset import ImagesDataset
from SHOW.renderer import Renderer
from SHOW.load_assets import load_assets
from SHOW.loggers.logger import setup_logger
from SHOW.save_tracker import save_tracker
from SHOW.utils import is_valid_json
from configs.cfg_ins import condor_cfg


@logger.catch
def SHOW_stage2(*args, **kwargs):

    machine_info = SHOW.get_machine_info()
    import pprint
    pprint.pprint(f'machine_info: {machine_info}')

    loggers = kwargs.get('loggers', None)

    tracker_cfg = SHOW.from_rela_path(__file__,
                                      './configs/mmcv_tracker_config.py')
    tracker_cfg.update(**kwargs)
    tracker_cfg.merge_from_dict(condor_cfg)

    if tracker_cfg.get('over_write_cfg', None):
        tracker_cfg.update(tracker_cfg.over_write_cfg)
    

    mmcv.dump(tracker_cfg, tracker_cfg.tracker_cfg_path)
    
    
    try:
        gpu_mem = machine_info['gpu_info']['gpu_Total']

        import platform
        if platform.system() == 'Linux':
            # 50.0 * 24220 / (65.0*1024)
            tracker_cfg.bs_at_a_time = int(50.0 * gpu_mem / (80.0 * 1024))
            logger.warning(f'bs_at_a_time: {tracker_cfg.bs_at_a_time}')
    except:
        import traceback
        traceback.print_exc()

    Path(tracker_cfg.mica_save_path).mkdir(exist_ok=True, parents=True)
    Path(tracker_cfg.mica_org_out_path).mkdir(exist_ok=True, parents=True)

    iters = tracker_cfg.iters
    sampling = tracker_cfg.sampling
    device = tracker_cfg.device
    tracker_cfg.dtype = dtype = SHOW.str_to_torch_dtype(tracker_cfg.dtype)

    face_ider = SHOW.build_ider(tracker_cfg.ider_cfg)
    img_folder = tracker_cfg.img_folder
    template_im = os.listdir(img_folder)[0]
    template_im = os.path.join(img_folder, template_im)

    assets = load_assets(
        tracker_cfg,
        face_ider=face_ider,
        template_im=template_im,
    )
    if assets is None:
        return

    setup_logger(tracker_cfg.mica_all_dir, filename='mica.log', mode='o')

    
    if not Path(tracker_cfg.ours_pkl_file_path).exists():
        logger.warning(
            f'ours_pkl_file_path not exists: {tracker_cfg.ours_pkl_file_path}')
        return False

    if not is_valid_json(tracker_cfg.final_losses_json_path):
        logger.warning(
            f'final_losses_json_path not valid: {tracker_cfg.final_losses_json_path}'
        )
        return False

    
    with default_timers['build_vars_stage']:
        face_ider = SHOW.build_ider(tracker_cfg.ider_cfg)
        person_face_emb = assets.person_face_emb
        face_detector_mediapipe = FaceDetector('google', device=device)
        face_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=device)

        body_model = load_smplx_model(dtype=dtype, **tracker_cfg.smplx_cfg)
        body_params_dict = load_save_pkl(tracker_cfg.ours_pkl_file_path,
                                         device)

        width = body_params_dict['width']
        height = body_params_dict['height']
        center = body_params_dict['center']
        camera_transl = body_params_dict['camera_transl']
        focal_length = body_params_dict['focal_length']
        total_batch_size = body_params_dict['batch_size']

        opt_bs = tracker_cfg.bs_at_a_time
        opt_iters = total_batch_size // opt_bs
        st_et_list = []
        for i in range(opt_iters):
            st = i * opt_bs
            et = (i + 1) * opt_bs
            if et > total_batch_size - 1:
                et = total_batch_size - 1
            st_et_list.append((st, et))

        op = op_base()
        smplx2flame_idx = assets.smplx2flame_idx

        mesh_file = Path(__file__).parent.joinpath(
            '../data/head_template_mesh.obj')

        diff_renderer = Renderer(torch.Tensor([[512, 512]]),
                                 obj_filename=mesh_file)

        flame_faces = load_obj(mesh_file)[1]
        flametex = FLAMETex(tracker_cfg.flame_cfg).to(device)

        mesh_rasterizer = MeshRasterizer(
            raster_settings=RasterizationSettings(image_size=[512, 512],
                                                  faces_per_pixel=1,
                                                  cull_backfaces=True,
                                                  perspective_correct=True))

        debug_renderer = MeshRenderer(
            rasterizer=mesh_rasterizer,
            shader=SoftPhongShader(device=device,
                                   lights=PointLights(
                                       device=device,
                                       location=((0.0, 0.0, -5.0), ),
                                       ambient_color=((0.5, 0.5, 0.5), ),
                                       diffuse_color=((0.5, 0.5, 0.5), ))))

    pre_frame_exp = None
    for opt_idx, (start_frame, end_frame) in enumerate(st_et_list):
        if assets.person_face_emb is not None:
            mica_part_file_path = f'w_mica_part_{start_frame}_{end_frame}_{opt_idx}_{opt_iters}.pkl'
            mica_part_pkl_path = os.path.join(tracker_cfg.mica_all_dir,
                                              mica_part_file_path)

            if Path(mica_part_pkl_path).exists():
                logger.info(
                    f'mica_part_pkl_path exists,skipping: {mica_part_pkl_path}'
                )
                pre_con = mmcv.load(mica_part_pkl_path)
                pre_frame_exp = pre_con['expression'][-1]
                pre_frame_exp = torch.Tensor(pre_frame_exp).to(device)
                continue

            opt_bs = end_frame - start_frame

            com_tex = torch.zeros(1, 150).to('cuda')
            com_sh = torch.zeros(1, 9, 3).to('cuda')

            use_shared_tex = 1
            if not use_shared_tex:
                opt_bs_tex = nn.Parameter(com_tex).expand(opt_bs, -1).detach()
            else:
                opt_bs_tex = nn.Parameter(com_tex).expand(1, -1).detach()

            opt_bs_sh = nn.Parameter(com_sh).expand(opt_bs, -1, -1).detach()

            logger.info(f'origin input data frame batchsize:{opt_bs}')
            
            with default_timers['load_dataset_stage']:

                debug = 0
                if debug: opt_bs = 30

                dataset = ImagesDataset(
                    tracker_cfg,
                    start_frame=start_frame,
                    face_ider=face_ider,
                    person_face_emb=person_face_emb,
                    face_detector_mediapipe=face_detector_mediapipe,
                    face_detector=face_detector)
                dataloader = DataLoader(dataset,
                                        batch_size=opt_bs,
                                        num_workers=0,
                                        shuffle=False,
                                        pin_memory=True,
                                        drop_last=False)
                iterator = iter(dataloader)
                batch = next(iterator)
                if not debug:
                    batch = SHOW.utils.to_cuda(batch)
                valid_bool = batch['is_person_deted'].bool()
                valid_bs = valid_bool.count_nonzero()
                logger.info(f'valid input data frame batchsize:{valid_bs}')
                logger.info(f'valid_bool: {valid_bool}')

                if valid_bs == 0:
                    logger.warning('valid bs == 0, skipping')
                    open(mica_part_pkl_path + '.empty', 'a').close()
                    continue

                bbox = batch['bbox']
                images = batch['cropped_image']
                landmarks = batch['cropped_lmk']
                h = batch['h']
                w = batch['w']
                py = batch['py']
                px = batch['px']

                diff_renderer.masking.set_bs(valid_bs)
                diff_renderer = diff_renderer.to(device)

            debug = 0
            report_wandb = 0
            use_opt_pose = 1
            save_traing_img = 0
            observe_idx_list = [4, 8]
            
            with default_timers['optimize_stage']:
                model_output = None

                def get_pose_opt(start_frame, end_frame):
                    tmp = body_params_dict['body_pose_axis'][
                        start_frame:end_frame, ...].clone().detach()
                    tmp = tmp.reshape(tmp.shape[0], -1, 3)
                    return torch.stack([tmp[:, 12 - 1, :], 
                                        tmp[:, 15 - 1, :]],
                                       dim=1)

                def clone_params_color(start_frame, end_frame):
                    opt_var_clone_detach = [
                        {
                            'params': [
                                nn.Parameter(
                                    body_params_dict['expression']
                                    [start_frame:end_frame].clone().detach())
                            ],
                            'lr':
                            0.025,
                            'name': ['exp']
                        },
                        {
                            'params': [
                                nn.Parameter(body_params_dict['leye_pose']
                                             [start_frame:end_frame].clone(
                                             ).clone().detach())
                            ],
                            'lr':
                            0.001,
                            'name': ['leyes']
                        },
                        {
                            'params': [
                                nn.Parameter(
                                    body_params_dict['reye_pose']
                                    [start_frame:end_frame].clone().detach())
                            ],
                            'lr':
                            0.001,
                            'name': ['reyes']
                        },
                        {
                            'params': [
                                nn.Parameter(
                                    body_params_dict['jaw_pose']
                                    [start_frame:end_frame].clone().detach())
                            ],
                            'lr':
                            0.001,
                            'name': ['jaw']
                        },
                        {
                            'params':
                            [nn.Parameter(opt_bs_sh.clone().detach())],
                            'lr': 0.01,
                            'name': ['sh']
                        },
                        {
                            'params':
                            [nn.Parameter(opt_bs_tex.clone().detach())],
                            'lr': 0.005,
                            'name': ['tex']
                        },
                    ]
                    if use_opt_pose:
                        opt_var_clone_detach.append({
                            'params': [
                                nn.Parameter(
                                    get_pose_opt(start_frame, end_frame))
                            ],
                            'lr':
                            0.005,
                            'name': ['body_pose']
                        })
                    return opt_var_clone_detach

                save_traing_img_dir = tracker_cfg.mica_process_path + f'_{start_frame}_{end_frame}'

                if save_traing_img:
                    Path(save_traing_img_dir).mkdir(parents=True,
                                                    exist_ok=True)

                with tqdm(total=iters * 3,
                          position=0,
                          leave=True,
                          bar_format="{percentage:3.0f}%|{bar}{r_bar}{desc}"
                          ) as pbar:
                    for k, scale in enumerate(sampling):

                        size = [int(512 * scale), int(512 * scale)]
                        img = F.interpolate(images.float().clone(),
                                            size,
                                            mode='bilinear',
                                            align_corners=False)

                        if k > 0:
                            img = gaussian_blur(img, [9, 9]).detach()

                        flipped = torch.flip(img, [2, 3])
                        flipped = flipped[valid_bool.bool(), ...]

                        best_loss = np.inf
                        prev_loss = np.inf

                        xb_min, xb_max, yb_min, yb_max = bbox.values()
                        box_w = xb_max - xb_min
                        box_h = yb_max - yb_min
                        box_w = box_w.int()
                        box_h = box_h.int()

                        image_size = size

                        diff_renderer.rasterizer.reset()
                        diff_renderer.set_size(image_size)
                        debug_renderer.rasterizer.raster_settings.image_size = size

                        image_lmks = landmarks * size[0] / 512
                        image_lmks = image_lmks[valid_bool.bool(), ...]

                        optimizer = torch.optim.Adam(
                            clone_params_color(start_frame, end_frame))
                        params = optimizer.param_groups
                        get_param = SHOW.utils.get_param

                        cur_tex = get_param('tex', params)
                        cur_sh = get_param('sh', params)

                        cur_exp = get_param('exp', params)
                        cur_leyes = get_param('leyes', params)
                        cur_reyes = get_param('reyes', params)
                        cur_jaw = get_param('jaw', params)

                        if use_opt_pose:
                            two_opt = get_param('body_pose', params)
                            frame_pose = body_params_dict['body_pose_axis'][
                                start_frame:end_frame]
                            bs = frame_pose.shape[0]
                            frame_pose = frame_pose.reshape(bs, -1, 3)
                            cur_pose = torch.cat(
                                [
                                    frame_pose[:, :11, :],
                                    two_opt[:, 0:1],  #11
                                    frame_pose[:, 12:14, :],
                                    two_opt[:, 1:2],  #14
                                    frame_pose[:, 15:, :]
                                ],
                                dim=1).reshape(bs, 1, -1)
                        else:
                            frame_pose = body_params_dict['body_pose_axis'][
                                start_frame:end_frame]
                            bs = frame_pose.shape[0]
                            cur_pose = frame_pose.reshape(bs, 1, -1)

                        cur_transl = body_params_dict['transl'][
                            start_frame:end_frame]
                        cur_global_orient = body_params_dict['global_orient'][
                            start_frame:end_frame]
                        cur_left_hand_pose = body_params_dict[
                            'left_hand_pose'][start_frame:end_frame]
                        cur_right_hand_pose = body_params_dict[
                            'right_hand_pose'][start_frame:end_frame]

                        R = torch.Tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
                        bs_image_size = torch.Tensor(image_size).repeat(
                            opt_bs, 1).to(device)
                        bs_camera_transl = (camera_transl).repeat(opt_bs,
                                                                  1).to(device)

                        # bs_center: torch.Size([10, 2])
                        bs_center = torch.Tensor(center).repeat(opt_bs,
                                                                1).to(device)
                        bs_box_min = torch.stack([xb_min, yb_min],
                                                 dim=-1).to(device)
                        bs_R = R.repeat(opt_bs, 1, 1).to(device)

                        s_w = size[0] / box_w
                        s_h = size[1] / box_h
                        s_1to2 = torch.stack([s_w, s_h], dim=1)
                        s_1to2 = s_1to2.to(device)
                        bs_pp = (bs_center - bs_box_min) * (s_1to2)

                        bs_pp[:, 0] = (bs_pp[:, 0] + px) * 512 / (w + 2 * px)
                        bs_pp[:, 1] = (bs_pp[:, 1] + py) * 512 / (h + 2 * py)

                        s_1to2[:, 0] = s_1to2[:, 0] * 512 / (w + 2 * px)
                        s_1to2[:, 1] = s_1to2[:, 1] * 512 / (h + 2 * py)

                        cam_cfg = dict(
                            principal_point=bs_pp,
                            focal_length=focal_length * s_1to2,
                            R=bs_R,
                            T=bs_camera_transl,
                            image_size=bs_image_size,
                        )

                        for key, val in cam_cfg.items():
                            cam_cfg[key] = val[valid_bool.bool(), ...]

                        cameras = PerspectiveCameras(**cam_cfg,
                                                     device=device,
                                                     in_ndc=False)

                        if True:
                            for p in range(iters):  # p=0
                                if (p + 1) % 32 == 0:
                                    diff_renderer.rasterizer.reset()
                                losses = {}
                                model_output = body_model(
                                    return_verts=True,
                                    jaw_pose=cur_jaw,
                                    leye_pose=cur_leyes,
                                    reye_pose=cur_reyes,
                                    expression=cur_exp,
                                    betas=body_params_dict['betas'],
                                    transl=cur_transl,
                                    body_pose=cur_pose,
                                    global_orient=cur_global_orient,
                                    left_hand_pose=cur_left_hand_pose,
                                    right_hand_pose=cur_right_hand_pose,
                                )

                                vertices = model_output.vertices[:,
                                                                 smplx2flame_idx
                                                                 .long(), :]
                                vertices = vertices[valid_bool.bool(), ...]

                                lmk68_all = model_output.joints[:, 67:67 + 51 +
                                                                17, :]
                                lmk68 = lmk68_all[valid_bool.bool(), ...]

                                proj_lmks = cameras.transform_points_screen(
                                    lmk68)[:, :, :2]
                                proj_lmks = torch.cat([
                                    proj_lmks[:, -17:, :],
                                    proj_lmks[:, :-17, :]
                                ],
                                                      dim=1)

                                I = torch.eye(3)[None].to(device)

                                if pre_frame_exp is not None and start_frame != 0:
                                    losses['pre_exp'] = 0.001 * torch.sum(
                                        (pre_frame_exp - cur_exp[0])**2)

                                if False:
                                    linear_rot_left = (axis_angle_to_matrix(
                                        cur_leyes[valid_bool.bool(), ...]))
                                    linear_rot_right = (axis_angle_to_matrix(
                                        cur_reyes[valid_bool.bool(), ...]))
                                    losses['eyes_sym_reg'] = torch.sum(
                                        (linear_rot_right - linear_rot_left)**
                                        2) / opt_bs
                                    losses['eyes_left_reg'] = torch.sum(
                                        (I - linear_rot_left)**2) / opt_bs
                                    losses['eyes_right_reg'] = torch.sum(
                                        (I - linear_rot_right)**2) / opt_bs

                                w_lmks = tracker_cfg.w_lmks
                                losses['lmk'] = SHOW.utils.lmk_loss(
                                    proj_lmks, image_lmks,
                                    image_size) * w_lmks * 8.0
                                losses[
                                    'lmk_mount'] = SHOW.utils.mouth_loss(  #(49, 68)
                                        proj_lmks, image_lmks,
                                        image_size) * w_lmks * 4.0 * 4
                                losses['lmk_oval'] = SHOW.utils.lmk_loss(
                                    proj_lmks[:, :17, ...], image_lmks[:, :17,
                                                                       ...],
                                    image_size) * w_lmks

                                losses['jaw_reg'] = torch.sum(
                                    (I - axis_angle_to_matrix(
                                        cur_jaw[valid_bool.bool(), ...]))**
                                    2) * 16.0 / opt_bs
                                losses['exp_reg'] = torch.sum(
                                    cur_exp[valid_bool.bool(),
                                            ...]**2) * 0.01 / opt_bs

                                if use_shared_tex:
                                    losses['tex_reg'] = torch.sum(cur_tex**
                                                                  2) * 0.02
                                else:
                                    losses['tex_reg'] = torch.sum(
                                        cur_tex[valid_bool.bool(),
                                                ...]**2) * 0.02 / opt_bs

                                def temporary_loss(o_w, i_w, gmof, param):
                                    assert param.shape[
                                        0] > 2, f'optimize batchsize must > 2 to enable temporary smooth'
                                    return (o_w**2) * (gmof(
                                        i_w *
                                        (param[2:, ...] + param[:-2, ...] -
                                         2 * param[1:-1, ...]))).mean()

                                def pow(x):
                                    return x.pow(2)

                                if cur_exp.shape[0] > 2:
                                    losses['loss_sexp'] = temporary_loss(
                                        1.0, 2.0, pow, cur_exp)
                                    losses['loss_sjaw'] = temporary_loss(
                                        1.0, 2.0, pow, cur_jaw)
                                    
                                def k_fun(k):
                                    return tracker_cfg.w_pho * 32.0 if k > 0 else tracker_cfg.w_pho

                                albedos = flametex(cur_tex) / 255.

                                if use_shared_tex:
                                    albedos = albedos.expand(
                                        valid_bs, -1, -1, -1)
                                else:
                                    albedos = albedos[valid_bool.bool(), ...]

                                ops = diff_renderer(
                                    vertices, albedos,
                                    cur_sh[valid_bool.bool(), ...], cameras)

                                grid = ops['position_images'].permute(
                                    0, 2, 3, 1)[:, :, :, :2]
                                sampled_image = F.grid_sample(
                                    flipped, grid, align_corners=False)
                                ops_mask = SHOW.utils.parse_mask(ops)
                                tmp_img = ops['images']

                                losses['pho'] = SHOW.utils.pixel_loss(
                                    tmp_img, sampled_image,
                                    ops_mask) * k_fun(k)

                                all_loss = 0.
                                for key in losses.keys():
                                    all_loss = all_loss + losses[key]
                                losses['all_loss'] = all_loss

                                log_str = SHOW.print_dict_losses(losses)

                                if report_wandb:
                                    if globals().get('wandb', None) is None:
                                        os.environ[
                                            'WANDB_API_KEY'] = 'xxx'
                                        os.environ['WANDB_NAME'] = 'tracker'
                                        import wandb
                                        wandb.init(
                                            reinit=True,
                                            resume='allow',
                                            project='tracker',
                                        )
                                        globals()['wandb'] = wandb

                                    if globals().get('wandb',
                                                     None) is not None:
                                        globals()['wandb'].log(losses)

                                if save_traing_img:

                                    def save_callback(frame, final_views):
                                        cur_idx = (frame + opt_bs * opt_idx)
                                        if cur_idx in observe_idx_list:

                                            observe_idx_frame_dir = os.path.join(
                                                save_traing_img_dir,
                                                f'{cur_idx:03d}')
                                            Path(observe_idx_frame_dir).mkdir(
                                                parents=True, exist_ok=True)

                                            cv2.imwrite(
                                                os.path.join(
                                                    observe_idx_frame_dir,
                                                    f'{k}_{p}.jpg'),
                                                final_views)

                                    save_tracker(
                                        img,
                                        valid_bool,
                                        valid_bs,
                                        ops,
                                        vertices,
                                        cameras,
                                        image_lmks,
                                        proj_lmks,
                                        flame_faces,
                                        mesh_rasterizer,
                                        debug_renderer,
                                        save_callback,
                                    )

                                if loggers is not None:
                                    loggers.log_bs(losses)
                                    if torch.isnan(all_loss).sum():
                                        loggers.alert(
                                            title='Nan error',
                                            msg=
                                            f'tracker nan in: {tracker_cfg.ours_output_folder}'
                                        )
                                        open(
                                            tracker_cfg.ours_output_folder +
                                            '/mica_opt_nan.info', 'a').close()
                                        break

                                else:
                                    pbar.set_description(log_str)
                                    pbar.update(1)

                                optimizer.zero_grad()
                                all_loss.backward()
                                optimizer.step()

                                if all_loss.item() < best_loss:
                                    best_loss = all_loss.item()
                                    opt_bs_tex = cur_tex.clone().detach()
                                    opt_bs_sh = cur_sh.clone().detach()
                                    body_params_dict['expression'][
                                        start_frame:end_frame] = cur_exp.clone(
                                        ).detach()
                                    body_params_dict['leye_pose'][
                                        start_frame:
                                        end_frame] = cur_leyes.clone().detach(
                                        )
                                    body_params_dict['reye_pose'][
                                        start_frame:
                                        end_frame] = cur_reyes.clone().detach(
                                        )
                                    body_params_dict['jaw_pose'][
                                        start_frame:end_frame] = cur_jaw.clone(
                                        ).detach()
                                    body_params_dict['body_pose_axis'][
                                        start_frame:
                                        end_frame] = cur_pose.clone().detach(
                                        ).squeeze()

        
        with default_timers['saving_stage']:
            if save_traing_img:
                for idx in observe_idx_list:
                    observe_idx_frame_dir = os.path.join(
                        save_traing_img_dir, f'{idx:03d}')
                    Path(observe_idx_frame_dir).mkdir(parents=True,
                                                      exist_ok=True)

                    if not SHOW.is_empty_dir(observe_idx_frame_dir):
                        images_to_video(
                            input_folder=observe_idx_frame_dir,
                            output_path=observe_idx_frame_dir + '.mp4',
                            img_format=None,
                            fps=30,
                        )

            dict_to_save = dict(
                expression=body_params_dict['expression']
                [start_frame:end_frame].clone().detach().cpu().numpy(),
                leye_pose=body_params_dict['leye_pose']
                [start_frame:end_frame].clone().detach().cpu().numpy(),
                reye_pose=body_params_dict['reye_pose']
                [start_frame:end_frame].clone().detach().cpu().numpy(),
                jaw_pose=body_params_dict['jaw_pose']
                [start_frame:end_frame].clone().detach().cpu().numpy(),
                body_pose_axis=body_params_dict['body_pose_axis']
                [start_frame:end_frame].clone().detach().cpu().numpy(),
                tex=opt_bs_tex.clone().detach().cpu().numpy(),
                sh=opt_bs_sh.clone().detach().cpu().numpy(),
            )
            mmcv.dump(dict_to_save, mica_part_pkl_path)
            logger.info(f'mica pkl part path: {mica_part_pkl_path}')
            pre_frame_exp = dict_to_save['expression'][-1]
            pre_frame_exp = torch.Tensor(pre_frame_exp).to(device)
            vertices_ = model_output.vertices.clone().detach().cpu()
            logger.info(
                f'mica render to origin path: {tracker_cfg.mica_org_out_path}')

            import platform
            if platform.system() == "Linux":
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
            else:
                if 'PYOPENGL_PLATFORM' in os.environ:
                    os.environ.__delitem__('PYOPENGL_PLATFORM')

            import pyrender
            input_renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                                        viewport_height=height,
                                                        point_size=1.0)

            for idx in tqdm(range(vertices_.shape[0]),
                            desc='saving ours final pyrender images'):  # idx=0

                cur_idx = idx + start_frame + 1

                input_img = SHOW.find_full_impath_by_name(
                    root=tracker_cfg.img_folder, name=f'{cur_idx:06d}')
                output_name = os.path.join(
                    tracker_cfg.mica_org_out_path,
                    f"{cur_idx:06}.{tracker_cfg.output_img_ext}")

                camera_pose = op.get_smplx_to_pyrender_K(camera_transl)

                meta_data = dict(
                    input_img=input_img,
                    output_name=output_name,
                )

                save_one_results(
                    vertices_[idx],
                    body_model.faces,
                    img_size=(height, width),
                    center=center,
                    focal_length=[focal_length, focal_length],
                    camera_pose=camera_pose,
                    meta_data=meta_data,
                    input_renderer=input_renderer,
                )
            input_renderer.delete()

            
            if tracker_cfg.save_final_vis:

                def save_callback(frame, final_views):
                    cur_idx = (frame + opt_bs * opt_idx)

                    if loggers is not None:
                        loggers.log_image(f"final_mica_img/{cur_idx:03d}",
                                          final_views / 255.0)

                    cv2.imwrite(
                        os.path.join(tracker_cfg.mica_save_path,
                                     f'{cur_idx:03d}.jpg'), final_views)

                if True:
                    save_tracker(
                        img,
                        valid_bool,
                        valid_bs,
                        ops,
                        vertices,
                        cameras,
                        image_lmks,
                        proj_lmks,
                        flame_faces,
                        mesh_rasterizer,
                        debug_renderer,
                        save_callback,
                    )

    load_data = mmcv.load(tracker_cfg.ours_pkl_file_path)[0]
    load_data = SHOW.replace_mica_exp(tracker_cfg.mica_all_dir, load_data)
    mmcv.dump([load_data], tracker_cfg.mica_merge_pkl)

    if not Path(tracker_cfg.mica_org_out_video).exists():
        if not SHOW.is_empty_dir(tracker_cfg.mica_org_out_path):
            images_to_video(
                input_folder=tracker_cfg.mica_org_out_path,
                output_path=tracker_cfg.mica_org_out_video,
                img_format=None,
                fps=30,
            )
    if not Path(tracker_cfg.mica_grid_video).exists():
        if not SHOW.is_empty_dir(tracker_cfg.mica_save_path):
            images_to_video(
                input_folder=tracker_cfg.mica_save_path,
                output_path=tracker_cfg.mica_grid_video,
                img_format=None,
                fps=30,
            )
