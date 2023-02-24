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
from easydict import EasyDict
import mmcv
import numpy as np
import trimesh
import os
import pickle
import os.path as osp
from tqdm import tqdm
import cv2
import os.path
from functools import reduce
from pathlib import Path
from loguru import logger
import mmcv
from pathlib import Path
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import mmcv
import time
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import PIL.Image as pil_img
import os
import pickle
import os.path as osp
from tqdm import tqdm
import cv2
import glob
import os.path
from functools import reduce
from pathlib import Path
from loguru import logger
import mmcv
from scipy.ndimage import distance_transform_edt


from pytorch3d.renderer import (
                                RasterizationSettings, MeshRenderer,
                                MeshRasterizer, 
                                SoftSilhouetteShader,
                                )


import torch
import torch.backends.cudnn
import torch.nn.functional as F
from smplx.lbs import vertices2landmarks

import SHOW
from SHOW.loggers.logger import setup_logger
from SHOW.load_models import load_save_pkl
from SHOW.utils import default_timers
from SHOW.utils.metric import MeterBuffer
from SHOW.load_models import load_save_pkl
from SHOW.load_models import load_smplx_model, load_vposer_model
from SHOW.datasets import op_dataset
from SHOW.prior import build_prior
from SHOW.load_assets import load_assets
from SHOW.parse_weight import parse_weight
from SHOW.losses import *
from SHOW.utils import is_valid_json
from configs.cfg_ins import condor_cfg
from SHOW.datasets.model_func_atach import atach_model_func


@logger.catch
def SHOW_stage1(*args, **kwargs):

    machine_info = SHOW.get_machine_info()
    import pprint
    pprint.pprint(f'machine_info: {machine_info}')
    smplifyx_cfg = SHOW.utils.from_rela_path(
        __file__, './configs/mmcv_smplifyx_config.py')
    smplifyx_cfg.merge_from_dict(kwargs)

    def update_betas_name_cfg():

        smplifyx_cfg.save_betas_name = smplifyx_cfg.betas_ver_temp.format(
            smplifyx_cfg.speaker_name)

        if (Path(smplifyx_cfg.save_betas_name).exists()
                and smplifyx_cfg.load_betas):
            logger.info(f'loading betas')
            smplifyx_cfg.merge_from_dict(smplifyx_cfg.betas_precompute)
        else:
            logger.info(f'not loading betas')
            smplifyx_cfg.merge_from_dict(smplifyx_cfg.betas_generate)

        if smplifyx_cfg.speaker_name == -1:
            smplifyx_cfg.use_height_constraint = False
            smplifyx_cfg.use_weight_constraint = False

        smplifyx_cfg.merge_from_dict(condor_cfg)

        if smplifyx_cfg.get('over_write_cfg', None):
            logger.info(
                f'over_write_cfg: {smplifyx_cfg.over_write_cfg.to_dict()}')
            smplifyx_cfg.update(smplifyx_cfg.over_write_cfg)

    update_betas_name_cfg()
    mmcv.mkdir_or_exist(smplifyx_cfg.ours_output_folder)
    mmcv.dump(dict(smplifyx_cfg),
              osp.join(smplifyx_cfg.ours_output_folder, 'conf.yaml'))
    setup_logger(smplifyx_cfg.ours_output_folder, mode='o')
    dtype = SHOW.str_to_torch_dtype(smplifyx_cfg.dtype)
    device = smplifyx_cfg.device
    smplifyx_cfg.dtype = dtype
    run_optimize_flag = False

    if (not is_valid_json(smplifyx_cfg.final_losses_json_path)):
        logger.warning('final_losses_json_path not valid')
        run_optimize_flag = True

    if (not Path(smplifyx_cfg.ours_pkl_file_path).exists()):
        logger.warning('ours_pkl_file_path not exists')
        run_optimize_flag = True

    if smplifyx_cfg.get('force_run', None) and smplifyx_cfg.force_run:
        run_optimize_flag = True

    if smplifyx_cfg.get('pure_pre_data', None) and smplifyx_cfg.pure_pre_data:
        logger.warning(f'pure_pre_data is True')
        SHOW.purge_dir(smplifyx_cfg.keyp_folder)
        SHOW.purge_dir(smplifyx_cfg.deca_mat_folder)
        SHOW.purge_dir(smplifyx_cfg.pixie_mat_folder)
        SHOW.purge_dir(smplifyx_cfg.fan_npy_folder)
        SHOW.purge_dir(smplifyx_cfg.mp_npz_folder)
        SHOW.purge_dir(smplifyx_cfg.pymaf_pkl_folder)

    if not run_optimize_flag:
        logger.warning('no need to optimize')
        return False

    with default_timers['load_dataset']:
        face_ider = SHOW.build_ider(smplifyx_cfg.ider_cfg)
        img_folder = smplifyx_cfg.img_folder
        template_im = os.listdir(img_folder)[0]
        template_im = os.path.join(img_folder, template_im)

        body_model = load_smplx_model(dtype=dtype,
                                      batch_size=1,
                                      **smplifyx_cfg.smplx_cfg)
        atach_model_func(body_model)
        smplifyx_cfg.cvt_hand_func = lambda *args, **kwargs: body_model.hand_axis_to_pca(
            body_model, *args, **kwargs)
        smplifyx_cfg.merge_from_dict(smplifyx_cfg.smplx_cfg)

        op = op_dataset(
            config=smplifyx_cfg,
            face_ider=face_ider,
            person_face_emb=None,
            batch_size=smplifyx_cfg.batch_size,
            device=smplifyx_cfg.device,
            dtype=smplifyx_cfg.dtype,
        )
        op.initialize()

        assets = load_assets(
            smplifyx_cfg,
            face_ider=face_ider,
            template_im=template_im,
        )
        if assets is None:
            return

        update_betas_name_cfg()
        if assets.speaker_shape_vertices is None:
            smplifyx_cfg.use_mica_shape = False

        smplx2flame_idx = assets['smplx2flame_idx'].long()
        face_mask = assets['FLAME_masks']['face']
        speaker_shape_vertices = assets['speaker_shape_vertices']
        lmk_faces_idx = assets['mp_lmk_emb']['lmk_face_idx']
        lmk_bary_coords = assets['mp_lmk_emb']['lmk_b_coords']
        mp_indices = assets['mp_lmk_emb']['landmark_indices']
        opt_weights_list = parse_weight(smplifyx_cfg, device, dtype)
        robustifier = GMoF(rho=100)
        op.person_face_emb = assets.person_face_emb
        ret = op.get_all()

        betas = ret['init_data']['betas']
        expression = ret['init_data']['exp']
        jaw_pose = ret['init_data']['jaw']
        right_hand_pose = ret['init_data']['rhand']
        left_hand_pose = ret['init_data']['lhand']
        pose = ret['init_data']['pose']
        global_orient = ret['init_data']['global_orient']
        cam_transl = ret['init_data']['cam_transl']
        mica_head_transl = ret['init_data']['mica_head_transl']
        leye_pose = ret['init_data']['leye_pose']
        reye_pose = ret['init_data']['reye_pose']
        transl = ret['init_data']['transl']

        (op_kpts_org_data, mp_kpts, deca_kpts, op_valid_flag, mp_valid_flag,
         deca_valid_flag, gt_seg) = ret['gt_data'].values()

        if op_valid_flag.sum() == 0:
            logger.warning(f'op_valid_flag is all False, skipping')
            return False

        update_betas_name_cfg()
        logger.info(f'save_betas_name: {smplifyx_cfg.save_betas_name}')
        if smplifyx_cfg.use_pre_compute_betas:
            pre_betas = torch.from_numpy(np.load(smplifyx_cfg.save_betas_name))
            if pre_betas.shape[-1] == smplifyx_cfg.smplx_cfg.num_betas:
                betas.data.copy_(pre_betas)
            else:
                logger.warning(f'pre betas exist but shape error')

        global_step = 0
        wandb_log_dict = {}
        losses_to_log = {}
        op_j_weight = op.get_modify_jt_weight()
        smplifyx_cfg.batch_size = batch_size = op.batch_size
        height, width = op.height, op.width
        center = [width / 2, height / 2]
        mp_gt_lmk_2d = torch.index_select(
            mp_kpts,
            1,  # mp_kpts：torch.Size([192, 478, 2]),
            mp_indices.long().view(-1)  # mp_indices:torch.Size([105])
        ).view(batch_size, -1, 2)

        deca_kpts = deca_kpts[:, :17 + 51, :]
        lmk_gt_outter = deca_kpts[:, :17, :]
        lmk_gt_inner = deca_kpts[:, 17:, :]

        # torch.Size([192, 105])
        lmk_faces_idx = lmk_faces_idx \
            .unsqueeze(dim=0).expand(batch_size, -1).contiguous().long()
        lmk_bary_coords = lmk_bary_coords \
            .unsqueeze(dim=0).repeat(batch_size, 1, 1).float()

        op_gt_conf = (op_kpts_org_data[:, :, -1][..., None])
        op_2dkpts = op_kpts_org_data[:, :, :2]

        meter = MeterBuffer(window_size=6)
        meter.reset()

        pbar = tqdm(position=0,
                    leave=True,
                    bar_format="{percentage:3.0f}%|{bar}{r_bar}{desc}")

        kpts = lmk_gt_outter[deca_valid_flag.bool(), :, :]
        diff_kpts = kpts[1:, :, :] - kpts[:-1, :, :]
        diff_kpts_dist = torch.sqrt(
            torch.square(diff_kpts[:, :, 0]) +
            torch.square(diff_kpts[:, :, 1]))
        body_kpts_diff_sum = diff_kpts_dist.mean()
        body_kpts_diff_sum_threshold = 0.5  # normal: 3.15; lmk_gt_outter
        if body_kpts_diff_sum < body_kpts_diff_sum_threshold:
            logger.warning(
                f'body_kpts_diff_sum: {body_kpts_diff_sum} < {body_kpts_diff_sum_threshold}, skipping'
            )
            SHOW.purge_dir(smplifyx_cfg.ours_images_path)
            SHOW.purge_dir(smplifyx_cfg.ours_pkl_file_path)
            SHOW.purge_dir(smplifyx_cfg.final_losses_json_path)
            SHOW.purge_dir(smplifyx_cfg.output_video_path)
            SHOW.purge_dir(smplifyx_cfg.w_mica_merge_pkl)
            SHOW.purge_dir(smplifyx_cfg.mica_all_dir)
            return False
        else:
            logger.info(
                f'body_kpts_diff_sum: {body_kpts_diff_sum} > {body_kpts_diff_sum_threshold}'
            )

    with default_timers['load_edt']:
        search_tree = None
        filter_faces = None
        pen_distance = None
        if smplifyx_cfg.use_bvh:
            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces

            search_tree = BVH(max_collisions=128)
            pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
                sigma=0.5,
                point2plane=False,
                vectorized=True,
                penalize_outside=True)

            with open(smplifyx_cfg.part_segm_fn, 'rb') as f:
                face_segm_data = pickle.load(f, encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            filter_faces = FilterFaces(faces_segm=faces_segm,
                                       faces_parents=faces_parents,
                                       ign_part_pairs=[
                                           "9,16", "9,17", "6,16", "6,17",
                                           "1,2", "12,22"
                                       ]).to(device=device)

        edt = None
        power = 0.25
        kernel_size = 7
        rasterizer_size = [height / 8, width / 8]
        rasterizer_size = [int(i) for i in rasterizer_size]

        pool = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=1,
                                  padding=(kernel_size // 2))

        def compute_edges(silhouette):
            return pool(silhouette) - silhouette

        if smplifyx_cfg.use_silhouette_loss:
            gt_seg = torch.from_numpy(gt_seg)[None]
            gt_seg = F.interpolate(gt_seg,
                                   size=rasterizer_size,
                                   mode='bilinear',
                                   align_corners=False)

            mask_edge = compute_edges(gt_seg).cpu().numpy()
            edt = distance_transform_edt(1 - (mask_edge > 0))**(power * 2)
            edt = torch.from_numpy(edt).to(device)
            logger.info(f'compute edt finished')

    with default_timers['load_models']:
        body_model = load_smplx_model(dtype=dtype,
                                      batch_size=batch_size,
                                      **smplifyx_cfg.smplx_cfg)
        atach_model_func(body_model)

        vposer = load_vposer_model(device, smplifyx_cfg.vposer_ckpt)
        angle_prior = build_prior(dict(type='SMPLifyAnglePrior',
                                       dtype=dtype)).to(device)

        if betas.shape[-1] > smplifyx_cfg.smplx_cfg.num_betas:
            betas = betas[..., :smplifyx_cfg.smplx_cfg.num_betas]
        if expression.shape[-1] > smplifyx_cfg.smplx_cfg.num_expression_coeffs:
            expression = expression[
                ..., :smplifyx_cfg.smplx_cfg.num_expression_coeffs]
        body_params = dict(
            expression=expression,
            jaw_pose=jaw_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            pose_embedding=vposer.encode(pose.reshape(batch_size, -1)).mean,
            mica_head_transl=mica_head_transl,
        )

        if (smplifyx_cfg.load_checkpoint
                and not Path(smplifyx_cfg.checkpoint_pkl_path).exists()):
            logger.warning(
                f'load_checkpoint is True but checkpoint_pkl_path not exist')

        if (smplifyx_cfg.load_checkpoint
                and Path(smplifyx_cfg.checkpoint_pkl_path).exists()
                and is_valid_json(smplifyx_cfg.checkpoint_json_path)):

            logger.warning(
                f'load_checkpoint: {smplifyx_cfg.checkpoint_pkl_path}')
            load_body_params = load_save_pkl(smplifyx_cfg.checkpoint_pkl_path)

            body_params.update(({
                key: load_body_params[key]
                for key in smplifyx_cfg.basic_param_keys
            }))

            logger.warning(
                f'load_checkpoint and jump to stage {smplifyx_cfg.load_ckpt_st_stage} {smplifyx_cfg.load_ckpt_ed_stage}'
            )

            smplifyx_cfg.start_stage = smplifyx_cfg.load_ckpt_st_stage
            smplifyx_cfg.end_stage = smplifyx_cfg.load_ckpt_ed_stage

        if smplifyx_cfg.re_optim_hands:
            logger.info(f'reload hands from PIXIE/PyMAF-X')
            body_params['left_hand_pose'] = left_hand_pose
            body_params['right_hand_pose'] = right_hand_pose

        body_params = cvt_dict_to_grad(body_params, device, dtype)
        tpose_vertices = get_tpose_vertice(body_model, body_params['betas'])
        body_params['mica_head_transl'].data.copy_(
            cal_smplx_head_transl(tpose_vertices, smplx2flame_idx))

        cam_params = {'focal': op.focal, 'cam_t': op.get_smplx_to_o3d_T()}

        cam_params = cvt_dict_to_grad(cam_params, device, dtype)
        cam_params['focal'].requires_grad = False

        camera_org = PerspectiveCameras(
            device=device,
            focal_length=cam_params['focal'].unsqueeze(0),
            T=cam_params['cam_t'].unsqueeze(0),
            R=torch.Tensor([op.get_smplx_to_o3d_R()]),
            image_size=torch.Tensor([[height, width]]),
            principal_point=torch.Tensor([[width / 2, height / 2]]),
            in_ndc=False)

        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=rasterizer_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            bin_size=0,
        )

        # Silhouette renderer
        renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(
            cameras=camera_org, raster_settings=raster_settings_soft),
                                           shader=SoftSilhouetteShader())

    sil_cnt = 0
    with default_timers['run_optimize']:

        opt_weights_list = opt_weights_list[smplifyx_cfg.
                                            start_stage:smplifyx_cfg.end_stage]

        for stage, curr_weights in enumerate(opt_weights_list):
            if (Path(smplifyx_cfg.checkpoint_pkl_path).exists()
                    and smplifyx_cfg.check_pkl_metric
                    and smplifyx_cfg.load_checkpoint):
                logger.info('jump to stage -1')
                curr_weights = opt_weights_list[-1]
                stage = 2

            curr_weights = EasyDict(curr_weights)

            meter.reset()
            prev_loss = None

            logger.info(f'stage: {stage}')
            for key in body_params.keys():
                body_params[key].requires_grad = bool(
                    curr_weights[f'{key}_en'])
                logger.info(f'{key}:{body_params[key].requires_grad}')

            if smplifyx_cfg.use_pre_compute_betas:
                body_params['betas'].requires_grad = False

            final_params = list(
                filter(lambda x: x.requires_grad, body_params.values()))
            optimizer = SHOW.build_optim(
                dict(params=final_params, **smplifyx_cfg.optimizer_config))
            optimizer.zero_grad()

            def loss_closure_finish_callback(losses, metric):
                nonlocal global_step
                nonlocal wandb_log_dict
                nonlocal losses_to_log
                global_step += 1
                losses_to_log = dict(**losses, **metric)
                log_str = SHOW.utils.print_dict_losses(losses_to_log)

                if (smplifyx_cfg.get('loggers', None)
                        and smplifyx_cfg.get('wandb_prefix', None)):
                    wandb_log_dict = {
                        f'{smplifyx_cfg.wandb_prefix}/{k}': v.item()
                        for k, v in losses_to_log.items()
                    }
                    smplifyx_cfg.loggers.log_bs(wandb_log_dict)
                else:
                    # logger.info(f'{stage}_{global_step}:{log_str}')
                    pbar.set_description(log_str)
                    pbar.update(1)

            meta_data = dict(step=0, pred_edge=None)
            closure = create_closure(
                optimizer,
                vposer,
                body_model,
                body_params,
                camera_org,
                lmk_faces_idx,
                lmk_bary_coords,
                op_2dkpts,
                op_j_weight,
                op_gt_conf,
                op_valid_flag,
                robustifier,
                curr_weights,
                batch_size,
                lmk_gt_inner,
                lmk_gt_outter,
                mp_valid_flag,
                mp_gt_lmk_2d,
                smplifyx_cfg,
                deca_valid_flag,
                height,
                width,
                speaker_shape_vertices,
                smplx2flame_idx,
                face_mask,
                angle_prior,
                device,
                loss_closure_finish_callback,
                renderer_silhouette,
                edt,
                compute_edges,
                meta_data=meta_data,
                search_tree=search_tree,
                filter_faces=filter_faces,
                pen_distance=pen_distance,
            )

            if (Path(smplifyx_cfg.checkpoint_pkl_path).exists()
                    and smplifyx_cfg.load_checkpoint
                    and smplifyx_cfg.check_pkl_metric):
                logger.warning(
                    'pkl exist and we only check th loss the metric,break stages loop'
                )
                all_loss = closure()
                break

            for n in range(smplifyx_cfg.maxiters):
                all_loss = optimizer.step(closure)

                if n > 1 and prev_loss is not None:
                    loss_rel_change = SHOW.utils.rel_change(
                        prev_loss, all_loss.item())
                    meter.update({'rel': loss_rel_change})
                    if meter['rel'].avg <= 1e-09:
                        logger.warning('rel exit')
                        break

                if all([
                        torch.abs(var.grad.view(-1).max()).item() < 1e-06
                        for var in final_params if var.grad is not None
                ]):
                    logger.warning('small grad')
                    break

                if (smplifyx_cfg.use_silhouette_loss
                        and curr_weights.wl_silhouette != 0):
                    sil_cnt += 1
                    if sil_cnt > 3:
                        break

                prev_loss = all_loss.item()

    with default_timers['final_output']:

        scalar_dict = {k: v.item() for k, v in losses_to_log.items()}
        logger.info(
            f'saving final_losses_json_path: {smplifyx_cfg.final_losses_json_path}'
        )
        mmcv.dump(scalar_dict, smplifyx_cfg.final_losses_json_path)
        if not SHOW.is_valid_json(smplifyx_cfg.final_losses_json_path):
            logger.warning(f"is_valid_json, not save, return")
            return False

        model_output, body_pose_axis = cal_model_output(
            vposer, body_model, body_params)
        vertices_ = model_output.vertices.detach().cpu().numpy()

        if smplifyx_cfg.save_template:
            obj_file_path = smplifyx_cfg.ours_output_folder + '/template.obj'
            logger.info(f'saving template obj: {obj_file_path}')
            tpose_vertices = get_tpose_vertice(body_model,
                                               body_params['betas'])
            out_mesh = trimesh.Trimesh(tpose_vertices.detach().cpu().numpy(),
                                       body_model.faces,
                                       process=False)
            out_mesh.export(obj_file_path)

        if (not Path(smplifyx_cfg.save_betas_name).exists()
                and smplifyx_cfg.save_betas):
            logger.info(f'saving betas npy: {smplifyx_cfg.save_betas_name}')
            Path(smplifyx_cfg.save_betas_name).parent.mkdir(parents=True,
                                                            exist_ok=True)
            np.save(smplifyx_cfg.save_betas_name,
                    body_params['betas'].detach().cpu().numpy())

        final_log_str = SHOW.utils.print_dict_losses(losses_to_log)
        logger.info(f'final_log_str:{final_log_str}')

        if (smplifyx_cfg.get('loggers', None)
                and smplifyx_cfg.get('wandb_prefix', None)):
            logger.info(f'logging final metric to server')
            final_metric = {
                f'final_metric/{k}': v
                for k, v in wandb_log_dict.items()
            }
            smplifyx_cfg.loggers.log_bs(final_metric, append=False)

        if (smplifyx_cfg.save_pkl_file or run_optimize_flag):
            logger.info(
                f'saving ours_pkl_file_path: {smplifyx_cfg.ours_pkl_file_path}'
            )
            mmcv.dump([{
                'losses_to_log':
                losses_to_log,
                'width':
                width,
                'height':
                height,
                'center':
                center,
                'batch_size':
                batch_size,
                'camera_transl':
                cam_params['cam_t'].detach().cpu().numpy(),
                'focal_length':
                cam_params['focal'].detach().cpu().numpy(),
                **SHOW.tensor2numpy(body_params),
                'body_pose_axis':
                body_pose_axis.detach().cpu().numpy(),
                'speaker_name':
                smplifyx_cfg.speaker_name,
            }], smplifyx_cfg.ours_pkl_file_path)

        if (smplifyx_cfg.save_ours_images
                or (SHOW.img_files_num_in_dir(smplifyx_cfg.ours_images_path) <
                    SHOW.img_files_num_in_dir(smplifyx_cfg.img_folder))
                or run_optimize_flag):
            logger.info(
                f'saving final images to: {smplifyx_cfg.ours_images_path}')
            from pkl2img import render_pkl_api
            render_pkl_api(**smplifyx_cfg)
