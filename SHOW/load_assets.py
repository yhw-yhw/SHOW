import sys
import os
import pickle
import torch
import numpy as np
from plyfile import PlyData
from easydict import EasyDict
from pathlib import Path
import random
import string
import mmcv
from loguru import logger
from modules.MICA.api_MICA import api_MICA
from SHOW.video_filter.MMposer import MMPoseAnalyzer
import cv2
import SHOW
import tempfile


def get_possible_person(poser, template_im):

    def is_small_person(bbox, org_im: np.ndarray):
        img_height, img_width, _ = org_im.shape
        box_height = bbox[3] - bbox[1]
        box_width = bbox[2] - bbox[0]
        is_small_person = 0
        if ((box_height / img_height) < 0.4 or (box_width / img_width) < 0.3):
            is_small_person = 1
        return is_small_person

    def is_kpts_whole(kpts):
        is_whole = 1
        poit_to_check = list(range(0, 13))
        for idx in poit_to_check:
            if kpts[idx][-1] < 0.3:
                is_whole = 0
                break
        return is_whole

    if isinstance(template_im, str):
        org_im = cv2.imread(template_im)
    else:
        org_im = template_im

    pose_results = poser.predict(template_im)

    #################
    logger.info(f'mmpose det length before: {len(pose_results)}')
    # pose_results = [i for i in pose_results if i['bbox'][-1]>0.3]
    # pose_results = [i for i in pose_results if is_small_person(i['bbox'],org_im)]
    pose_results = [i for i in pose_results if is_kpts_whole(i['keypoints'])]
    if len(pose_results) == 0:
        logger.info(f'no whole person detected')
        return None
    logger.info(f'mmpose det length after: {len(pose_results)}')
    #################

    for idx in range(len(pose_results)):
        bbox = pose_results[idx]['bbox']
        box_height = bbox[3] - bbox[1]
        box_width = bbox[2] - bbox[0]
        pose_results[idx]['size'] = box_height * box_width

    pose_results.sort(key=lambda x: x['size'], reverse=True)
    pose_results_size_list = [i['size'] for i in pose_results]
    logger.info(f'pose_results_size_list: {pose_results_size_list}')

    max_ret = pose_results[0]
    left, top, right, bottom = [int(i) for i in max_ret['bbox'][:4]]
    max_person_crop_im = org_im[top:bottom, left:right, :]
    logger.info(
        f'cropped image from left:{left},top:{top},right:{right},bottom:{bottom}'
    )
    #################

    return max_person_crop_im


def read_shape(speaker_ply_file_path):
    # return： (5023, 3)
    ply_data = PlyData.read(speaker_ply_file_path)
    speaker_shape = np.stack([
        ply_data['vertex']['x'], ply_data['vertex']['y'],
        ply_data['vertex']['z']
    ], 1)
    return speaker_shape


def load_assets(config, face_ider=None, template_im=None, **kwargs):

    assets_root = config.assets_root
    dtype = config.dtype
    device = config.device
    ret_dict = EasyDict({})
    
    shape_res_factory_dir = f'{assets_root}/id_pic/shape_factory2'
    emb_res_factory_dir = f'{assets_root}/id_pic/{config.ider_cfg.npy_folder_name}'
    emb_res_factory_path = f'{assets_root}/id_pic/emb_factory2.pkl'

    emb_res_factory_is_changed_flag = False
    if Path(emb_res_factory_path).exists():
        emb_res_factory = mmcv.load(emb_res_factory_path)
    else:
        emb_res_factory = {'0' * 12: [np.zeros((512, ))]}
        emb_res_factory_is_changed_flag = True

    use_direct_face_emb = False
    max_person_crop_im = None
    shape_id = config.speaker_name
    logger.info(f'shape_id/speaker_name: {shape_id}')
    registered_id = ['oliver', 'seth', 'conan', 'chemistry']

    if shape_id not in registered_id:
        logger.info(f'current shape_id: {shape_id}')

        poser = MMPoseAnalyzer()
        config.load_betas = False
        config.save_betas = False

        max_person_crop_im = get_possible_person(poser, template_im)
        if max_person_crop_im is None:
            logger.error(f'max_person_crop_im is None')
            return

        face_ider_ret = face_ider.get(max_person_crop_im)
        if face_ider_ret is None:
            logger.error(f'face_ider_ret is None')
            ret_dict.person_face_emb = None
        else:
            cur_speaker_feat = face_ider_ret[0].normed_embedding
            use_direct_face_emb = True
        
            shape_id=os.urandom(24).hex()
            mica_out_ply = os.path.join(tempfile.gettempdir(), f'{shape_id}.ply')
            
            mica_out_img = None
            mica_out_npy = None
            
            ret_dict.person_face_emb = cur_speaker_feat
            
        del poser

    else:
        person_face_emb_path = os.path.abspath(
            os.path.join(assets_root, emb_res_factory_dir, f'{shape_id}.npy'))

        if Path(person_face_emb_path).exists():
            ret_dict.person_face_emb = np.load(person_face_emb_path)
        else:
            ret_dict.person_face_emb = None

        logger.info(f'loaded specific speaker: {shape_id}')
        mica_out_ply = os.path.join(shape_res_factory_dir, shape_id,
                                    'out.ply')
        mica_out_img = os.path.join(shape_res_factory_dir, shape_id,
                                        'out.jpg')
        mica_out_npy = os.path.join(shape_res_factory_dir, shape_id,
                                        'out.npy')
        
    #######################################save pkl
    if emb_res_factory_is_changed_flag:
        logger.info(f'saving emb_res_factory...')
        mmcv.dump(emb_res_factory, emb_res_factory_path)


    #######################################run MICA
    if ret_dict.person_face_emb is not None:
        if not Path(mica_out_ply).exists():
            mica_ins = api_MICA()
            mica_ins.predict(
                input_img_path=max_person_crop_im
                if max_person_crop_im is not None else template_im,
                output_ply_path=mica_out_ply,
                output_render_path=mica_out_img,
                output_param_npy_path=mica_out_npy
            )
            del mica_ins

        if Path(mica_out_ply).exists():
            ret_dict.speaker_shape_vertices = torch.from_numpy(
                read_shape(mica_out_ply)).to(device).to(dtype)
        else:
            ret_dict.speaker_shape_vertices = None

        if use_direct_face_emb:
            # mica_out_ply_f.close()
            if os.path.exists(mica_out_ply):
                os.remove(mica_out_ply)
    else:
        ret_dict.speaker_shape_vertices = None

    #######################################others
    config.speaker_name = shape_id
    logger.info(f'shape_id/speaker_name: {shape_id}')


    flame2020to2019_exp_trafo = './flame2020to2019_exp_trafo.npy'
    flame2020to2019_exp_trafo = os.path.abspath(
        os.path.join(assets_root, flame2020to2019_exp_trafo))
    flame2020to2019_exp_trafo = np.load(flame2020to2019_exp_trafo)
    flame2020to2019_exp_trafo = torch.from_numpy(flame2020to2019_exp_trafo).to(
        device).to(dtype)
    ret_dict.flame2020to2019_exp_trafo = flame2020to2019_exp_trafo

    mediapipe_landmark_embedding = './mediapipe_landmark_embedding__smplx.npz'
    mediapipe_landmark_embedding = os.path.abspath(
        os.path.join(assets_root, mediapipe_landmark_embedding))
    mp_lmk_emb = {}
    mediapipe_landmark_embedding = np.load(mediapipe_landmark_embedding)
    mp_lmk_emb['lmk_face_idx'] = torch.from_numpy(
        mediapipe_landmark_embedding['lmk_face_idx'].astype(int)).to(
            device).to(dtype).long()
    mp_lmk_emb['lmk_b_coords'] = torch.from_numpy(
        mediapipe_landmark_embedding['lmk_b_coords'].astype(float)).to(
            device).to(dtype)
    mp_lmk_emb['landmark_indices'] = torch.from_numpy(
        mediapipe_landmark_embedding['landmark_indices'].astype(int)).to(
            device).to(dtype)
    ret_dict.mp_lmk_emb = mp_lmk_emb

    FLAME_masks = './FLAME_masks.pkl'
    FLAME_masks = os.path.abspath(os.path.join(assets_root, FLAME_masks))
    with open(FLAME_masks, 'rb') as f:
        FLAME_masks = pickle.load(f, encoding='latin1')
    for key in FLAME_masks.keys():
        FLAME_masks[key] = torch.from_numpy(
            FLAME_masks[key]).to(device).to(dtype)
    ret_dict.FLAME_masks = FLAME_masks

    smplx2flame_idx = './SMPL-X__FLAME_vertex_ids.npy'
    smplx2flame_idx = os.path.abspath(
        os.path.join(assets_root, smplx2flame_idx))
    smplx2flame_idx = np.load(smplx2flame_idx)
    smplx2flame_idx = torch.from_numpy(smplx2flame_idx).to(device).to(
        dtype).long()
    ret_dict.smplx2flame_idx = smplx2flame_idx

    vertex_colors = r'./smplx_verts_colors.txt'
    vertex_colors = os.path.abspath(os.path.join(assets_root, vertex_colors))
    vertex_colors = np.loadtxt(vertex_colors)
    ret_dict.vertex_colors = vertex_colors

    return ret_dict
