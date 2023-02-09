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

    # 丢进该算法的数据前提：
    # 1、所有的图片都要包含人体
    # 2、第一张图片用于决定重要的参数，比如face id
    # 3、包含一个或者多个人体（多个人的时候而且没有指定speaker的时候的策略？）
    # 4、选取目标的准则：bbox的面积？；同时处理多人？

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

        # 尝试找到图片中的person, note: 该方法只能用于单人
        max_person_crop_im = get_possible_person(poser, template_im)
        if max_person_crop_im is None:
            logger.error(f'max_person_crop_im is None')
            return

        # 当人体box也检测不到人脸的时候，person_face_emb = None
        face_ider_ret = face_ider.get(max_person_crop_im)
        if face_ider_ret is None:
            logger.error(f'face_ider_ret is None')
            ret_dict.person_face_emb = None
            # return
        else:
            cur_speaker_feat = face_ider_ret[0].normed_embedding
            if False:
                # TODO: 使用聚类分析
                is_match_sucess = False
                for ref_shape_id, emb_list in emb_res_factory.items():
                    sim_list = [
                        face_ider.cal_emb_sim(cur_speaker_feat, ref_emb)
                        for ref_emb in emb_list
                    ]
                    if any([sim > 0.6 for sim in sim_list]):
                        logger.info(f'match sucess.')
                        shape_id = ref_shape_id
                        is_match_sucess = True
                        if any([sim < 0.6 for sim in sim_list]):
                            logger.info(f'adding current feat to shape_id: {shape_id}')
                            emb_res_factory[shape_id].append(cur_speaker_feat)
                            emb_res_factory_is_changed_flag = True
                        break
                if not is_match_sucess:
                    shape_id = ''.join(
                        random.sample(string.ascii_letters + string.digits, 12))
                    emb_res_factory[shape_id] = [cur_speaker_feat]
                    emb_res_factory_is_changed_flag = True
                    logger.info(f'match failed, add new: {shape_id}')
                    
            else:
                logger.warning('not using cluster analysis')
                use_direct_face_emb = True
                
                if False:
                    mica_out_ply_f = tempfile.NamedTemporaryFile(suffix='.ply',
                                        prefix='mica_')
                    logger.warning(f'mica_out_ply_f: {mica_out_ply_f.name}')
                    mica_out_ply = mica_out_ply_f.name
                    shape_id=Path(mica_out_ply).stem
                    
                    mica_out_img = None
                    mica_out_npy = None
                else:
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


    # FLAME的exp转SMPLX的矩阵
    flame2020to2019_exp_trafo = './flame2020to2019_exp_trafo.npy'
    flame2020to2019_exp_trafo = os.path.abspath(
        os.path.join(assets_root, flame2020to2019_exp_trafo))
    flame2020to2019_exp_trafo = np.load(flame2020to2019_exp_trafo)
    flame2020to2019_exp_trafo = torch.from_numpy(flame2020to2019_exp_trafo).to(
        device).to(dtype)
    ret_dict.flame2020to2019_exp_trafo = flame2020to2019_exp_trafo

    # mediapipe的landmark转smplx的index
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

    # FLAME的各种mask
    FLAME_masks = './FLAME_masks.pkl'
    FLAME_masks = os.path.abspath(os.path.join(assets_root, FLAME_masks))
    with open(FLAME_masks, 'rb') as f:
        FLAME_masks = pickle.load(f, encoding='latin1')
    for key in FLAME_masks.keys():
        FLAME_masks[key] = torch.from_numpy(
            FLAME_masks[key]).to(device).to(dtype)
    ret_dict.FLAME_masks = FLAME_masks

    # SMPLX转FLAME的index
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

    ref_tex = r'./002.flame'
    ref_tex = os.path.abspath(os.path.join(assets_root, ref_tex))
    ref_tex = torch.load(ref_tex)
    ret_dict.ref_tex = ref_tex

    return ret_dict
