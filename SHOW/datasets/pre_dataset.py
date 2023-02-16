from scipy.io import savemat, loadmat
from collections import namedtuple
import numpy as np
import json
import torch
import glob
import cv2
import os.path as osp
import os
import pickle
from ..image import lmk2d_to_bbox
import mmcv

deca_exp_to_smplx_X = np.load(
    osp.join(os.path.dirname(__file__),
             '../../../data/flame2020to2019_exp_trafo.npy')
)


def deca_exp_to_smplx(e_deca):
    e_deca = np.concatenate([e_deca, np.zeros(50)])
    e_smplx = deca_exp_to_smplx_X.dot(e_deca)
    e_smplx = e_smplx[:50]
    return e_smplx


def read_mp(mp_npz_file, height, width):
    try:
        mp_npz = np.load(mp_npz_file, allow_pickle=True)
    except:
        import traceback
        traceback.print_exc()
        return None

    mp_npz = list(mp_npz.values())
    return_list = []
    for ret in mp_npz:
        # (478,2)
        return_list.append({
            'lmk2d': ret[0],
        })

    return return_list


def read_pixie(pixie_mat_file, height, width, cvt_hand_func=None):
    pixie_ret_list = mmcv.load(pixie_mat_file)
    
    assert cvt_hand_func, 'cvt_hand_func must set'

    return_list = []
    for ret in pixie_ret_list:
        for key, val in ret.items():
            if isinstance(val, np.ndarray) and val.shape[0] == 1:
                ret[key] = ret[key][0]

        face_bbox = lmk2d_to_bbox(ret['face_kpt'], height, width)
        
        if 1:
            lhand, rhand = cvt_hand_func(
                ret['left_hand_pose'],
                ret['right_hand_pose'],
            )
            ret['left_hand_pose'] = lhand
            ret['right_hand_pose'] = rhand
        
        ret['face_bbox'] = face_bbox

        return_list.append(ret)

    return return_list


def read_deca(deca_mat_file):
    assert(osp.exists(deca_mat_file))

    deca_ret_list = mmcv.load(deca_mat_file)

    assert(deca_ret_list != [])
    return_list = []

    for ret in deca_ret_list:
        for key, val in ret.items():
            if isinstance(val, np.ndarray) and val.shape[0] == 1:
                ret[key] = ret[key][0]

        deca_lmk = torch.tensor(ret['landmarks2d'])
        org2deca_tform = torch.tensor(ret['tform'])

        # deca_lmk=deca_lmk[:, :, :2]
        deca_lmk = deca_lmk[None, ...][:, :, :2]
        deca_lmk[:, :, 0] = (deca_lmk[:, :, 0] + 1) * 0.5 * 224
        deca_lmk[:, :, 1] = (deca_lmk[:, :, 1] + 1) * 0.5 * 224

        tform_T = torch.inverse(org2deca_tform[None, ...]).transpose(1, 2)
        bs, n_points, _ = deca_lmk.shape
        tmp_one = torch.ones(
            [bs, n_points, 1], device=deca_lmk.device, dtype=deca_lmk.dtype)
        deca_lmk = torch.cat([deca_lmk, tmp_one], dim=-1)
        org_deca_lmk = torch.bmm(deca_lmk, tform_T)[:, :, :2]

        smplx_exp = deca_exp_to_smplx(ret['expression_params'])

        return_list.append({
            'face_bbox': ret['bbox'],  # array([774., 177., 969., 372.])
            'lmk2d': org_deca_lmk,  # torch.Size([1, 68, 2])
            'exp': smplx_exp,  # shape:(50,)
            'jaw': ret['pose_params'][3:],
        })

    return return_list
