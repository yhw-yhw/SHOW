import smplx
from human_body_prior.tools.model_loader import load_vposer
import torch
import torch.nn as nn
import os
import pickle
import os.path as osp
from .datasets import op_dataset
import numpy as np
import mmcv

DEFAULT_SMPLX_CONFIG2 = dict(
    dtype=torch.float32,
    num_betas=200,
    num_expression_coeffs=50,
    num_pca_comps=12,
    flat_hand_mean=True,
    use_pca=True,
    model_type='smplx',
    use_face_contour=True,
)

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


class JointMapper(nn.Module):

    def __init__(self, joint_maps=None):
        super().__init__()
        self.register_buffer('joint_maps',
                             torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        return torch.index_select(joints, 1, self.joint_maps)


def load_save_pkl(ours_pkl_file_path, device='cuda'):
    data = mmcv.load(ours_pkl_file_path)[0]

    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            data[key] = torch.from_numpy(data[key]).to(device)
    data['batch_size'] = data['expression'].shape[0]

    return data


def load_smplx_model(device='cuda', **kwargs):
    body_model = smplx.create(joint_mapper=JointMapper(
        op_dataset.smpl_to_openpose()),
                              **DEFAULT_SMPLX_CONFIG,
                              **kwargs).to(device=device)
    return body_model


def load_vposer_model(device='cuda', vposer_ckpt=''):
    vposer_ckpt = osp.expandvars(vposer_ckpt)
    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
    vposer = vposer.to(device=device)
    vposer.eval()
    return vposer