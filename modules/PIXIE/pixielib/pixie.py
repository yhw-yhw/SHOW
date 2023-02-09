# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at pixie@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread
import cv2

from .models.encoders import ResnetEncoder, MLP, HRNEncoder
from .models.moderators import TempSoftmaxFusion
from .models.FLAME import FLAMETex
from .models.SMPLX import SMPLX
from .utils import util
from .utils.util import perspective_projection, estimate_translation_np, dict_tensor2npy
from .utils import rotation_converter as converter
from .utils import tensor_cropper
from .utils.config import cfg
from scipy.io import savemat, loadmat

# from test4 import


class PIXIE(object):
    def __init__(self, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device
        # parameters setting
        self.param_list_dict = {}
        for lst in self.cfg.params.keys():
            param_list = cfg.params.get(lst)
            self.param_list_dict[lst] = {
                i: cfg.model.get('n_'+i) for i in param_list}

        # Build the models
        self._create_model()
        # Set up the cropping modules used to generate face/hand crops from the body predictions
        self._setup_cropper()

    def _setup_cropper(self):
        self.Cropper = {}
        for crop_part in ['head', 'hand']:
            data_cfg = self.cfg.dataset[crop_part]
            scale_size = (data_cfg.scale_min + data_cfg.scale_max)*0.5
            self.Cropper[crop_part] = tensor_cropper.Cropper(
                crop_size=data_cfg.image_size,
                scale=[scale_size, scale_size],
                trans_scale=0)

    def _create_model(self):
        self.model_dict = {}
        # Build all image encoders
        # Hand encoder only works for right hand, for left hand, flip inputs and flip the results back
        self.Encoder = {}
        for key in self.cfg.network.encoder.keys():
            if self.cfg.network.encoder.get(key).type == 'resnet50':
                self.Encoder[key] = ResnetEncoder().to(self.device)
            elif self.cfg.network.encoder.get(key).type == 'hrnet':
                self.Encoder[key] = HRNEncoder().to(self.device)
            self.model_dict[f'Encoder_{key}'] = self.Encoder[key].state_dict()

        # Build the parameter regressors
        self.Regressor = {}
        for key in self.cfg.network.regressor.keys():
            n_output = sum(self.param_list_dict[f'{key}_list'].values())
            channels = [2048] + \
                self.cfg.network.regressor.get(key).channels + [n_output]
            if self.cfg.network.regressor.get(key).type == 'mlp':
                self.Regressor[key] = MLP(channels=channels).to(self.device)
            self.model_dict[f'Regressor_{key}'] = self.Regressor[key].state_dict(
            )

        # Build the extractors
        # to extract separate head/left hand/right hand feature from body feature
        self.Extractor = {}
        for key in self.cfg.network.extractor.keys():
            channels = [2048] + \
                self.cfg.network.extractor.get(key).channels + [2048]
            if self.cfg.network.extractor.get(key).type == 'mlp':
                self.Extractor[key] = MLP(channels=channels).to(self.device)
            self.model_dict[f'Extractor_{key}'] = self.Extractor[key].state_dict(
            )

        # Build the moderators
        self.Moderator = {}
        for key in self.cfg.network.moderator.keys():
            share_part = key.split('_')[0]
            detach_inputs = self.cfg.network.moderator.get(key).detach_inputs
            detach_feature = self.cfg.network.moderator.get(key).detach_feature
            channels = [2048*2] + \
                self.cfg.network.moderator.get(key).channels + [2]
            self.Moderator[key] = TempSoftmaxFusion(
                detach_inputs=detach_inputs, detach_feature=detach_feature,
                channels=channels).to(self.device)
            self.model_dict[f'Moderator_{key}'] = self.Moderator[key].state_dict(
            )

        class JointMapper(nn.Module):
            def __init__(self, joint_maps=None):
                super(JointMapper, self).__init__()
                if joint_maps is None:
                    self.joint_maps = joint_maps
                else:
                    self.register_buffer('joint_maps',
                                         torch.tensor(joint_maps, dtype=torch.long))

            def forward(self, joints, **kwargs):
                if self.joint_maps is None:
                    return joints
                else:
                    return torch.index_select(joints, 1, self.joint_maps)
        # import smplx
        # smplx_model_path = r'C:\Users\lithiumice\code\smplify-x\models'
        # joint_mapper = JointMapper(
        #     torch.tensor(
        #         [55,  12,  17,  19,  21,  16,  18,  20,   0,   2,   5,   8,   1,   4,
        #          7,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  20,  37,  38,
        #          39,  66,  25,  26,  27,  67,  28,  29,  30,  68,  34,  35,  36,  69,
        #          31,  32,  33,  70,  21,  52,  53,  54,  71,  40,  41,  42,  72,  43,
        #          44,  45,  73,  49,  50,  51,  74,  46,  47,  48,  75,  76,  77,  78,
        #          79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
        #          93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
        #          107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        #          121, 122, 123, 124, 125, 126]))
        # model_params = dict(model_path=smplx_model_path,
        #                     # joint_mapper=joint_mapper,
        #                     create_global_orient=True,
        #                     create_body_pose=False,
        #                     create_betas=True,
        #                     create_left_hand_pose=True,
        #                     create_right_hand_pose=True,
        #                     create_expression=True,
        #                     create_jaw_pose=True,
        #                     create_leye_pose=True,
        #                     create_reye_pose=True,
        #                     create_transl=False,
        #                     dtype=torch.float64,
        #                     use_pca=False,
        #                     use_face_contour=True,
        #                     model_type='smplx')

        # # male_model = smplx.create(gender='male', **model_params)
        # # # SMPL-H has no gender-neutral model
        # # if args.get('model_type') != 'smplh':
        # self.neutral_model = smplx.create(gender='neutral', **model_params)
        # # female_model = smplx.create(gender='female', **model_params)

        # Build the SMPL-X body model, which we also use to represent faces and
        # hands, using the relevant parts only
        self.smplx = SMPLX(self.cfg.model).to(self.device)
        self.part_indices = self.smplx.part_indices
        # Build the FLAME texture space
        if self.cfg.model.use_tex:
            self.flametex = FLAMETex(self.cfg.model).to(self.device)

        # -- resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            for key in self.model_dict.keys():
                util.copy_state_dict(self.model_dict[key], checkpoint[key])
        else:
            print(f'pixie trained model path: {model_path} does not exist!')
            exit()
        # eval mode
        for module in [self.Encoder, self.Regressor, self.Moderator, self.Extractor]:
            for net in module.values():
                net.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
        return code_dict

    def part_from_body(self, image, part_key, points_dict, crop_joints=None):
        ''' crop part(head/left_hand/right_hand) out from body data, joints also change accordingly
        '''
        assert part_key in ['head', 'left_hand', 'right_hand']
        assert 'smplx_kpt' in points_dict.keys()
        if part_key == 'head':
            # use face 68 kpts for cropping head image
            indices_key = 'face'
        elif part_key == 'left_hand':
            indices_key = 'left_hand'
        elif part_key == 'right_hand':
            indices_key = 'right_hand'

        # get points for cropping
        part_indices = self.part_indices[indices_key]
        if crop_joints is not None:
            points_for_crop = crop_joints[:, part_indices]
        else:
            points_for_crop = points_dict['smplx_kpt'][:, part_indices]

        # crop
        cropper_key = 'hand' if 'hand' in part_key else part_key
        points_scale = image.shape[-2:]
        cropped_image, tform = self.Cropper[cropper_key].crop(
            image,
            points_for_crop,
            points_scale
        )
        # transform points(must be normalized to [-1.1]) accordingly
        cropped_points_dict = {}
        for points_key in points_dict.keys():
            points = points_dict[points_key]
            cropped_points = self.Cropper[cropper_key].transform_points(
                points, tform, points_scale, normalize=True)
            cropped_points_dict[points_key] = cropped_points
        return cropped_image, cropped_points_dict

    @torch.no_grad()
    def encode(self, data, threthold=True, keep_local=True, copy_and_paste=False, body_only=False):
        ''' Encode images to smplx parameters
        Args:
            data: dict
                key: image_type (body/head/hand)
                value: 
                    image: [bz, 3, 224, 224], range [0,1]
                    image_hd(needed if key==body): a high res version of image, only for cropping parts from body image
                    head_image: optinal, well-cropped head from body image
                    left_hand_image: optinal, well-cropped left hand from body image
                    right_hand_image: optinal, well-cropped right hand from body image
        Returns:
            param_dict: dict
                key: image_type (body/head/hand)
                value: param_dict
        '''
        for key in data.keys():
            assert key in ['body', 'head', 'hand']

        feature = {}
        param_dict = {}

        # Encode features
        for key in data.keys():
            part = key
            # encode feature
            feature[key] = {}
            feature[key][part] = self.Encoder[part](data[key]['image'])

            # for head/hand image
            if key == 'head' or key == 'hand':
                # predict head/hand-only parameters from part feature
                part_dict = self.decompose_code(self.Regressor[part](
                    feature[key][part]), self.param_list_dict[f'{part}_list'])
                # if input is part data, skip feature fusion: share feature is the same as part feature
                # then predict share parameters
                feature[key][f'{key}_share'] = feature[key][key]
                share_dict = self.decompose_code(
                    self.Regressor[f'{part}_share'](
                        feature[key][f'{part}_share']),
                    self.param_list_dict[f'{part}_share_list'])
                # compose parameters
                param_dict[key] = {**share_dict, **part_dict}

            # for body image
            if key == 'body':
                fusion_weight = {}
                f_body = feature['body']['body']
                # extract part feature
                for part_name in ['head', 'left_hand', 'right_hand']:
                    feature['body'][f'{part_name}_share'] = self.Extractor[f'{part_name}_share'](
                        f_body)

                # -- check if part crops are given, if not, crop parts by coarse body estimation
                if 'head_image' not in data[key].keys() \
                        or 'left_hand_image' not in data[key].keys() \
                        or 'right_hand_image' not in data[key].keys():
                    # - run without fusion to get coarse estimation, for cropping parts
                    # body only
                    body_dict = self.decompose_code(self.Regressor[part](
                        feature[key][part]), self.param_list_dict[part+'_list'])
                    # head share
                    head_share_dict = self.decompose_code(self.Regressor['head'+'_share'](
                        feature[key]['head'+'_share']), self.param_list_dict['head'+'_share_list'])
                    # right hand share
                    right_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                        feature[key]['right_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                    # left hand share
                    left_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                        feature[key]['left_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                    # change the dict name from right to left
                    left_hand_share_dict['left_hand_pose'] = left_hand_share_dict.pop(
                        'right_hand_pose')
                    left_hand_share_dict['left_wrist_pose'] = left_hand_share_dict.pop(
                        'right_wrist_pose')
                    param_dict[key] = {**body_dict, **head_share_dict,
                                       **left_hand_share_dict, **right_hand_share_dict}
                    if body_only:
                        param_dict['moderator_weight'] = None
                        return param_dict
                    prediction_body_only = self.decode(
                        param_dict[key], param_type='body')
                    # crop
                    for part_name in ['head', 'left_hand', 'right_hand']:
                        part = part_name.split('_')[-1]
                        points_dict = {
                            'smplx_kpt': prediction_body_only['smplx_kpt'],
                            'trans_verts': prediction_body_only['transformed_vertices']
                        }
                        cropped_image, cropped_joints_dict = self.part_from_body(
                            data['body']['image_hd'], part_name, points_dict)
                        data[key][part_name+'_image'] = cropped_image

                # os.makedirs('part_images2',exist_ok=True)
                # sub_dir_name=os.path.join('part_images2',data['body']['name'])
                # os.makedirs(sub_dir_name,exist_ok=True)

                # for part_name in ['head', 'left_hand', 'right_hand']:
                #     part = part_name.split('_')[-1]
                #     cropped_image = data[key][part_name+'_image']
                #     tmp=cropped_image[0].cpu().numpy()
                #     tmp=tmp.transpose(1,2,0)
                #     tmp=tmp*255
                #     tmp=cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)
                #     cv2.imwrite(os.path.join(sub_dir_name,part_name+'.jpg'),tmp)

                # -- encode features from part crops, then fuse feature using the weight from moderator
                for part_name in ['head', 'left_hand', 'right_hand']:
                    part = part_name.split('_')[-1]
                    cropped_image = data[key][part_name+'_image']
                    # if left hand, flip it as if it is right hand
                    if part_name == 'left_hand':
                        cropped_image = torch.flip(cropped_image, dims=(-1,))
                    # run part regressor
                    f_part = self.Encoder[part](cropped_image)
                    part_dict = self.decompose_code(self.Regressor[part](
                        f_part), self.param_list_dict[f'{part}_list'])
                    part_share_dict = self.decompose_code(self.Regressor[f'{part}_share'](
                        f_part), self.param_list_dict[f'{part}_share_list'])
                    param_dict['body_' +
                               part_name] = {**part_dict, **part_share_dict}

                    # moderator to assign weight, then integrate features
                    f_body_out, f_part_out, f_weight = self.Moderator[f'{part}_share'](
                        feature['body'][f'{part_name}_share'], f_part, work=True)
                    if copy_and_paste:
                        # copy and paste strategy always trusts the results from part
                        feature['body'][f'{part_name}_share'] = f_part
                    elif threthold and part == 'hand':
                        # for hand, if part weight > 0.7 (very confident, then fully trust part)
                        part_w = f_weight[:, [1]]
                        part_w[part_w > 0.7] = 1.
                        f_body_out = feature['body'][f'{part_name}_share']*(
                            1. - part_w) + f_part*part_w
                        feature['body'][f'{part_name}_share'] = f_body_out
                    else:
                        feature['body'][f'{part_name}_share'] = f_body_out
                    fusion_weight[part_name] = f_weight
                # save weights from moderator, that can be further used for optimization/running specific tasks on parts
                param_dict['moderator_weight'] = fusion_weight

                # -- predict parameters from fused body feature
                # head share
                head_share_dict = self.decompose_code(self.Regressor['head'+'_share'](
                    feature[key]['head'+'_share']), self.param_list_dict['head'+'_share_list'])
                # right hand share
                right_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                    feature[key]['right_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                # left hand share
                left_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                    feature[key]['left_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                # change the dict name from right to left
                left_hand_share_dict['left_hand_pose'] = left_hand_share_dict.pop(
                    'right_hand_pose')
                left_hand_share_dict['left_wrist_pose'] = left_hand_share_dict.pop(
                    'right_wrist_pose')
                param_dict['body'] = {
                    **body_dict, **head_share_dict, **left_hand_share_dict, **right_hand_share_dict}
                # copy tex param from head param dict to body param dict
                param_dict['body']['tex'] = param_dict['body_head']['tex']
                param_dict['body']['light'] = param_dict['body_head']['light']

                if keep_local:
                    # for local change that will not affect whole body and produce unnatral pose, trust part
                    param_dict[key]['exp'] = param_dict['body_head']['exp']
                    param_dict[key]['right_hand_pose'] = param_dict['body_right_hand']['right_hand_pose']
                    param_dict[key]['left_hand_pose'] = param_dict['body_left_hand']['right_hand_pose']

        return param_dict

    def convert_pose(self, param_dict, param_type):
        ''' Convert pose parameters to rotation matrix
        Args:
            param_dict: smplx parameters
            param_type: should be one of body/head/hand
        Returns:
            param_dict: smplx parameters 
        '''
        assert param_type in ['body', 'head', 'hand']

        # convert pose representations: the output from network are continous repre or axis angle,
        # while the input pose for smplx need to be rotation matrix
        for key in param_dict:
            if "pose" in key and 'jaw' not in key:
                param_dict[key] = converter.batch_cont2matrix(param_dict[key])
        if param_type == 'body' or param_type == 'head':
            param_dict['jaw_pose'] = converter.batch_euler2matrix(param_dict['jaw_pose'])[
                :, None, :, :]

        # complement params if it's not in given param dict
        if param_type == 'head':
            batch_size = param_dict['shape'].shape[0]
            param_dict['abs_head_pose'] = param_dict['head_pose'].clone()
            param_dict['global_pose'] = param_dict['head_pose']
            param_dict['partbody_pose'] = self.smplx.body_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1)[:, :self.param_list_dict['body_list']['partbody_pose']]
            param_dict['neck_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['left_wrist_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['left_hand_pose'] = self.smplx.left_hand_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['right_wrist_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['right_hand_pose'] = self.smplx.right_hand_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
        elif param_type == 'hand':
            batch_size = param_dict['right_hand_pose'].shape[0]
            param_dict['abs_right_wrist_pose'] = param_dict['right_wrist_pose'].clone()
            dtype = param_dict['right_hand_pose'].dtype
            device = param_dict['right_hand_pose'].device
            x_180_pose = torch.eye(
                3, dtype=dtype, device=device).unsqueeze(0).repeat(1, 1, 1)
            x_180_pose[0, 2, 2] = -1.
            x_180_pose[0, 1, 1] = -1.
            param_dict['global_pose'] = x_180_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['shape'] = self.smplx.shape_params.expand(
                batch_size, -1)
            param_dict['exp'] = self.smplx.expression_params.expand(
                batch_size, -1)
            param_dict['head_pose'] = self.smplx.head_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['neck_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['jaw_pose'] = self.smplx.jaw_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['partbody_pose'] = self.smplx.body_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1)[:, :self.param_list_dict['body_list']['partbody_pose']]
            param_dict['left_wrist_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['left_hand_pose'] = self.smplx.left_hand_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
        elif param_type == 'body':
            # the predcition from the head and hand share regressor is always absolute pose
            batch_size = param_dict['shape'].shape[0]
            param_dict['abs_head_pose'] = param_dict['head_pose'].clone()
            param_dict['abs_right_wrist_pose'] = param_dict['right_wrist_pose'].clone()
            param_dict['abs_left_wrist_pose'] = param_dict['left_wrist_pose'].clone()
            # the body-hand share regressor is working for right hand
            # so we assume body network get the flipped feature for the left hand. then get the parameters
            # then we need to flip it back to left, which matches the input left hand
            param_dict['left_wrist_pose'] = util.flip_pose(
                param_dict['left_wrist_pose'])
            param_dict['left_hand_pose'] = util.flip_pose(
                param_dict['left_hand_pose'])
        else:
            exit()

        return param_dict

    def decode(self, param_dict, param_type, extra=None,**kwargs):
        ''' Decode model parameters to smplx vertices & joints & texture
        Args:
            param_dict: smplx parameters
            param_type: should be one of body/head/hand
        Returns:
            predictions: smplx predictions
        '''
        if 'jaw_pose' in param_dict.keys() and len(param_dict['jaw_pose'].shape) == 2:
            self.convert_pose(param_dict, param_type)
        elif param_dict['right_wrist_pose'].shape[-1] == 6:
            self.convert_pose(param_dict, param_type)

        # concatenate body pose
        partbody_pose = param_dict['partbody_pose']
        param_dict['body_pose'] = torch.cat(
            [partbody_pose[:, :11],
             param_dict['neck_pose'],
             partbody_pose[:, 11:11+2],
             param_dict['head_pose'],
             partbody_pose[:, 13:13+4],
             param_dict['left_wrist_pose'],
             param_dict['right_wrist_pose']], dim=1)

        # change absolute head&hand pose to relative pose according to rest body pose
        if param_type == 'head' or param_type == 'body':
            param_dict['body_pose'] = self.smplx.pose_abs2rel(
                param_dict['global_pose'],
                param_dict['body_pose'],
                abs_joint='head')
        if param_type == 'hand' or param_type == 'body':
            param_dict['body_pose'] = self.smplx.pose_abs2rel(
                param_dict['global_pose'],
                param_dict['body_pose'],
                abs_joint='left_wrist')
            param_dict['body_pose'] = self.smplx.pose_abs2rel(
                param_dict['global_pose'],
                param_dict['body_pose'],
                abs_joint='right_wrist')

        if self.cfg.model.check_pose:
            # check if pose is natural (relative rotation), if not, set relative to 0 (especially for head pose)
            # xyz: pitch(positive for looking down), yaw(positive for looking left), roll(rolling chin to left)
            for pose_ind in [14]:  # head [15-1, 20-1, 21-1]:
                curr_pose = param_dict['body_pose'][:, pose_ind]
                euler_pose = converter._compute_euler_from_matrix(curr_pose)
                for i, max_angle in enumerate([20, 70, 10]):
                    euler_pose_curr = euler_pose[:, i]
                    euler_pose_curr[
                        euler_pose_curr !=
                        torch.clamp(
                            euler_pose_curr,
                            min=-max_angle*np.pi/180,
                            max=max_angle*np.pi/180)
                    ] = 0.
                param_dict['body_pose'][:,
                                        pose_ind] = converter.batch_euler2matrix(euler_pose)

        # SMPLX
        # joints: 145，在crop image下的3d坐标
        # verts：torch.Size([1, 10475, 3])
        # landmarks：68
        verts, landmarks, joints = self.smplx(
            # torch.Size([1, 200])
            shape_params=param_dict['shape'],
            # torch.Size([1, 50])
            expression_params=param_dict['exp'],
            # torch.Size([1, 1, 3, 3])
            global_pose=param_dict['global_pose'],
            # torch.Size([1, 21, 3, 3])
            body_pose=param_dict['body_pose'],
            # torch.Size([1, 1, 3, 3])
            jaw_pose=param_dict['jaw_pose'],
            # torch.Size([1, 15, 3, 3])
            left_hand_pose=param_dict['left_hand_pose'],
            # torch.Size([1, 15, 3, 3])
            right_hand_pose=param_dict['right_hand_pose'])

        # change the order of face keypoints, to be the same as "standard" 68 keypoints
        landmarks = torch.cat([landmarks[:, -17:], landmarks[:, :-17]], dim=1)

        # extra1=joints[:,55+68:55+68+22,:]
        # landmarks1=joints[:,55:55+68,:]
        # pose1=joints[:,0:55,:]
        param_dict_maxtrix = param_dict.copy()
        param_dict_axis = {}
        for key, value in param_dict.items():
            if key in ['global_pose', 'body_pose', 'jaw_pose', 'left_hand_pose', 'right_hand_pose']:
                tmp = converter.batch_matrix2axis(value[0]).unsqueeze_(0)
                param_dict_axis[key] = tmp.to('cuda')

        # self.neutral_model=self.neutral_model.to('cuda')
        # model_output = self.neutral_model(
        #     return_verts=True,
        #     # torch.Size([1, 10])
        #     shape_params=param_dict['shape'],
        #     # torch.Size([1, 10])
        #     expression=param_dict['exp'],
        #     # torch.Size([1, 3])
        #     global_orient=param_dict_axis['global_pose'].reshape(-1,3),
        #     # torch.Size([1, 63])
        #     body_pose=param_dict_axis['body_pose'].reshape(-1,63),
        #     # torch.Size([1, 3])
        #     jaw_pose=param_dict_axis['jaw_pose'].reshape(-1,3),
        #     # torch.Size([1, 45])
        #     left_hand_pose=param_dict_axis['left_hand_pose'].reshape(-1,45),
        #     # torch.Size([1, 45])
        #     right_hand_pose=param_dict_axis['right_hand_pose'].reshape(-1,45)            ,
        #     return_full_pose=True)
        # verts2=model_output.vertices.detach().float()
        # joints2=model_output.joints.detach().float()
        # reg_joints = torch.einsum(
        #         'ji,bik->bjk', self.smplx.extra_joint_regressor, verts2)
        # joints2[:, self.smplx.source_idxs.long()] = (
        #     joints2[:, self.smplx.source_idxs.long()].detach() * 0.0 +
        #     reg_joints[:, self.smplx.target_idxs.long()] * 1.0
        # )
        # landmarks2=joints2[:,55+21:55+21+68,:]
        # extra2=joints2[:,55:55+21,:]
        # pose2=joints2[:,0:55,:]
        # joints=torch.cat([pose2,landmarks2,extra2,torch.zeros(1,1,3,device='cuda')],dim=1)
        # verts=verts2
        # landmarks=landmarks2

        # if 0:
        if param_type == 'body' and extra != None:
            org_img_height, org_img_width = extra['batch']['org_img_size']
            pixie_bbox = extra['batch']['bbox']
            pixie_offsetx2org = pixie_bbox[0]
            pixie_offsety2org = pixie_bbox[1]
            pixie_bbox_w = pixie_bbox[2]-pixie_bbox[0]
            pixie_bbox_h = pixie_bbox[3]-pixie_bbox[1]
            week_cam = param_dict[param_type + '_cam']

            # TODO deca_landmarks2d_org为deca在原图的2d坐标
            if extra.get('landmarks2d') is not None:
                deca_face_bbox = extra['deca_face_bbox']
                deca_landmarks2d_org = extra['landmarks2d']

                deca_startx = deca_face_bbox[0]+pixie_offsetx2org
                deca_starty = deca_face_bbox[1]+pixie_offsety2org
                deca_width = deca_face_bbox[2]-deca_face_bbox[0]
                deca_height = deca_face_bbox[3]-deca_face_bbox[1]
                deca_landmarks2d_org[:, :, 0] = (
                    deca_landmarks2d_org[:, :, 0]+1)*0.5*deca_width+deca_startx
                deca_landmarks2d_org[:, :, 1] = (
                    deca_landmarks2d_org[:, :, 1]+1)*0.5*deca_height+deca_starty

                # out
                loc_3d = landmarks.clone()
                loc_2d = deca_landmarks2d_org.clone()

            else:

                org2pixie_tform = extra['batch']['tform']

                # joints (x,y)
                predicted_joints = util.batch_orth_proj(
                    joints, week_cam)[:, :, :2]
                # h, w, depth = image.shape
                predicted_joints[:, :, 0] = (#axis x
                    predicted_joints[:, :, 0]+1)*0.5*224
                predicted_joints[:, :, 1] = (#axis y
                    predicted_joints[:, :, 1]+1)*0.5*224

                tform_T = torch.inverse(org2pixie_tform[None, ...]).transpose(1, 2)
                # tform_T = org2pixie_tform.T[None, ...]
                batch_size, n_points, _ = predicted_joints.shape
                r=torch.cat(
                        [predicted_joints,
                         torch.ones(
                             [batch_size, n_points, 1],
                             device=predicted_joints.device,
                             dtype=predicted_joints.dtype)],
                        dim=-1)
                t = torch.bmm( r,tform_T)
                t=t[:,:,:2]
                # out
                loc_3d = joints.clone()
                loc_2d = t.clone()

            # TODO 拟合出perspective camera相机在(原图/crop_img)的translation
            focal=kwargs['focal']
            # focal_1080p = 5000
            # focal_per_pixel = focal_1080p/1080.
            camera_intrics = np.array([[focal, 0.00000000e+00, org_img_width/2],
                                       [0.00000000e+00, focal, org_img_height/2],
                                       [0.00000000e+00, 0.00000000e+00, 1]])

            trans_cam = estimate_translation_np(
                loc_3d.cpu().numpy()[0],
                loc_2d.cpu().numpy()[0],
                np.ones(loc_2d[0].shape[0]),
                camera_intrics)

            # TODO 用perspective camera重新投影到原图的坐标
            predicted_landmarks = perspective_projection(
                landmarks,
                rotation=torch.eye(3, device='cuda').unsqueeze(
                    0).expand(1, -1, -1),
                translation=torch.tensor(
                    [trans_cam], device='cuda', dtype=torch.float),
                camera_intrics=camera_intrics)[:, :, :-1]

            trans_verts = perspective_projection(
                verts,
                rotation=torch.eye(3, device='cuda').unsqueeze(
                    0).expand(1, -1, -1),
                translation=torch.tensor(
                    [trans_cam], device='cuda', dtype=torch.float),
                camera_intrics=camera_intrics)
            predicted_joints = perspective_projection(
                joints,
                rotation=torch.eye(3, device='cuda').unsqueeze(
                    0).expand(1, -1, -1),
                translation=torch.tensor(
                    [trans_cam], device='cuda', dtype=torch.float),
                camera_intrics=camera_intrics)[:, :, :-1]

            # TODO 将坐标转换到裁剪后的图片
            trans_verts[:, :, 0] = trans_verts[:, :, 0]-pixie_offsetx2org
            trans_verts[:, :, 1] = trans_verts[:, :, 1]-pixie_offsety2org
            trans_verts[:, :, 0] = trans_verts[:, :, 0]*2/pixie_bbox_w-1
            trans_verts[:, :, 1] = trans_verts[:, :, 1]*2/pixie_bbox_h-1
            trans_verts[:, :, 2] = verts[:, :, 2]

            face_lmks_on_origin_width=predicted_landmarks.clone()
            predicted_landmarks[:, :,
                                0] = predicted_landmarks[:, :, 0]-pixie_offsetx2org
            predicted_landmarks[:, :,
                                1] = predicted_landmarks[:, :, 1]-pixie_offsety2org
            predicted_landmarks[:, :,
                                0] = predicted_landmarks[:, :, 0]*2/pixie_bbox_w-1
            predicted_landmarks[:, :,
                                1] = predicted_landmarks[:, :, 1]*2/pixie_bbox_h-1

            predicted_joints[:, :, 0] = predicted_joints[:,
                                                         :, 0]-pixie_offsetx2org
            predicted_joints[:, :, 1] = predicted_joints[:,
                                                         :, 1]-pixie_offsety2org
            predicted_joints[:, :, 0] = predicted_joints[:,
                                                         :, 0]*2/pixie_bbox_w-1
            predicted_joints[:, :, 1] = predicted_joints[:,
                                                         :, 1]*2/pixie_bbox_h-1

        else:
            # projection
            trans_cam = None
            week_cam = param_dict[param_type + '_cam']
            trans_verts = util.batch_orth_proj(verts, week_cam)
            face_lmks_on_origin_width=landmarks.clone()
            predicted_landmarks = util.batch_orth_proj(
                landmarks, week_cam)[:, :, :2]
            predicted_joints = util.batch_orth_proj(joints, week_cam)[:, :, :2]

            # focal_1080p = 5000
            # focal_per_pixel = focal_1080p/1080.
            # pixie_landmarks3d = landmarks.clone()
            # camera_intrics = np.array([[5000., 0.00000000e+00, org_img_width/2],
            #                             [0.00000000e+00, 5000., org_img_height/2],
            #                             [0.00000000e+00, 0.00000000e+00, 1]])

            # loc_3d = pixie_landmarks3d.clone()
            # loc_2d = deca_landmarks2d_org.clone()
            # trans_cam = estimate_translation_np(
            #     loc_3d.cpu().numpy()[0],
            #     loc_2d.cpu().numpy()[0],
            #     np.ones(loc_2d[0].shape[0]),
            #     camera_intrics)

        # save predcition
        prediction = {
            # 世界坐标系下3d点
            'vertices': verts,
            'smplx_kpt3d': joints,
            'joints': joints,
            
            # 在若透视相机投影的2d点
            'transformed_vertices': trans_verts,
            'face_kpt': face_lmks_on_origin_width,
            'smplx_kpt': predicted_joints,

            # 相机参数
            'trans_cam': trans_cam,
            'week_cam': week_cam,
            
            # smplx参数            
            'shape': param_dict['shape'],
            'exp': param_dict['exp'],
            
            # ['global_pose', 'body_pose', 'jaw_pose', 'left_hand_pose', 'right_hand_pose']
            'param_dict_maxtrix': param_dict_maxtrix,
            'param_dict_axis': param_dict_axis
        }
        # if my_flag==1:
        #     savemat('tmp2.mat',dict_tensor2npy(prediction))
        # savemat('tmp.mat',dict_tensor2npy(prediction))

        # return albedo map
        # if 'tex' in param_dict.keys() and self.cfg.model.use_tex:
        #     albedo = self.flametex(param_dict['tex'])
        #     prediction['albedo'] = albedo
        #     prediction['light'] = param_dict['light'].reshape(-1, 9, 3)
        # elif self.cfg.model.use_tex:
        #     texcode = torch.zeros(
        #         [verts.shape[0], self.cfg.model.n_tex], device=self.device, dtype=torch.float32)
        #     albedo = self.flametex(texcode)
        #     prediction['albedo'] = albedo
        # else:
        #     prediction['albedo'] = torch.zeros(
        #         [verts.shape[0], 3, self.cfg.model.uv_size, self.cfg.model.uv_size], device=self.device, dtype=torch.float32)
        return prediction

    def decode_Tpose(self, param_dict):
        ''' return body mesh in T pose, support body and head param dict only
        '''
        verts, _, _ = self.smplx(
            shape_params=param_dict['shape'],
            expression_params=param_dict['exp'],
            jaw_pose=param_dict['jaw_pose'])
        return verts
