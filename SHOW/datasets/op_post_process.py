from scipy.io import savemat, loadmat
from collections import namedtuple
from collections import defaultdict
import numpy as np
import json
import torch
import glob
import cv2
import os.path as osp
import os
from pathlib import Path
from .pre_dataset import *
from typing import Union
from functools import reduce, partial
from ..utils import cvt_dict_to_tensor
from loguru import logger


class op_post_process(object):
    def __init__(self, all_processed_item, device, dtype):
        self.all_processed_item = all_processed_item
        self.device = device
        self.dtype = dtype

    def run(self):
        self.parse_batch()
        self.merge_list_to_tensor()
        return self.parse_data

    def check_valid(self):
        self.valid_dict = defaultdict(list)

        for out in self.all_processed_item:
            for key in ['pixie', 'mp', 'op', 'deca']:
                self.valid_dict[key].append(
                    1 if out.get(key, None) is not None else 0)

        for key, val in self.valid_dict.items():
            logger.info(f'{key}:{val}')

    def merge_list_to_tensor(self,):

        self.exp = np.stack(self.exp_list)  # (bs, 50)
        self.pose = np.stack(self.pose_list)  # (bs, 21, 3)
        self.jaw = np.stack(self.jaw_list)  # (batch_size, 3)

        self.global_orient = np.stack(
            self.global_orient_list)  # (batch_size, 3)
        self.cam_transl = np.stack(self.cam_transl_list)  # (batch_size, 3)
        self.rhand = np.concatenate(
            self.rhand_list, axis=0)  # (batch_size, 12)
        self.lhand = np.concatenate(
            self.lhand_list, axis=0)  # (batch_size, 12)

        for idx in range(len(self.all_processed_item)):
            if self.all_processed_item[idx].get('pixie',None) is not None:
                self.betas = self.all_processed_item[idx]['pixie']['shape']  # (200,)
                break
        
        if not hasattr(self, 'betas'):
            self.betas = np.zeros(200)

        self.op_kpts = np.stack(self.op_kpts_list, axis=0)  # (bs, 135, 3)
        self.mp_kpts = np.stack(self.mp_kpts_list, axis=0)  # (bs, 478, 2)
        self.deca_kpts = np.concatenate(
            self.deca_kpts_list, axis=0)  # (bs, 68, 2)

        self.op_valid_flag = np.array(self.valid_dict['op'])  # (bs)
        self.mp_valid_flag = np.array(self.valid_dict['mp'])  # (bs)
        self.deca_valid_flag = np.array(self.valid_dict['deca'])  # (bs)
        # op_valid_flag=torch.tensor(op_valid_flag,device=self.device).long()

        batch_size = self.exp.shape[0]
        self.seg_stack=np.stack(self.seg_list)
        
        self.exp = np.concatenate([(self.exp), np.zeros(
            (batch_size, 50))], axis=-1)  # torch.Size([bs, 100])
        self.betas = np.concatenate([(self.betas), np.zeros(100)])[
            None, ...]  # torch.Size([1, 300])

        self.mica_head_transl = np.zeros((1, 3))
        self.leye_pose = np.zeros((batch_size, 3))
        self.reye_pose = np.zeros((batch_size, 3))
        self.transl = np.zeros((batch_size, 3))
        
        ret_dict=dict(
            init_data=dict(
                betas=self.betas, 
                exp=self.exp, 
                jaw=self.jaw, 
                rhand=self.rhand,
                lhand=self.lhand, 
                pose=self.pose, 
                global_orient=self.global_orient, 
                cam_transl=self.cam_transl,
                mica_head_transl=self.mica_head_transl, 
                leye_pose=self.leye_pose, 
                reye_pose=self.reye_pose,
                transl=self.transl
            ),
            gt_data=dict(
                op_kpts=self.op_kpts, 
                mp_kpts=self.mp_kpts, 
                deca_kpts=self.deca_kpts,
                op_valid_flag=self.op_valid_flag, 
                mp_valid_flag=self.mp_valid_flag, 
                deca_valid_flag=self.deca_valid_flag,
                seg_stack=self.seg_stack
            )
        )
        self.parse_data = cvt_dict_to_tensor(ret_dict,self.device,self.dtype)
        
        #expression (bs, 50)
        #body_pose_axis (bs, 21, 3)
        #jaw_pose (bs,3)
        #global_orient (bs,3)
        #transl (bs,3)
        #left_hand_pose (bs,12)
        #right_hand_pose (bs,12)
        #betas (200)
        
        # jaw: (B,3)
        # body: (B,21,3)
        # hands: 2*(B,15,3)
        return self.parse_data

    def parse_batch(self):
        
        self.global_orient_list = []
        self.cam_transl_list = []
        self.jaw_list = []
        self.exp_list = []
        self.lhand_list = []
        self.rhand_list = []
        self.pose_list = []
        self.mp_kpts_list = []
        self.op_kpts_list = []
        self.deca_kpts_list = []
        self.seg_list = []
        last_nonempty=dict(
            pixie=None,
            deca=None,
        )
        for i in range(len(self.all_processed_item)):
            item_pos=len(self.all_processed_item)-1-i
            item=self.all_processed_item[item_pos]
            for key in last_nonempty.keys():
                out_val=item.get(key,None)
                if out_val is not None:
                    last_nonempty[key]=out_val
                if out_val is None:
                    self.all_processed_item[item_pos][key]=last_nonempty[key]
        
        for i in range(len(self.all_processed_item)):
            item_pos=i
            item=self.all_processed_item[item_pos]
            for key in last_nonempty.keys():
                out_val=item.get(key,None)
                if out_val is not None:
                    last_nonempty[key]=out_val
                if out_val is None:
                    self.all_processed_item[item_pos][key]=last_nonempty[key]
        
        logger.info(f'after filling init datas: ')
        self.check_valid()
        
        for out in self.all_processed_item:
            #########################
            out_pixie = out.get('pixie', None)
            if out_pixie is None:
                out_pixie = {
                    'body_pose_63': np.zeros((21, 3)),
                    'left_hand_pose': np.zeros((1, 12)),
                    'right_hand_pose': np.zeros((1, 12)),
                    'global_orient': np.zeros((1, 3)),
                    'transl': np.zeros(3),
                }
            self.pose_list.append(out_pixie['body_pose_63'])
            self.lhand_list.append(out_pixie['left_hand_pose'])
            self.rhand_list.append(out_pixie['right_hand_pose'])
            self.cam_transl_list.append(out_pixie['transl'])
            self.global_orient_list.append(out_pixie['global_orient'])

            #########################
            out_deca = out.get('deca', None)
            if out_deca is None:
                out_deca = {
                    'exp': np.zeros(50),
                    'jaw': np.zeros(3),
                    'lmk2d': np.zeros((1, 68, 2)),
                }
            self.exp_list.append(out_deca['exp'])
            self.jaw_list.append(out_deca['jaw'])
            self.deca_kpts_list.append(out_deca['lmk2d'])

            #########################
            out_mp = out.get('mp', None)
            if out_mp is None:
                out_mp = {
                    'lmk2d': np.zeros((478, 2))
                }
            self.mp_kpts_list.append(out_mp['lmk2d'])

            #########################
            out_op = out.get('op', None)
            if out_op is None:
                out_op = np.zeros((135, 3))
            self.op_kpts_list.append(out_op)
            
            #########################
            out_seg = out.get('seg', None)
            self.seg_list.append(out_seg)
