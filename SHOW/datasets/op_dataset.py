
from collections import defaultdict
import numpy as np
import json
import torch
import glob
import cv2
import os.path as osp
import os
import sys
from pathlib import Path
from .op_post_process import op_post_process
from .pre_dataset import *
from ..image import *
from typing import Union
from functools import reduce, partial
from easydict import EasyDict
from loguru import logger
from .op_base import op_base
from ..utils import glob_exts_in_path
from ..utils import default_timers
from ..utils import is_empty_dir,ext_files_num_in_dir,img_files_num_in_dir,run_openpose
from ..face_iders import match_faces
from .pre_runner import run_pymafx
from tqdm import tqdm


from modules.PIXIE.demos.api_multi_pixie import api_multi_body
from modules.DECA.demos.api_multi_deca import api_multi_deca
from SHOW.detector.face_detector import FaceDetector
from SHOW.video_filter.deeplab_seg import deeplab_seg
from SHOW.utils.video import images_to_video
import joblib
import SHOW


class op_dataset(op_base):
    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(
        self,
        
        dtype=torch.float32,
        device='cpu',
        batch_size=-1,
        config=None,
        face_ider=None,
        person_face_emb: np.ndarray = None,
    ):
        self.config = config
        self.use_hands = config.use_hands
        self.use_face = config.use_face
        self.use_face_contour = config.use_face_contour
        
        self.person_face_emb = person_face_emb
        

        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        self.all_processed_item = []
        self.corr_center_pymaf = None
        self.correspond_center = None
        self.correspond_bbox = None
        self.face_ider = face_ider
        self.cnt = 0

        assert(osp.exists(self.config.img_folder))

        self.get_all_dict=None
        self.focal = np.array(5000)
        self.match_thres = 0.08  # 30/720=0.04
        self.num_joints = (self.NUM_BODY_JOINTS + 2 * self.NUM_HAND_JOINTS * self.use_hands)
        self.int_id_to_item = defaultdict(dict)
        # self.all_item_list = self.int_id_to_item = {
        #     0: {
        #         'img': 'path/img'
        #     }
        #     1: {
        #         'img': 'path/img'
        #         'fan': 'path/fan'
        #     }
        # }
        self.all_processed_item = []
        # [
        #     {
        #         'img': np.array,
        #         'pixie': {
        #             'face_kpt': np.array
        #         },
        #     }
        #     {
        #         'img': np.array,
        #         'pixie': {
        #             'face_kpt': np.array
        #         },
        #         'fan': None, # match failed
        #     }
        # ]
        self.find_item_template_list={}
        ref_len = img_files_num_in_dir(self.config.img_folder)
        
        
        def match_by_bbox(ret_list, get_bbox_center_func):
            if (len(ret_list) < 1 or ret_list is None):
                logger.error(f'length of ret_list < 1')
                return None
            
            if self.correspond_center is not None:
                dist1 = dist2 = np.inf
                for ret in ret_list:
                    bbox, cur_center = get_bbox_center_func(ret)
                    dist1 = np.abs(cur_center[0]-self.correspond_center[0])/self.width
                    dist2 = np.abs(cur_center[1]-self.correspond_center[1])/self.height
                    # 根据xy方向的偏移的比率判断
                    if dist1 < self.match_thres and dist2 < self.match_thres:
                        if bbox is not None:
                            self.correspond_bbox = bbox
                        self.correspond_center = cur_center
                        return ret
            else:
                logger.info(f'corresponding center is None')
                ret = ret_list[0]
                bbox, cur_center = get_bbox_center_func(ret)
                self.correspond_bbox = bbox
                self.correspond_center = cur_center
                return ret
              
            # self.correspond_center is not None and match failed
            if len(ret_list)==1:
                return ret_list[0]
            return None

        # images -----------------------------------
        if True:
            def empty_func(*args, **kwargs):
                pass
            
            find_item_template_img={
                'name': 'img',
                'exts': ['png', 'jpg'],
                'dir': self.config.img_folder,
                'judge_prepare_data_func': empty_func,
                'run_prepare_data_func': empty_func,
                'post_prepare_data_func': empty_func,
                'match_item_func': empty_func,
                'read_item_func': empty_func,
            }
            self.find_item_template_list['img']=find_item_template_img

        # FAN -----------------------------------
        if True:
            def judge_prepare_data_func_fan():
                return ext_files_num_in_dir(
                    self.config.fan_npy_folder, 
                    exts=['*.npy', '*.npy.empty']
                ) < ref_len
                
            def run_prepare_data_func_fan():
                fan = SHOW.detector.fan_detector.FAN_Detector()
                fan.predict(self.config.img_folder, 
                            self.config.fan_npy_folder,
                            save_vis=self.config.saveVis,
                            fan_vis_dir=self.config.fan_npy_folder_vis)
                del fan
                
            def post_prepare_data_func_fan():
                if (self.config.saveVis and
                    not Path(self.config.fan_npy_folder_v).exists() and
                    not is_empty_dir(self.config.fan_npy_folder_vis)
                ):
                    logger.info(f'converting {self.config.fan_npy_folder_vis} to {self.config.fan_npy_folder_v}')
                    images_to_video(
                        input_folder=self.config.fan_npy_folder_vis,
                        output_path=self.config.fan_npy_folder_v,
                        img_format = None,
                        fps=30,
                    )

            def match_item_func_fan(fan_ret_list):
                def get_bbox_center_func(input_ret):
                    face_kpt = input_ret
                    bbox = lmk2d_to_bbox(face_kpt, self.height, self.width)
                    xmin, ymin, xmax, ymax = [int(i) for i in bbox]
                    cur_center = [int((xmin+xmax)/2), int((ymin+ymax)/2)]
                    return bbox, cur_center
                return match_by_bbox(
                    fan_ret_list, 
                    get_bbox_center_func)
                
            def read_item_func_fan(lmk_path):
                return np.load(lmk_path, allow_pickle=True)
            
            def post_read_func_fan(template):
                # fan_valid_flag = [0 if i is None else 1 for i in template['files_list']]
                fan_valid_flag = [0 if i.get('fan') is None else 1 for i in self.all_processed_item]
                self.get_all_dict['fan_valid']=np.array(fan_valid_flag)
                
                fan_fill = [np.zeros((68,2)) if i.get('fan') is None else i['fan'] for i in self.all_processed_item]
                self.get_all_dict['fan'] = np.array(fan_fill)
                
            find_item_template_fan={
                'name': 'fan',
                'exts': ['npy'],
                'files_list': [],
                'dir': self.config.fan_npy_folder,
                'judge_prepare_data_func': judge_prepare_data_func_fan,
                'run_prepare_data_func': run_prepare_data_func_fan,
                'post_prepare_data_func': post_prepare_data_func_fan,
                'match_item_func': match_item_func_fan,
                'read_item_func': read_item_func_fan,
                'post_read_func': post_read_func_fan,
            }
            self.find_item_template_list['fan']=find_item_template_fan
          
        # openpose -----------------------------------
        if True:
            def judge_prepare_data_func_op():
                return is_empty_dir(self.config.keyp_folder)
            
            def run_prepare_data_func_op():
                with default_timers['run_openpose']:
                    run_openpose(
                        low_res=self.config.low_res,
                        openpose_root_path=self.config.openpose_root_path,
                        openpose_bin_path=self.config.openpose_bin_path,
                        img_dir=self.config.img_folder,
                        out_dir=self.config.keyp_folder,
                        img_out=self.config.keyp_folder_vis if self.config.saveVis else None,
                    )
                    
            def post_prepare_data_func_op():
                if (self.config.saveVis and
                    not Path(self.config.keyp_folder_v).exists() and
                    not is_empty_dir(self.config.keyp_folder_vis)
                ):
                    logger.info(f'converting {self.config.keyp_folder_vis} to {self.config.keyp_folder_v}')
                    images_to_video(
                        input_folder=self.config.keyp_folder_vis,
                        output_path=self.config.keyp_folder_v,
                        img_format = None,
                        fps=30,
                    )

            def match_item_func_op(keypoints):
                def get_bbox_center_func(keypoint):
                    body_j = keypoint[0:25, :]
                    head_j = body_j[0]
                    x, y, _ = head_j
                    cur_center = [x, y]
                    return None, cur_center
                return match_by_bbox(
                    keypoints, 
                    get_bbox_center_func)

            def read_item_func_op(file_path):
                keypoints, _ = self.read_keypoints(keypoint_fn=file_path)
                return keypoints
            
            find_item_template_op={
                'name': 'op',
                'exts': ['json'],
                'dir':self.config.keyp_folder,
                'judge_prepare_data_func': judge_prepare_data_func_op,
                'run_prepare_data_func': run_prepare_data_func_op,
                'post_prepare_data_func': post_prepare_data_func_op,
                'match_item_func': match_item_func_op,
                'read_item_func': read_item_func_op,
            }
            self.find_item_template_list['op']=find_item_template_op

        # deca -----------------------------------
        if True:
            def judge_prepare_data_func_deca():
                return ext_files_num_in_dir(
                    self.config.deca_mat_folder, 
                    exts=['*.pkl', '*.pkl.empty']
                ) < ref_len
            
            def run_prepare_data_func_deca():
                with default_timers['api_multi_deca']:
                    logger.info(f'running api_multi_deca')
                    api_multi_deca(
                        inputpath=self.config.img_folder,
                        savefolder=self.config.deca_mat_folder,
                        visfolder=self.config.deca_mat_folder_vis,
                        saveVis=self.config.saveVis,
                        face_detector=self.config.face_detector
                    )
                
            def post_prepare_data_func_deca():
                if (self.config.saveVis and
                    not Path(self.config.deca_mat_folder_v).exists() and
                    not is_empty_dir(self.config.deca_mat_folder_vis)
                ):
                    logger.info(f'converting {self.config.deca_mat_folder_vis} to {self.config.deca_mat_folder_v}')
                    images_to_video(
                        input_folder=self.config.deca_mat_folder_vis,
                        output_path=self.config.deca_mat_folder_v,
                        img_format = None,
                        fps=30,
                    )

            def match_item_func_deca(deca_ret_list):
                def get_bbox_center_func(input_ret):
                    bbox = input_ret['face_bbox']
                    xmin, ymin, xmax, ymax = [int(i) for i in bbox]
                    cur_center = [int((xmin+xmax)/2), int((ymin+ymax)/2)]
                    return bbox, cur_center
                return match_by_bbox(
                    deca_ret_list, 
                    get_bbox_center_func)

            def read_item_func_deca(deca):
                return read_deca(deca)
                
            find_item_template_deca={
                'name': 'deca',
                'exts': ['pkl'],
                'dir':self.config.deca_mat_folder,
                'judge_prepare_data_func': judge_prepare_data_func_deca,
                'run_prepare_data_func': run_prepare_data_func_deca,
                'post_prepare_data_func': post_prepare_data_func_deca,
                'match_item_func': match_item_func_deca,
                'read_item_func': read_item_func_deca,
            }
            self.find_item_template_list['deca']=find_item_template_deca
            
        # pixie -----------------------------------
        if True:
            def judge_prepare_data_func_pixie():
                return ext_files_num_in_dir(
                    self.config.pixie_mat_folder, 
                    exts=['*.pkl', '*.pkl.empty']) < ref_len
            
            def run_prepare_data_func_pixie():
                with default_timers['api_multi_body']:
                    logger.info(r'running pixie')
                    api_multi_body(
                        imgfolder=self.config.img_folder,
                        savefolder=self.config.pixie_mat_folder,
                        visfolder=self.config.pixie_mat_folder_vis,
                        saveVis=self.config.saveVis,
                        rasterizer_type=self.config.rasterizer_type
                    )
                
            def post_prepare_data_func_pixie():
                if (self.config.saveVis and
                    not Path(self.config.pixie_mat_folder_v).exists() and
                    not is_empty_dir(self.config.pixie_mat_folder_vis)
                ):
                    logger.info(f'converting {self.config.pixie_mat_folder_vis} to {self.config.pixie_mat_folder_v}')
                    images_to_video(
                        input_folder=self.config.pixie_mat_folder_vis,
                        output_path=self.config.pixie_mat_folder_v,
                        img_format = None,
                        fps=30,
                    )

            def match_item_func_pixie(pixie_ret_list):
                def get_bbox_center_func(input_ret):
                    bbox = input_ret['face_bbox']
                    xmin, ymin, xmax, ymax = [int(i) for i in bbox]
                    cur_center = [int((xmin+xmax)/2), int((ymin+ymax)/2)]
                    return bbox, cur_center
                return match_by_bbox(
                    pixie_ret_list, 
                    get_bbox_center_func)

            def read_item_func_pixie(pixie):
                return read_pixie(
                    pixie, self.height, self.width,
                    cvt_hand_func=self.config.cvt_hand_func)
                
            find_item_template_pixie={
                'name': 'pixie',
                'exts': ['pkl'],
                'dir':self.config.pixie_mat_folder,
                'judge_prepare_data_func': judge_prepare_data_func_pixie,
                'run_prepare_data_func': run_prepare_data_func_pixie,
                'post_prepare_data_func': post_prepare_data_func_pixie,
                'match_item_func': match_item_func_pixie,
                'read_item_func': read_item_func_pixie,
            }
            self.find_item_template_list['pixie']=find_item_template_pixie

        # mp -----------------------------------
        if True:   
            def judge_prepare_data_func_mp():
                return self.config.use_mp_loss and (
                    ext_files_num_in_dir(self.config.mp_npz_folder, 
                                         exts=['*.npz', '*.npz.empty']) < ref_len
                )
                
            def run_prepare_data_func_mp():
                if self.config.use_mp_loss:
                    with default_timers['face_detector']:
                        logger.info(f'running face detection')
                        if self.__dict__.get('face_detector',None) is None:
                            self.face_detector = FaceDetector()
                            
                        self.face_detector.predict_batch(
                            img_folder=self.config.img_folder,
                            savefolder=self.config.mp_npz_folder,
                            visfolder=self.config.mp_npz_folder_vis,
                            saveVis=self.config.saveVis
                        )
                    
            def post_prepare_data_func_mp():
                if (self.config.saveVis and
                    not Path(self.config.mp_npz_folder_v).exists() and
                    not is_empty_dir(self.config.mp_npz_folder_vis)
                ):
                    logger.info(f'converting {self.config.mp_npz_folder_vis} to {self.config.mp_npz_folder_v}')
                    images_to_video(
                        input_folder=self.config.mp_npz_folder_vis,
                        output_path=self.config.mp_npz_folder_v,
                        img_format = None,
                        fps=30,
                    )
                    
            def match_item_func_mp(mp_ret_list):
                def get_bbox_center_func(input_ret):
                    face_kpt = input_ret['lmk2d']
                    bbox = lmk2d_to_bbox(face_kpt, self.height, self.width)
                    xmin, ymin, xmax, ymax = [int(i) for i in bbox]
                    cur_center = [int((xmin+xmax)/2), int((ymin+ymax)/2)]
                    return bbox, cur_center
                return match_by_bbox(
                    mp_ret_list, 
                    get_bbox_center_func)

            def read_item_func_mp(mp):
                return read_mp(mp, self.height, self.width)
                
            find_item_template_mp={
                'name': 'mp',
                'exts': ['npz'],
                'dir':self.config.mp_npz_folder,
                'judge_prepare_data_func': judge_prepare_data_func_mp,
                'run_prepare_data_func': run_prepare_data_func_mp,
                'post_prepare_data_func': post_prepare_data_func_mp,
                'match_item_func': match_item_func_mp,
                'read_item_func': read_item_func_mp,
            }
            self.find_item_template_list['mp']=find_item_template_mp
            
        # seg -----------------------------------
        if True:   
            def judge_prepare_data_func_seg():
                return self.config.use_silhouette_loss and (
                     ext_files_num_in_dir(self.config.seg_img_folder, 
                                          exts=['*.jpg', '*.png']) < ref_len
                 )
                 
            def run_prepare_data_func_seg():
                if self.config.use_silhouette_loss:
                    with default_timers['deeplab_seg']:
                        logger.info(f'running deeplab segmentation')
                        if not hasattr(self,'deeplab_seg'):
                            self.deeplab_seg=deeplab_seg()
                            
                        self.deeplab_seg.predict_batch(
                            img_folder=self.config.img_folder,
                            savefolder=self.config.seg_img_folder,
                            saveVis=True
                        )
                    
            def post_prepare_data_func_seg():
                if (self.config.saveVis and
                    is_empty_dir(self.config.seg_img_folder_vis)
                ):
                    if not hasattr(self,'deeplab_seg'):
                        self.deeplab_seg=deeplab_seg()
                    logger.info(f'running deeplab segmentation visualization')
                    self.deeplab_seg.predict_batch(
                        img_folder=self.config.img_folder,
                        savefolder=self.config.seg_img_folder_vis,
                        saveVis=True,
                        save_mode='mask',
                    )
                    
                if (self.config.saveVis and
                    not Path(self.config.seg_img_folder_v).exists() and
                    not is_empty_dir(self.config.seg_img_folder)
                ):
                    logger.info(f'convert {self.config.seg_img_folder} to video')
                    images_to_video(
                        input_folder=self.config.seg_img_folder_vis,
                        output_path=self.config.seg_img_folder_v,
                        img_format = None,
                        fps=30,
                    )
                
            def match_item_func_seg(seg):
                return seg
                
            def read_item_func_seg(seg):
                return cv2.imread(seg,cv2.IMREAD_GRAYSCALE)/255
                
            find_item_template_seg={
                'name': 'seg',
                'exts': ['jpg'],
                'dir':self.config.seg_img_folder,
                'judge_prepare_data_func': judge_prepare_data_func_seg,
                'run_prepare_data_func': run_prepare_data_func_seg,
                'post_prepare_data_func': post_prepare_data_func_seg,
                'match_item_func': match_item_func_seg,
                'read_item_func': read_item_func_seg,
            }
            self.find_item_template_list['seg']=find_item_template_seg
            
        # pymaf -----------------------------------
        if True:   
            def judge_prepare_data_func_pymaf():
                return self.config.use_pymaf_hand
                 
            def run_prepare_data_func_pymaf():
                if self.config.use_pymaf_hand:
                    if (not Path(self.config.pymaf_pkl_path).exists()):
                        logger.info('Running pymaf-x')
                        run_pymafx(
                            image_folder=self.config.img_folder,
                            output_folder=self.config.pymaf_pkl_folder,
                            no_render=not self.config.saveVis,
                            pymaf_code_dir=os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                '../../modules/PyMAF')
                        )
                        
            def post_prepare_data_func_pymaf():
                if (
                    self.config.saveVis and
                    not Path(self.config.pymaf_pkl_folder_v).exists() and
                    not is_empty_dir(self.config.pymaf_folder_vis)
                ):
                    images_to_video(
                        input_folder=self.config.pymaf_folder_vis,
                        output_path=self.config.pymaf_pkl_folder_v,
                        img_format = None,
                        fps=30,
                    )
 
            def match_item_func_pymaf():
                pass
                
            def read_item_func_pymaf():
                pass 
                
            def post_read_func_pymaf(template):
       
                if (
                    self.config.use_pymaf_hand and
                    Path(self.config.pymaf_pkl_path).exists()
                ):
                    logger.info(f'load pymaf file %s' % self.config.pymaf_pkl_path)
                    pymaf_out_data=joblib.load(self.config.pymaf_pkl_path)
                    smplx_params=pymaf_out_data['smplx_params']
                    joints2d=pymaf_out_data['joints2d']
                    nose_j_list=[i[0] for i in joints2d]
                    
                    self.nose_j_list=nose_j_list
                    self.smplx_params=smplx_params
                    
                    matched_smplx_params=[]
                    for idx,nose_j in enumerate(self.nose_j_list):
                        dist1 = np.abs(nose_j[0]-self.corr_center_pymaf[0])/self.width
                        dist2 = np.abs(nose_j[1]-self.corr_center_pymaf[1])/self.height
                        if dist1 < self.match_thres and dist2 < self.match_thres:
                            self.corr_center_pymaf=nose_j
                            matched_smplx_params.append(self.smplx_params[idx])

                    lhand_list=[e['left_hand_pose'] for e in matched_smplx_params]#(1,15,3,3)
                    rhand_list=[e['right_hand_pose'] for e in matched_smplx_params]
                    lhand_rot=torch.cat(lhand_list,dim=0)
                    rhand_rot=torch.cat(rhand_list,dim=0)
                    
                    from pytorch3d.transforms import matrix_to_axis_angle
                    lhand_axis=matrix_to_axis_angle(lhand_rot)
                    rhand_axis=matrix_to_axis_angle(rhand_rot)

                    cvt_hand_func=self.config.cvt_hand_func
                    lhand_pca, rhand_pca=cvt_hand_func(lhand_axis,rhand_axis)
                    
                    self.pymaf_lhand_pca=lhand_pca
                    self.pymaf_rhand_pca=rhand_pca
                    
                    if 1:
                        logger.info(f'matched pymaf_lhand_pca shape {self.pymaf_lhand_pca.shape}')
                        self.pymaf_lhand_pca=self.pymaf_lhand_pca[:self.batch_size,:]
                        self.pymaf_rhand_pca=self.pymaf_rhand_pca[:self.batch_size,:]
                    
                    if self.pymaf_lhand_pca.shape[0]==self.batch_size:
                        logger.warning(f'replaced r&l hand with pymaf')
                        self.get_all_dict['init_data']['lhand']=self.pymaf_lhand_pca
                        self.get_all_dict['init_data']['rhand']=self.pymaf_rhand_pca
                        
                
            find_item_template_pymaf={
                'name': 'pymaf',
                'exts': ['xxx'],
                'dir':self.config.pymaf_pkl_folder,
                'judge_prepare_data_func': judge_prepare_data_func_pymaf,
                'run_prepare_data_func': run_prepare_data_func_pymaf,
                'post_prepare_data_func': post_prepare_data_func_pymaf,
                'match_item_func': match_item_func_pymaf,
                'read_item_func': read_item_func_pymaf,
                'post_read_func': post_read_func_pymaf,
            }
            self.find_item_template_list['pymaf']=find_item_template_pymaf
          
    
    def initialize(self):
        for template in self.find_item_template_list.values():
            
            if template['judge_prepare_data_func']():
                template['run_prepare_data_func']()
            template['post_prepare_data_func']()
            template['files_list'] = glob_exts_in_path(
                template['dir'], img_ext = template['exts'])
            
            for file_path in template['files_list']:
                fn_id, _ = osp.splitext(osp.split(file_path)[1])
                fn_id = int(fn_id.split('_')[0].split('.')[0])
                self.int_id_to_item[fn_id][template['name']] = file_path
                
        self.all_item_list = [[k, v] for k, v in self.int_id_to_item.items() if v.get('img') is not None]
        self.all_item_list = sorted(self.all_item_list, key=lambda x: x[0])
        self.all_item_list = [i[1] for i in self.all_item_list]
        
        assert(Path(self.all_item_list[0]['img']).exists())
        self.height, self.width, _ = cv2.imread(self.all_item_list[0]['img']).shape

        if self.batch_size == -1:
            self.batch_size = len(self.all_item_list)
        else:
            self.all_item_list = self.all_item_list[:self.batch_size]

        if self.batch_size>300:
            self.batch_size=300
  
    def __len__(self):
        return self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= self.__len__():
            raise StopIteration

        assert(self.cnt >= 0)
        self.cnt += 1

        return self.__getitem__(self.cnt-1)

    def __getitem__(self, idx):

        img_path = self.all_item_list[idx]['img']
        
        # get_bbox_by_emb_and_deca
        assert(Path(img_path).exists())
        img = cv2.imread(img_path).astype(np.float32)
        assert(img is not None)
        img_fn, self.img_ext = osp.splitext(osp.split(img_path)[1])
        self.height, self.width, _ = img.shape
        # bgr,hw3,uint8

        if self.person_face_emb is not None:
            if self.correspond_center is None:
                self.correspond_center, self.correspond_bbox = match_faces(
                    img, self.face_ider, 
                    self.person_face_emb)
                if self.corr_center_pymaf is None:
                    self.corr_center_pymaf=self.correspond_center
                if self.correspond_center is None:
                    logger.warning("correspond_center return None")

        img = img[:, :, ::-1] / 255.0
        img = img.transpose(2, 0, 1)
        # c,h,w; rgb 0-1
            
        correspond_info = {}
        for item_name,item_path in self.all_item_list[idx].items():
            item_dict = self.find_item_template_list[item_name]
            item_con = item_dict['read_item_func'](item_path)
            correspond_info[item_name] = item_dict['match_item_func'](item_con)
        
        correspond_info.update(
            img=img,
            img_fn=img_fn,
            img_path=img_path)
        return EasyDict(correspond_info)


    def get_all(self):

        for idx in tqdm(list(range(self.batch_size)),desc='reading raw files'):
            assert(type(idx) == int)
            self.all_processed_item.append(
                self.__getitem__(idx)
            )

        self.pp = op_post_process(self.all_processed_item,
                                  device=self.device,
                                  dtype=self.dtype)
        self.get_all_dict = self.pp.run()
        
        for template in self.find_item_template_list.values():
            post_read_func=template.get('post_read_func')
            if post_read_func:
                post_read_func(template)
            
        return self.get_all_dict

    def get_modify_jt_weight(self) -> torch.Tensor:
        # @return optim_weights: [1,135,1]
        self.get_joint_weights()
        
        self.optim_weights[:, 2, :] *= self.config.op_shoulder_conf_weight  # shoulder
        self.optim_weights[:, 5, :] *= self.config.op_shoulder_conf_weight  # shoulder
        self.optim_weights[:, 8, :] *= self.config.op_root_conf_weight  # root
        
        if 0:
            self.optim_weights[:, :25] = 1  # body
            self.optim_weights[:, 25:67] = 2  # hand
            self.optim_weights[:, 67:] = 0  # face

        # print(self.optim_weights)
        return self.optim_weights
