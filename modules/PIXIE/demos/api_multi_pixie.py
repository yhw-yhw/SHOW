import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
import platform
if platform.system() == 'Windows':
    matplotlib.use('TkAgg')
    
from scipy.io import loadmat, savemat
import imageio
import cv2
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import get_cfg_defaults
from pathlib import Path
import mmcv


def api_multi_body(
    imgfolder='',
    savefolder='',
    visfolder='',
    focal=5000,
    
    device='cuda',
    iscrop=True,
    saveVis=True,
    saveMat=True,
    rasterizer_type='pytorch3d'
    ):
    pixie_cfg=get_cfg_defaults()
    Path(savefolder).mkdir(exist_ok=True,parents=True)
    os.makedirs(visfolder, exist_ok=True)

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    print(f'results in {savefolder}')

    posedata = TestData(imgfolder, iscrop=iscrop,body_detector='rcnn')
    pixie_cfg.model.use_tex = False
    pixie = PIXIE(config=pixie_cfg, device=device)
    visualizer = Visualizer(
        render_size=1024, config=pixie_cfg,
        device=device, rasterizer_type=rasterizer_type)


    for _, batch_list in enumerate(tqdm(posedata, dynamic_ncols=True)):
            
            if isinstance(batch_list,dict) and batch_list.get('is_missing',None):
                data_name = batch_list['name']
                open(os.path.join(savefolder, f'{data_name}.pkl.empty'), 'a').close()
                print(f'no face detected! skip: {data_name}')
                continue


            original_image=None
            pixie_save_data_list=[]
            for index,batch in enumerate(batch_list):
                data_name = batch['name']
                util.move_dict_to_device(batch, device)
                batch['image'] = batch['image'].unsqueeze(0)
                batch['image_hd'] = batch['image_hd'].unsqueeze(0)
                pixie_param_dict = pixie.encode({'body': batch})
                codedict = pixie_param_dict['body']
                pixie_opdict = pixie.decode(
                    codedict,
                    param_type='body',
                    extra={'batch':batch},
                    focal=focal
                    )

                tform = torch.inverse(batch['tform'][None, ...]).transpose(1, 2)
                
                if original_image is None:
                    original_image = batch['original_image'][None, ...]
                elif visdict['color_shape_images'] is not None:
                    original_image=visdict['color_shape_images']
                
                visualizer.recover_position(
                    pixie_opdict, batch, tform, original_image)
                visdict = visualizer.render_results(
                    pixie_opdict, batch['image_hd'],
                    moderator_weight=pixie_param_dict['moderator_weight'],
                    overlay=True)
                
                pixie_save_data={
                    'face_kpt': pixie_opdict['face_kpt'],#(68, 2)
                    'transl': pixie_opdict['trans_cam'],
                    'exp': pixie_opdict['exp'],
                    'shape':pixie_opdict['shape'],
                    
                    'body_pose_63':pixie_opdict['param_dict_axis']['body_pose'],
                    'left_hand_pose': pixie_opdict['param_dict_axis']['left_hand_pose'],
                    'right_hand_pose': pixie_opdict['param_dict_axis']['right_hand_pose'],
                    'global_orient': pixie_opdict['param_dict_axis']['global_pose'],
                    
                    'focal':focal,
                    'body_box':batch['bbox'],
                }
                ret=util.dict_tensor2npy(pixie_save_data)
                pixie_save_data_list.append(ret)

            if saveVis:
                save_img_path=os.path.join(visfolder, f'{data_name}.jpg')
                cv2.imwrite(
                    save_img_path,
                    visualizer.visualize_grid(
                        {'pose_ref_shape': visdict['color_shape_images'].clone()}, size=512)
                )
                
            if saveMat: 
                mmcv.dump(pixie_save_data_list,os.path.join(savefolder, f'{data_name}.pkl'))
              
if __name__ == '__main__':
    api_multi_body(
        imgfolder=r'C:\Users\lithiumice\code\speech2gesture_dataset\crop\oliver\test_video\1-00_00_00-00_00_01\image',
        savefolder=r'C:\Users\lithiumice\code\speech2gesture_dataset\crop\oliver\test_video\1-00_00_00-00_00_01\test'
    )
# prediction = {
#     'vertices': verts,

#     'transformed_vertices': trans_verts,
#     'face_kpt': predicted_landmarks,
#     'smplx_kpt': predicted_joints,

#     'smplx_kpt3d': joints,
#     'joints': joints,

#     'trans_cam': trans_cam,
#     'week_cam': week_cam,
#     'shape': param_dict['shape'],
#     'exp': param_dict['exp'],

#     'param_dict_maxtrix': param_dict_maxtrix,
#     'param_dict_axis': param_dict_axis
# }
