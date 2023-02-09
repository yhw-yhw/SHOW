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
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat,loadmat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    
    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config = deca_cfg, device=device)
    
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        tobj_list=testdata[i]
        if not tobj_list:
            print('no face detected! skip')
            continue
        
        original_image=None
        tobj_returns_list={}
        for index,tobj in enumerate(tobj_list):
            name = tobj['imagename']
            images = tobj['dst_image'].to(device)[None,...]

            with torch.no_grad():
                codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict) #tensor
                if args.render_orig:
                    tform = tobj['tform'][None, ...]
                    tform = torch.inverse(tform).transpose(1,2).to(device)
                    
                    if original_image is None:
                        original_image = tobj['original_image'][None, ...].to(device)
                    elif orig_visdict['shape_images'] is not None:
                        original_image=orig_visdict['shape_images']
                    # else:
                    #     raise ValueError('org img error')
                    
                    _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    

            tobj_returns = util.dict_tensor2npy(
                {
                'bbox':tobj['bbox'],
                'tform': tobj['tform'],
                'imagename': tobj['imagename'],
                'landmarks2d':opdict['landmarks2d'],
                'landmarks3d':opdict['landmarks3d'],
                'shape_params':codedict['shape'], 
                'expression_params':codedict['exp'], 
                'pose_params':codedict['pose']
            })
            # tobj_returns_list.append(tobj_returns)
            tobj_returns_list[f'face{index}']=tobj_returns
            
            
        if args.saveVis:
            cv2.imwrite(
                os.path.join(savefolder, name + '.jpg'), 
                deca.visualize({
                "shape_images":orig_visdict['shape_images'],
                }))
        
        if args.saveMat:
            savemat(
                os.path.join(savefolder, name + '.mat'), 
                tobj_returns_list
                )
            
            
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    
    parser.add_argument('--use_spec_pos', default=False, type=lambda x: x.lower() in ['true', '1'])
    # parser.add_argument('--save_result_image', default=True, type=lambda x: x.lower() in ['true', '1'])
    
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())