import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat,loadmat
import argparse
from tqdm import tqdm
import torch
import mmcv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import get_cfg_defaults
from decalib.utils.tensor_cropper import transform_points
from loguru import logger

@logger.catch
def api_multi_deca(
    savefolder,
    visfolder,
    inputpath,
    
    device='cuda',
    saveVis=True,
    saveMat=True,
    rasterizer_type='pytorch3d',
    face_detector='mtcnn',
    ):
    deca_cfg=get_cfg_defaults()
    os.makedirs(savefolder, exist_ok=True)
    os.makedirs(visfolder, exist_ok=True)
    print(f'-- please check the results in {savefolder}')
    testdata = datasets.TestData(inputpath, iscrop=True, face_detector=face_detector)
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = rasterizer_type
    deca = DECA(config = deca_cfg, device=device)
    
    for batch_list in tqdm(testdata):
        if isinstance(batch_list,dict) and batch_list.get('is_missing',None):
            name=batch_list['name']
            open(os.path.join(savefolder, f'{name}.pkl.empty'), 'a').close()
            print('no face detected! skip： {name}')
            continue

        original_image=None
        tobj_returns_list=[]
        for index,tobj in enumerate(batch_list):
            name = tobj['imagename']
            images = tobj['dst_image'].to(device)[None,...]

            with torch.no_grad():
                codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict) #tensor
                
                # args.render_orig:
                tform = tobj['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                
                if original_image is None:
                    original_image = tobj['original_image'][None, ...].to(device)
                elif orig_visdict['shape_images'] is not None:
                    original_image=orig_visdict['shape_images']
                
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    

            tobj_returns = util.dict_tensor2npy(
                {
                'bbox':tobj['bbox'],#xmin,ymin,xmax,ymax
                'tform': tobj['tform'],
                'imagename': tobj['imagename'],
                'landmarks2d':opdict['landmarks2d'],
                'shape_params':codedict['shape'], 
                'expression_params':codedict['exp'], 
                'pose_params':codedict['pose']
            })
            tobj_returns_list.append(tobj_returns)
            
            
        if saveVis:
            cv2.imwrite(
                os.path.join(visfolder, name + '.jpg'), 
                deca.visualize({
                "shape_images":orig_visdict['shape_images'],
                }))
        
        if saveMat:
            mmcv.dump(tobj_returns_list,os.path.join(savefolder, name + '.pkl'))

    print('deca ended')
            
            
        