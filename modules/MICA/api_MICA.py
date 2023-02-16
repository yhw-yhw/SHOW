import argparse
import os
import random
from glob import glob
from pathlib import Path

import matplotlib
import platform
if platform.system() == 'Windows':
    matplotlib.use('TkAgg')

import sys
sys.path.insert(0,
    os.path.join(
        os.path.dirname(__file__),'./'
    )
)
    
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from pytorch3d.io import save_ply
from skimage.io import imread
from tqdm import tqdm

from .micalib.config import get_cfg_defaults
from .datasets.creation.util import get_arcface_input, get_center
from .micalib import util

from typing import Union,NewType,Optional

class api_MICA(object):
    def __init__(self):
        cfg = get_cfg_defaults()
        device = 'cuda:0'
        cfg.model.testing = True
        self.mica = util.find_model_using_name(model_dir='micalib', model_name=cfg.model.name)(cfg, device)
        self.load_checkpoint(self.mica,model_path=cfg.pretrained_model_path)
        self.mica.eval()

        self.app = FaceAnalysis(name='antelopev2', 
                                providers=['CPUExecutionProvider'],
                                # providers=['CUDAExecutionProvider']
                                )
        self.app.prepare(ctx_id=0, det_size=(224, 224))
        logger.info('MICA api init done.')

    def read_arcface(self,input_img_path):
        def process(img:Union[np.ndarray,str],
                    image_size=224
        ):
            if isinstance(img,str):
                img = cv2.imread(img)
            bboxes, kpss = self.app.det_model.detect(
                img, 
                max_num=0, 
                metric='default'
            )
            
            if bboxes.shape[0] == 0:
                # return None, None
                logger.error(f'No face detected in {img_path}')
                import sys
                sys.exit(0)
            
            i = get_center(bboxes, img)
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(
                bbox=bbox, 
                kps=kps, 
                det_score=det_score
            )
            blob, aimg = get_arcface_input(face, img)
            crop_im=face_align.norm_crop(img, 
                                        landmark=face.kps, 
                                        image_size=image_size)
            np.save('tmp.npy', blob)
            cv2.imwrite('tmp.jpg', crop_im)
            return blob,crop_im
        
        blob,crop_im=process(input_img_path)
                    
        if 0:
            img_path='tmp.jpg'
            image = imread(img_path)[:, :, :3]
        else:
            image=crop_im
            
        image = image / 255.
        image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
        image = torch.tensor(image).cuda()[None]

        if 0:
            path='tmp.npy'
            arcface = np.load(path)
        else:
            arcface=blob
        
        arcface = torch.tensor(arcface).cuda()[None]
        # logger.info('arcface shape: {}'.format(arcface.shape))
        logger.info(f'read and process arcface done.')
        return image, arcface

    def load_checkpoint(self,
                        mica,
                        model_path:Union[str,Path], 
    ):
        checkpoint = torch.load(model_path)
        if 'arcface' in checkpoint:
            mica.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            mica.flameModel.load_state_dict(checkpoint['flameModel'])

    def predict(self,
                input_img_path,
                output_ply_path=None,
                output_render_path=None,
                output_param_npy_path=None,
    ):
        faces = self.mica.render.faces[0].cpu()
        with torch.no_grad():
            images, arcface = self.read_arcface(input_img_path)
            
            codedict = self.mica.encode(images, arcface)
            opdict = self.mica.decode(codedict)
            meshes = opdict['pred_canonical_shape_vertices']
            code = opdict['pred_shape_code']
            mesh = meshes[0]
            
            rendering = self.mica.render.render_mesh(mesh[None])
            image = (rendering[0]
                     .cpu()
                     .numpy()
                     .transpose(1, 2, 0)
                     .copy() 
                     * 255)[:, :, [2, 1, 0]]
            
            image = np.minimum(
                np.maximum(image, 0), 255
            ).astype(np.uint8)
            
            if output_render_path:
                Path(output_render_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_render_path, image)
                logger.info(f'MICA img: {output_render_path}')
                
            if output_ply_path:
                Path(output_ply_path).parent.mkdir(parents=True, exist_ok=True)
                save_ply(output_ply_path, 
                        verts=mesh.cpu() * 1000.0, 
                        faces=faces)
                logger.info(f'MICA ply: {output_ply_path}')
                
            if output_param_npy_path:
                Path(output_param_npy_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(output_param_npy_path, 
                        code[0].cpu().numpy())
                logger.info(f'MICA npy: {output_param_npy_path}')

