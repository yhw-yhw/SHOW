import os
import sys
import torch
import mmcv
import cv2
import numpy as np
from pathlib import Path
from easydict import EasyDict
from .builder import IDER
from .base import ider_base
from loguru import logger

default_weight_path=os.path.join(os.path.dirname(__file__),
                                     '../../../models/arcface/glink360k_cosface_r100_fp16_0.1.pth')

@IDER.register_module()
class arcface_ider(ider_base):
    def __init__(self,
                 weight=default_weight_path,
                 name='r100', fp16=True, 
                 det='fan', threshold=0.45, **kwargs
                 ):
        
        self.threshold = threshold
        self.det = det

        from modules.arcface_torch.backbones import get_model
        self.net = get_model(name, fp16=fp16)
        self.net.load_state_dict(torch.load(weight))
        self.net.eval()

        if self.det == 'fan':
            import face_alignment
            self.fan = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, 
                flip_input=False)
        if self.det == 'mtcnn':
            from facenet_pytorch import MTCNN as mtcnn
            self.mt = mtcnn(keep_all=True)

    @torch.no_grad()
    def get_bbox_mtcnn(self, img):
        # image: 0-255, uint8, rgb, [h, w, 3]
        out = self.mt.detect(img[None, ...])
        # [747.456    94.97711 889.748   282.031  ]a[0]
        if out[0].any():
            return out[0].squeeze(0), 'bbox' 
        else:
            logger.warning('img det return None bbox')
            return (None, None)

    @torch.no_grad()
    def get_bbox_fan(self, img):
        # image: 0-255, uint8, rgb, [h, w, 3]
        h, w, _ = img.shape
        lmk_list = self.fan.get_landmarks(img)

        if lmk_list:
            bbox_list = []
            for lmk in lmk_list:
                kpt = lmk.squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                bbox_list.append(bbox)
            # [[746.0, 140.0, 894.0, 283.0]]
            return bbox_list, 'kpt68'
        
        logger.warning('img det return None bbox')
        return (None, None)

    @torch.no_grad()
    def get_face_info_from_img(self, img):
        # img: rgb,hw3,uint8
        ret_list = []

        if self.det == 'fan':
            bboxes, _ = self.get_bbox_fan(img)
        if self.det == 'mtcnn':
            bboxes, _ = self.get_bbox_mtcnn(img)
            
        if bboxes is None or (
            isinstance(bboxes,np.ndarray) and
            not bboxes.any()
        ):
            logger.warning(f'img det return None bbox')
            return None

        crop_im_bs = mmcv.image.imcrop(img, np.array(bboxes))

        for crop_im, bbox in zip(crop_im_bs, bboxes):
            _img = cv2.resize(crop_im, (112, 112))
            _img = np.transpose(_img, (2, 0, 1))
            _img = torch.from_numpy(_img).unsqueeze(0).float()
            _img.div_(255).sub_(0.5).div_(0.5)


            feat = self.net(_img).numpy()[0]
            feat = feat/np.linalg.norm(feat)

            ret_list.append(EasyDict({
                'normed_embedding': feat, 
                'crop_im': crop_im,
                'bbox': bbox}))

        return ret_list

    def get(self, img):
        # img: bgr,hw3,uint8
        if isinstance(img, str):
            if Path(img).exists():
                img = cv2.imread(img)
            else:
                logger.info(f'img not exists: {img}')
                return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_info = self.get_face_info_from_img(img)
        
        return face_info
