#  -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Authors: paper author. 
# Special Acknowlegement:  Wojciech Zielonka and Justus Thies
# Contact: ps-license@tuebingen.mpg.de

import os
from glob import glob
from pathlib import Path

import PIL.Image as Image
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

import face_alignment
from .detector.face_detector import FaceDetector
from .image import landmark_crop, crop_image_bbox, crop_image, squarefiy
from SHOW.face_iders import match_faces
from SHOW.image import lmk2d_to_bbox
from .utils.decorator import to_tensor
from loguru import logger
import SHOW


class ImagesDataset(Dataset):
    def __init__(
        self,
        config,
        start_frame=0,
        bbox=None,

        person_face_emb: np.ndarray = None,
        face_ider=None,
        face_detector=None,
        face_detector_mediapipe=None,
    ):
        
        if False:
            ref_len = SHOW.img_files_num_in_dir(config.img_folder)
            if SHOW.ext_files_num_in_dir(config.img_sup_folder, exts=['*.png', '*.jpg']) < ref_len:
                SHOW.datasets.pre_runner.run_psfr(
                    image_folder=config.img_folder,
                    image_sup_folder=config.img_sup_folder,
                    log_cmds=True,
                )
            
            source = Path(config.img_sup_folder)
        else:
            source = Path(config.img_folder)
            

        self.face_ider = face_ider
        self.person_face_emb = person_face_emb

        self.device = 'cpu'
        self.start_frame = start_frame
        self.source = source
        self.actor_name = source.name
        self.face_detector = None
        self.bbox = bbox

        self.config = config
        self.bbox_scale = config.bbox_scale
        self.make_image_square = config.make_image_square
        self.square_size = config.square_size

        self.face_detector_mediapipe = face_detector_mediapipe
        self.face_detector = face_detector


        assert(Path(self.source).exists())

        # if self.source == '/top1/top2/source.mp4'
        # then self.source = '/top1/top2/image'
        if self.source.suffix in ['.mp4', '.avi']:
            parent = Path(self.source).parent
            img_path = Path(parent).joinpath('images')
            save_folder = Path(parent).joinpath('save_folder')
            self.config.save_folder = str(save_folder)

            if not Path(img_path).exists():
                img_path.mkdir(parents=True, exist_ok=True)
                os.system(
                    f'ffmpeg -i {self.source} -vf fps=10 -q:v 1 {img_path}/%05d.jpg')

            self.images = sorted(
                glob(f'{img_path}/*.jpg') + glob(f'{img_path}/*.png'))
            self.img_path = img_path
        else:
            self.images = sorted(
                glob(f'{self.source}/*.jpg') + glob(f'{self.source}/*.png'))
            if len(self.images)==0:
                print(f'image len == 0, exit')
                import sys
                sys.exit()
                
        starting = self.start_frame
        if self.config.use_keyframes:
            starting = self.config.keyframes[0]
            end = self.config.keyframes[1]
            print(f'[WARN] starting:{starting}, end:{end}')
            self.images = self.images[starting:end]
        else:
            self.images = self.images[starting:]

        if config.shape_path == '' and Path(config.shape_path).exists():
            self.shape_path = (config.shape_path)
        else:
            from modules.MICA.api_MICA import api_MICA
            mica_ins = api_MICA()
            
            import tempfile
            shape_id=os.urandom(24).hex()
            mica_out_npy = os.path.join(tempfile.gettempdir(), f'{shape_id}.npy')
            self.shape_path = mica_out_npy
            mica_ins.predict(
                input_img_path=self.images[0],
                output_ply_path=None,
                output_render_path=None,
                output_param_npy_path=mica_out_npy
            )
            del mica_ins
            
        if self.shape_path != '':
            assert(Path(self.shape_path).exists())
            self.shape = np.load(self.shape_path)
            self.shape = torch.from_numpy(self.shape)
        else:
            self.shape = np.zeros((1, 3))
            
        self.correspond_center=None

    def get_lmk_face(self, image, lmk_path, save_npz=True, save_empty=True):
        lmks = self.face_detector.get_landmarks(image)  # list

        Path(lmk_path).parent.mkdir(parents=True, exist_ok=True)

        if lmks is not None:
            lmks = np.array(lmks)
            if save_npz:
                np.save(lmk_path, lmks)
        return lmks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imagepath = self.images[index]
        image_tem = Path(imagepath).stem
        cv2_image = cv2.imread(imagepath)
        pil_image = Image.open(imagepath).convert("RGB")
        orig_width, orig_height = pil_image.size
        image = F.to_tensor(pil_image)

        ############################################################
        lmk_root = Path(self.config.fan_npy_folder)
        lmk_root.mkdir(exist_ok=True, parents=True)
        lmk_path = lmk_root.joinpath(f'{image_tem}.npy')
        
        if not os.path.exists(lmk_path):
            lmk_list = self.get_lmk_face(np.array(pil_image), lmk_path)
        else:
            lmk_list = np.load(lmk_path, allow_pickle=True)

        lmk=None
        dist1=dist2=np.inf
        is_person_deted = False
        
        if lmk_list is not None and lmk_list.any():
            
            if self.config.get('no_use_face_match',None) and self.config.no_use_face_match:
                
                lmk=lmk_list[0]
                is_person_deted=True
                logger.info(f'no_use_face_match')
                
            else:
                    
                if self.correspond_center is None:
                    self.correspond_center, _ = match_faces(
                        cv2_image, 
                        self.face_ider, 
                        self.person_face_emb)
                if self.correspond_center is None:
                    logger.warning(f'correspond_center is None, idx:{index}')
                    logger.info(f'lmk_list len:{lmk_list.shape}')

                if (
                    len(lmk_list)==1
                ):
                    lmk=lmk_list[0]
                    is_person_deted=True
                    logger.info(f'lmk_list len == 1')
                elif (
                    self.correspond_center is not None and
                    is_person_deted == False
                ):
                    for face_kpt in (lmk_list):
                        bbox = lmk2d_to_bbox(face_kpt, orig_height, orig_width)
                        xmin, ymin, xmax, ymax = [int(i) for i in bbox]
                        cur_center = [int((xmin+xmax)/2), int((ymin+ymax)/2)]
                        dist1 = np.abs(cur_center[0]-self.correspond_center[0])
                        dist2 = np.abs(cur_center[1]-self.correspond_center[1])
                        dist1/=orig_width
                        dist2/=orig_height
                        self.match_thres=0.09#30/720=0.04
                        if dist1 < self.match_thres and dist2 < self.match_thres:
                            lmk = face_kpt
                            self.correspond_center=cur_center
                            is_person_deted=True
                            logger.info('matched face from lmk')
                            break
                    logger.info(f'match failed: {dist1} {dist2}')
                        
        ####################################################
        if is_person_deted:
            cropped_lmk, cropped_dense_lmk, cropped_image, self.bbox = landmark_crop(
                    image, lmk, lmk, bb_scale=self.bbox_scale)

            px = py = -1
            cropped_image, cropped_lmk, cropped_dense_lmk, px, py = squarefiy(
                cropped_image, cropped_lmk, cropped_dense_lmk, size=self.square_size)

            _, h, w = cropped_image.shape
            bbox = self.bbox

        else:
            logger.info(f'is_person_deted False')
            self.correspond_center=None
            
            lmk = np.zeros((68, 2))
            cropped_lmk = np.zeros((68, 2))
            
            cropped_image = torch.zeros((3, 512, 512))
            cropped_dense_lmk = np.zeros((478, 2))
            
            bbox = {'xb_min': 0, 'xb_max': 0, 'yb_min': 0, 'yb_max': 0}
            _, h, w = cropped_image.shape
            # h=w=512
            px = py = -1


        return {
            'image': image,
            'lmk': torch.from_numpy(lmk).float(),
            'cropped_lmk': torch.from_numpy(cropped_lmk).float(),
            'cropped_image': cropped_image,
            
            'index': index,
            'is_person_deted': is_person_deted,
            'actor_name': self.actor_name,
            'shape': (self.shape),
            'bbox': bbox,
            
            'h':h,
            'w':w,
            'px':px,
            'py': py,
            
            'info': {
                'height': h,
                'width': w,
                'orig_width': orig_width,
                'orig_height': orig_height,
                'px': px,
                'py': py},
        }
