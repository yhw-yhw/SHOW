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
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

from . import detectors

def video2sequence(video_path):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, testpath=None, iscrop=True, crop_size=224, scale=1.25, face_detector='fan'):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        def glob_exts_in_path(path, img_ext=['png', 'jpg']):
            from functools import reduce
            return reduce(
                lambda before, ext: before+glob(
                    os.path.join(path, f'*.{ext}')
                ),
                [[]]+img_ext)
            
        if testpath==None:
            self.imagepath_list = []
        elif isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list=glob_exts_in_path(testpath)
            # self.imagepath_list=[os.path.join(testpath, i) for i in os.listdir(testpath)]
            # self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            import pdb;pdb.set_trace()
            raise ValueError("testpath error")
            
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
        elif face_detector == 'mtcnn':
            self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center
    
    def cvt_data(self,image):
        return self.__getitem__(index=0,extra=image)

    def __getitem__(self, index, extra=None):
        if extra is None:
            imagepath = self.imagepath_list[index]
            imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
            image = np.array(imread(imagepath))
        else:
            image=extra
            imagename='000000'
            
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]
        h, w, _ = image.shape

        bbox_list, bbox_type = self.face_detector.run(image)
        image = image/255.
        
        ###############
        if bbox_list is None or not bbox_list.any():
            return {'name': imagename,'is_missing':True}
        
        return_list=[]
        for bbox in bbox_list:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([
                [center[0]-size/2, center[1]-size/2],
                [center[0] - size/2, center[1]+size/2],
                [center[0]+size/2, center[1]-size/2]])
            
            # xmin,ymin,xmax,ymax
            new_bbox=[center[0]-size/2, center[1]-size/2,center[0] + size/2, center[1]+size/2]
            DST_PTS = np.array([
                [0,0], 
                [0,self.resolution_inp - 1], 
                [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            
            dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
            dst_image = dst_image.transpose(2,0,1)
            
            return_item= {
                    'dst_image': torch.tensor(dst_image).float(),
                    'imagename': imagename,
                    'tform': torch.tensor(tform.params).float(),
                    'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    'bbox':new_bbox
                    }
            return_list.append(return_item)
        return return_list
        ###############
        