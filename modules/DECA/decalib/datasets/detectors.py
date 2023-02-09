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

import numpy as np
import torch

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image,**kwargs):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        h,w,_=image.shape
        out = self.model.get_landmarks(image)
        
        if out:
            bbox_list=[]
            for i in out:
                kpt=i.squeeze()
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                bbox = [left,top, right, bottom]
                bbox_list.append(bbox)  
            bbox_list=np.stack(bbox_list,axis=0)          
            return (bbox_list, 'kpt68')
        else:
            return (None,None)
            

class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
        
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        out=out[0]
        
        if out.any():
            return (out.squeeze(0), 'bbox')
        else:
            return (None,None)
        


