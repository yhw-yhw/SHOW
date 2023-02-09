from abc import ABCMeta, abstractmethod
from loguru import logger
import torch
from pathlib import Path
import numpy as np
import os.path as osp
import shutil
import os


@logger.catch
def img_preprocess(img):
    # img: 0-1
    # img: tensor or ndarray
    # img: (1,c,h,w)
    # img: (c,h,w)
    # img: (h,w,c)
    # img: cpu or gpu
    # img: frad or no_grad
    if isinstance(img,torch.Tensor):
        img=img.cpu().detach()
        if img.ndimension()==4:
            img=img[0]
        if img.shape[0]==3 or img.shape[0]==1:
            img=img.permute(1,2,0)
            
    if isinstance(img,np.ndarray):
        if img.ndim==4:
            img=img[0]
        if img.shape[0]==3 or img.shape[0]==1:
            img=img.transpose(1,2,0)
    # return: (h,w,3);0-1
    return img



class my_logger(ABCMeta):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def log(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def log_bs(self,*args,**kwargs):
        pass

    @abstractmethod
    def update_config(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def log_image(self,*args,**kwargs):
        pass
    
    def alert(self,*args,**kwargs):
        pass