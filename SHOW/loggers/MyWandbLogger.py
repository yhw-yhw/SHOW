


import torch
from pathlib import Path
import numpy as np
import os.path as osp
import shutil
import os

from mmcv.runner.hooks.logger import WandbLoggerHook
from loguru import logger
from .builder import MMYLOGGER
from .base import *


    
@MMYLOGGER.register_module()
class MyWandbLogger(WandbLoggerHook):
    def __init__(self,wandb_key,wandb_name,*args, **kwargs):
        os.environ['WANDB_API_KEY'] = wandb_key
        os.environ['WANDB_NAME'] = wandb_name
        super().__init__(*args, **kwargs)
        
    def initialize(self):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
            
    @logger.catch
    def log(self, tag_name,tag_value,**kwargs):
        self.wandb.log({tag_name:tag_value})
        
    @logger.catch
    def log_bs(self,dict_to_log={},**kwargs):
        self.wandb.log(dict_to_log)

    @logger.catch
    def update_config(self,config_dict):
        self.wandb.config.update(config_dict)
        
    @logger.catch
    def alert(self,title='',msg=''):
        self.wandb.alert(
            title=title, 
            text=msg
        )
        
    @logger.catch
    def log_image(self,key:str,img):
        img=img_preprocess(img)*255
        upload_list=[self.wandb.Image(img,caption=key)]
        self.wandb.log({key:upload_list})
        
        
     