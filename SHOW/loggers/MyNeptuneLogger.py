


import torch
from pathlib import Path
import numpy as np
import os.path as osp
import shutil
import os

from mmcv.runner.hooks.logger import NeptuneLoggerHook
from loguru import logger
from .builder import MMYLOGGER
from .base import *



@MMYLOGGER.register_module()
class MyNeptuneLogger(NeptuneLoggerHook):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        from neptune.new.types import File
        self.File=File
        
    def initialize(self):
        if self.init_kwargs:
            self.run = self.neptune.init(**self.init_kwargs)
        else:
            self.run = self.neptune.init()

    @logger.catch
    def log(self, tag_name:str, tag_value, append=True):
        if append:
            self.run[tag_name].log(tag_value)
        else:
            self.run[tag_name]=(tag_value)
    
    @logger.catch
    def log_bs(self,dict_to_log={},append=True):
        for key,val in dict_to_log.items():
            # if self.is_scalar(val):
                self.log(key,val,append=append)

    @logger.catch
    def log_image(self,key:str,img):
        img=img_preprocess(img)
        self.run[key].upload(self.File.as_image(img))
        
    @logger.catch
    def update_config(self,config_dict):
        self.run['parameters']=config_dict
        
    @logger.catch
    def create_proj(self,
                    workshops='lithiumice',
                    proj_name='smplifyx',
    ):
        from neptune import management
        api_token=self.init_kwargs.api_token

        management.get_project_list(api_token=api_token)
        management.create_project(
            name=f'{workshops}/{proj_name}', 
            key=proj_name, 
            api_token=api_token, 
            visibility='pub')