
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
class MyTextLogger(object):
    def __init__(self,save_dir,filename="log.txt", mode="a",*args, **kwargs):
        from SHOW.loggers.logger import setup_logger
        setup_logger(save_dir,filename,mode=mode)
        
    @logger.catch
    def log(self, tag_name:str, tag_value,print_to_screen=False,**kwargs):
        logger.log(f"{tag_name}:{tag_value}")
        
    @logger.catch
    def log_bs(self,append=True,print_to_screen=False,**kwargs):
        for key,val in kwargs.items():
            # if self.is_scalar(val):
                self.log(key,val,print_to_screen=print_to_screen)
