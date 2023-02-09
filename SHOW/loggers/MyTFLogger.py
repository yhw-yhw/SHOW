


import torch
from pathlib import Path
import numpy as np
import os.path as osp
import shutil
import os

from mmcv.runner.hooks.logger import TensorboardLoggerHook
from loguru import logger
from .builder import MMYLOGGER
from .base import *



@MMYLOGGER.register_module()
class MyTFLogger(TensorboardLoggerHook,my_logger):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if Path(self.log_dir).exists():
            shutil.rmtree(self.log_dir)
    
    @logger.catch
    def log(self, tags:dict,iters=0):
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, iters)
            else:
                self.writer.add_scalar(tag, val, iters)
