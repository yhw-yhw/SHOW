import os
import matplotlib
import platform

if platform.system() == 'Windows':
    matplotlib.use('TkAgg')
    
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from .utils import *
from .face_iders import *
from .loggers import *
from .image import *
from .detector import *

from mmcv.runner import OPTIMIZERS
from mmcv.runner.builder import RUNNERS


def build_optim(cfg):
    return OPTIMIZERS.build(cfg)

def build_runner(cfg):
    return RUNNERS.build(cfg)