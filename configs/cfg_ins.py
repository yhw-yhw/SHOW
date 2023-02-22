
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())

import platform
import SHOW
import os

condor_cfg = SHOW.from_rela_path(
    __file__, 
    './configs/condor_mmcv_cfg.py')

condor_cfg.is_linux = 1 if platform.system() == "Linux" else 0
gpu_info = SHOW.get_gpu_info()

condor_cfg.merge_from_dict(SHOW.from_rela_path(
    __file__,
    './configs/MPI_machine_cfg.py'))

if condor_cfg.is_linux and gpu_info is not None:
    SHOW.set_op_path_by_gpu_name(condor_cfg, gpu_info['gpu_name'])
