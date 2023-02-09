import os
import sys
from loguru import logger
import subprocess
import torch

class path_enter(object):
    def __init__(self,target_path=None):
        self.origin_path=None
        self.target_path=target_path
    
    def __enter__(self):
        if sys.path[0]!=self.target_path:
            sys.path.insert(
                0,self.target_path
            )
            
        if self.target_path:
            self.origin_path=os.getcwd()
            os.chdir(self.target_path)
            logger.info(f'entered: {self.target_path}; origin_path: {self.origin_path}')
    
    def __exit__(self, exc_type, exc_value, trace):
        if self.origin_path:
            os.chdir(self.origin_path)
            logger.info(f'exit to origin_path: {self.origin_path}')

   
def run_smplifyx_org(
    image_folder, 
    output_folder,
    smplifyx_code_dir,
    log_cmds=True,
    **kwargs,
):        
    with path_enter(smplifyx_code_dir):
        data_folder=os.path.dirname(image_folder)
        cmds=[
            'python smplifyx/main.py --config cfg_files/fit_smplx.yaml',
            '--data_folder', data_folder,
            '--output_folder',output_folder,
            '--img_folder','image',
            '--keyp_folder','op',
            '--model_folder ../../../models/smplx_model',
            '--vposer_ckpt ../../../models/vposer_v1_0',
            '--visualize="True"',
            # '--visualize="False"',
        ]
        cmds=[str(i) for i in cmds]
        cmds=' '.join(cmds)
        if log_cmds:
            logger.info(f'log_cmds: {cmds}')
        subprocess.run(cmds,shell=True)
        logger.info(f'done')
        
        
def run_pymafx(
    image_folder, 
    output_folder,
    pymaf_code_dir,
    log_cmds=True,
    no_render=True,
):
    with path_enter(pymaf_code_dir):
        cmds=[
            'python apps/demo_smplx.py',
            # 'python -m apps.demo_smplx',
            '--image_folder',f'"{image_folder}"',
            '--output_folder',f'"{output_folder}"',
            '--detection_threshold 0.3',
            ]
        
        if no_render:
            cmds+=['--no_render']
            
        cmds+=[
            '--pretrained_model data/pretrained_model/PyMAF-X_model_checkpoint.pt',
            '--misc TRAIN.BHF_MODE full_body MODEL.EVAL_MODE True MODEL.PyMAF.HAND_VIS_TH 0.1'
        ]
        
        cmds=[str(i) for i in cmds]
        cmds=' '.join(cmds)
        if log_cmds:
            logger.info(f'log_cmds: {cmds}')
        subprocess.run(cmds,shell=True)
        logger.info(f'run_pymafx done')
        

def run_psfr(
    image_folder,
    image_sup_folder,
    log_cmds=True,
):
    psfr_code_dir=os.path.join(os.path.dirname(__file__),'../../modules/PSFRGAN')
    with path_enter(psfr_code_dir):
        cmds=[
           'python test_enhance_dir_unalign.py',
           '--src_dir',f'"{image_folder}"', 
           '--results_dir',f'"{image_sup_folder}"', 
        ]
        cmds=[str(i) for i in cmds]
        cmds=' '.join(cmds)
        if log_cmds:
            logger.info(f'log_cmds: {cmds}')
        subprocess.run(cmds,shell=True)
        logger.info(f'done')
        
     