_base_ = [
    './base/face_ider.py',
    './base/log_config.py',
    './base/optimizer_config.py',
    
    './base/model_smplx_config.py',
    './base/model_flame_config.py',
    './base/smplifyx_weights.py',
    './base/smplifyx_loss_configs.py',
    './base/smplifyx_prior_config.py',
    
    './base/betas_generate.py',
    './base/betas_precompute.py',
    
    './base/smoothnet_cfg.py',
]

basic_param_keys=[
    'expression','jaw_pose','global_orient','transl',
    'left_hand_pose','right_hand_pose','leye_pose','reye_pose','pose_embedding'
]

use_bvh=False
batch_size = -1
save_betas=False
load_betas=False
shape_path = ''

save_objs = False
save_template = False
save_smplpix = True
save_ours_images = True
save_pkl_file=True

focal_length = 5000
use_vposer = True
device = 'cuda'
dtype = "float32"
output_img_ext = 'png'
img_save_mode='origin'

load_tracker_checkpoint=False
tracker_checkpoint_root=''

start_stage=0
end_stage=None
load_checkpoint=False
check_pkl_metric=False
load_ckpt_st_stage=-1
load_ckpt_ed_stage=None
checkpoint_pkl_path=''

use_pre_compute_betas=False
use_hand_pose_filter=False
use_smoothnet_hands=False
use_smoothnet_pose=False
re_optim_hands=False
use_pymaf_hand=True
# csv parser will: 
# 1. add paths configs
# 2. modify config above