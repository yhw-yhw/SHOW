_base_ = [
    './base/model_smplx_config.py',
    './base/model_flame_config.py',
    './base/log_config.py',
    './base/face_ider.py',
]
batch_size = -1
shape_path = ''

device = 'cuda'
dtype = "float32"
output_img_ext = 'png'

exp_weight = 0.02
batch_size = 1
w_pho = 10.
w_lmks = 2.

sampling = 0
keyframes = []
use_keyframes = False
warmup_steps = 1
bbox_scale = 2.5
make_image_square = True
square_size = 512
use_kinect = False
use_mediapipe = True
optimize_shape = False
image_size = [512, 512]
config_name = ''
save_final_vis=True
save_final_flame=True

iters=149
bs_at_a_time=3
sampling=[1/2,1,2]

use_face_upsample=False
