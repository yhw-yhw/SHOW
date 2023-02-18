ffmpeg_path = 'ffmpeg'
# openpose_root_path = r'C:\Users\lithiumice\code\openpose'
# openpose_bin_path = r'bin\OpenPoseDemo.exe'
openpose_root_path = '/content/openpose'
openpose_bin_path = 'build/examples/openpose'

video_out_base_path: str = '{{ fileDirname }}/../../../speech2gesture_dataset/crop'
# intervals_csv_path: str = "{{ fileDirname }}/../data_csv/data_merge.csv"
intervals_csv_path: str = r'C:\Users\lithiumice\code\speech2gesture_dataset\raw_videos\TED\TED.csv'
intervals_csv_path_debug: str = "{{ fileDirname }}/../data_csv/test.csv"

folder_version: int = 1
low_res: int = 1
fps=15

# 跑face tracker的bs
bs_at_a_time=3
sampling=[1/2,1,2]
# sampling=[1/4,1/2,1]

# 计算COAP pen loss时的vertice bs
coap_bs_at_a_time=150

# 计算sil loss时pytorch3d的bs
o3d_opt_bs_at_a_time=1

saveVis=True
rasterizer_type='pytorch3d'
# face_detector='mtcnn'
face_detector='fan'
debug_mode=True


# 只有在MPI服务器上才开启
use_silhouette_loss=False
use_pymaf_hand=False