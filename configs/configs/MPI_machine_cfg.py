
ffmpeg_path = '/usr/bin/ffmpeg'
openpose_root_path = '/content/openpose'
openpose_bin_path = 'build/examples/openpose/openpose.bin'

video_out_base_path: str = '{{ fileDirname }}/../../../speech2gesture_dataset/crop4'
intervals_csv_path: str = "{{ fileDirname }}/../data_csv/intervals_sub4.csv"
intervals_csv_path_debug: str = "{{ fileDirname }}/../data_csv/test.csv"

folder_version: int = 1
low_res: int = 1
fps=30

bs_at_a_time=50
coap_bs_at_a_time=400
o3d_opt_bs_at_a_time=400

saveVis=False
rasterizer_type='pytorch3d'
face_detector='fan'
debug_mode=False

request_gpus = 1