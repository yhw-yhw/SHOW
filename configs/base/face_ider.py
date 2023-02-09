ider_cfg=dict(
    type='arcface_ider',
    threshold=0.40,
    npy_folder_name='arcface',
    weight='{{ fileDirname }}/../../../models/arcface/glink360k_cosface_r100_fp16_0.1.pth',
    name='r100',fp16=True,det='mtcnn',
)