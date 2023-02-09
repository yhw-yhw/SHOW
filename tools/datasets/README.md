# Dataset Description

- speaker=oliver/chemistry/conan/seth
- The maximum length of video clip is 10s with 30 fps

- Format of files in the compressed package:
  - `{speaker}_wav_tar.tar.gz`:
    - The path format of each file is: `speaker/video_fn/seq_fn.wav`
    - Audio obtained from the original video at 22k sampling rate
  - `{speaker}_pkl_tar.tar.gz`:
    - The path format of each file is: `speaker/video_fn/seq_fn.pkl`
    - Data contained in the pkl file:
      - width，height: the video width and height
      - center: the center point of the video
      - batch_size: the sequence length
      - camera_transl: the displacement of the camera
      - focal_length: the pixel focal length of a camera
      - body_pose_axis: (bs, 21, 3)
      - expression: (bs, 100)
      - jaw_pose: (bs,3)
      - betas: (300)
      - global_orient: (bs,3)
      - transl: (bs,3)
      - left_hand_pose: (bs,12)
      - right_hand_pose: (bs,12)
      - leye_pose: (bs,3)
      - reye_pose: (bs,3)
      - pose_embedding: (bs,32)
  
    - smplx version is 2020/neutral
    - Set the config of smplx model as follows:
  
    ```python
      smplx_cfg=dict(
          model_path='path_to_smplx_model'
          model_type= 'smplx',
          gender= 'neutral',
          use_face_contour= True,
          use_pca= True,
          flat_hand_mean= False,
          use_hands= True,
          use_face= True,
          num_pca_comps= 12,
          num_betas= 300,
          num_expression_coeffs= 100,
      )
    ```

- In practice, global orient and transl parameters should be fixed as the first frame and the lower part of the body pose should be fixed as sitting or standing position: [code](post_process.py)