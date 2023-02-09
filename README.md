# SHOW
<b>Generating Holistic 3D Human Motion from Speech</b>

[[Project Page](https://talkshow.is.tue.mpg.de)] [[Arxiv](export.arxiv.org/abs/2212.04420)] 

![Teaser SHOW](doc/images/overview.png)

## Description

> This repository provides the official implementation of SHOW(Synchronous HOlistic body in the Wild)

> Generating Holistic 3D Human Motion from Speech: This work addresses the problem of generating 3D holistic body motions from human speech. Given a speech recording, we synthesize sequences of 3D body poses, hand gestures, and facial expressions that are realistic and diverse. To achieve this, we first build a high-quality dataset of 3D holistic body meshes with synchronous speech. 

## Demo

![demo](doc/images/rec_results_detial.png)

## Installation

To install SHOW, execute:
```bash
pip install git+https://github.com/yhw-yhw/SHOW.git
cd SHOW && pip install -v -e .
```

Note that Pytorch3D may require manuall installation (see instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)).

```bash
conda create -n env_SHOW python=3.9

eval "$(conda shell.bash hook)"

conda activate env_SHOW

conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

pip install -r tools/env/req_parts/t0.txt
pip install -r tools/env/req_parts/t1.txt
pip install -r tools/env/req_parts/t2.txt
pip install -r tools/env/req_parts/t3.txt
pip install -r tools/env/req_parts/t4.txt
```

in `{path_to_envs}/env_SHOW/lib/python3.9/site-packages/torchgeometry/core/conversions.py` change:

```python
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~ mask_d0_d1)
    mask_c2 = (~ mask_d2) * mask_d0_nd1
    mask_c3 = (~ mask_d2) * (~ mask_d0_nd1)
```

## Datasets
  
### Download Links

- [[Dropbox](https://www.dropbox.com/sh/f1gu531w5s2sbqd/AAA2I7oLolEkcXnWI6tnwUpAa?dl=0)]
- Google Drive: [TODO]

### Dataset Videos Download

#### Prerequisits for data download

download all videos from `TODO` or youtube, please refer to (https://github.com/amirbar/speech2gesture), or using the following script: ```tools\datasets\download_youtube.py```, remember to install `yt-dlp`.


csv columns in `tools\datasets\SHOW_intervals_subject4.csv`:
```
speaker
video_fn
interval_id
dataset
start_time
end_time
```

### Visualize Dataset

```bash
python render_pkl_release.py \
--pkl_file_path test/all.pkl \
--out_images_path test/ours_images \
--output_video_path test/ours.mp4 \
--smplx_model_path models/smplx/SMPLX_NEUTRAL_2020_org.npz
```

### Dataset Description

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


## Code Structure

[code readme](doc/code.md)

this code `main.py` does the following things:

- crop intervals from videos
- prepare data from PIXIE、DECA、Openpose、PyMAF-X、FAN, etc.
- do smplifyx optimazation
- refine facial expression

## Citation

```
@misc{yi2022generating,
    title={Generating Holistic 3D Human Motion from Speech},
    author={Hongwei Yi and Hualin Liang and Yifei Liu and Qiong Cao and Yandong Wen and Timo Bolkart and Dacheng Tao and Michael J. Black},
    year={2022},
    eprint={2212.04420},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

For questions, please contact `hongwei.yi@tuebingen.mpg.de` or `fthualinliang@mail.scut.edu.cn`
