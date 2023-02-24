import os
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())

import SHOW
import pandas as pd
import datetime
import os.path as osp
from easydict import EasyDict
from .cfg_ins import condor_cfg
from configs.configs.speaker_info import speaker_info
from loguru import logger


def gen_path_from_ours_root(
    speaker_name,
    all_top_dir,
    ours_name='ours',
    mica_name='ours_exp',
    **kwargs
):
    ###########################################
    ours_output_folder = osp.join(
        all_top_dir, ours_name)
    
    mica_all_dir= osp.join(
        all_top_dir, mica_name)
    
    mica_org_out_video_old=osp.join(
        all_top_dir, 'mica0', 'mica_org.mp4')
    
    checkpoint_pkl_path=osp.join(
        all_top_dir, ours_name, 'all.pkl')

    all_path_dict=EasyDict(dict(
        speaker_name=speaker_name,
        all_top_dir=all_top_dir,
        mica_org_out_video_old=mica_org_out_video_old
    ))

    sep=os.path.sep
    small_video_dir_name=all_top_dir.split(sep)[-1]
    interval_video_fn=all_top_dir.split(sep)[-2]
    speaker_name=all_top_dir.split(sep)[-3]
    
    # 不需要变的文件
    raw_image_video_path = osp.join(
        all_top_dir, f'{small_video_dir_name}_org.mp4')
    
    # 不需要变的文件
    audio_output_fn=os.path.join(
        all_top_dir, f'{small_video_dir_name}.wav')
    
    # 导出用到的音频路径和别名
    audio_output_fn_arcname=os.path.join(
        speaker_name,
        interval_video_fn,
        small_video_dir_name,
        f'{small_video_dir_name}.wav')
    
    # 导出用到的pkl路径和别名
    w_mica_merge_pkl_arcname=osp.join(
        speaker_name,
        interval_video_fn,
        small_video_dir_name,
        f'{small_video_dir_name}.pkl')
    
    # wandb等用到的prefix
    logger_prefix=os.path.join(
        speaker_name,
        interval_video_fn,
        small_video_dir_name
    ).replace('\\','/')
    
    # sample等用到的prefix
    v_id_name=osp.join(
        speaker_name,
        interval_video_fn,
        small_video_dir_name)
    
    all_path_dict.update(dict(
        v_id_name=v_id_name,
        logger_prefix=logger_prefix,
        w_mica_merge_pkl_arcname=w_mica_merge_pkl_arcname,
        audio_output_fn_arcname=audio_output_fn_arcname,
        raw_image_video_path=raw_image_video_path, 
        audio_output_fn=audio_output_fn,
    ))
    
    ###########################################
    info=speaker_info.get(
        speaker_name,
        speaker_info['oliver']
    )

    extra_info=dict(
        speaker_weight = info['weight'],
        speaker_height = info['height'],
        
        load_checkpoint=False,
        load_ckpt_st_stage=-2,
        load_ckpt_ed_stage=None,
        load_tracker_checkpoint=True,
        
        tracker_checkpoint_root=mica_all_dir,
        checkpoint_pkl_path=checkpoint_pkl_path,
        checkpoint_json_path=checkpoint_pkl_path.replace(
                                            'all.pkl', 
                                            'final_metric.json'),
    )
       
            

    # 需要准备的数据文件夹
    org_v = osp.join(
        all_top_dir, 'org.mp4')
    img_folder = osp.join(
        all_top_dir, 'image')
    img_sup_folder = osp.join(
        all_top_dir, 'image_sup')
    pifpaf_output_path = osp.join(
        all_top_dir, 'paf')
    keyp_folder = osp.join(
        all_top_dir, 'op')
    deca_mat_folder = osp.join(
        all_top_dir, 'deca')
    pixie_mat_folder = osp.join(
        all_top_dir, 'pixie')
    mp_npz_folder = osp.join(
        all_top_dir, 'mp')
    fan_npy_folder = osp.join(
        all_top_dir, 'fan')
    seg_img_folder= osp.join(
        all_top_dir, 'seg')
    pymaf_pkl_folder= osp.join(
        all_top_dir, 'pymaf')
    pymaf_pkl_path= osp.join(
        all_top_dir, 'pymaf', 'image/output.pkl')
    
    all_path_dict.update(dict(
        img_folder=img_folder,
        pifpaf_output_path=pifpaf_output_path,
        mp_npz_folder=mp_npz_folder,
        keyp_folder=keyp_folder,
        deca_mat_folder=deca_mat_folder,
        pixie_mat_folder=pixie_mat_folder,
        seg_img_folder=seg_img_folder,
        fan_npy_folder=fan_npy_folder,
        pymaf_pkl_folder=pymaf_pkl_folder,
        pymaf_pkl_path=pymaf_pkl_path,
        img_sup_folder=img_sup_folder,
    ))
    
    # 可视化mp4路径
    # 可视化图片文件夹
    all_path_dict.update(dict(
        keyp_folder_vis=keyp_folder+'_vis',
        deca_mat_folder_vis=deca_mat_folder+'_vis',
        pixie_mat_folder_vis=pixie_mat_folder+'_vis',
        mp_npz_folder_vis=mp_npz_folder+'_vis',
        fan_npy_folder_vis=fan_npy_folder+'_vis',
        seg_img_folder_vis=seg_img_folder+'_vis',
        pymaf_folder_vis= os.path.join(
            all_top_dir, 'pymaf', 'image/image_output'
        ),
        
        keyp_folder_v=keyp_folder+'_video.mp4',
        deca_mat_folder_v=deca_mat_folder+'_video.mp4',
        pixie_mat_folder_v=pixie_mat_folder+'_video.mp4',
        mp_npz_folder_v=mp_npz_folder+'_video.mp4',
        fan_npy_folder_v=fan_npy_folder+'_video.mp4',
        seg_img_folder_v=seg_img_folder+'_video.mp4',
        pymaf_pkl_folder_v=pymaf_pkl_folder+'_video.mp4',
    
    ))
    
    
    # ours算法生成的数据
    all_path_dict.update(dict(
        ours_output_folder=ours_output_folder,
        ours_images_path = osp.join(
            ours_output_folder, 'out_img'),
        ours_pkl_file_path = osp.join(
            ours_output_folder, 'all.pkl'),
        final_losses_json_path= osp.join(
            ours_output_folder, 'final_metric.json'),
        output_video_path = osp.join(
            ours_output_folder, 'out.mp4'),
        w_mica_merge_pkl=osp.join(
            ours_output_folder, 'all_mica.pkl' ),
    ))
    
    
    # mica算法跑出的结果    
    render_ver=3
    all_path_dict.update(dict(
        mica_org_out_path=osp.join(
            mica_all_dir, 'mica_org'),
        mica_process_path=osp.join(
            mica_all_dir, 'mica_pro'),
        mica_merge_pkl=osp.join(
            mica_all_dir, 'final_all.pkl'),
        mica_obj_root=osp.join(
            mica_all_dir, 'obj'),
        mica_org_out_video=osp.join(
            mica_all_dir, 'final_vis.mp4'),
        mica_grid_video=osp.join(
            mica_all_dir, 'mica_grid.mp4'),
        tracker_cfg_path=osp.join(
            mica_all_dir, 'mica_cfg.yaml'),
        mica_save_path=osp.join(
            mica_all_dir, 'mica_render'),
        mica_all_dir=mica_all_dir,
        
        new_render_img_folder = osp.join(
            all_top_dir, f'new_render_img_{render_ver}'),
        new_render_img_video = osp.join(
            all_top_dir, f'new_render_{render_ver}.mp4'),
        merge_video_path = osp.join(
            all_top_dir, f'merge_ours_{render_ver}.mp4'),
        merge_pymaf_video_path = osp.join(
            all_top_dir, f'merge_pymaf_{render_ver}.mp4'),
    ))
    
    all_path_dict.update(extra_info)
    return all_path_dict