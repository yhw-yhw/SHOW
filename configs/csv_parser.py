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

def csv_parse(
    interval,
    **kwargs
):
    return csv_parse_verbose(
        interval,
        video_out_base_path=condor_cfg.video_out_base_path,
        downloaded_video_base_path=condor_cfg.downloaded_video_base_path,
        folder_version=condor_cfg.folder_version,
        **kwargs
    )


def csv_parse_verbose(
    interval,
    video_out_base_path,
    downloaded_video_base_path,
    log_info=True,
    **kwargs
):
    all_path_dict=EasyDict({})
    
    interval_index = interval.index
    speaker_name = interval['speaker']
    interval_video_fn = interval['video_fn']

    if 'interval_id' in interval.index:
        interval_id = int(interval['interval_id'])
    else:
        interval_id = 0
    if 'start_time' in interval.index:
        start_time_str = interval['start_time']
    else:
        start_time_str = '00:00:00'
    if 'end_time' in interval.index:
        end_time_str = interval['end_time']
    else:
        end_time_str = '00:00:00'

    start_time_struct = datetime.datetime.strptime(start_time_str, "%H:%M:%S")
    end_time_struct = datetime.datetime.strptime(end_time_str, "%H:%M:%S")

    start_time_str = datetime.datetime.strftime(start_time_struct, "%H:%M:%S")
    end_time_str = datetime.datetime.strftime(end_time_struct, "%H:%M:%S")

    duration_time = end_time_struct-start_time_struct
    over_flow_flag = 1 if duration_time.total_seconds() > 10 else 0
    short_dur_flag = 1 if duration_time.total_seconds() < 5 else 0

    start_time_10 = start_time_struct+datetime.timedelta(seconds=10)
    start_time_10 = datetime.datetime.strftime(start_time_10, "%H:%M:%S")

    all_path_dict.update(dict(

        interval_index=interval_index,
        interval_video_fn=interval_video_fn,
        interval_id=interval_id,

        start_time_str=start_time_str,
        end_time_str=end_time_str,
        start_time_struct=start_time_struct,
        end_time_struct=end_time_struct,

        duration_time=duration_time,
        start_time_10=start_time_10,
        over_flow_flag=over_flow_flag,
        short_dur_flag=short_dur_flag,
    ))

    if speaker_name == -1:
        speaker_name = 'TED'
        
    speaker_video_path = osp.join(
        downloaded_video_base_path,
        speaker_name, "videos",
        interval_video_fn)

    small_video_dir_name='{}-{}-{}'.format(
        interval_id, 
        start_time_str,
        end_time_str
    ).replace(':', '_')

    big_video_dir = osp.join(
        video_out_base_path,
        speaker_name)
    
    all_top_dir=osp.join(
        video_out_base_path,
        speaker_name,
        interval_video_fn,
        small_video_dir_name)
    

    
    all_path_dict.update(dict(
        big_video_dir=big_video_dir,
        small_video_dir_name=small_video_dir_name,
        speaker_video_path=speaker_video_path,
    ))
    

    
    all_path_dict.update(
        gen_path_from_ours_root(
            speaker_name,
            all_top_dir,
        )
    )
    
    key_to_print=[
        'speaker_name',
        'start_time_str',
        'end_time_str',
        'small_video_dir_name',
        'interval_video_fn',
        ]
    for key in key_to_print:
        logger.info(f'{key}: {all_path_dict[key]}')

    return EasyDict(all_path_dict)


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
            mica_all_dir, 'all.pkl'),
        mica_obj_root=osp.join(
            mica_all_dir, 'obj'),
        mica_org_out_video=osp.join(
            mica_all_dir, 'mica_org.mp4'),
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