from easydict import EasyDict
import subprocess
from loguru import logger
import os
import glob
import json
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

DEFAULT_FONT_FILE_PATH = os.path.join(
    os.path.dirname(__file__), '../../../data/AdobeHeitiStd-Regular2.otf')


def ffmpeg_merge_api(
    print_cmd=False,
    loglevel='error',
    fontcolor='red',
    out_name='out.mp4',
    ffmpeg_path='ffmpeg',
    font_file_path=DEFAULT_FONT_FILE_PATH,
    title2vpath_map=None,
    resolution=None,
    
    fontsize = 'h/15',
    y = 'text_h+60',
    x = '30',
    
    addition_map='-map 0:a',
    crop=None,
):
    # crop='w:h:x:y'
    '''
        title2vpath_map=[
        [{'org':'out1.mp4'},{'pixie':'out1.mp4'},{'deca':'out1.mp4',}],
        [{'op':'out1.mp4'},{'mp':'out1.mp4'},{'ours':'out1.mp4',}]
    ]
    '''
    assert isinstance(title2vpath_map, list)
    assert isinstance(title2vpath_map[0], list)
    
    Path(out_name).parent.mkdir(exist_ok=True,parents=True)


    font_desc = f'fontfile={font_file_path}:'+\
                f'fontcolor={fontcolor}: '+\
                f'fontsize={fontsize}: x={x}: y={y}'

    row_length = len(title2vpath_map)
    col_length = len(title2vpath_map[0])

    todo_list = []
    for row_idx, row in enumerate(title2vpath_map):
        for col_idx, value in enumerate(row):
            title, vpath = list(value.items())[0]
            v_idx = row_idx * col_length + col_idx
            v_info = EasyDict(
                row_idx=row_idx,
                col_idx=col_idx,
                title=title,
                vpath=vpath,
                v_idx=v_idx,
            )
            todo_list.append(v_info)


    # if resolution is not None:
    #     height, width = resolution
    #     f = lambda info: f'[{info.v_idx}]scale={width}:{height},drawtext=text="{info.title}":{font_desc}[{info.v_idx}:v];'
    #     # f = lambda info: f'[{info.v_idx}]drawtext=text="{info.title}":{font_desc},scale={width}:{height},setdar=16/9[{info.v_idx}:v];'
    # else:
    #     f = lambda info: f'[{info.v_idx}]drawtext=text="{info.title}":{font_desc}[{info.v_idx}:v];'

    def f(info):
        ret_str=f'[{info.v_idx}]'
        if resolution is not None:
            height, width = resolution
            ret_str+=f'scale={width}:{height},'
        if crop is not None:
            ret_str+=f'crop={crop},'
        if info.title!='':
            ret_str+=f'drawtext=text="{info.title}":{font_desc},'
        ret_str=ret_str.strip(',')
        ret_str+=f'[{info.v_idx}:v];'
        return ret_str


    total_cmds = []
    total_cmds += [ffmpeg_path]
    total_cmds += [f'-i "{info.vpath}"' for info in todo_list]
    # total_cmds += ['-lavfi']
    total_cmds += ['-filter_complex']
    total_cmds += ['"']
    total_cmds += [f(info) for info in todo_list]

    vstack_list = []
    output_symbol = 'v'
    

    for row_idx in range(row_length):
        prefix = ''.join([
            f'[{info.v_idx}:v]' for info in todo_list
            if info.row_idx == row_idx
        ])
        
        if col_length > 1:
            total_cmds += [f'{prefix}hstack=inputs={col_length}[row_{row_idx}];']
            output_symbol = f'row_{row_idx}'
            vstack_list.append(f'[{output_symbol}]')
        else:
            vstack_list.append(f'{prefix}')


    if row_length > 1:
        output_symbol = 'v'
        vstack_list = ''.join(vstack_list)
        total_cmds += [
            f'{vstack_list}vstack=inputs={row_length}[{output_symbol}]'
        ]
    else:
        total_cmds[-1] = total_cmds[-1].rstrip(';')

    total_cmds += ['"']
    # total_cmds += ['-map 0:a']
    
    if addition_map!='':
        total_cmds += [addition_map]
    total_cmds += [
        '-map',
        f'"[{output_symbol}]"',
        f'"{out_name}"',
        '-y',
    ]

    if 1:
        total_cmds += [
            '-hide_banner',
            f'-loglevel {loglevel}'
        ]

    total_cmds = ' '.join(total_cmds)
    if print_cmd:
        logger.info(f'cmd: {total_cmds}')

    subprocess.run(
        total_cmds,
        shell=True,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
    )


def temporal_concat_video(
    input_path_list: List[str],
    input_title_list: List[str],
    output_path: str,
    resolution=(720, 1080 * 3),
    remove_raw_files: bool = False,
    disable_log: bool = False,
    font_file_path=DEFAULT_FONT_FILE_PATH,
    fontcolor='red',
    fontsize = 'h/20',
    y = 'text_h+60',
    x = '30',
) -> None:
    # resolution: Union[Tuple[int, int],
    #                 Tuple[float, float]] = (720, 1080*3),
    
    for path in input_path_list:
        check_input_path(path,
                         allowed_suffix=['.gif', '.mp4'],
                         tag='input video',
                         path_type='file')

    prepare_output_path(output_path,
                        allowed_suffix=['.gif', '.mp4'],
                        tag='output video',
                        path_type='file',
                        overwrite=True)


    font_desc = f'fontfile={font_file_path}:'+\
                f'fontcolor={fontcolor}: '+\
                f'fontsize={fontsize}: x={x}: y={y}'

    command = ['ffmpeg']
    concat_command = []
    scale_command = []
    for index, vid_file in enumerate(input_path_list):
        command.append('-i')
        command.append(vid_file)

        if resolution is not None:
            height, width = resolution
            scale_command.append(
                '[%d:v]drawtext=text="%s":%s,scale=%d:%d,setdar=16/9[v%d];' %
                (index, input_title_list[index], font_desc, width, height,
                 index))
        else:
            scale_command.append(
                '[%d:v]drawtext=text="%s":%s[v%d];' %
                (index, input_title_list[index], font_desc, index))

        # scale_command.append(
        #     '[%d:v]drawtext=text="%s":%s,scale=%d:%d:force_original_aspect_ratio=0[v%d];' %
        #     (index,
        #      input_title_list[index], font_desc,
        #      width, height,
        #      index))

        # scale_command.append(
        #     '[%d:v]scale=%d:%d:force_original_aspect_ratio=0[v%d];' %
        #     (index, width, height, index))

        concat_command.append('[v%d]' % index)

    concat_command = ''.join(concat_command)
    scale_command = ''.join(scale_command)
    command += [
        '-filter_complex',
        '%s%sconcat=n=%d:v=1:a=0[v]' %
        (scale_command, concat_command, len(input_path_list)), '-loglevel',
        'error', '-map', '[v]', '-c:v', 'libx264', '-y', output_path
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    if remove_raw_files:
        command = ['rm'] + input_path_list
        subprocess.call(command)


if __name__ == "__main__":

    if False:
        temporal_concat_video(
            input_path_list=[
                r'C:\Users\lithiumice\Desktop\merge.mp4',
                r'C:\Users\lithiumice\Desktop\merge.mp4',
                r'C:\Users\lithiumice\Desktop\merge.mp4'
            ],
            input_title_list=['test', 'test', 'test'],
            output_path=r'C:\Users\lithiumice\Desktop\merge2.mp4',
            fontcolor='red',
        )

    if False:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\merge.mp4',
                         title2vpath_map=[
                             [{
                                 'orgin':
                                 r'C:\Users\lithiumice\Desktop\img.mp4'
                             }, {
                                 'origin_algorithms':
                                 r'C:\Users\lithiumice\Desktop\out.avi'
                             }, {
                                 'with_texture_loss':
                                 r'C:\Users\lithiumice\Desktop\mica_org.avi',
                             }],
                         ])

    if 0:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\merge.mp4',
                        #  resolution=(720, 1080),
                        #  fontcolor='white',
                         title2vpath_map=[
                             [{
                                 '1':
                                 r'C:\Users\lithiumice\code\speech2gesture_dataset\crop\seth\Hey_Puerto_Rico-u5U0q-hh5B4.webm\96109-00_00_32-00_00_42\96109-00_00_32-00_00_42_org.mp4'
                             }, {
                                 '2':
                                 r'C:\Users\lithiumice\code\speech2gesture_dataset\crop\seth\Hey_Puerto_Rico-u5U0q-hh5B4.webm\96109-00_00_32-00_00_42\op_video.mp4'
                             }],
                         ])
    if 0:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\merge0.mp4',
                         resolution=(720, 1080),
                         fontcolor='blue',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\org.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\smplifyxout.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\pixie2_video.mp4'
                             }]
                         ])
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\merge1.mp4',
                         resolution=(720, 1080),
                         fontcolor='blue',
                         addition_map='',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\pymaf_video.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\mica_org.mp4'
                             }],
                         ])
    if 0:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\org_deeplab_mica.mp4',
                         fontcolor='blue',
                         addition_map='',
                         crop='590:650:530:50',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\org.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\wwwout.mp4'
                             },{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\mica_org.mp4'
                             }],
                         ])
        
    if 0:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\org_deeplab_mica.mp4',
                         fontcolor='blue',
                         addition_map='',
                         crop='590:650:530:50',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\org.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\wwwout.mp4'
                             },{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\mica_org.mp4'
                             }],
                         ])
        
    if 1:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\merge_all.mp4',
                         fontcolor='blue',
                         addition_map='',
                         crop='590:650:530:50',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\org.mp4'
                             },
                             {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\oooout.mp4'
                             },
                             {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\owoout.mp4'
                             },
                             {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\wwoout.mp4'
                             },
                             {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\wwwout.mp4'
                             }],
                         ])

    if 0:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\org_ooo.mp4',
                         fontcolor='blue',
                         addition_map='',
                         crop='590:650:530:50',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\org.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\oooout.mp4'
                             }],
                         ])
    if 0:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\org_pymaf.mp4',
                         fontcolor='blue',
                         addition_map='',
                         crop='590:650:530:50',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\org.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\pymaf_video.mp4'
                             }],
                         ])

    if 0:
        ffmpeg_merge_api(print_cmd=True,
                         out_name=r'C:\Users\lithiumice\Desktop\org_pymaf2.mp4',
                         fontcolor='blue',
                         addition_map='',
                         crop='590:650:530:50',
                         title2vpath_map=[
                             [{
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\org2.mp4'
                             }, {
                                 '_':
                                 r'C:\Users\lithiumice\Desktop\pymaf2.mp4'
                             }],
                         ]) 
    print(f'test done...')
