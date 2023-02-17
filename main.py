#  -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Authors: paper author. 
# Special Acknowlegement:  Wojciech Zielonka and Justus Thies
# Contact: ps-license@tuebingen.mpg.de

import mmcv
import argparse
import os
import SHOW
from stage1_main import SHOW_stage1
from stage2_main import SHOW_stage2
from configs.csv_parser import gen_path_from_ours_root
from SHOW import attr_dict
from pathlib import Path


def cvt_cfg(val):
    if val == 'True':
        return True
    elif val == 'False':
        return False
    elif val.isdigit():
        return int(val)
    else:
        return val


def parse_other_Cfg(other_cfg: list):
    parse_dict = {}
    parse_key = None
    parse_val = None

    def reg_key():
        nonlocal parse_dict
        nonlocal parse_key
        nonlocal parse_val
        if parse_key != None:
            parse_dict[parse_key] = parse_val
            parse_key = None
            parse_val = None

    for val in other_cfg:
        if val.startswith('--'):
            reg_key()
            parse_key = val.strip('--')
        else:
            if val.isdigit():
                val = int(val)
            if val == 'False':
                val = False
            if val == 'True':
                val = True
            if parse_val is None:
                parse_val = val
            elif isinstance(parse_val, list):
                parse_val.append(val)
            else:
                parse_val = [parse_val, val]
    reg_key()
    return parse_dict


def parse_overwrite_flag(over_write_cfg):

    over_write_cfg = over_write_cfg.strip(',')
    over_write_cfg = over_write_cfg.split(',')

    ret = {}
    for i in over_write_cfg:
        k, v = i.split('=')
        ret[k] = cvt_cfg(v)

    return ret


if __name__ == '__main__':
    SHOW.utils.work_seek_init()
    SHOW.utils.platform_init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker_name',
                        type=str,
                        default='-1',
                        help='if speaker_name == -1, configs are auto loaded')
    parser.add_argument(
        '--all_top_dir',
        type=str,
        default='./test/half.mp4',
    )
    parser.add_argument('--others_cfg',
                        type=str,
                        nargs=argparse.REMAINDER,
                        default=[])

    args = parser.parse_args()
    cfg = mmcv.Config(vars(args))

    cfg.over_write_cfg = parse_other_Cfg(cfg.others_cfg)
    cfg.all_top_dir = os.path.abspath(cfg.all_top_dir)

    tmp_dir = Path(cfg.all_top_dir)
    if tmp_dir.suffix.lower() in ['.mp4', '.avi', '.mov']:
        print(f'tmp_dir: {tmp_dir}')
        # all_top_dir/video.mp4
        parent = tmp_dir.parent
        cfg.all_top_dir = parent.__str__()
        img_path = parent.joinpath('image')

        if not os.path.exists(img_path):
            img_path.mkdir(parents=True, exist_ok=True)
            os.system(
                f'ffmpeg -i {tmp_dir} -vf fps=30 -q:v 1 {img_path}/%06d.png'
            )

    temp_cfg = gen_path_from_ours_root(speaker_name=cfg.speaker_name,
                                       all_top_dir=cfg.all_top_dir,
                                       **cfg.over_write_cfg)
    temp_cfg.update(cfg.over_write_cfg)
    temp_cfg = attr_dict(temp_cfg)
    temp_cfg.over_write_cfg = cfg.over_write_cfg

    if (temp_cfg.speaker_name == '-1' or temp_cfg.speaker_name == None):
        temp_cfg.speaker_name = -1

    SHOW_stage1(**temp_cfg)
    SHOW_stage2(**temp_cfg)
