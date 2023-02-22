import os
import shutil
import mmcv
from pathlib import Path
import os.path as osp
import glob
from functools import reduce


def get_file_size(fpath):
    statinfo = os.stat(fpath)
    size = statinfo.st_size
    return size
    
        
def recursive_walk(rootdir):
    """
    Yields:
        str: All files in rootdir, recursively.
    """
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def glob_exts_in_path(path, img_ext=['png', 'jpg']):
    return reduce(
        lambda before, ext: before+glob.glob(
            os.path.join(path, f'*.{ext}')
        ),
        [[]]+img_ext)


def find_full_impath_by_name(root, name):
    for ext in ['jpg', 'png', 'bmp', 'jpeg']:
        input_img = os.path.join(root, f'{name}.{ext}')
        if Path(input_img).exists():
            return input_img
    return None


def files_num_in_dir(dir_name):
    if not Path(dir_name).exists():
        return -1
    return len(os.listdir(dir_name))


def ext_files_num_in_dir(dir_name, exts=['*.pkl', '*.pkl.empty']):
    if not Path(dir_name).exists():
        return -1
    all_list = []
    for ext in exts:
        all_list += glob.glob(os.path.join(dir_name, ext))
    return len(all_list)


def img_files_num_in_dir(dir_name):
    if not Path(dir_name).exists():
        return -1
    i = glob.glob(os.path.join(dir_name, '*.jpg')) +\
        glob.glob(os.path.join(dir_name, '*.png'))
    return len(i)


def is_empty_dir(dir_name):
    if not os.path.exists(dir_name):
        return 1
    return int(files_num_in_dir(dir_name) == 0)


def is_notexist_file(dir_name):
    return 0 if os.path.exists(dir_name) else 1


def purge_dir(target_dir):
    if os.path.exists(target_dir):
        if os.path.isfile(target_dir):
            os.remove(target_dir)
        else:
            shutil.rmtree(target_dir)
    # os.makedirs(target_dir, exist_ok=True)

def check_makedir(dir):
    dir = osp.abspath(dir)
    root = osp.dirname(dir)
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)


def remake_dir(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)


def parse_abs_path(cureent_file_path, path):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(cureent_file_path), path)
    )


def from_rela_path(cureent_file_path, path):
    return mmcv.Config.fromfile(
        parse_abs_path(cureent_file_path, path)
    )
