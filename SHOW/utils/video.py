import shutil
from glob import glob
import os
import cv2
from tqdm import tqdm
import subprocess
import os.path as osp
from loguru import logger
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union


def video():
    for actor in tqdm(filter(os.path.isdir, glob(f'./test_results/*_rgb_*'))):
        os.system(f'ffmpeg -y -framerate 30 -pattern_type glob -i \'{actor}/video/*.jpg\' -c:v libx264 {actor}.mp4')


@logger.catch
def video_to_images(
    vid_file,
    prefix='',
    start=0,
    duration=10,
    img_folder=None,
    return_info=False,
    fps=15
):
    '''
    From https://github.com/mkocabas/VIBE/blob/master/lib/utils/demo_utils.py

    fps will sample the video to this rate.
    '''
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-r', str(fps),
               '-ss', str(start),
               '-t', str(duration),
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/{prefix}%06d.jpg']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')


@logger.catch
def frame_to_video(
    image_path,
    video_name,
    fps=9,
    cut=-1,
    size = (960,540)
):
    # fps=15,
    # size = (960,540)
    # size = (480,720)
    logger.info(f'video_name: {video_name}')
    f_name=Path(video_name).stem
    ext_name=Path(video_name).suffixes[-1]
    
    
    
    if not Path(image_path).exists():
        logger.warning(f'image_path not exist!')
        return
        
    filelist = os.listdir(image_path)
    if filelist == []:
        logger.warning(f'the filelist is empty!')
        return
    
    # assert(filelist!=[],'the filelist is empty')
    
    filelist=filter(lambda x:(x[-3:]=='jpg') or (x[-3:]=='png'),filelist)
    filelist=list(filelist)
    filelist.sort(key= lambda x:int(x[:-4].split('_')[0]))
    
    if cut!=-1: 
        filelist=filelist[:cut]
    if size==-1:
        size = cv2.imread(os.path.join(image_path, filelist[0])).shape[:2][::-1]
        
    if ext_name=='.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif ext_name=='.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'I420')* 'mp4v'
    else:
        # raise ValueError("Unknown ext_name")
        logger.warning(f'Unknown ext_name:{ext_name}')
        return
    
    
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    for item in tqdm(filelist,desc='frame to video ing...'):
        img_path=os.path.join(image_path, item)
        if not os.path.exists(img_path):
            logger.warning(f'{img_path} not exit...')
            break
        
        if item.endswith('.jpg') or item.endswith('.png'):
            img = cv2.imread(img_path)
            img2=cv2.resize(img,size)
            video.write(img2)
    video.release()
    logger.info('frame_to_video finished')
    

def images_to_sorted_images(input_folder, output_folder, img_format='%06d'):
    """Copy and rename a folder of images into a new folder following the
    `img_format`.

    Args:
        input_folder (str): input folder.
        output_folder (str): output folder.
        img_format (str, optional): image format name, do not need extension.
            Defaults to '%06d'.

    Returns:
        str: image format of the rename images.
    """
    img_format = img_format.rsplit('.', 1)[0]
    file_list = []
    os.makedirs(output_folder, exist_ok=True)
    pngs = glob(os.path.join(input_folder, '*.png'))
    if pngs:
        ext = 'png'
    file_list.extend(pngs)
    jpgs = glob(os.path.join(input_folder, '*.jpg'))
    if jpgs:
        ext = 'jpg'
    file_list.extend(jpgs)
    file_list.sort()
    for index, file_name in enumerate(file_list):
        shutil.copy(
            file_name,
            os.path.join(output_folder, (img_format + '.%s') % (index, ext)))
    return img_format + '.%s' % ext


def images_to_video(input_folder: str,
                    output_path: str,
                    remove_raw_file: bool = False,
                    img_format: str = '%06d.png',
                    fps: Union[int, float] = 30,
                    resolution: Optional[Union[Tuple[int, int],
                                               Tuple[float, float]]] = None,
                    start: int = 0,
                    end: Optional[int] = None,
                    disable_log: bool = False) -> None:
    """Convert a folder of images to a video.

    Args:
        input_folder (str): input image folder
        output_path (str): output video file path
        remove_raw_file (bool, optional): whether remove raw images.
            Defaults to False.
        img_format (str, optional): format to name the images].
            Defaults to '%06d.png'.
        fps (Union[int, float], optional): output video fps. Defaults to 30.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of output.
            defaults to None.
        start (int, optional): start frame index. Inclusive.
            If < 0, will be converted to frame_index range in [0, frame_num].
            Defaults to 0.
        end (int, optional): end frame index. Exclusive.
            Could be positive int or negative int or None.
            If None, all frames from start till the last frame are included.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None
    """
    # check_input_path(
    #     input_folder,
    #     allowed_suffix=[],
    #     tag='input image folder',
    #     path_type='dir')
    # prepare_output_path(
    #     output_path,
    #     allowed_suffix=['.mp4'],
    #     tag='output video',
    #     path_type='file',
    #     overwrite=True)
    
    input_folderinfo = Path(input_folder)
    num_frames = len(os.listdir(input_folder))
    start = (min(start, num_frames - 1) + num_frames) % num_frames
    end = (min(end, num_frames - 1) +
           num_frames) % num_frames if end is not None else num_frames
    temp_input_folder = None
    if img_format is None:
        temp_input_folder = os.path.join(input_folderinfo.parent,
                                         input_folderinfo.name + '_temp')
        img_format = images_to_sorted_images(input_folder, temp_input_folder)

    command = [
        'ffmpeg',
        '-y',
        '-threads',
        '4',
        '-start_number',
        f'{start}',
        '-r',
        f'{fps}',
        '-i',
        f'{input_folder}/{img_format}'
        if temp_input_folder is None else f'{temp_input_folder}/{img_format}',
        '-frames:v',
        f'{end - start}',
        '-profile:v',
        'baseline',
        '-level',
        '3.0',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-an',
        '-v',
        'error',
        '-loglevel',
        'error',
        output_path,
    ]
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
        command.insert(1, '-s')
        command.insert(2, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if remove_raw_file:
        if Path(input_folder).is_dir():
            shutil.rmtree(input_folder)
    if temp_input_folder is not None:
        if Path(temp_input_folder).is_dir():
            shutil.rmtree(temp_input_folder)

