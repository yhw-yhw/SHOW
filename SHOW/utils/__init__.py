from .render import *
from .op_utils import *
from .disp_img import *
from .misc import *
from .decorator import *
from .paths import *
from .video import *
from .timer import *
from .fun_factory import *
from .ffmpeg import *
from .attrdict import *

import os
import cv2
import numpy as np
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map
from torchvision.transforms.functional import gaussian_blur
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import PIL
import time
from pathlib import Path
import mmcv
from loguru import logger
import math
from . import tensor2numpy



def test_temporary_val(param: np.ndarray) -> np.ndarray:
    tmp = (param[2:, ...] + param[:-2, ...] - 2 * param[1:-1, ...])
    return np.sum(np.abs(tmp))


def replace_mica_exp(mica_parts_dir,load_data, check_all_part=False):
    w_mica_parts = glob.glob( mica_parts_dir + '/w_mica_part*.pkl')
    all_num = int(Path(w_mica_parts[0]).stem.split('_')[-1])
    idx_list = [
        int(Path(part).stem.split('_')[-2]) for part in w_mica_parts
    ]
    logger.warning(f'idx_list:{idx_list}, all_num:{all_num}')
    if check_all_part:
        if len(set(idx_list)) != all_num:
            return False

    for part in w_mica_parts:  
        # part=w_mica_parts[1]
        # st,et,idx,all_num
        name = Path(part).stem
        tmp = name.split('_')
        st = int(tmp[-4])
        et = int(tmp[-3])
        logger.info(f'merging {part}')
        part_data = tensor2numpy(
            mmcv.load(part)               
        )
        merge_pkl_data_keys = [
            'expression', 
            'leye_pose', 
            'reye_pose', 
            'jaw_pose'
        ]
        for key in merge_pkl_data_keys:
            load_data[key][st:et] = part_data[key]

        for val in load_data.values():
            if isinstance(val, np.ndarray):
                if np.isnan(val).any():
                    logger.error(f'nan found')
                    return False
                
    return load_data
     
            

def cvt_cfg(val):
    if val=='True':
        return True
    elif val=='False':
        return False
    elif val.isdigit():
        return int(val)
    else:
        return val


def is_valid_json(json_path, judge_map=dict(
    all_loss=50*10000,
    loss_deca_outter=3500
)):
    
    if not Path(json_path).exists():
        logger.warning(f'not exist: {json_path}')
        return False
    
    def check_loss_val(loss_json,key,threshold):
        val=loss_json.get(key,None)
        if val is None:
            logger.warning(f'file: {loss_json}, {key} no exist, invalid')
            return False
        if math.isnan(val):
            logger.error(f'file: {loss_json}, {k}: {v} is nan, invalid')
            return False
        if val > threshold:
            logger.warning(f'file: {loss_json}, {key}: {val} > {threshold}, invalid')
            return False
        return True
    
    loss_json=mmcv.load(json_path)
    for k,v in judge_map.items():
        if not check_loss_val(
            loss_json,
            key=k,
            threshold=v
        ):
            return False
    
    return True


def to_cuda(batch):
    for key in batch.keys():
        # if 'torch.Tensor' in type(batch[key]):
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
    return batch



def parse_mask(ops):
    a = (ops['mask_images_rendering'] > 0.).float() * 0.01
    b = ((ops['alpha_images'] * ops['mask_images']) > 0.).float()
    return (a + b).detach()


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


def get_param(name, param_groups):
    for param in param_groups:
        if name in param['name']:
            return param['params'][param['name'].index(name)]



def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()


def lmk_loss(opt_lmks, target_lmks, image_size):
    h, w = image_size
    size = torch.tensor(
        [1 / w, 1 / h]).float().to(target_lmks.device)[None, None, ...]
    loss = torch.sqrt(torch.sum(torch.square(
        (opt_lmks - target_lmks) * size), dim=-1))
    return torch.mean(torch.mean(loss, 1))


def eye_closure_loss(opt_lmks, target_lmks):
    upper_eyelid_lmk_ids = [37, 38, 43, 44]
    lower_eyelid_lmk_ids = [41, 40, 47, 46]

    diff_opt = opt_lmks[:, upper_eyelid_lmk_ids, :] - \
        opt_lmks[:, lower_eyelid_lmk_ids, :]
    diff_target = target_lmks[:, upper_eyelid_lmk_ids,
                              :] - target_lmks[:, lower_eyelid_lmk_ids, :]
    loss = torch.sqrt(torch.sum(torch.square(
        diff_opt - diff_target.detach()), dim=-1))
    return torch.mean(torch.mean(loss, 1))


def mouth_loss(opt_lmks, target_lmks, image_size):
    mouth_ids = [i for i in range(49, 68)]
    h, w = image_size
    size = torch.tensor(
        [1 / w, 1 / h]).float().to(target_lmks.device)[None, None, ...]
    diff_diff = opt_lmks[:, mouth_ids, :] - target_lmks[:, mouth_ids, :]
    loss = torch.sqrt(torch.sum(torch.square(diff_diff * size), dim=-1))
    return torch.mean(torch.mean(loss, 1))


def pixel_loss(opt_img, target_img, mask=None):
    if mask is None:
        mask = torch.ones_like(opt_img)
    n_pixels = torch.sum(mask[:, 0, ...].int()).detach().float()
    loss = (mask * (opt_img - target_img)).abs()
    loss = torch.sum(loss) / n_pixels
    return loss


def reg_loss(params):
    return torch.mean(torch.sum(torch.square(params), dim=1))


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + \
        (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device)
                     * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def plot_verts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    kpts = kpts.copy().astype(np.int32)
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, c, 5)

    return image

def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color='g', isScale=False):
    # images,landmarks, gt_landmarks: 3,h,w
    if len(images.shape) == 3:
        images = images[None]
        landmarks = landmarks[None]
        if gt_landmarks is not None:
            gt_landmarks = gt_landmarks[None]

    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()
        image = (image * 255)
        if isScale:
            predicted_landmark = predicted_landmarks[i]
            predicted_landmark[:, 0] = (
                predicted_landmark[:, 0] + 1.0) / 2. * image.shape[1]
            predicted_landmark[:, 1] = (
                predicted_landmark[:, 1] + 1.0) / 2. * image.shape[0]
        else:
            predicted_landmark = predicted_landmarks[i]

        if predicted_landmark.shape[0] != 68:
            image_landmarks = plot_all_kpts(image, predicted_landmark, color)
        elif predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks, gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2, 'r')
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks, gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2, 'r')

        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.  # , dtype=torch.float32)
    return vis_landmarks




def plot_kpts(image, kpts, color='r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    c = (0, 100, 255)
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)

    image = image.copy()
    kpts = kpts.copy()

    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1
    # for j in range(kpts.shape[0] - 17):
    for j in range(kpts.shape[0]):
        # i = j + 17
        st = kpts[j, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)
        if j in end_list:
            continue
        ed = kpts[j + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(
            ed[0]), int(ed[1])), (255, 255, 255), 1)

    return image


def plot_all_kpts(image, kpts, color='b'):
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    elif color == 'p':
        c = (255, 100, 100)

    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)

    return image


def get_gaussian_pyramid(levels, images):
    pyramid = []
    for k, level in enumerate(reversed(levels)):
        image_size, iters = level
        size = [int(image_size[0]), int(image_size[1])]
        images = F.interpolate(
            images, size, mode='bilinear', align_corners=False)

        if k == 0:
            pyramid.append((images, iters, size, image_size))
        else:
            images = gaussian_blur(images, [9, 9])
            pyramid.append((images.detach(), iters, size, image_size))

    return list(reversed(pyramid))


def generate_triangles(h, w, margin_x=2, margin_y=5, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    for x in range(margin_x, w - 1 - margin_x):
        for y in range(margin_y, h - 1 - margin_y):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


def get_aspect_ratio(images):
    h, w = images.shape[2:4]
    ratio = w / h
    if ratio > 1.0:
        aspect_ratio = torch.tensor([1. / ratio, 1.0]).float().cuda()[None]
    else:
        aspect_ratio = torch.tensor([1.0, ratio]).float().cuda()[None]
    return aspect_ratio


def is_optimizable(name, param_groups):
    for param in param_groups:
        if name in param['name']:
            return True
    return False


def merge_views(views):
    # img: c,h,w
    grid = []
    for view in views:
        grid.append(np.concatenate(view, axis=2))
    grid = np.concatenate(grid, axis=1)

    # tonemapping
    grid_image = (grid.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
    return grid_image


def dump_point_cloud(name, view):
    _, _, h, w = view.shape
    np.savetxt(f'pc_{name}.xyz', view.permute(0, 2, 3, 1).reshape(
        h * w, 3).detach().cpu().numpy(), fmt='%f')
