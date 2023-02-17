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

import numpy as np
import torch
import torch.nn.functional as F


def lmk2d_to_bbox(lmks,h,w,bb_scale=2.0):
    # lmks:68,2
    x_min, x_max, y_min, y_max = np.min(lmks[:, 0]), np.max(lmks[:, 0]), np.min(lmks[:, 1]), np.max(lmks[:, 1])
    x_center, y_center = int((x_max + x_min) / 2.0), int((y_max + y_min) / 2.0)
    size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))
    xb_min, xb_max, yb_min, yb_max = max(x_center - size // 2, 0), min(x_center + size // 2, w - 1), \
                                     max(y_center - size // 2, 0), min(y_center + size // 2, h - 1)

    yb_max = min(yb_max, h-1)
    xb_max = min(xb_max, w-1)
    yb_min = max(yb_min, 0)
    xb_min = max(xb_min, 0)
    return [xb_min,yb_min, xb_max,yb_max]


def landmark_crop(image, lmks, dense_lmk, bb_scale=2.0):
    h, w = image.shape[1:]
    
    xb_min,yb_min, xb_max,yb_max=lmk2d_to_bbox(lmks,h,w,bb_scale=bb_scale)

    if (xb_max - xb_min) % 2 != 0:
        xb_min += 1

    if (yb_max - yb_min) % 2 != 0:
        yb_min += 1

    cropped_image = crop_image(image, xb_min, yb_min, xb_max, yb_max)
    cropped_image_lmks = np.vstack((lmks[:, 0] - xb_min, lmks[:, 1] - yb_min)).T
    cropped_dense_lmk = np.vstack((dense_lmk[:, 0] - xb_min, dense_lmk[:, 1] - yb_min)).T
    return cropped_image_lmks, cropped_dense_lmk, cropped_image, {'xb_min': xb_min, 'xb_max': xb_max, 'yb_min': yb_min, 'yb_max': yb_max}


def crop_image(image, x_min, y_min, x_max, y_max):
    # image：C,H,W or c,y,x
    return image[:, max(y_min, 0):min(y_max, image.shape[1] - 1), max(x_min, 0):min(x_max, image.shape[2] - 1)]


def squarefiy(image, lmk, dense_lmk, size=512):
    _, h, w = image.shape
    px = py = 0
    max_wh = max(w, h)
    
    if w != h:
        px = int((max_wh - w) / 2)
        py = int((max_wh - h) / 2)
        image = F.pad(image, (px, px, py, py), 'constant', 0)

    img = F.interpolate(image[None], (size, size), mode='bilinear', align_corners=False)[0]
    
    if False:
        scale_x = size / (w + px)
        scale_y = size / (h + py)
        lmk[:, 0] *= scale_x
        lmk[:, 1] *= scale_y
        dense_lmk[:, 0] *= scale_x
        dense_lmk[:, 1] *= scale_y
    else:
        lmk[:, 0] = (lmk[:, 0] + px)*size/max_wh
        lmk[:, 1] = (lmk[:, 1] + py)*size/max_wh
        dense_lmk[:, 0] = (dense_lmk[:, 0] + px)*size/max_wh
        dense_lmk[:, 1] = (dense_lmk[:, 1] + py)*size/max_wh

    return img, lmk, dense_lmk, px, py


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        input_image = torch.clamp(input_image, -1.0, 1.0)
        image_tensor = input_image.data
    else:
        return input_image.reshape(3, 512, 512).transpose()
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def get_heatmap( values):
    import cv2
    l2 = tensor2im(values)
    l2 = cv2.cvtColor(l2, cv2.COLOR_RGB2BGR)
    l2 = cv2.normalize(l2, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(l2, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(cv2.addWeighted(heatmap, 0.75, l2, 0.25, 0).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)
    return heatmap

def crop_image_bbox(image, lmks, dense_lmk, bbox):
    xb_min = bbox['xb_min']
    yb_min = bbox['yb_min']
    xb_max = bbox['xb_max']
    yb_max = bbox['yb_max']
    cropped = crop_image(image, xb_min, yb_min, xb_max, yb_max)
    cropped_image_lmks = np.vstack((lmks[:, 0] - xb_min, lmks[:, 1] - yb_min)).T
    cropped_image_dense_lmk = np.vstack((dense_lmk[:, 0] - xb_min, dense_lmk[:, 1] - yb_min)).T
    return cropped_image_lmks, cropped_image_dense_lmk, cropped
