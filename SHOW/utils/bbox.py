import torch
import numpy as np
import cv2


def msk_to_xywh(msk):
    """
    calculate box [left upper width height] from mask.
    :param msk: nd.array, single-channel or 3-channels mask
    :return: float, iou score    
    """
    if len(msk.shape) > 2:
        msk = msk[..., 0]
    nonzeros = np.nonzero(msk.astype(np.uint8))
    u, l = np.min(nonzeros, axis=1)
    b, r = np.max(nonzeros, axis=1)
    return np.array((l, u, r-l+1, b-u+1))

def msk_to_xyxy(msk):
    """
    calculate box [left upper right bottom] from mask.
    :param msk: nd.array, single-channel or 3-channels mask
    :return: float, iou score    
    """
    if len(msk.shape) > 2:
        msk = msk[..., 0]
    nonzeros = np.nonzero(msk.astype(np.uint8))
    u, l = np.min(nonzeros, axis=1)
    b, r = np.max(nonzeros, axis=1)
    return np.array((l, u, r+1, b+1))

def get_edges(msk):
    """
    get edge from mask
    :param msk: nd.array, single-channel or 3-channel mask
    :return: edges: nd.array, edges with same shape with mask
    """
    msk_sp = msk.shape
    if len(msk_sp) == 2:
        c = 1 # single channel
    elif (len(msk_sp) == 3) and (msk_sp[2] == 3):
        c = 3 # 3 channels
        msk = msk[:, :, 0] != 0        
    edges = np.zeros(msk_sp[:2])
    edges[:-1, :] = np.logical_and(msk[:-1, :] != 0, msk[1:, :] == 0) + edges[:-1, :]
    edges[1:, :] = np.logical_and(msk[1:, :] != 0, msk[:-1, :] == 0) + edges[1:, :]
    edges[:, :-1] = np.logical_and(msk[:, :-1] != 0, msk[:, 1:] == 0) + edges[:, :-1]
    edges[:, 1:] = np.logical_and(msk[:, 1:] != 0, msk[:, :-1] == 0) + edges[:, 1:]
    if c == 3:
        return np.dstack((edges, edges, edges))
    else:
        return edges