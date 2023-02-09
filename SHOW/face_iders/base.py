import os
import sys
import mmcv
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, NewType
from .builder import IDER


class ider_base(object):
    def get_all_emb(self, im: np.ndarray = None) -> np.ndarray:
        # im：bgr
        faces = self.app.get(im)
        return faces

    def get_face_emb(self, im: np.ndarray = None) -> np.ndarray:
        # im：bgr
        faces = self.app.get(im)
        emb = faces[0].normed_embedding
        return emb

    def cal_emb_sim(self, emb1: np.ndarray = None, emb2: np.ndarray = None) -> np.ndarray:
        return np.dot(emb1, emb2.T)

    def get(self, img):
        return self.app.get(img)


@IDER.register_module()
class insightface_ider(ider_base):
    def __init__(self, threshold=0.6, **kwargs):
        self.threshold = threshold

        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
