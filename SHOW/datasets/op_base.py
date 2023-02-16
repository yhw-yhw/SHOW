
from scipy.io import savemat, loadmat
from collections import namedtuple
from collections import defaultdict
import numpy as np
import json
import torch
import glob
import cv2
import os.path as osp
import os
from pathlib import Path
from typing import Union
from functools import reduce, partial
from easydict import EasyDict
from loguru import logger

return_item_tuple = namedtuple(
    'return_item_tuple',
    ['keypoints_2d', 'gender_gt']
)
return_item_tuple.__new__.__defaults__ = (None,)*len(return_item_tuple._fields)


class op_base(object):

    def get_joint_weights(self) -> torch.Tensor:
        # @return optim_weights: [1,135,1]
        self.optim_weights = torch.ones(
            self.num_joints +
            2 * self.use_hands +
            51 * self.use_face +
            17 * self.use_face_contour,
            dtype=self.dtype
        ).to(self.device)

        self.optim_weights = self.optim_weights[None, ..., None]

        return self.optim_weights

    def get_smplx_to_o3d_R(self,) -> np.ndarray:
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        return R

    def get_smplx_to_pyrender_R(self,) -> np.ndarray:
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        return R

    def get_smplx_to_pyrender_T(self,) -> np.ndarray:
        cam_transl = self.pp.cam_transl
        if isinstance(cam_transl, torch.Tensor):
            cam_transl = cam_transl.cpu().detach().numpy()
        cam_t = cam_transl[0].copy()
        cam_t[0] *= -1
        return cam_t

    def get_smplx_to_o3d_T(self,) -> np.ndarray:
        # assert(self.pp.cam_transl is not None)
        return self.cvt_pixie_cam_to_o3d(self.pp.cam_transl)

    def cvt_pixie_cam_to_o3d(self, cam_transl) -> np.ndarray:
        if isinstance(cam_transl, torch.Tensor):
            cam_transl = cam_transl.cpu().detach().numpy()
        cam_t = cam_transl[0].copy()
        cam_t[0] *= -1
        cam_t[1] *= -1
        return cam_t

    def get_smplx_to_pyrender_K(self, cam_transl) -> np.ndarray:
        if isinstance(cam_transl, torch.Tensor):
            T = cam_transl.detach().cpu().numpy()
        T[1] *= -1
        K = np.eye(4)
        K[:3, 3] = T
        return K

    def get_smplx_to_pyrender_K2(self,) -> np.ndarray:
        from smplx.lbs import transform_mat
        R = self.get_smplx_to_pyrender_R()
        T = self.get_smplx_to_pyrender_T()
        K = transform_mat(torch.from_numpy(R).unsqueeze(0),
                          torch.from_numpy(T).unsqueeze(0).unsqueeze(dim=-1)
                          ).numpy()[0]
        return K

    @classmethod
    def smpl_to_openpose(
        cls,
        use_face=True,
        use_hands=True,
        use_face_contour=True,
    ) -> np.ndarray:
        # smplx-->openpose_body_25
        body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                63, 64, 65], dtype=np.int32)  # 25
        mapping = [body_mapping]

        if use_hands:
            lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                      67, 28, 29, 30, 68, 34, 35, 36, 69,
                                      31, 32, 33, 70], dtype=np.int32)
            rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                      43, 44, 45, 73, 49, 50, 51, 74, 46,
                                      47, 48, 75], dtype=np.int32)
            mapping += [lhand_mapping, rhand_mapping]

        if use_face:
            #  end_idx = 127 + 17 * use_face_contour
            face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                     dtype=np.int32)
            mapping += [face_mapping]

        return np.concatenate(mapping)

    @classmethod
    def read_keypoints(
        cls,
        keypoint_fn=None,
        use_hands=True,
        use_face=True,
        use_face_contour=True
    ):

        with open(keypoint_fn) as f:
            data = json.load(f)

        # body_size_list = []
        keypoints = []
        gender_gt = []

        for _, person_data in enumerate(data['people']):
            body_keypoints = np.array(
                person_data['pose_keypoints_2d'], dtype=np.float32)
            body_keypoints = body_keypoints.reshape([-1, 3])

            if use_hands:
                left_hand_keyp = np.array(
                    person_data['hand_left_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])
                right_hand_keyp = np.array(
                    person_data['hand_right_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])
                body_keypoints = np.concatenate(
                    [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)

            if use_face:
                face_keypoints = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
                contour_keyps = np.array(
                    [], dtype=body_keypoints.dtype).reshape(0, 3)
                if use_face_contour:
                    contour_keyps = np.array(
                        person_data['face_keypoints_2d'],
                        dtype=np.float32).reshape([-1, 3])[:17, :]
                body_keypoints = np.concatenate(
                    [body_keypoints, face_keypoints, contour_keyps], axis=0)
            keypoints.append(body_keypoints)

            if 'gender_gt' in person_data:
                gender_gt.append(person_data['gender_gt'])

        # keypoints: [B,135,3]
        return return_item_tuple(
            keypoints,
            gender_gt
        )
