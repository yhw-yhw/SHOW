# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import pickle

import numpy as np
# Modified from smplx code for FLAME
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from .lbs import lbs



def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given FLAME parameters for shape, pose, and expression, this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        print("Creating the FLAME Decoder")
        with open(config.flame_geom_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.num_shape_params], shapedirs[:, :, 300:300 + config.num_exp_params]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long();
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Register default parameters
        self._register_default_params('neck_pose_params', 6)
        self._register_default_params('jaw_pose_params', 6)
        self._register_default_params('eye_pose_params', 12)
        self._register_default_params('shape_params', config.num_shape_params)
        self._register_default_params('expression_params', config.num_exp_params)

        # Static and Dynamic Landmark embeddings for FLAME
        # static_lmk_embedding = np.load(config.flame_static_lmk_path,allow_pickle=True)
        static_lmk_embedding = np.load(config.flame_static_lmk_path)
        dynamic_lmk_embedding = np.load(config.flame_dynamic_lmk_path, allow_pickle=True, encoding='latin1').item()
        self.num_lmk_angles = len(dynamic_lmk_embedding['lmk_face_idx'])

        self.register_buffer('lmk_faces_idx', torch.from_numpy(static_lmk_embedding['lmk_face_idx'][17:].astype(int)).to(torch.int64))
        self.register_buffer('lmk_bary_coords', torch.from_numpy(static_lmk_embedding['lmk_b_coords'][17:, :]).to(self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', torch.from_numpy(np.array(dynamic_lmk_embedding['lmk_face_idx']).astype(int)).to(torch.int64))
        self.register_buffer('dynamic_lmk_bary_coords', torch.from_numpy(np.array(dynamic_lmk_embedding['lmk_b_coords'])).to(self.dtype))

        neck_kin_chain = [];
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 6), 1,
                                     neck_kin_chain)

        rot_mats = rotation_6d_to_matrix(aa_pose.view(-1, 6)).view([batch_size, -1, 3, 3])
        rel_rot_mat = torch.eye(3, device=pose.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)

        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=(self.num_lmk_angles - 1) // 2)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-(self.num_lmk_angles - 1) // 2).to(dtype=torch.long)
        neg_vals = mask * (self.num_lmk_angles - 1) + (1 - mask) * ((self.num_lmk_angles - 1) // 2 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def forward(self, shape_params, trans_params=None, rot_params=None, neck_pose_params=None, jaw_pose_params=None, eye_pose_params=None, expression_params=None):

        """
            Input:
                trans_params: N X 3 global translation
                rot_params: N X 3 global rotation around the root joint of the kinematic tree (rotation is NOT around the origin!)
                neck_pose_params (optional): N X 3 rotation of the head vertices around the neck joint
                jaw_pose_params (optional): N X 3 rotation of the jaw
                eye_pose_params (optional): N X 6 rotations of left (parameters [0:3]) and right eyeball (parameters [3:6])
                shape_params (optional): N X number of shape parameters
                expression_params (optional): N X number of expression parameters
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        I = matrix_to_rotation_6d(torch.eye(3)[None].cuda())

        if trans_params is None:
            trans_params = torch.zeros(batch_size, 3).cuda()
        if rot_params is None:
            rot_params = I.clone()
        if neck_pose_params is None:
            neck_pose_params = I.clone()
        if jaw_pose_params is None:
            jaw_pose_params = self.jaw_pose_params.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose_params.expand(batch_size, -1)
        if shape_params is None:
            shape_params = self.shape_params.expand(batch_size, -1)
        if expression_params is None:
            expression_params = self.expression_params.expand(batch_size, -1)

        # Concatenate identity shape and expression parameters
        betas = torch.cat([shape_params, expression_params], dim=1)

        # The pose vector contains global rotation, and neck, jaw, and eyeball rotations
        full_pose = torch.cat([rot_params, neck_pose_params, jaw_pose_params, eye_pose_params], dim=1)

        # FLAME models shape and expression deformations as vertex offset from the mean face in 'zero pose', called v_template
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Use linear blendskinning to model pose roations
        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        # Add translation to the vertices
        vertices = vertices + trans_params[:, None, :]

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)

        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
        landmarks2d = self._vertices2landmarks(vertices, self.faces, lmk_faces_idx, lmk_bary_coords)

        return vertices, landmarks2d

    def _register_default_params(self, param_fname, dim):
        default_params = torch.zeros([1, dim], dtype=self.dtype, requires_grad=False)
        self.register_parameter(param_fname, nn.Parameter(default_params, requires_grad=False))


class FLAMETex(nn.Module):
    """
    current FLAME texture are adapted from BFM Texture Model
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        tex_space = np.load(config.tex_space_path)
        mu_key = 'MU'
        pc_key = 'PC'   
        n_pc = 199
        texture_mean = tex_space[mu_key].reshape(1, -1)
        texture_basis = tex_space[pc_key].reshape(-1, n_pc)
        n_tex = config.tex_params
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...] * 255.0
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...] * 255.0
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)
        self.image_size = config.image_size

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, self.image_size, mode='bilinear')
        texture = texture[:, [2, 1, 0], :, :]
        return texture
