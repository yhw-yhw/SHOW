import os
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from smplx.lbs import vertices2landmarks
import SHOW


from pytorch3d.transforms import so3_exp_map
from torchvision.transforms.functional import gaussian_blur
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, Transform3d, axis_angle_to_matrix
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras
from enum import Enum

def temporary_loss(o_w, i_w, gmof, param):
    return (o_w ** 2) * (gmof(
        i_w*(param[2:, ...] + param[:-2, ...] - 2 * param[1:-1, ...]))).mean()


def cal_deg_delta(deta, deg_interval):
    # deg_loss=0

    theta = torch.arctan((deta[:, 2]) / (deta[:, 1]))
    theta_deg = torch.rad2deg(theta)

    diff_up = theta_deg - deg_interval[1]
    diff_up = torch.where(diff_up > 0, diff_up, torch.zeros(
        diff_up.shape, device=diff_up.device))
    diff_down = deg_interval[0] - theta_deg
    diff_down = torch.where(diff_down > 0, diff_down, torch.zeros(
        diff_down.shape, device=diff_down.device))

    return diff_up, diff_down, theta_deg


def get_body_height(verts, faces):
    if verts.shape[0] == 0:
        return 0.0
    else:
        head_faces = faces[2581]
        head_verts = verts[head_faces]
        top_head_v = 0.8277337276382795 * head_verts[0] + 0.1422200962169292 * head_verts[1] + \
            0.030046176144791284 * head_verts[2]

        feet_faces = faces[15605]
        feet_verts = verts[feet_faces]
        bot_feet_v = feet_verts[1]

        return top_head_v[1] - bot_feet_v[1]


def compute_mass(tris, DENSITY=1):
    ''' Computes the mass from volume and average body density
    '''

    x = tris[:, :, :, 0]
    y = tris[:, :, :, 1]
    z = tris[:, :, :, 2]
    volume = (
        -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
        x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
        x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
        x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
        x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
        x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
    ).sum(dim=1).abs() / 6.0
    return volume * DENSITY


def cvt_dict_to_grad(params,device,dtype):
    # params：{ np.ndarray/torch.Tensor(cpu/gpu) }
    for key in params.keys():
        if isinstance(params[key],np.ndarray):
            params[key]=torch.from_numpy(params[key])
        params[key]=nn.Parameter(params[key].clone().detach().to(device).type(dtype))
    return params


def cal_model_output(vposer=None,body_model=None,body_params=None):
    cur_pose = None
    if vposer is not None:
        cur_pose = vposer.decode(
            body_params['pose_embedding'],
            output_type='aa'
        ).view(-1,63)
        cur_bs=cur_pose.shape[0]
    
    model_output = body_model(
        return_verts=True,
        return_full_pose=True,

        betas=body_params['betas'],
        jaw_pose=body_params['jaw_pose'],
        leye_pose=body_params['leye_pose'],
        reye_pose=body_params['reye_pose'],
        expression=body_params['expression'],
    
        transl=body_params['transl'],
        body_pose=cur_pose if vposer is not None else body_params['body_pose_axis'],
        global_orient=body_params['global_orient'],
        left_hand_pose=body_params['left_hand_pose'],
        right_hand_pose=body_params['right_hand_pose'],
    ) 
    return model_output,cur_pose

def get_tpose_vertice(body_model,betas,**kwargs):
    batch_size=body_model.batch_size
    tpose_body = body_model(
        return_verts=True,
        body_pose=torch.zeros(batch_size, 63).type_as(betas),
        betas=betas.expand(batch_size, -1),
        **kwargs
    )
    tpose_vertices = tpose_body.vertices[0]
    return tpose_vertices
    
def cal_smplx_head_transl(tpose_vertices,smplx2flame_idx):
    pre_flame_vertices = torch.index_select(
        tpose_vertices, 0, smplx2flame_idx)
    smplx_shape_mean = pre_flame_vertices.mean(0)
    return smplx_shape_mean

class GMoF(nn.Module):
    def __init__(self, rho=100):
        super().__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist

local_step=0
def create_closure(
    optimizer,
    vposer,body_model, 
    body_params,
    camera_org,
    
    lmk_faces_idx,
    lmk_bary_coords,
    
    op_2dkpts,op_j_weight,
    op_gt_conf,op_valid_flag,
    
    robustifier,
    curr_weights,
    batch_size,
    
    lmk_gt_inner,
    lmk_gt_outter,
    mp_valid_flag,
    mp_gt_lmk_2d,
    
    smplifyx_cfg,
    
    deca_valid_flag,
    height, width,
    speaker_shape_vertices,
    smplx2flame_idx,
    face_mask,
    angle_prior,
    device,
    loss_closure_finish_callback,
    
    renderer_silhouette,

    
    edt=None,
    compute_edges=None,
    meta_data={},

    search_tree=None,
    filter_faces=None,
    pen_distance=None,
):
    def closure():
        loss_scale = 1080*720/(height*width)
        
        losses = {}
        metric = {}

        optimizer.zero_grad()
        I = torch.eye(3)[None].to(device)

        model_output, _ = cal_model_output(vposer, body_model, body_params)

        model_output_vertices = model_output.vertices.float()
        body_model_face = body_model.faces_tensor.long()
        model_joints = model_output.joints
        batch_size = model_joints.shape[0]

        lmkall_proj = camera_org.transform_points_screen(model_joints)[
            :, :, :2]
        lmk_face68 = lmkall_proj[:, 67:67 + 51+17, :]
        lmk_pre_outter = lmk_face68[:, 51:, :]
        lmk_pre_inner = lmk_face68[:, :51, :]

        tpose_vertices = get_tpose_vertice(
            body_model, body_params['betas'])

        # torch.Size([192, 105, 2])
        mp_pre_lmk = vertices2landmarks(
            model_output_vertices, body_model_face, lmk_faces_idx, lmk_bary_coords)
        mp_pre_lmk_2d = camera_org.transform_points_screen(mp_pre_lmk)[
            :, :, :2]
        
        
        ###################################
        if smplifyx_cfg.use_silhouette_loss and curr_weights.wl_silhouette != 0:
            alpha_list=[]
            opt_bs_at_a_time=smplifyx_cfg.o3d_opt_bs_at_a_time
            org_B = model_output_vertices.shape[0]
            iter_num=org_B//opt_bs_at_a_time
            if iter_num-org_B/opt_bs_at_a_time!=0:
                iter_num+=1
                
            for i in range(iter_num):#i=0
                vertices=model_output_vertices[i*opt_bs_at_a_time:(i+1)*opt_bs_at_a_time]
                B = vertices.shape[0]
                V = vertices.shape[1]
                
                faces = body_model_face[None].repeat(B, 1, 1)
                meshes_world = Meshes(verts=vertices.float(), faces=faces.long())
                
                lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
                images_predicted = renderer_silhouette(meshes_world, cameras=camera_org, lights=lights)
                predicted_silhouette = images_predicted[..., 3]
                alpha_list.append(predicted_silhouette)
                
            pred_mask=torch.cat(alpha_list, dim=0)
            temp=compute_edges(pred_mask)
            diff=curr_weights.wl_silhouette*(edt*temp).sum()/batch_size
            losses['sil_loss']=diff
            meta_data['step']=meta_data['step']+1
        
        ###################################
        #torch.Size([192, 135, 2])
        kpts_diff = lmkall_proj-op_2dkpts
        kpts_diff_rb = robustifier(kpts_diff)*(op_j_weight*op_gt_conf**2)

        if curr_weights.hand_joints_weight != 0:
            losses['loss_kpts_hand'] = 2*curr_weights.hand_joints_weight*torch.sum(
                kpts_diff_rb[:, 25:67, :][op_valid_flag.bool(), :, :]) / batch_size
            metric['hands_joints'] = losses['loss_kpts_hand'].clone().detach() / \
                curr_weights.hand_joints_weight

        if curr_weights.body_joints_weight != 0:
            losses['loss_kpts_body'] = curr_weights.body_joints_weight*torch.sum(
                kpts_diff_rb[:, :25, :][op_valid_flag.bool(), :, :]) / batch_size
            metric['body_joints'] = losses['loss_kpts_body'].clone().detach() / \
                curr_weights.body_joints_weight

        ###################################
        if curr_weights.w_deca_inner != 0:
            losses['loss_deca_inner'] = curr_weights.w_deca_inner*torch.sum(
                robustifier(lmk_pre_inner-lmk_gt_inner)[deca_valid_flag.bool(), :, :]) / batch_size
        if curr_weights.w_deca_outter != 0:
            losses['loss_deca_outter'] = curr_weights.w_deca_outter*torch.sum(
                robustifier(lmk_pre_outter-lmk_gt_outter)[deca_valid_flag.bool(), :, :]) / batch_size

        ###################################
        if curr_weights.mp_weight != 0 and smplifyx_cfg.use_mp_loss:
            if mp_valid_flag.sum()!=0:
                image_size = (height, width)
                mp_diff = robustifier(mp_gt_lmk_2d - mp_pre_lmk_2d)
                losses['mp_loss'] = 0.4*((curr_weights.mp_weight**2)
                                        * mp_diff[mp_valid_flag.bool(), :, :]).sum(1).mean()
                # losses['lmk_mount'] = SHOW.utils.mouth_loss(lmk_face68, image_lmks, image_size)
                # losses['lmk_oval'] = SHOW.utils.lmk_loss(lmk_face68[:, -17:, ...], image_lmks[:, -17:, ...], image_size)

        ###################################
        if smplifyx_cfg.use_const_velocity and (curr_weights.w_s3d_body != 0 or curr_weights.w_s3d_hand != 0):
            # torch.Size([190, 135, 3])
            smooth_j3d_total = robustifier(
                1000 * (model_joints[2:, :, :] + model_joints[:-2, :, :] - 2 * model_joints[1:-1, :, :]))

            if curr_weights.body_joints_weight != 0:
                losses['loss_s3d_body'] = (
                    curr_weights.w_s3d_body ** 2) * (smooth_j3d_total[:, :25, :]).sum(1).mean()
                
            if curr_weights.hand_joints_weight != 0:
                losses['loss_s3d_hand'] = (
                    curr_weights.w_s3d_hand ** 2) * (smooth_j3d_total[:, 25:67, :]).sum(1).mean()
                
            if curr_weights.transl_en:
                losses['loss_stransl'] = temporary_loss(
                    curr_weights.w_transl_smooth, 100, robustifier, body_params['transl'])
                
            if curr_weights.global_orient_en:
                losses['loss_sorient'] = temporary_loss(
                    curr_weights.w_orient_smooth, 10, robustifier, body_params['global_orient'])
            
            # if curr_weights.expression_en:
            #     losses['loss_sexp'] = temporary_loss(
            #         4, 5.0, lambda x: x.pow(2), body_params['expression'])

        ###################################
        if smplifyx_cfg.use_const_velocity and (curr_weights.w_s2d_body != 0 or curr_weights.w_s2d_hand != 0):
            smooth_j2d_total = (op_j_weight[:, :, :]*op_gt_conf[1:-1, :, :]**2) * robustifier(
                (lmkall_proj[2:, :, :] + lmkall_proj[:-2, :, :] - 2 * lmkall_proj[1:-1, :, :]))

            if curr_weights.body_joints_weight != 0:
                losses['loss_s2d_body'] = (
                    curr_weights.w_s2d_body ** 2) * (smooth_j2d_total[op_valid_flag.bool()[1:-1], :25, :]).mean()
            
            if curr_weights.hand_joints_weight != 0:
                losses['loss_s2d_hand'] = (
                    curr_weights.w_s2d_hand ** 2) * (smooth_j2d_total[op_valid_flag.bool()[1:-1], 25:67, :]).mean()

        ###################################
        if curr_weights.w_svposer != 0 and smplifyx_cfg.use_vposer_smoooth:
            losses['loss_svposer'] = 32*(curr_weights.w_svposer ** 2)*(
                body_params['pose_embedding'][1:, :] - (body_params['pose_embedding'][:-1, :])).pow(2).mean()

        ###################################
        if curr_weights.w_spca_hand != 0 and smplifyx_cfg.use_hand_pca_smoooth:
            losses['loss_spca_lhand'] = 12*(curr_weights.w_spca_hand ** 2)*(
                body_params['left_hand_pose'][1:, :] - (body_params['left_hand_pose'][:-1, :])).pow(2).mean()
            losses['loss_spca_rhand'] = 12*(curr_weights.w_spca_hand ** 2)*(
                body_params['right_hand_pose'][1:, :] - (body_params['right_hand_pose'][:-1, :])).pow(2).mean()

        ###################################
        if speaker_shape_vertices is not None and curr_weights.mica_weight != 0 and smplifyx_cfg.use_mica_shape:
            gt_mica_shape_vertices = speaker_shape_vertices.detach() * 0.001 + \
                body_params['mica_head_transl']
            pre_flame_vertices = torch.index_select(
                tpose_vertices, 0, smplx2flame_idx)
            mica_diff_row = pre_flame_vertices - gt_mica_shape_vertices
            mica_diff_row = torch.index_select(
                mica_diff_row, 0, face_mask.long())
            mica_loss = 1000000 * curr_weights.mica_weight * \
                (mica_diff_row.pow(2).sum())
            losses['mica_loss'] = mica_loss

            metric['mica_diff_max'] = torch.linalg.norm(
                mica_diff_row.clone().detach(), ord=2, dim=1).max()
            metric['mica_diff_mean'] = torch.linalg.norm(
                mica_diff_row.clone().detach(), ord=2, dim=1).mean()

        ###################################
        if curr_weights.w_body_weight != 0 and smplifyx_cfg.use_weight_constraint:
            target_density = 985
            target_weight = smplifyx_cfg.speaker_weight
            triangles = torch.index_select(
                tpose_vertices[None], 1,  # (5,10475,3)
                body_model_face.view(-1)).view(1, -1, 3, 3)  # 62724=20908*3
            body_weight = compute_mass(triangles, target_density)[0]
            losses['body_weight_loss'] = curr_weights.w_body_weight * \
                abs(100 * (body_weight - target_weight))
            if 1:
                metric['curr_body_weight'] = body_weight.clone().detach()

        ###################################
        if curr_weights.w_body_height != 0 and smplifyx_cfg.use_height_constraint:
            target_height = smplifyx_cfg.speaker_height
            body_height = get_body_height(
                tpose_vertices, body_model_face)
            losses['body_height_loss'] = curr_weights.w_body_height * \
                abs(1000 * (body_height - target_height))
            if 1:
                metric['curr_body_height'] = body_height.clone().detach()

        ###################################
        if curr_weights.w_deg_range != 0 and smplifyx_cfg.use_head_loss:
            body_angle_loss = 0
            delta = model_joints[:, 15] - model_joints[:, 12]  # [bs,2]
            diff_up, diff_down, theta_deg = cal_deg_delta(
                delta, smplifyx_cfg.head_deg_range)
            body_angle_loss += curr_weights.w_deg_range * \
                robustifier(diff_up).sum()
            body_angle_loss += curr_weights.w_deg_range * \
                robustifier(diff_down).sum()
            metric['theta_head_max'] = theta_deg.clone().detach().max()
            metric['theta_head_min'] = theta_deg.clone().detach().min()

            delta = model_joints[:, 9] - model_joints[:, 3]  # [bs,2]
            diff_up, diff_down, theta_deg = cal_deg_delta(
                delta, smplifyx_cfg.body_deg_range)
            body_angle_loss += curr_weights.w_deg_range * \
                robustifier(diff_up).sum()
            body_angle_loss += curr_weights.w_deg_range * \
                robustifier(diff_down).sum()
            metric['theta_body_max'] = theta_deg.clone().detach().max()
            metric['theta_body_min'] = theta_deg.clone().detach().min()
            losses['body_angle_loss'] = body_angle_loss

        ###################################

        linear_rot_left = (axis_angle_to_matrix(body_params['leye_pose']))
        linear_rot_right = (axis_angle_to_matrix(body_params['reye_pose']))
        losses['eyes_sym_reg'] = torch.sum(
            (linear_rot_right - linear_rot_left) ** 2)/batch_size
        losses['eyes_left_reg'] = torch.sum(
            (I - linear_rot_left) ** 2)/batch_size
        losses['eyes_right_reg'] = torch.sum(
            (I - linear_rot_right) ** 2)/batch_size

        ###################################
        losses['vposer_reg'] = torch.sum(
            body_params['pose_embedding'] ** 2) * curr_weights.body_pose_weight ** 2 / batch_size
        # losses['shape_reg'] = torch.sum(body_params['betas'] ** 2) *  curr_weights.shape_weight** 2 / batch_size
        losses['shape_reg'] = 1.5*torch.sum(model_output.betas ** 2) * \
            curr_weights.shape_weight ** 2 / batch_size
        losses['angle_prior'] = torch.sum(angle_prior(
            model_output.full_pose[:, 3:66])) * curr_weights.bending_prior_weight / batch_size
        losses['jaw_reg'] = torch.sum((body_params['jaw_pose'].mul(
            curr_weights.jaw_prior_weight)) ** 2) / batch_size
        # losses['jaw_reg'] = torch.sum((I - axis_angle_to_matrix(body_params['jaw_pose'])) ** 2) * curr_weights.jaw_prior_weight*16.0/batch_size
        losses['exp_reg'] = (body_params['expression'] ** 2).sum() * \
            curr_weights.expr_prior_weight ** 2/batch_size
        losses['lhand_reg'] = torch.sum(
            body_params['left_hand_pose'] ** 2) * curr_weights.hand_prior_weight ** 2/batch_size
        losses['rhand_reg'] = torch.sum(
            body_params['right_hand_pose'] ** 2) * curr_weights.hand_prior_weight ** 2/batch_size
        ###################################

        if (curr_weights.selfpen_weight!=0 and smplifyx_cfg.use_bvh):
            
            triangles = torch.index_select(
                model_output.vertices, 1,  # (5,10475,3)
                body_model_face).view(batch_size, -1, 3, 3)  # 62724=20908*3

            with torch.no_grad():
                collision_idxs = search_tree(triangles)

            if filter_faces is not None:
                collision_idxs = filter_faces(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum( pen_distance(triangles, collision_idxs) ) / batch_size

            metric['pen_loss'] = pen_loss*curr_weights.selfpen_weight
        ###################################
            
            
        all_loss = 0.
        for key in losses.keys():
            losses[key] = losses[key] * loss_scale
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        
        # all_loss*=0.7

        loss_closure_finish_callback(
            losses,metric
        )
        

        all_loss.backward(create_graph=False)
        return all_loss
    return closure