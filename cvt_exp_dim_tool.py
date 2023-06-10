from dataclasses import dataclass
import tyro
import trimesh
import mmcv
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch
import smplx

DEFAULT_SMPLX_CONFIG = dict(
    create_global_orient=True,
    create_body_pose=True,
    create_betas=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_expression=True,
    create_jaw_pose=True,
    create_leye_pose=True,
    create_reye_pose=True,
    create_transl=True,
)

def load_smplx_model(device='cuda', **kwargs):
    body_model = smplx.create(
        **DEFAULT_SMPLX_CONFIG,
        **kwargs).to(device=device)
    return body_model



@dataclass
class Config:
    lr = 0.1
    device = 'cuda'
    target_exp_dim = 75
    pkl_path = 'C:/Users/lithiumice/Downloads/mica_all.pkl'
    model_path='../models/smplx/SMPLX_MALE_shape2019_exp2020.npz'
    


args = tyro.cli(Config)
device = torch.device(args.device)
dtype = torch.float32
comm_cfg = EasyDict(
    dtype=dtype,
    model_path=args.model_path,
    use_face_contour=True,
    flat_hand_mean=False,
    use_hands=True,
    use_face=True,
)

# all_var = mmcv.load(args.pkl_path)
with open(args.pkl_path, 'rb') as f:
    import pickle
    all_var = pickle.load(f, encoding='latin1')
    
if isinstance(all_var, list):
    all_var = all_var[0]
all_var = EasyDict(all_var)
comm_cfg.use_pca=True
batch_size = comm_cfg.batch_size=all_var.batch_size
num_pca_comps=all_var.left_hand_pose.shape[1]
comm_cfg.num_pca_comps=num_pca_comps
st_et_list = [(0, all_var.batch_size)]
save_data = np.zeros((batch_size, 165+args.target_exp_dim), dtype=np.float32)
    
    
for opt_idx, (start_frame, end_frame) in enumerate(st_et_list):
    
    print(f'opt_idx:{opt_idx}, start_frame:{start_frame}, end_frame:{end_frame})')
    
    comm_cfg.batch_size = batch_size = end_frame-start_frame
    
    all_var = {k:v[start_frame:end_frame] for k,v in all_var.items() if isinstance(v, np.ndarray) and len(v.shape)>=1 and v.shape[0]==all_var.batch_size}
    all_var['betas'] = np.zeros((batch_size, 100), dtype=np.float32)
    all_var['transl'] = np.zeros((batch_size, 3), dtype=np.float32)
    all_var = EasyDict(all_var)

    smlx2flame_idx = torch.from_numpy(
        np.load(
            '../data/SMPL-X__FLAME_vertex_ids.npy'
        )
    ).to(dtype=torch.long, device=device)

    with open('../data/generic_model.pkl', 'rb') as f:
        import pickle
        class Struct(object):
            def __init__(self, **kwargs):
                for key, val in kwargs.items():
                    setattr(self, key, val)
        ss = pickle.load(f, encoding='latin1')
        ff = Struct(**ss)
        faces = ff.f
        
    body_model = load_smplx_model(**comm_cfg,
                                    num_betas=all_var.betas.shape[1],
                                    num_expression_coeffs=all_var.expression.shape[1],
                                    ).to(device)

    target_body_model = load_smplx_model(**comm_cfg,
                                    num_betas=all_var.betas.shape[1],
                                    num_expression_coeffs=args.target_exp_dim,
                                    ).to(device)

    smplx_params = dict(
        body_pose=all_var.body_pose_axis,
        betas=all_var.betas,
        global_orient=all_var.global_orient,
        transl=all_var.transl,
        left_hand_pose=all_var.left_hand_pose,
        right_hand_pose=all_var.right_hand_pose,
        jaw_pose=all_var.jaw_pose,
        leye_pose=all_var.leye_pose,
        reye_pose=all_var.reye_pose,
        expression=all_var.expression,
    )
    for key, val in smplx_params.items():
        smplx_params[key] = torch.from_numpy(smplx_params[key]).to(device)
            
    model_output = body_model(return_verts=True, **smplx_params)
    target_vertices_ = model_output.vertices.detach()#important!
    target_vertices_ = target_vertices_[:, smlx2flame_idx]

    smplx_params['expression']=nn.Parameter(torch.zeros(batch_size,args.target_exp_dim).detach().to(device).type(dtype))
    opt_params=[smplx_params['expression']]
    optimizer = torch.optim.LBFGS(opt_params, 
                                  lr=args.lr, max_iter=20, 
                                  history_size=10, tolerance_grad=1e-05)
    loss_fn = torch.nn.MSELoss()

    max_iter = 50
    loss_history = []
    pbar = tqdm(range(max_iter), desc='Optimizing expression parameters')
    def closure():
        optimizer.zero_grad()
        model_output = target_body_model(return_verts=True, **smplx_params)
        vertices_ = model_output.vertices
        vertices_ = vertices_[:, smlx2flame_idx]
        loss = 1000 * 1000 * loss_fn(vertices_, target_vertices_)
        ss = (f'Iteration {i}, loss {loss.item()}')
        pbar.set_description(ss)
        
        loss_history.append(loss.item())
        loss.backward()
        return loss


    def rel_change(prev_val, curr_val):
        return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])
    prev_loss = None
    from SHOW.utils.metric import MeterBuffer
    meter = MeterBuffer(window_size=6)
    for i in pbar:
        loss = optimizer.step(closure)
        if i > 1 and prev_loss is not None:
            loss_rel_change = rel_change(
                prev_loss, loss.item())
            meter.update({'rel': loss_rel_change})
            if meter['rel'].avg <= 1e-09:
                logger.warning('rel exit')
                break
        if all([torch.abs(var.grad.view(-1).max()).item() < 1e-06
                for var in opt_params if var.grad is not None]):
            logger.warning('small grad')
            break
        prev_loss = loss.item()
        
        
    # smplx_params****
    # jaw_pose=i[:,0:3],
    # leye_pose=i[:,3:6],
    # reye_pose=i[:,6:9],
    # global_orient=i[:,9:12],
    # body_pose_axis=i[:,12:75],
    # left_hand_pose=i[:,75:120],
    # right_hand_pose=i[:,120:165],
    # expression=i[:,165:]

    save_data[start_frame:end_frame] = np.concatenate([smplx_params['jaw_pose'].detach().cpu().numpy(),
                                smplx_params['leye_pose'].detach().cpu().numpy(),
                                smplx_params['reye_pose'].detach().cpu().numpy(),
                                smplx_params['global_orient'].detach().cpu().numpy(),
                                smplx_params['body_pose'].detach().cpu().numpy(),
                                smplx_params['left_hand_pose'].detach().cpu().numpy(),
                                smplx_params['right_hand_pose'].detach().cpu().numpy(),
                                smplx_params['expression'].detach().cpu().numpy()],axis=1)

    print(f'save_data.shape:{save_data.shape}')

    model_output = target_body_model(return_verts=True, **smplx_params)
    vertices_ = model_output.vertices
    vertices_ = vertices_[:, smlx2flame_idx]

    target_mesh = trimesh.Trimesh(vertices_[0].detach().cpu().numpy(), faces)
    _ = target_mesh.export(f'target_mesh_{args.target_exp_dim}.obj')

    ref_mesh = trimesh.Trimesh(target_vertices_[0].detach().cpu().numpy(), faces)
    _ = ref_mesh.export(f'ref_mesh_{args.target_exp_dim}.obj')
    
np.save(f'rich_expdim{args.target_exp_dim}.npy',save_data)
        