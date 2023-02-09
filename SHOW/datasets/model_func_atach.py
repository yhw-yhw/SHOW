import torch
import smplx
from smplx.body_models import SMPLXLayer
import numpy as np


# (PIXIE_init_hand-hand_mean)@inv_hand_comp=hand_pca_delta
# hand_pca_delta@hand_comp+hand_mean=PIXIE_init_hand
# hand_pca_full@hand_comp=PIXIE_init_hand


def hand_pca_to_axis(self, lhand_pca, rhand_pca):
    # device=self.left_hand_mean.device
    lhand_axis = torch.einsum('bi,ij->bj', [lhand_pca, self.left_hand_components])
    rhand_axis = torch.einsum('bi,ij->bj', [rhand_pca, self.right_hand_components])
    
    if not self.flat_hand_mean:
        lhand_axis=lhand_axis+self.left_hand_mean
        rhand_axis=rhand_axis+self.right_hand_mean
        
    return lhand_axis,rhand_axis
    
    
def hand_axis_to_pca(self, lhand_axis, rhand_axis):
    device=self.left_hand_mean.device
    
    if isinstance(lhand_axis, np.ndarray):
        lhand_axis = torch.from_numpy(lhand_axis)
    if isinstance(rhand_axis, np.ndarray):
        rhand_axis = torch.from_numpy(rhand_axis)
        
    lhand_axis = lhand_axis.reshape(-1, 45).to(device)
    rhand_axis = rhand_axis.reshape(-1, 45).to(device)
    
    if not self.flat_hand_mean:
        lhand_axis=lhand_axis-self.left_hand_mean
        rhand_axis=rhand_axis-self.right_hand_mean
        
    lhand_pca = torch.einsum('bi,ij->bj', [lhand_axis, self.l_comp])
    rhand_pca = torch.einsum('bi,ij->bj', [rhand_axis, self.r_comp])
    
    # return lhand_pca, rhand_pca
    return lhand_pca.to('cpu'), rhand_pca.to('cpu')


def atach_model_func(model):
    if not hasattr(model, 'hand_axis_to_pca'):
        setattr(model, 'hand_axis_to_pca',hand_axis_to_pca)
        
    if not hasattr(model, 'hand_pca_to_axis'):
        setattr(model, 'hand_pca_to_axis',hand_pca_to_axis)

    if not hasattr(model, 'l_comp'):
        l_comp = torch.linalg.pinv(model.left_hand_components)
        r_comp = torch.linalg.pinv(model.right_hand_components)
        setattr(model, 'l_comp', l_comp)
        setattr(model, 'r_comp', r_comp)